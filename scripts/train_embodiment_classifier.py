#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import pickle
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from scar.environment import SCAR_ROOT


def log_step(message: str) -> None:
    print(f"[embodiment-cls] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small single-frame embodiment classifier on Robotwin LMDB "
            "episodes using the adaptation split manifest."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def canonical_embodiment_name(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    mapping = {
        "aloha": "aloha",
        "aloha-agilex": "aloha",
        "arx": "arx",
        "arx-x5": "arx",
        "franka": "franka",
        "ur5": "ur5",
    }
    if key not in mapping:
        raise KeyError(f"Unsupported embodiment label: {value}")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_rebase_scar_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.exists():
        return str(path.resolve())

    path_str = str(path).replace("\\", "/")
    needle = "/SCAR/"
    if needle in path_str:
        suffix = path_str.split(needle, 1)[1]
        return str((SCAR_ROOT / suffix).resolve())
    if path_str.endswith("/SCAR"):
        return str(SCAR_ROOT.resolve())
    return str(path)


def resolve_output_dir(cfg, config_path: Path) -> Path:
    output_dir = str(cfg.train.output_dir or "").strip()
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        SCAR_ROOT
        / "outputs"
        / "embodiment_image_classifier"
        / f"{config_path.stem}_{timestamp}"
    ).resolve()


@dataclass(frozen=True)
class DatasetSpec:
    class_name: str
    class_index: int
    dataset_path: str
    episode_indices: list[int]


@dataclass(frozen=True)
class FrameRecord:
    dataset_id: int
    episode_index: int
    frame_index: int
    class_index: int


class SmallEmbodimentCNN(nn.Module):
    def __init__(self, channels: list[int], num_classes: int) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"Expected exactly 3 channel widths, got {channels}")
        c1, c2, c3 = [int(value) for value in channels]
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, int(num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class LMDBFrameDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list[str],
        records: list[FrameRecord],
        *,
        cache_size: int = 8,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        self.dataset_paths = [str(Path(path).resolve()) for path in dataset_paths]
        self.records = list(records)
        self.cache_size = max(int(cache_size), 0)
        self.normalize_mean = None
        self.normalize_std = None
        if normalize_mean is not None and normalize_std is not None:
            self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32).view(3, 1, 1)
            self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32).view(3, 1, 1)
        self._env_by_dataset_id: dict[int, Any] = {}
        self._episode_cache: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        episode = self._load_episode(record.dataset_id, record.episode_index)
        observations = episode["observations"]
        image = torch.from_numpy(observations[record.frame_index]).to(torch.float32)
        if self.normalize_mean is not None and self.normalize_std is not None:
            image = (image - self.normalize_mean) / self.normalize_std
        return image, int(record.class_index)

    def _get_env(self, dataset_id: int):
        env = self._env_by_dataset_id.get(dataset_id)
        if env is None:
            import lmdb

            lmdb_path = Path(self.dataset_paths[dataset_id]) / "episodes.lmdb"
            env = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )
            self._env_by_dataset_id[dataset_id] = env
        return env

    def _load_episode(self, dataset_id: int, episode_index: int) -> dict[str, Any]:
        cache_key = (int(dataset_id), int(episode_index))
        episode = self._episode_cache.get(cache_key)
        if episode is not None:
            self._episode_cache.move_to_end(cache_key)
            return episode

        env = self._get_env(dataset_id)
        with env.begin() as txn:
            raw = txn.get(f"episode_{episode_index:04d}".encode())
        if raw is None:
            raise KeyError(
                f"Missing episode_{episode_index:04d} in dataset "
                f"{self.dataset_paths[dataset_id]}"
            )
        episode = pickle.loads(raw)
        if self.cache_size > 0:
            self._episode_cache[cache_key] = episode
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.cache_size:
                self._episode_cache.popitem(last=False)
        return episode


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest payload in {path}")
    return payload


def build_dataset_specs(
    manifest: dict[str, Any],
    *,
    embodiments: list[str],
    episode_key: str,
) -> list[DatasetSpec]:
    requested = [canonical_embodiment_name(name) for name in embodiments]
    entries_by_class: dict[str, dict[str, Any]] = {}
    for entry in list(manifest.get("datasets") or []):
        class_name = canonical_embodiment_name(entry["embodiment"])
        entries_by_class[class_name] = entry

    specs: list[DatasetSpec] = []
    for class_index, class_name in enumerate(requested):
        entry = entries_by_class.get(class_name)
        if entry is None:
            raise KeyError(f"No dataset entry found for embodiment={class_name}")
        dataset_path = maybe_rebase_scar_path(str(entry["dataset_path"]))
        episode_indices = [int(value) for value in list(entry[episode_key] or [])]
        specs.append(
            DatasetSpec(
                class_name=class_name,
                class_index=int(class_index),
                dataset_path=dataset_path,
                episode_indices=episode_indices,
            )
        )
    return specs


def build_frame_records(
    specs: list[DatasetSpec],
    *,
    frame_stride: int,
    max_frames_per_class: int,
    seed: int,
) -> tuple[list[str], list[FrameRecord], dict[str, Any]]:
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    import lmdb

    rng = random.Random(seed)
    dataset_paths: list[str] = []
    dataset_id_by_path: dict[str, int] = {}
    all_records: list[FrameRecord] = []
    summary: dict[str, Any] = {}

    for spec in specs:
        dataset_id = dataset_id_by_path.get(spec.dataset_path)
        if dataset_id is None:
            dataset_id = len(dataset_paths)
            dataset_id_by_path[spec.dataset_path] = dataset_id
            dataset_paths.append(spec.dataset_path)

        lmdb_path = Path(spec.dataset_path) / "episodes.lmdb"
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        class_records: list[FrameRecord] = []
        episode_lengths: list[int] = []
        with env.begin() as txn:
            for episode_index in spec.episode_indices:
                raw = txn.get(f"episode_{episode_index:04d}".encode())
                if raw is None:
                    raise KeyError(
                        f"Missing episode_{episode_index:04d} in dataset {spec.dataset_path}"
                    )
                episode = pickle.loads(raw)
                length = int(episode["observations"].shape[0])
                if length <= 0:
                    continue
                episode_lengths.append(length)
                for frame_index in range(0, length, frame_stride):
                    class_records.append(
                        FrameRecord(
                            dataset_id=int(dataset_id),
                            episode_index=int(episode_index),
                            frame_index=int(frame_index),
                            class_index=int(spec.class_index),
                        )
                    )
        env.close()

        if max_frames_per_class > 0 and len(class_records) > max_frames_per_class:
            class_records = rng.sample(class_records, int(max_frames_per_class))

        all_records.extend(class_records)
        summary[spec.class_name] = {
            "dataset_path": spec.dataset_path,
            "num_episodes": len(spec.episode_indices),
            "num_frames": len(class_records),
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        }

    return dataset_paths, all_records, summary


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            total_correct += int((preds == labels).sum().item())
            for true_class in range(num_classes):
                mask = labels == true_class
                if not mask.any():
                    continue
                pred_counts = torch.bincount(preds[mask], minlength=num_classes)
                confusion[true_class] += pred_counts.to(confusion.dtype).cpu()

    per_class_accuracy: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        denom = int(confusion[class_index].sum().item())
        if denom <= 0:
            per_class_accuracy[class_name] = 0.0
        else:
            per_class_accuracy[class_name] = float(
                confusion[class_index, class_index].item() / denom
            )

    return {
        "loss": float(total_loss / max(total_count, 1)),
        "top1": float(total_correct / max(total_count, 1)),
        "num_samples": int(total_count),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(class_names),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
    epoch_index: int,
    print_every: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    start_time = time.time()

    for step_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

        if print_every > 0 and step_index % print_every == 0:
            elapsed = time.time() - start_time
            log_step(
                f"epoch={epoch_index:03d} step={step_index:05d} "
                f"loss={total_loss / max(total_count, 1):.4f} "
                f"top1={total_correct / max(total_count, 1):.4f} "
                f"elapsed={elapsed:.1f}s"
            )

    return {
        "loss": float(total_loss / max(total_count, 1)),
        "top1": float(total_correct / max(total_count, 1)),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config_path)

    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg.train.device))
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    set_seed(int(cfg.train.seed))

    save_json(output_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
    log_step(f"config={config_path}")
    log_step(f"output_dir={output_dir}")

    manifest_path = Path(str(cfg.data.split_manifest)).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    class_names = [canonical_embodiment_name(name) for name in list(cfg.data.embodiments)]

    train_specs = build_dataset_specs(
        manifest,
        embodiments=class_names,
        episode_key=str(cfg.data.train_episode_key),
    )
    val_specs = build_dataset_specs(
        manifest,
        embodiments=class_names,
        episode_key=str(cfg.data.val_episode_key),
    )

    train_dataset_paths, train_records, train_summary = build_frame_records(
        train_specs,
        frame_stride=int(cfg.data.frame_stride),
        max_frames_per_class=int(cfg.data.max_train_frames_per_class),
        seed=int(cfg.train.seed),
    )
    val_dataset_paths, val_records, val_summary = build_frame_records(
        val_specs,
        frame_stride=int(cfg.data.frame_stride),
        max_frames_per_class=int(cfg.data.max_val_frames_per_class),
        seed=int(cfg.train.seed) + 1,
    )

    log_step(
        f"train_frames={len(train_records)} val_frames={len(val_records)} "
        f"classes={class_names}"
    )
    for class_name in class_names:
        train_info = train_summary[class_name]
        val_info = val_summary[class_name]
        log_step(
            f"{class_name}: train_eps={train_info['num_episodes']} "
            f"train_frames={train_info['num_frames']} "
            f"val_eps={val_info['num_episodes']} val_frames={val_info['num_frames']}"
        )

    normalize_mean = list(cfg.data.normalize_mean) if "normalize_mean" in cfg.data else None
    normalize_std = list(cfg.data.normalize_std) if "normalize_std" in cfg.data else None
    cache_size = int(cfg.data.cache_size)
    train_dataset = LMDBFrameDataset(
        train_dataset_paths,
        train_records,
        cache_size=cache_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    val_dataset = LMDBFrameDataset(
        val_dataset_paths,
        val_records,
        cache_size=cache_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    num_workers = int(cfg.train.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.eval_batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=num_workers > 0,
    )

    model = SmallEmbodimentCNN(
        channels=[int(value) for value in list(cfg.model.channels)],
        num_classes=len(class_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_top1 = float("-inf")
    best_epoch = -1
    best_val_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch_index in range(1, int(cfg.train.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler if amp_enabled else None,
            epoch_index=epoch_index,
            print_every=int(cfg.train.print_every),
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            class_names=class_names,
        )
        record = {
            "epoch": int(epoch_index),
            "train_loss": float(train_metrics["loss"]),
            "train_top1": float(train_metrics["top1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_top1": float(val_metrics["top1"]),
            "val_per_class_accuracy": dict(val_metrics["per_class_accuracy"]),
        }
        history.append(record)
        log_step(
            f"epoch={epoch_index:03d} "
            f"train_loss={record['train_loss']:.4f} train_top1={record['train_top1']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_top1={record['val_top1']:.4f}"
        )

        save_json(output_dir / "history.json", {"history": history})

        checkpoint_payload = {
            "epoch": int(epoch_index),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "class_names": class_names,
            "val_top1": float(val_metrics["top1"]),
        }
        torch.save(checkpoint_payload, checkpoints_dir / "last.pt")

        if float(val_metrics["top1"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            best_epoch = int(epoch_index)
            best_val_metrics = dict(val_metrics)
            torch.save(checkpoint_payload, checkpoints_dir / "best.pt")
            save_json(output_dir / "best_val_confusion_matrix.json", val_metrics)

    summary = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "class_names": class_names,
        "train_dataset": train_summary,
        "val_dataset": val_summary,
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_top1),
        "best_val_metrics": best_val_metrics or {},
        "final_epoch": int(cfg.train.epochs),
        "final_metrics": history[-1] if history else {},
    }
    save_json(output_dir / "summary.json", summary)
    log_step(
        f"done: best_epoch={best_epoch} best_val_top1={best_val_top1:.4f} "
        f"summary={output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
