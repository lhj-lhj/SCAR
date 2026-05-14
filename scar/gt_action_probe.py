from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

def extract_gt_action_sequence(batch: Mapping[str, Any]) -> torch.Tensor:
    for key in ("actions", "gt_actions", "action", "robot_actions"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    available = ", ".join(sorted(str(k) for k in batch.keys()))
    raise KeyError(
        "Unable to find a ground-truth action tensor in batch. "
        f"Available keys: {available}"
    )


def _build_sinusoidal_positional_encoding(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half_dim = max(1, dim // 2)
    div_term = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    )
    angles = position * div_term.unsqueeze(0)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, : pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class GroundTruthActionTransformerProbe(nn.Module):
    def __init__(
        self,
        latent_action_dim: int,
        gt_action_dim: int,
        *,
        dim_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_action_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(dim_model, gt_action_dim)

    def forward(self, latent_actions: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(latent_actions)
        x = x + _build_sinusoidal_positional_encoding(
            x.shape[1], x.shape[2], device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        x = self.encoder(x)
        return self.output_proj(x)


def _align_probe_sequences(
    latent_actions: torch.Tensor,
    gt_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_actions.ndim == 2:
        gt_actions = gt_actions.unsqueeze(1)
    target_len = min(latent_actions.shape[1], gt_actions.shape[1])
    if target_len <= 0:
        raise ValueError(
            "Ground-truth action probe requires a non-empty action trajectory."
        )
    return latent_actions[:, :target_len], gt_actions[:, :target_len]


@dataclass
class GroundTruthActionProbeResult:
    steps: int
    train_loss: float
    train_mse: float
    train_l1: float
    sequence_length: int
    latent_action_dim: int
    gt_action_dim: int
    num_windows: int
    source: str
    subset_manifest: str | None


def _load_probe_subset_manifest(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GT-action probe subset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("tasks"), list):
        raise ValueError(f"Invalid probe subset manifest: missing tasks list in {path}")
    return payload


def _collate_window_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty probe window batch.")
    keys = samples[0].keys()
    collated: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, torch.Tensor):
            collated[key] = torch.stack(values)
        elif hasattr(first, "dtype") and hasattr(first, "shape"):
            collated[key] = torch.as_tensor(values)
        elif isinstance(first, (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values
    return collated


def _batch_dict_to_batch(batch_dict: dict[str, Any], device: torch.device):
    from .dataloader import Batch, to_device

    payload = {key: to_device(value, device) for key, value in batch_dict.items()}
    return Batch(**payload)


@contextlib.contextmanager
def _temporarily_frozen_probe_modules(idm, lvp):
    from .runtime import get_lvp_module_map, temporarily_frozen_eval, unwrap_module

    modules: list[nn.Module] = []
    if idm is not None:
        modules.append(unwrap_module(idm))
    if lvp is not None:
        for module in get_lvp_module_map(lvp).values():
            base_module = unwrap_module(module)
            if base_module is not None:
                modules.append(base_module)

    with contextlib.ExitStack() as stack:
        seen: set[int] = set()
        for module in modules:
            marker = id(module)
            if marker in seen:
                continue
            seen.add(marker)
            stack.enter_context(temporarily_frozen_eval(module))
        yield


def _extract_probe_subset_tensors(
    args: Any,
    *,
    subset_manifest_path: str | Path,
    idm,
    lvp,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if getattr(args, "lvp_action_source", "idm") != "idm":
        raise ValueError("GT-action probe subset extraction requires model.lvp_action_source=idm")
    if idm is None or lvp is None:
        raise ValueError("GT-action probe subset extraction requires both IDM and LVP objects.")

    from .dataloader import EpisodeWindowDataset, _preprocess_episode, load_episodes
    from .objectives import encode_lvp_video_latents, resolve_conditioning_actions

    probe_cfg = getattr(args, "gt_action_probe")
    subset_manifest = _load_probe_subset_manifest(subset_manifest_path)
    seq_len = int(subset_manifest.get("seq_len", int(getattr(args.data, "seq_len", 49))))
    shift = int(subset_manifest.get("shift", int(getattr(args.data, "shift", 1))))
    target_action_dim = int(getattr(args.data, "action_dim", 0)) or None
    batch_size = int(getattr(probe_cfg, "batch_size", 16))
    num_workers = int(getattr(probe_cfg, "num_workers", 0))

    latent_chunks: list[torch.Tensor] = []
    gt_chunks: list[torch.Tensor] = []

    with _temporarily_frozen_probe_modules(idm, lvp):
        for task_entry in subset_manifest["tasks"]:
            selected_indices = [int(index) for index in task_entry.get("selected_window_indices", [])]
            if not selected_indices:
                continue

            dataset_path = str(task_entry["dataset_path"])
            selected_episode_indices = task_entry.get("selected_episode_indices", None)
            if selected_episode_indices is not None:
                selected_episode_indices = [int(index) for index in selected_episode_indices]
            episodes = load_episodes(dataset_path, episode_indices=selected_episode_indices)
            processed = [
                _preprocess_episode(ep, target_action_dim=target_action_dim)
                for ep in episodes
            ]
            dataset = EpisodeWindowDataset(
                processed,
                seq_len=seq_len,
                shift=shift,
                embodiment_id=int(task_entry.get("embodiment_id", 0)),
                cover_tail=bool(subset_manifest.get("cover_tail", False)),
            )
            expected_windows = int(task_entry.get("num_total_windows", len(dataset)))
            if len(dataset) != expected_windows:
                raise ValueError(
                    "Probe subset manifest no longer matches dataset windowing: "
                    f"{dataset_path} expected {expected_windows}, got {len(dataset)} "
                    f"(seq_len={seq_len}, shift={shift})"
                )
            subset = Subset(dataset, selected_indices)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_window_batch,
                drop_last=False,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(num_workers > 0),
            )

            for batch_dict in loader:
                batch = _batch_dict_to_batch(batch_dict, device)
                with torch.no_grad():
                    _, video_lat = encode_lvp_video_latents(
                        lvp,
                        batch,
                        lvp_trainable_modules=set(),
                    )
                    conditioning_actions, _ = resolve_conditioning_actions(
                        action_source="idm",
                        batch=batch,
                        video_lat=video_lat,
                        idm=idm,
                        target_frame_tokens=int(batch.observations.shape[1]),
                        return_idm_output=True,
                    )
                    gt_actions = extract_gt_action_sequence(batch.__dict__)
                latent_chunks.append(conditioning_actions.detach().float().cpu())
                gt_chunks.append(gt_actions.detach().float().cpu())

    if not latent_chunks or not gt_chunks:
        raise ValueError(
            "Probe subset manifest yielded no usable windows. "
            f"manifest={Path(subset_manifest_path).resolve()}"
        )

    latent_actions = torch.cat(latent_chunks, dim=0).to(device=device)
    gt_actions = torch.cat(gt_chunks, dim=0).to(device=device)
    return latent_actions, gt_actions, subset_manifest


def _evaluate_probe_dataset(
    probe: nn.Module,
    latent_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float, float]:
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    probe.eval()
    with torch.no_grad():
        for start in range(0, latent_actions.shape[0], batch_size):
            end = start + batch_size
            latent_batch = latent_actions[start:end]
            gt_batch = gt_actions[start:end]
            pred_batch = probe(latent_batch)
            total_squared_error += float(
                F.mse_loss(pred_batch, gt_batch, reduction="sum").detach().item()
            )
            total_absolute_error += float(
                F.l1_loss(pred_batch, gt_batch, reduction="sum").detach().item()
            )
            total_elements += int(gt_batch.numel())

    total_elements = max(total_elements, 1)
    mse = total_squared_error / total_elements
    l1 = total_absolute_error / total_elements
    return mse, mse, l1


def train_and_save_gt_action_probe(
    args: Any,
    *,
    global_step: int,
    output_dir: str | Path,
    device: torch.device,
    idm=None,
    lvp=None,
    log_fn=None,
    log_every: int = 0,
) -> GroundTruthActionProbeResult | None:
    probe_cfg = getattr(args, "gt_action_probe", None)
    if probe_cfg is None or not bool(getattr(probe_cfg, "enabled", False)):
        return None
    train_steps = int(getattr(probe_cfg, "train_steps", 0))
    if train_steps <= 0:
        return None
    subset_manifest_path = getattr(probe_cfg, "subset_manifest", None)
    if not subset_manifest_path:
        raise ValueError(
            "gt_action_probe.subset_manifest must be set when gt_action_probe.enabled=true"
        )

    latent_actions, gt_actions, _subset_manifest = _extract_probe_subset_tensors(
        args,
        subset_manifest_path=subset_manifest_path,
        idm=idm,
        lvp=lvp,
        device=device,
    )
    source = "fixed_subset_manifest"
    subset_manifest_path = str(Path(subset_manifest_path).resolve())
    latent_actions, gt_actions = _align_probe_sequences(latent_actions, gt_actions)

    probe_batch_size = int(getattr(probe_cfg, "batch_size", 16))
    probe_seed = int(getattr(probe_cfg, "seed", 0))
    fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(probe_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(probe_seed)

        probe = GroundTruthActionTransformerProbe(
            latent_action_dim=latent_actions.shape[-1],
            gt_action_dim=gt_actions.shape[-1],
            dim_model=int(getattr(probe_cfg, "dim_model", 256)),
            n_heads=int(getattr(probe_cfg, "n_heads", 4)),
            n_layers=int(getattr(probe_cfg, "n_layers", 2)),
            dim_feedforward=int(getattr(probe_cfg, "dim_feedforward", 1024)),
            dropout=float(getattr(probe_cfg, "dropout", 0.1)),
        ).to(device=device)

        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=float(getattr(probe_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(probe_cfg, "weight_decay", 1e-4)),
        )

        train_dataset = TensorDataset(latent_actions, gt_actions)
        train_loader = DataLoader(
            train_dataset,
            batch_size=probe_batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(probe_seed),
            drop_last=False,
        )
        train_iter = iter(train_loader)

        probe.train()
        for step_idx in range(train_steps):
            try:
                latent_batch, gt_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                latent_batch, gt_batch = next(train_iter)

            pred_actions = probe(latent_batch)
            loss = F.mse_loss(pred_actions, gt_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if log_fn is not None and log_every > 0 and ((step_idx + 1) % log_every == 0):
                log_fn(
                    {
                        "probe/train_step": int(step_idx + 1),
                        "probe/train_step_loss": float(loss.detach().item()),
                    }
                )

        train_loss, train_mse, train_l1 = _evaluate_probe_dataset(
            probe,
            latent_actions,
            gt_actions,
            batch_size=probe_batch_size,
        )

    result = GroundTruthActionProbeResult(
        steps=train_steps,
        train_loss=train_loss,
        train_mse=train_mse,
        train_l1=train_l1,
        sequence_length=int(latent_actions.shape[1]),
        latent_action_dim=int(latent_actions.shape[-1]),
        gt_action_dim=int(gt_actions.shape[-1]),
        num_windows=int(latent_actions.shape[0]),
        source=source,
        subset_manifest=subset_manifest_path,
    )

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "probe_config": {
            "dim_model": int(getattr(probe_cfg, "dim_model", 256)),
            "n_heads": int(getattr(probe_cfg, "n_heads", 4)),
            "n_layers": int(getattr(probe_cfg, "n_layers", 2)),
            "dim_feedforward": int(getattr(probe_cfg, "dim_feedforward", 1024)),
            "dropout": float(getattr(probe_cfg, "dropout", 0.1)),
            "lr": float(getattr(probe_cfg, "lr", 1e-4)),
            "weight_decay": float(getattr(probe_cfg, "weight_decay", 1e-4)),
            "train_steps": int(train_steps),
            "batch_size": int(probe_batch_size),
            "seed": int(probe_seed),
            "subset_manifest": subset_manifest_path,
        },
        "metrics": {
            "loss": result.train_loss,
            "mse": result.train_mse,
            "l1": result.train_l1,
        },
        "shape": {
            "sequence_length": result.sequence_length,
            "latent_action_dim": result.latent_action_dim,
            "gt_action_dim": result.gt_action_dim,
            "num_windows": result.num_windows,
        },
        "source": result.source,
        "state_dict": probe.state_dict(),
    }
    step_name = f"gt_action_probe_step_{int(global_step):07d}.pt"
    torch.save(payload, checkpoint_dir / step_name)
    torch.save(payload, checkpoint_dir / "gt_action_probe_latest.pt")
    return result
