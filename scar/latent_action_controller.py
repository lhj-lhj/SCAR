from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .fixed_subset import load_fixed_window_subset_manifest
from .gt_action_probe import extract_gt_action_sequence
from .models import ContextGTActionToLatentActionTransformer
from .objectives import run_idm_on_video_lat
from .runtime import get_lvp_module_map, select_rgb_channels, temporarily_frozen_eval, to_lvp_range, unwrap_module


def _collate_window_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty controller window batch.")
    keys = samples[0].keys()
    collated: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(first, np.ndarray):
            collated[key] = torch.from_numpy(np.stack(values, axis=0))
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


def _spatial_pool_context_tokens(video_lat: torch.Tensor) -> torch.Tensor:
    if video_lat.ndim != 5:
        raise ValueError(
            "Expected context video latents to have shape [B, C, T, H, W], got "
            f"{tuple(video_lat.shape)}"
        )
    return video_lat.mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()


def _align_controller_sequences(
    gt_actions: torch.Tensor,
    latent_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_actions.ndim == 2:
        gt_actions = gt_actions.unsqueeze(1)
    target_len = min(gt_actions.shape[1], latent_targets.shape[1])
    if target_len <= 0:
        raise ValueError("Latent-action controller requires a non-empty action trajectory.")
    return gt_actions[:, :target_len], latent_targets[:, :target_len]


def _select_deterministic_latent_actions(idm_output: Any) -> torch.Tensor:
    la_mean = getattr(idm_output, "la_mean", None)
    if torch.is_tensor(la_mean):
        return la_mean
    latent_actions = getattr(idm_output, "la", None)
    if torch.is_tensor(latent_actions):
        return latent_actions
    raise ValueError("IDM output does not contain a latent action tensor.")


@contextlib.contextmanager
def _temporarily_frozen_teacher_modules(idm, lvp):
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


def _extract_controller_subset_tensors(
    controller_cfg: Any,
    *,
    subset_manifest_path: str | Path,
    idm,
    lvp,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if idm is None or lvp is None:
        raise ValueError("Latent-action controller extraction requires both IDM and LVP objects.")

    from .dataloader import EpisodeWindowDataset, _preprocess_episode, load_episodes

    subset_manifest = load_fixed_window_subset_manifest(subset_manifest_path)
    seq_len = int(subset_manifest.get("seq_len", 49))
    shift = int(subset_manifest.get("shift", 1))
    target_action_dim = int(getattr(controller_cfg, "action_dim", 0)) or None
    batch_size = int(getattr(controller_cfg, "batch_size", 16))
    num_workers = int(getattr(controller_cfg, "num_workers", 0))
    context_len = int(getattr(controller_cfg, "context_len", 17))
    if context_len <= 0 or context_len > seq_len:
        raise ValueError(f"context_len must be in [1, {seq_len}], got {context_len}")

    context_chunks: list[torch.Tensor] = []
    gt_action_chunks: list[torch.Tensor] = []
    latent_target_chunks: list[torch.Tensor] = []

    with _temporarily_frozen_teacher_modules(idm, lvp):
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
                    "Controller subset manifest no longer matches dataset windowing: "
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
                videos = select_rgb_channels(batch.observations)
                videos = to_lvp_range(videos).contiguous()

                with torch.no_grad():
                    full_video_lat = lvp.encode_video(videos.permute(0, 2, 1, 3, 4))
                    context_video_lat = lvp.encode_video(
                        videos[:, :context_len].permute(0, 2, 1, 3, 4)
                    )
                    idm_output = run_idm_on_video_lat(
                        idm,
                        full_video_lat,
                        target_frame_tokens=int(batch.observations.shape[1]),
                        return_output=True,
                    )
                    latent_targets = _select_deterministic_latent_actions(idm_output)
                    gt_actions = extract_gt_action_sequence(batch.__dict__)

                gt_actions, latent_targets = _align_controller_sequences(
                    gt_actions,
                    latent_targets,
                )
                context_tokens = _spatial_pool_context_tokens(context_video_lat)
                context_chunks.append(context_tokens.detach().float().cpu())
                gt_action_chunks.append(gt_actions.detach().float().cpu())
                latent_target_chunks.append(latent_targets.detach().float().cpu())

    if not context_chunks or not gt_action_chunks or not latent_target_chunks:
        raise ValueError(
            "Controller subset manifest yielded no usable windows. "
            f"manifest={Path(subset_manifest_path).resolve()}"
        )

    context_tokens = torch.cat(context_chunks, dim=0)
    gt_actions = torch.cat(gt_action_chunks, dim=0)
    latent_targets = torch.cat(latent_target_chunks, dim=0)
    return context_tokens, gt_actions, latent_targets, subset_manifest


def extract_controller_subset_tensors(
    controller_cfg: Any,
    *,
    subset_manifest_path: str | Path,
    idm,
    lvp,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    return _extract_controller_subset_tensors(
        controller_cfg,
        subset_manifest_path=subset_manifest_path,
        idm=idm,
        lvp=lvp,
        device=device,
    )


def _evaluate_controller_dataset(
    controller: nn.Module,
    context_tokens: torch.Tensor,
    gt_actions: torch.Tensor,
    latent_targets: torch.Tensor,
    *,
    batch_size: int,
    context_len: int,
    device: torch.device,
) -> tuple[float, float]:
    total_squared_error = 0.0
    total_elements = 0
    total_future_squared_error = 0.0
    total_future_elements = 0

    controller.eval()
    with torch.no_grad():
        for start in range(0, context_tokens.shape[0], batch_size):
            end = start + batch_size
            context_batch = context_tokens[start:end].to(device=device)
            action_batch = gt_actions[start:end].to(device=device)
            latent_batch = latent_targets[start:end].to(device=device)
            pred_batch = controller(context_batch, action_batch)
            total_squared_error += float(
                F.mse_loss(pred_batch, latent_batch, reduction="sum").detach().item()
            )
            total_elements += int(latent_batch.numel())
            if latent_batch.shape[1] > context_len:
                future_pred = pred_batch[:, context_len:]
                future_target = latent_batch[:, context_len:]
                total_future_squared_error += float(
                    F.mse_loss(future_pred, future_target, reduction="sum").detach().item()
                )
                total_future_elements += int(future_target.numel())

    mse = total_squared_error / max(total_elements, 1)
    future_mse = total_future_squared_error / max(total_future_elements, 1)
    return mse, future_mse


def evaluate_controller_dataset(
    controller: nn.Module,
    context_tokens: torch.Tensor,
    gt_actions: torch.Tensor,
    latent_targets: torch.Tensor,
    *,
    batch_size: int,
    context_len: int,
    device: torch.device,
) -> tuple[float, float]:
    return _evaluate_controller_dataset(
        controller,
        context_tokens,
        gt_actions,
        latent_targets,
        batch_size=batch_size,
        context_len=context_len,
        device=device,
    )


@dataclass
class LatentActionControllerResult:
    steps: int
    train_loss: float
    train_mse: float
    train_future_mse: float
    sequence_length: int
    context_sequence_length: int
    context_latent_dim: int
    latent_action_dim: int
    gt_action_dim: int
    num_windows: int
    source: str
    subset_manifest: str | list[str] | None
    eval_trace: list[dict[str, float | int]]
    latest_checkpoint: str
    best_eval_checkpoint: str | None
    best_eval_step: int | None
    best_eval_mse: float | None
    saved_checkpoints: list[str]


def _build_controller_checkpoint_payload(
    *,
    controller: nn.Module,
    global_step: int,
    controller_step: int,
    controller_cfg: Any,
    train_steps: int,
    batch_size: int,
    controller_seed: int,
    context_len: int,
    subset_manifest_path: str | list[str],
    result: LatentActionControllerResult | None = None,
    context_tokens: torch.Tensor | None = None,
    gt_actions: torch.Tensor | None = None,
    latent_targets: torch.Tensor | None = None,
    metrics: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    if result is not None:
        shape = {
            "sequence_length": result.sequence_length,
            "context_sequence_length": result.context_sequence_length,
            "context_latent_dim": result.context_latent_dim,
            "latent_action_dim": result.latent_action_dim,
            "gt_action_dim": result.gt_action_dim,
            "num_windows": result.num_windows,
        }
    else:
        if context_tokens is None or gt_actions is None or latent_targets is None:
            raise ValueError("Raw tensors are required when result is not provided.")
        shape = {
            "sequence_length": int(latent_targets.shape[1]),
            "context_sequence_length": int(context_tokens.shape[1]),
            "context_latent_dim": int(context_tokens.shape[-1]),
            "latent_action_dim": int(latent_targets.shape[-1]),
            "gt_action_dim": int(gt_actions.shape[-1]),
            "num_windows": int(latent_targets.shape[0]),
        }

    payload_metrics = dict(metrics or {})
    if result is not None and not payload_metrics:
        payload_metrics = {
            "loss": float(result.train_loss),
            "mse": float(result.train_mse),
            "future_mse": float(result.train_future_mse),
        }

    return {
        "global_step": int(global_step),
        "controller_step": int(controller_step),
        "controller_config": {
            "dim_model": int(getattr(controller_cfg, "dim_model", 128)),
            "n_heads": int(getattr(controller_cfg, "n_heads", 4)),
            "n_layers": int(getattr(controller_cfg, "n_layers", 1)),
            "dim_feedforward": int(getattr(controller_cfg, "dim_feedforward", 256)),
            "dropout": float(getattr(controller_cfg, "dropout", 0.1)),
            "architecture": str(
                getattr(
                    controller_cfg,
                    "architecture",
                    ContextGTActionToLatentActionTransformer.ARCH_LATENT_RESIDUAL_CROSS_ATTENTION,
                )
            ),
            "lr": float(getattr(controller_cfg, "lr", 1e-4)),
            "weight_decay": float(getattr(controller_cfg, "weight_decay", 1e-4)),
            "train_steps": int(train_steps),
            "batch_size": int(batch_size),
            "seed": int(controller_seed),
            "context_len": int(context_len),
            "eval_every": int(getattr(controller_cfg, "eval_every", 0)),
            "ckpt_every": int(getattr(controller_cfg, "ckpt_every", 0)),
            "subset_manifest": subset_manifest_path,
        },
        "metrics": payload_metrics,
        "eval_trace": list(result.eval_trace) if result is not None else [],
        "shape": shape,
        "source": result.source if result is not None else "fixed_subset_manifest",
        "state_dict": controller.state_dict(),
    }


def _save_controller_checkpoint(
    path: Path,
    *,
    controller: nn.Module,
    global_step: int,
    controller_step: int,
    controller_cfg: Any,
    train_steps: int,
    batch_size: int,
    controller_seed: int,
    context_len: int,
    subset_manifest_path: str | list[str],
    result: LatentActionControllerResult | None = None,
    context_tokens: torch.Tensor | None = None,
    gt_actions: torch.Tensor | None = None,
    latent_targets: torch.Tensor | None = None,
    metrics: Mapping[str, float | int] | None = None,
) -> None:
    payload = _build_controller_checkpoint_payload(
        controller=controller,
        global_step=global_step,
        controller_step=controller_step,
        controller_cfg=controller_cfg,
        train_steps=train_steps,
        batch_size=batch_size,
        controller_seed=controller_seed,
        context_len=context_len,
        subset_manifest_path=subset_manifest_path,
        result=result,
        context_tokens=context_tokens,
        gt_actions=gt_actions,
        latent_targets=latent_targets,
        metrics=metrics,
    )
    torch.save(payload, path)


def train_and_save_latent_action_controller(
    controller_cfg: Any,
    *,
    global_step: int,
    output_dir: str | Path,
    device: torch.device,
    idm=None,
    lvp=None,
    eval_context_tokens: torch.Tensor | None = None,
    eval_gt_actions: torch.Tensor | None = None,
    eval_latent_targets: torch.Tensor | None = None,
    log_fn=None,
    status_fn=None,
    log_every: int = 0,
    ckpt_every: int = 0,
) -> LatentActionControllerResult | None:
    if controller_cfg is None or not bool(getattr(controller_cfg, "enabled", False)):
        return None
    train_steps = int(getattr(controller_cfg, "train_steps", 0))
    if train_steps <= 0:
        return None
    subset_manifest_paths = list(getattr(controller_cfg, "subset_manifests", []) or [])
    if not subset_manifest_paths:
        subset_manifest_path = getattr(controller_cfg, "subset_manifest", None)
        if subset_manifest_path:
            subset_manifest_paths = [subset_manifest_path]
    if not subset_manifest_paths:
        raise ValueError(
            "latent_action_controller.subset_manifest or subset_manifests must be set "
            "when enabled=true"
        )

    resolved_manifest_paths = [str(Path(path).resolve()) for path in subset_manifest_paths]
    context_chunks: list[torch.Tensor] = []
    gt_action_chunks: list[torch.Tensor] = []
    latent_target_chunks: list[torch.Tensor] = []
    for subset_manifest_path in resolved_manifest_paths:
        context_part, gt_part, latent_part, _subset_manifest = extract_controller_subset_tensors(
            controller_cfg,
            subset_manifest_path=subset_manifest_path,
            idm=idm,
            lvp=lvp,
            device=device,
        )
        context_chunks.append(context_part)
        gt_action_chunks.append(gt_part)
        latent_target_chunks.append(latent_part)

    context_tokens = torch.cat(context_chunks, dim=0)
    gt_actions = torch.cat(gt_action_chunks, dim=0)
    latent_targets = torch.cat(latent_target_chunks, dim=0)
    source = "fixed_subset_manifests" if len(resolved_manifest_paths) > 1 else "fixed_subset_manifest"
    subset_manifest_path: str | list[str]
    subset_manifest_path = (
        resolved_manifest_paths[0]
        if len(resolved_manifest_paths) == 1
        else list(resolved_manifest_paths)
    )

    batch_size = int(getattr(controller_cfg, "batch_size", 16))
    controller_seed = int(getattr(controller_cfg, "seed", 0))
    context_len = int(getattr(controller_cfg, "context_len", 17))
    eval_every = max(int(getattr(controller_cfg, "eval_every", 0)), 0)
    ckpt_every = max(int(getattr(controller_cfg, "ckpt_every", ckpt_every)), 0)
    fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    eval_trace: list[dict[str, float | int]] = []
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / "latent_action_controller_latest.pt"
    best_eval_checkpoint_path = checkpoint_dir / "latent_action_controller_best_eval.pt"
    saved_checkpoints: list[str] = []
    best_eval_mse = float("inf")
    best_eval_step: int | None = None
    last_eval_payload: dict[str, float | int] | None = None

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(controller_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(controller_seed)

        controller = ContextGTActionToLatentActionTransformer(
            context_latent_dim=int(context_tokens.shape[-1]),
            gt_action_dim=int(gt_actions.shape[-1]),
            latent_action_dim=int(latent_targets.shape[-1]),
            dim_model=int(getattr(controller_cfg, "dim_model", 128)),
            n_heads=int(getattr(controller_cfg, "n_heads", 4)),
            n_layers=int(getattr(controller_cfg, "n_layers", 1)),
            dim_feedforward=int(getattr(controller_cfg, "dim_feedforward", 256)),
            dropout=float(getattr(controller_cfg, "dropout", 0.1)),
            architecture=str(
                getattr(
                    controller_cfg,
                    "architecture",
                    ContextGTActionToLatentActionTransformer.ARCH_LATENT_RESIDUAL_CROSS_ATTENTION,
                )
            ),
        ).to(device=device)

        optimizer = torch.optim.AdamW(
            controller.parameters(),
            lr=float(getattr(controller_cfg, "lr", 1e-4)),
            weight_decay=float(getattr(controller_cfg, "weight_decay", 1e-4)),
        )

        train_dataset = TensorDataset(context_tokens, gt_actions, latent_targets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(controller_seed),
            drop_last=False,
        )
        train_iter = iter(train_loader)

        controller.train()
        for step_idx in range(train_steps):
            try:
                context_batch, action_batch, latent_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                context_batch, action_batch, latent_batch = next(train_iter)

            context_batch = context_batch.to(device=device)
            action_batch = action_batch.to(device=device)
            latent_batch = latent_batch.to(device=device)

            pred_latent = controller(context_batch, action_batch)
            loss = F.mse_loss(pred_latent, latent_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if log_fn is not None and log_every > 0 and ((step_idx + 1) % log_every == 0):
                log_fn(
                    {
                        "controller/train_step": int(step_idx + 1),
                        "controller/train_step_loss": float(loss.detach().item()),
                    }
                )

            should_run_eval = (
                eval_every > 0
                and (
                    ((step_idx + 1) % eval_every == 0)
                    or ((step_idx + 1) == train_steps)
                )
            )
            if should_run_eval:
                train_eval_mse, train_eval_future_mse = evaluate_controller_dataset(
                    controller,
                    context_tokens,
                    gt_actions,
                    latent_targets,
                    batch_size=batch_size,
                    context_len=context_len,
                    device=device,
                )
                eval_payload = {
                    "step": int(step_idx + 1),
                    "train_mse": float(train_eval_mse),
                    "train_future_mse": float(train_eval_future_mse),
                }
                if (
                    eval_context_tokens is not None
                    and eval_gt_actions is not None
                    and eval_latent_targets is not None
                ):
                    eval_mse, eval_future_mse = evaluate_controller_dataset(
                        controller,
                        eval_context_tokens,
                        eval_gt_actions,
                        eval_latent_targets,
                        batch_size=batch_size,
                        context_len=context_len,
                        device=device,
                    )
                    eval_payload["eval_mse"] = float(eval_mse)
                    eval_payload["eval_future_mse"] = float(eval_future_mse)

                eval_trace.append(eval_payload)
                last_eval_payload = dict(eval_payload)
                if log_fn is not None:
                    wandb_payload = {
                        "controller/eval_step": int(step_idx + 1),
                        "controller/eval_step_train_mse": float(eval_payload["train_mse"]),
                        "controller/eval_step_train_future_mse": float(eval_payload["train_future_mse"]),
                    }
                    if "eval_mse" in eval_payload:
                        wandb_payload["controller/eval_step_eval_mse"] = float(eval_payload["eval_mse"])
                        wandb_payload["controller/eval_step_eval_future_mse"] = float(
                            eval_payload["eval_future_mse"]
                        )
                    log_fn(wandb_payload)
                if status_fn is not None:
                    status = (
                        f"eval step={int(step_idx + 1)} "
                        f"train_mse={float(eval_payload['train_mse']):.6f} "
                        f"train_future_mse={float(eval_payload['train_future_mse']):.6f}"
                    )
                    if "eval_mse" in eval_payload:
                        status += (
                            f" eval_mse={float(eval_payload['eval_mse']):.6f} "
                            f"eval_future_mse={float(eval_payload['eval_future_mse']):.6f}"
                        )
                    status_fn(status)
                if "eval_mse" in eval_payload and float(eval_payload["eval_mse"]) < best_eval_mse:
                    best_eval_mse = float(eval_payload["eval_mse"])
                    best_eval_step = int(step_idx + 1)
                    _save_controller_checkpoint(
                        best_eval_checkpoint_path,
                        controller=controller,
                        global_step=global_step,
                        controller_step=int(step_idx + 1),
                        controller_cfg=controller_cfg,
                        train_steps=train_steps,
                        batch_size=batch_size,
                        controller_seed=controller_seed,
                        context_len=context_len,
                        subset_manifest_path=subset_manifest_path,
                        context_tokens=context_tokens,
                        gt_actions=gt_actions,
                        latent_targets=latent_targets,
                        metrics=eval_payload,
                    )
                    if status_fn is not None:
                        status_fn(
                            f"saved best_eval checkpoint step={int(step_idx + 1)} "
                            f"eval_mse={float(eval_payload['eval_mse']):.6f} "
                            f"path={best_eval_checkpoint_path}"
                        )

            should_save_checkpoint = (
                ckpt_every > 0
                and (
                    ((step_idx + 1) % ckpt_every == 0)
                    or ((step_idx + 1) == train_steps)
                )
            )
            if should_save_checkpoint:
                checkpoint_metrics: dict[str, float | int] = {
                    "train_step_loss": float(loss.detach().item()),
                }
                if last_eval_payload is not None:
                    checkpoint_metrics.update(last_eval_payload)
                step_checkpoint_path = checkpoint_dir / (
                    f"latent_action_controller_step_{int(step_idx + 1):07d}.pt"
                )
                _save_controller_checkpoint(
                    step_checkpoint_path,
                    controller=controller,
                    global_step=global_step,
                    controller_step=int(step_idx + 1),
                    controller_cfg=controller_cfg,
                    train_steps=train_steps,
                    batch_size=batch_size,
                    controller_seed=controller_seed,
                    context_len=context_len,
                    subset_manifest_path=subset_manifest_path,
                    context_tokens=context_tokens,
                    gt_actions=gt_actions,
                    latent_targets=latent_targets,
                    metrics=checkpoint_metrics,
                )
                _save_controller_checkpoint(
                    latest_checkpoint_path,
                    controller=controller,
                    global_step=global_step,
                    controller_step=int(step_idx + 1),
                    controller_cfg=controller_cfg,
                    train_steps=train_steps,
                    batch_size=batch_size,
                    controller_seed=controller_seed,
                    context_len=context_len,
                    subset_manifest_path=subset_manifest_path,
                    context_tokens=context_tokens,
                    gt_actions=gt_actions,
                    latent_targets=latent_targets,
                    metrics=checkpoint_metrics,
                )
                saved_checkpoints.append(str(step_checkpoint_path))
                if status_fn is not None:
                    status_fn(
                        f"saved controller checkpoint step={int(step_idx + 1)} "
                        f"path={step_checkpoint_path}"
                    )

        train_loss, train_future_mse = evaluate_controller_dataset(
            controller,
            context_tokens,
            gt_actions,
            latent_targets,
            batch_size=batch_size,
            context_len=context_len,
            device=device,
        )

    final_step_checkpoint_path = checkpoint_dir / f"latent_action_controller_step_{int(train_steps):07d}.pt"

    result = LatentActionControllerResult(
        steps=train_steps,
        train_loss=train_loss,
        train_mse=train_loss,
        train_future_mse=train_future_mse,
        sequence_length=int(latent_targets.shape[1]),
        context_sequence_length=int(context_tokens.shape[1]),
        context_latent_dim=int(context_tokens.shape[-1]),
        latent_action_dim=int(latent_targets.shape[-1]),
        gt_action_dim=int(gt_actions.shape[-1]),
        num_windows=int(latent_targets.shape[0]),
        source=source,
        subset_manifest=subset_manifest_path,
        eval_trace=eval_trace,
        latest_checkpoint=str(latest_checkpoint_path),
        best_eval_checkpoint=(
            str(best_eval_checkpoint_path)
            if best_eval_step is not None and best_eval_checkpoint_path.is_file()
            else None
        ),
        best_eval_step=best_eval_step,
        best_eval_mse=(best_eval_mse if best_eval_step is not None else None),
        saved_checkpoints=list(saved_checkpoints),
    )
    if str(final_step_checkpoint_path) not in saved_checkpoints:
        _save_controller_checkpoint(
            final_step_checkpoint_path,
            controller=controller,
            global_step=global_step,
            controller_step=int(train_steps),
            controller_cfg=controller_cfg,
            train_steps=train_steps,
            batch_size=batch_size,
            controller_seed=controller_seed,
            context_len=context_len,
            subset_manifest_path=subset_manifest_path,
            result=result,
        )
        result.saved_checkpoints.append(str(final_step_checkpoint_path))
    _save_controller_checkpoint(
        latest_checkpoint_path,
        controller=controller,
        global_step=global_step,
        controller_step=int(train_steps),
        controller_cfg=controller_cfg,
        train_steps=train_steps,
        batch_size=batch_size,
        controller_seed=controller_seed,
        context_len=context_len,
        subset_manifest_path=subset_manifest_path,
        result=result,
    )
    return result


__all__ = [
    "ContextGTActionToLatentActionTransformer",
    "LatentActionControllerResult",
    "evaluate_controller_dataset",
    "extract_controller_subset_tensors",
    "train_and_save_latent_action_controller",
]
