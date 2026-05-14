from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from export_split_videos import save_split_outputs
from scar.cycle_api import (
    build_sampling_batch,
    encode_lvp_video_latents,
    run_idm_on_video_lat,
    sample_lvp_video,
    select_rgb_channels,
    to_lvp_range,
    trim_batch,
)
from scar.dataloader import Batch, to_device
from scar.gt_action_probe import extract_gt_action_sequence
from scar.metrics import MetricBundle, compute_lvp_style_eval_video_metrics_from_batches
from scar.models import ContextGTActionToLatentActionTransformer
from scar.runtime import (
    get_lvp_module_map,
    temporarily_frozen_eval,
    temporarily_unwrap_lvp_ddp,
    unwrap_module,
)


@dataclass(frozen=True)
class ControllerFDMCache:
    split: str
    full_video_lat: torch.Tensor
    context_tokens: torch.Tensor
    gt_actions: torch.Tensor
    teacher_latent_actions: torch.Tensor
    seq_len: int
    context_len: int
    input_video: torch.Tensor | None = None

    @property
    def num_windows(self) -> int:
        return int(self.full_video_lat.shape[0])


def select_deterministic_latent_actions(idm_output: Any) -> torch.Tensor:
    la_mean = getattr(idm_output, "la_mean", None)
    if torch.is_tensor(la_mean):
        return la_mean
    latent_actions = getattr(idm_output, "la", None)
    if torch.is_tensor(latent_actions):
        return latent_actions
    raise ValueError("IDM output does not contain a latent action tensor.")


def align_controller_sequences(
    gt_actions: torch.Tensor,
    latent_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_actions.ndim == 2:
        gt_actions = gt_actions.unsqueeze(1)
    target_len = min(gt_actions.shape[1], latent_targets.shape[1])
    if target_len <= 0:
        raise ValueError("Controller-FDM pipeline requires a non-empty action trajectory.")
    return gt_actions[:, :target_len], latent_targets[:, :target_len]


def spatial_pool_context_tokens(video_lat: torch.Tensor) -> torch.Tensor:
    if video_lat.ndim != 5:
        raise ValueError(
            "Expected context video latents to have shape [B, C, T, H, W], got "
            f"{tuple(video_lat.shape)}"
        )
    return video_lat.mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()


def load_controller_checkpoint(
    controller_ckpt_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], ContextGTActionToLatentActionTransformer]:
    payload = torch.load(controller_ckpt_path, map_location="cpu", weights_only=False)
    controller_config = dict(payload.get("controller_config", {}))
    shape = dict(payload.get("shape", {}))
    controller = ContextGTActionToLatentActionTransformer(
        context_latent_dim=int(shape["context_latent_dim"]),
        gt_action_dim=int(shape["gt_action_dim"]),
        latent_action_dim=int(shape["latent_action_dim"]),
        dim_model=int(controller_config.get("dim_model", 128)),
        n_heads=int(controller_config.get("n_heads", 4)),
        n_layers=int(controller_config.get("n_layers", 1)),
        dim_feedforward=int(controller_config.get("dim_feedforward", 256)),
        dropout=float(controller_config.get("dropout", 0.1)),
        architecture=str(
            controller_config.get(
                "architecture",
                ContextGTActionToLatentActionTransformer.ARCH_SUMMARY_ADD,
            )
        ),
    ).to(device=device)
    controller.load_state_dict(payload["state_dict"])
    controller.eval()
    return payload, controller


def load_controller_summary(controller_ckpt_path: Path) -> tuple[Path | None, dict[str, Any]]:
    controller_run_dir = controller_ckpt_path.parent.parent
    summary_path = controller_run_dir / "summary.json"
    if not summary_path.is_file():
        return None, {}
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_backbone_run_and_ckpt(
    *,
    run_dir: str | Path | None,
    ckpt: str | Path | None,
    controller_summary: dict[str, Any],
) -> tuple[Path, Path]:
    run_dir_path = Path(run_dir).resolve() if run_dir is not None else None
    ckpt_path = Path(ckpt).resolve() if ckpt is not None else None

    if run_dir_path is None:
        run_dir_value = controller_summary.get("run_dir")
        if run_dir_value:
            run_dir_path = Path(str(run_dir_value)).resolve()
    if ckpt_path is None:
        ckpt_value = controller_summary.get("ckpt_path")
        if ckpt_value:
            ckpt_path = Path(str(ckpt_value)).resolve()

    if run_dir_path is None and ckpt_path is None:
        raise ValueError(
            "Unable to infer SCAR run_dir / checkpoint from controller summary. "
            "Provide run_dir or ckpt explicitly."
        )
    if run_dir_path is None:
        assert ckpt_path is not None
        if ckpt_path.parent.name == "checkpoints":
            run_dir_path = ckpt_path.parent.parent.resolve()
        else:
            raise ValueError("Cannot infer run_dir from checkpoint outside a checkpoints/ directory.")
    if ckpt_path is None:
        assert run_dir_path is not None
        ckpt_path = (run_dir_path / "checkpoints" / "latest.pt").resolve()

    if not run_dir_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return run_dir_path, ckpt_path


def collate_window_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty list of window samples.")
    keys = samples[0].keys()
    collated: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, np.ndarray):
            collated[key] = np.stack(values, axis=0)
        elif isinstance(first, (int, float, np.integer, np.floating)):
            collated[key] = np.asarray(values)
        else:
            collated[key] = values
    return collated


@contextlib.contextmanager
def _temporarily_frozen_teacher_modules(idm, lvp):
    modules = []
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


@contextlib.contextmanager
def temporarily_eval_controller_and_lvp(controller, lvp):
    controller_module = unwrap_module(controller)
    modules = [controller_module]
    modules.extend(
        unwrap_module(module)
        for module in get_lvp_module_map(lvp).values()
        if module is not None
    )
    training_states = [(module, module.training) for module in modules if module is not None]
    try:
        for module, _was_training in training_states:
            module.eval()
        with temporarily_unwrap_lvp_ddp(lvp):
            yield controller_module
    finally:
        for module, was_training in training_states:
            module.train(was_training)


def build_controller_fdm_cache(
    *,
    split_name: str,
    records,
    device: torch.device,
    seq_len: int,
    context_len: int,
    batch_size: int,
    lvp,
    idm,
    include_input_video: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> ControllerFDMCache:
    full_video_lat_chunks: list[torch.Tensor] = []
    context_token_chunks: list[torch.Tensor] = []
    gt_action_chunks: list[torch.Tensor] = []
    teacher_latent_chunks: list[torch.Tensor] = []
    input_video_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        with _temporarily_frozen_teacher_modules(idm, lvp):
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                batch_np = collate_window_samples([record.sample_np for record in batch_records])
                batch = Batch(**to_device(batch_np, device))
                batch = trim_batch(batch, seq_len)

                input_video, full_video_lat = encode_lvp_video_latents(lvp, batch)
                videos = select_rgb_channels(batch.observations)
                videos = to_lvp_range(videos).contiguous()
                context_video_lat = lvp.encode_video(
                    videos[:, :context_len].permute(0, 2, 1, 3, 4)
                )

                idm_output = run_idm_on_video_lat(
                    idm,
                    full_video_lat,
                    target_frame_tokens=int(batch.observations.shape[1]),
                    return_output=True,
                )
                teacher_latent_actions = select_deterministic_latent_actions(idm_output)
                gt_actions = extract_gt_action_sequence(batch.__dict__)
                gt_actions, teacher_latent_actions = align_controller_sequences(
                    gt_actions,
                    teacher_latent_actions,
                )

                full_video_lat_chunks.append(full_video_lat.detach().float().cpu())
                context_token_chunks.append(
                    spatial_pool_context_tokens(context_video_lat).detach().float().cpu()
                )
                gt_action_chunks.append(gt_actions.detach().float().cpu())
                teacher_latent_chunks.append(teacher_latent_actions.detach().float().cpu())
                if include_input_video:
                    input_video_chunks.append(input_video.detach().float().cpu())

                if status_fn is not None:
                    status_fn(
                        f"{split_name}: cached_windows="
                        f"{min(start + len(batch_records), len(records))}/{len(records)}"
                    )

    if not full_video_lat_chunks:
        raise ValueError(f"No controller-FDM cache entries were built for split={split_name}.")

    return ControllerFDMCache(
        split=split_name,
        full_video_lat=torch.cat(full_video_lat_chunks, dim=0),
        context_tokens=torch.cat(context_token_chunks, dim=0),
        gt_actions=torch.cat(gt_action_chunks, dim=0),
        teacher_latent_actions=torch.cat(teacher_latent_chunks, dim=0),
        seq_len=int(seq_len),
        context_len=int(context_len),
        input_video=torch.cat(input_video_chunks, dim=0) if input_video_chunks else None,
    )


def run_controller_fdm_eval_suite_from_cache(
    *,
    cache: ControllerFDMCache,
    controller,
    lvp,
    device: torch.device,
    batch_size: int,
    metric_names: list[str],
    metric_batch_size: int,
    hist_len: int,
    n_metrics_frames: int | None,
    prompt_text: str,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    output_dir: Path,
    save_video_count: int,
    save_video_fps: int,
    seed: int,
    status_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if cache.input_video is None:
        raise ValueError(
            "Controller-FDM eval requires cache.input_video. Build the cache with include_input_video=True."
        )

    start_time = time.time()
    video_pred_batches: list[torch.Tensor] = []
    video_gt_batches: list[torch.Tensor] = []
    latent_sqerr = 0.0
    latent_count = 0
    future_sqerr = 0.0
    future_count = 0
    saved_video_paths: list[str] = []

    with torch.no_grad():
        with temporarily_eval_controller_and_lvp(controller, lvp) as controller_module:
            for batch_index, start in enumerate(range(0, cache.num_windows, batch_size)):
                end = min(start + batch_size, cache.num_windows)
                input_video = cache.input_video[start:end].to(device=device)
                context_tokens = cache.context_tokens[start:end].to(device=device)
                gt_actions = cache.gt_actions[start:end].to(device=device)
                teacher_latent_actions = cache.teacher_latent_actions[start:end].to(device=device)

                pred_latent_actions = controller_module(context_tokens, gt_actions)
                batch = Batch(states=None, observations=input_video, actions=None)
                sampling_batch = build_sampling_batch(
                    lvp,
                    batch,
                    pred_latent_actions,
                    prompt_text=prompt_text,
                    prompt_embed=prompt_embed,
                    prompt_embed_len=prompt_embed_len,
                    negative_prompt_embed=negative_prompt_embed,
                    negative_prompt_embed_len=negative_prompt_embed_len,
                )
                pred_video = sample_lvp_video(lvp, sampling_batch, seed=seed + batch_index)

                video_pred_batches.append(pred_video.detach().cpu())
                video_gt_batches.append(input_video.detach().cpu())
                latent_sqerr += float(
                    F.mse_loss(
                        pred_latent_actions,
                        teacher_latent_actions,
                        reduction="sum",
                    ).detach().cpu()
                )
                latent_count += int(teacher_latent_actions.numel())
                if teacher_latent_actions.shape[1] > cache.context_len:
                    future_pred = pred_latent_actions[:, cache.context_len :]
                    future_teacher = teacher_latent_actions[:, cache.context_len :]
                    future_sqerr += float(
                        F.mse_loss(
                            future_pred,
                            future_teacher,
                            reduction="sum",
                        ).detach().cpu()
                    )
                    future_count += int(future_teacher.numel())

                remaining_to_save = max(int(save_video_count) - len(saved_video_paths), 0)
                for local_idx in range(min(remaining_to_save, int(input_video.shape[0]))):
                    sample_output_dir = output_dir / f"sample_{len(saved_video_paths):03d}"
                    saved_paths = save_split_outputs(
                        output_dir=sample_output_dir,
                        input_video=input_video[local_idx : local_idx + 1],
                        pred_video=pred_video[local_idx : local_idx + 1],
                        hist_len=hist_len,
                        fps=save_video_fps,
                    )
                    saved_video_paths.append(saved_paths["input_vs_pred"])

    video_metrics = compute_lvp_style_eval_video_metrics_from_batches(
        metric_names,
        video_pred_batches=video_pred_batches,
        video_gt_batches=video_gt_batches,
        hist_len=hist_len,
        n_metrics_frames=n_metrics_frames,
        split_batch_size=metric_batch_size,
        max_video_count=None,
    )
    metric_bundle = video_metrics.with_updates(
        latent_mse=latent_sqerr / max(latent_count, 1),
        latent_future_mse=future_sqerr / max(future_count, 1),
    )
    elapsed = time.time() - start_time
    if status_fn is not None:
        status_fn(
            f"{cache.split}: windows={cache.num_windows} elapsed={elapsed:.2f}s "
            + " ".join(f"{key}={value:.6f}" for key, value in metric_bundle.items())
        )
        for video_path in saved_video_paths:
            status_fn(f"{cache.split}: saved_video={video_path}")

    return {
        "split": cache.split,
        "num_windows": int(cache.num_windows),
        "elapsed_seconds": float(elapsed),
        "metrics": MetricBundle(metric_bundle).to_dict(),
        "saved_videos": saved_video_paths,
    }


__all__ = [
    "ControllerFDMCache",
    "align_controller_sequences",
    "build_controller_fdm_cache",
    "collate_window_samples",
    "load_controller_checkpoint",
    "load_controller_summary",
    "resolve_backbone_run_and_ckpt",
    "run_controller_fdm_eval_suite_from_cache",
    "select_deterministic_latent_actions",
    "spatial_pool_context_tokens",
    "temporarily_eval_controller_and_lvp",
]
