from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from export_split_videos import save_split_outputs
from infer_action_controller import (
    _align_sequences,
    _select_deterministic_latent_actions,
    _spatial_pool_context_tokens,
)
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
from scar.metrics import (
    MetricBundle,
    build_eval_table_metric_bundle,
    build_namespaced_log_payload,
    compute_lvp_style_eval_video_metrics_from_batches,
)


def collate_window_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty list of window samples.")
    keys = samples[0].keys()
    collated: dict[str, Any] = {}
    for key in keys:
        vals = [sample[key] for sample in samples]
        first = vals[0]
        if isinstance(first, np.ndarray):
            collated[key] = np.stack(vals, axis=0)
        elif isinstance(first, (int, float, np.integer, np.floating)):
            collated[key] = np.asarray(vals)
        else:
            collated[key] = vals
    return collated


def compute_controller_video_batch(
    *,
    records,
    device: torch.device,
    seq_len: int,
    context_len: int,
    controller,
    lvp,
    idm: torch.nn.Module,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    prompt_text: str,
    seed: int,
) -> tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_np = collate_window_samples([record.sample_np for record in records])
    batch = Batch(**to_device(batch_np, device))
    batch = trim_batch(batch, seq_len)

    with torch.no_grad():
        _, full_video_lat = encode_lvp_video_latents(lvp, batch)
        videos = select_rgb_channels(batch.observations)
        videos = to_lvp_range(videos).contiguous()
        context_video_lat = lvp.encode_video(videos[:, :context_len].permute(0, 2, 1, 3, 4))
        context_tokens = _spatial_pool_context_tokens(context_video_lat)

        idm_output = run_idm_on_video_lat(
            idm,
            full_video_lat,
            target_frame_tokens=int(batch.observations.shape[1]),
            return_output=True,
        )
        teacher_latent_actions = _select_deterministic_latent_actions(idm_output)
        gt_actions = extract_gt_action_sequence(batch.__dict__)
        gt_actions, teacher_latent_actions = _align_sequences(gt_actions, teacher_latent_actions)
        pred_latent_actions = controller(context_tokens, gt_actions)

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
        pred_video = sample_lvp_video(lvp, sampling_batch, seed=seed)

    input_video = to_lvp_range(select_rgb_channels(batch.observations))
    return batch, input_video, pred_video, teacher_latent_actions, pred_latent_actions


def run_controller_fdm_eval_suite(
    *,
    split_name: str,
    records,
    device: torch.device,
    seq_len: int,
    context_len: int,
    batch_size: int,
    controller,
    lvp,
    idm: torch.nn.Module,
    metric_names: list[str],
    metric_batch_size: int,
    hist_len: int,
    n_metrics_frames: int | None,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    prompt_text: str,
    output_dir: Path,
    save_video_count: int,
    save_video_fps: int,
    seed: int,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    start_time = time.time()
    video_pred_batches: list[torch.Tensor] = []
    video_gt_batches: list[torch.Tensor] = []
    latent_sqerr = 0.0
    latent_count = 0
    future_sqerr = 0.0
    future_count = 0
    saved_video_paths: list[str] = []

    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch_records = records[start : start + batch_size]
        (
            _batch,
            input_video,
            pred_video,
            teacher_latent_actions,
            pred_latent_actions,
        ) = compute_controller_video_batch(
            records=batch_records,
            device=device,
            seq_len=seq_len,
            context_len=context_len,
            controller=controller,
            lvp=lvp,
            idm=idm,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            negative_prompt_embed=negative_prompt_embed,
            negative_prompt_embed_len=negative_prompt_embed_len,
            prompt_text=prompt_text,
            seed=seed + batch_index,
        )

        video_pred_batches.append(pred_video.detach().cpu())
        video_gt_batches.append(input_video.detach().cpu())

        latent_sqerr += float(
            torch.nn.functional.mse_loss(
                pred_latent_actions,
                teacher_latent_actions,
                reduction="sum",
            ).detach().cpu()
        )
        latent_count += int(teacher_latent_actions.numel())
        if teacher_latent_actions.shape[1] > context_len:
            future_pred = pred_latent_actions[:, context_len:]
            future_teacher = teacher_latent_actions[:, context_len:]
            future_sqerr += float(
                torch.nn.functional.mse_loss(
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
    if log_fn is not None:
        log_fn(
            f"{split_name}: windows={len(records)} elapsed={elapsed:.2f}s "
            + " ".join(f"{key}={value:.6f}" for key, value in metric_bundle.items())
        )
        for video_path in saved_video_paths:
            log_fn(f"{split_name}: saved_video={video_path}")

    return {
        "split": split_name,
        "num_windows": int(len(records)),
        "elapsed_seconds": float(elapsed),
        "metrics": metric_bundle.to_dict(),
        "saved_videos": saved_video_paths,
    }


def build_controller_fdm_wandb_payload(
    *,
    step: int,
    main_result: dict[str, Any] | None,
    right_target_result: dict[str, Any] | None,
    main_namespace: str = "controller_fdm_eval",
    main_table_namespace: str = "controller_fdm_eval_table",
    right_namespace: str = "controller_fdm_eval_right_target",
    right_table_namespace: str = "controller_fdm_eval_table_right_target",
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if main_result is not None:
        main_metrics = MetricBundle(dict(main_result.get("metrics", {})))
        payload.update(
            build_namespaced_log_payload(
                main_namespace,
                step=step,
                metrics=main_metrics,
                extra_scalars={
                    "num_windows": float(main_result.get("num_windows", 0)),
                    "elapsed_seconds": float(main_result.get("elapsed_seconds", 0.0)),
                },
            )
        )
        payload.update(
            build_namespaced_log_payload(
                main_table_namespace,
                step=step,
                metrics=build_eval_table_metric_bundle(main_metrics),
            )
        )
    if right_target_result is not None:
        right_metrics = MetricBundle(dict(right_target_result.get("metrics", {})))
        payload.update(
            build_namespaced_log_payload(
                right_namespace,
                step=step,
                metrics=right_metrics,
                extra_scalars={
                    "num_windows": float(right_target_result.get("num_windows", 0)),
                    "elapsed_seconds": float(right_target_result.get("elapsed_seconds", 0.0)),
                },
            )
        )
        payload.update(
            build_namespaced_log_payload(
                right_table_namespace,
                step=step,
                metrics=build_eval_table_metric_bundle(right_metrics),
            )
        )
    return payload


__all__ = [
    "build_controller_fdm_wandb_payload",
    "collate_window_samples",
    "compute_controller_video_batch",
    "run_controller_fdm_eval_suite",
]
