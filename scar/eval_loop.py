from __future__ import annotations

"""Evaluation orchestration for SCAR cycle training.

This module owns the periodic evaluation control flow used by ``train_cycle``:
- running full finite eval passes over the explicit eval loader
- optionally running cross-cycle eval forwards
- generating eval videos and cross-cycle comparison videos
- reducing eval metrics locally on rank 0
- assembling eval log payloads for console / WandB output

This module intentionally does not own:
- model / objective definitions
- optimizer or scheduler updates
- checkpoint save / restore logic
- train-step loss construction
"""

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .cross_cycle_objectives import compute_cross_cycle_objectives
from .evaluation import (
    all_gather_object_payload,
    build_sampling_batch,
    rollout_lvp_latent_trainable,
    save_eval_videos,
    video_tensor_to_uint8_numpy,
)
from .metrics import (
    MetricBundle,
    VideoMetricConfig,
    build_eval_table_metric_bundle,
    build_eval_video_metric,
    build_metric_bundle,
    build_namespaced_log_payload,
    compute_lvp_style_eval_video_metrics_from_batches,
    format_eval_summary as format_eval_metric_summary,
)
from .runtime import (
    reduce_pair_mean,
    select_rgb_channels,
    set_lvp_mode,
    take_batch_prefix,
    temporarily_unwrap_lvp_ddp,
    temporarily_disable_grads,
    to_lvp_range,
    trim_batch,
    unwrap_module,
)


@dataclass
class EvalLoopContext:
    """Holds pre-built, step-independent resources used by periodic eval."""

    eval_video_metric: Any | None
    video_metric_config: VideoMetricConfig


@dataclass
class EvalLoopResult:
    """Collects the outputs produced by one periodic evaluation round."""

    reduced_metrics: MetricBundle
    video_metrics: MetricBundle
    saved_videos: list[Path]
    right_target_reduced_metrics: MetricBundle = field(default_factory=MetricBundle)
    right_target_video_metrics: MetricBundle = field(default_factory=MetricBundle)
    right_target_saved_videos: list[Path] = field(default_factory=list)
    main_eval_seconds: float = 0.0
    right_target_eval_seconds: float = 0.0
    total_eval_seconds: float = 0.0


@dataclass
class _EvalVideoResult:
    """Carries the outputs from one eval video generation pass."""

    video_metrics: MetricBundle
    saved_videos: list[Path]


def _empty_eval_loop_result() -> EvalLoopResult:
    """Returns an empty eval result for ranks that do not own an eval suite."""

    return EvalLoopResult(
        reduced_metrics=MetricBundle(),
        video_metrics=MetricBundle(),
        saved_videos=[],
    )


def _has_suite_outputs(result: EvalLoopResult) -> bool:
    """Checks whether a single-suite eval result contains any meaningful outputs."""

    return bool(
        result.reduced_metrics
        or result.video_metrics
        or result.saved_videos
    )


def build_eval_loop_context(
    *,
    args,
    device: torch.device,
) -> EvalLoopContext:
    """Builds eval-only shared state once during startup.

    This keeps metric configuration out of the main training loop and gives
    ``train_cycle`` a single object to pass into periodic eval.
    """

    requested_eval_metric_names = list(args.metrics.video.names)
    eval_video_metric = None
    eval_video_metric_names: list[str] = []
    if requested_eval_metric_names:
        eval_video_metric, eval_video_metric_names = build_eval_video_metric(
            requested_eval_metric_names,
            device=device,
            split_batch_size=args.metrics.video.batch_size,
        )

    return EvalLoopContext(
        eval_video_metric=eval_video_metric,
        video_metric_config=VideoMetricConfig(
            names=tuple(eval_video_metric_names),
            batch_size=int(args.metrics.video.batch_size),
            max_video_count=max(int(args.metrics.video.max_video_count), 0),
        ),
    )


def _build_eval_scalar_metrics(eval_output) -> MetricBundle:
    """Maps objective outputs into the eval scalar metric namespace."""

    return eval_output.metrics


def _apply_cross_cycle_eval_metrics(
    *,
    eval_metrics_list: list[MetricBundle],
    cross_output,
) -> list[MetricBundle]:
    """Attaches cross-cycle eval metrics to every per-batch eval metric dict."""

    cross_metrics = (
        cross_output.metrics
        if cross_output is not None
        else build_metric_bundle("cross_cycle")
    )
    return [metrics.merge(cross_metrics) for metrics in eval_metrics_list]


def run_eval_video_generation(
    *,
    args,
    step: int,
    lvp,
    idm,
    eval_loader,
    seq_len: int,
    lvp_cfg,
    lvp_trainable_modules: set[str],
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    dist_ctx,
    device,
    output_dir: Path,
    wandb_run,
    ctx: EvalLoopContext,
    Batch,
    to_device,
    allow_non_main_process: bool = False,
) -> _EvalVideoResult:
    """Generates eval videos, computes video metrics, and saves sample videos."""

    from .objectives import (
        build_lvp_condition_latents,
        encode_lvp_video_latents,
        resolve_conditioning_actions,
    )

    eval_video_metrics = MetricBundle()
    saved_eval_videos: list[Path] = []
    local_metric_video_preds: list[torch.Tensor] = []
    local_metric_video_gts: list[torch.Tensor] = []

    if not dist_ctx.is_main_process and not allow_non_main_process:
        return _EvalVideoResult(video_metrics=MetricBundle(), saved_videos=[])

    metric_needs_full_pass = ctx.eval_video_metric is not None
    max_saved_videos = (
        max(int(args.eval_video_count), 0)
        if args.eval_generate_video and dist_ctx.is_main_process
        else 0
    )

    saved_any_videos = False
    for metric_np in eval_loader:

        metric_batch = Batch(**to_device(metric_np, device))
        metric_batch = trim_batch(metric_batch, seq_len)

        _, metric_video_lat = encode_lvp_video_latents(
            lvp,
            metric_batch,
            lvp_trainable_modules=lvp_trainable_modules,
        )
        metric_conditioning_actions = resolve_conditioning_actions(
            action_source=args.lvp_action_source,
            batch=metric_batch,
            video_lat=metric_video_lat,
            idm=idm,
            target_frame_tokens=seq_len,
        )
        sampling_batch = build_sampling_batch(
            lvp,
            metric_batch,
            metric_conditioning_actions.detach(),
            prompt_text=args.prompt,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            negative_prompt_embed=negative_prompt_embed,
            negative_prompt_embed_len=negative_prompt_embed_len,
        )
        _, sampled_batch, _, sampled_video = rollout_lvp_latent_trainable(
            lvp,
            sampling_batch,
            build_lvp_condition_latents_fn=build_lvp_condition_latents,
            video_lat=metric_video_lat.detach(),
            decode_video=True,
        )
        if sampled_video is None:
            raise RuntimeError("Expected decoded eval video from latent rollout.")

        if ctx.eval_video_metric is not None:
            local_metric_video_preds.append(sampled_video.detach().cpu())
            local_metric_video_gts.append(sampled_batch["videos"].detach().cpu())

        if (
            args.eval_generate_video
            and max_saved_videos > 0
            and not saved_any_videos
        ):
            saved_eval_videos = save_eval_videos(
                output_dir,
                step,
                video_gt=sampled_batch["videos"],
                video_pred=sampled_video,
                fps=args.eval_video_fps,
                hist_len=int(lvp_cfg.algorithm.hist_len),
                max_count=max_saved_videos,
                wandb_run=wandb_run,
            )
            saved_any_videos = True

        if not metric_needs_full_pass and saved_any_videos:
            break

    if ctx.eval_video_metric is not None:
        eval_video_metrics = compute_lvp_style_eval_video_metrics_from_batches(
            ctx.eval_video_metric,
            video_pred_batches=local_metric_video_preds,
            video_gt_batches=local_metric_video_gts,
            hist_len=int(lvp_cfg.algorithm.hist_len),
            n_metrics_frames=getattr(
                lvp_cfg.algorithm.logging,
                "n_metrics_frames",
                None,
            ),
            split_batch_size=ctx.video_metric_config.batch_size,
            max_video_count=None,
        )

    return _EvalVideoResult(
        video_metrics=eval_video_metrics,
        saved_videos=saved_eval_videos,
    )


def run_cross_cycle_eval_video(
    *,
    cross_cycle,
    args,
    step: int,
    lvp,
    idm,
    seq_len: int,
    lvp_cfg,
    lvp_trainable_modules: set[str],
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    device,
    output_dir: Path,
    wandb_run,
    Batch,
    to_device,
    eval_batch,
) -> list[Path]:
    """Generates cross-cycle comparison videos for eval and saves them."""

    from .objectives import (
        build_lvp_condition_latents,
        encode_lvp_video_latents,
        resolve_conditioning_actions,
    )
    from utils.video_utils import write_numpy_to_mp4

    max_count = max(args.eval_video_count, 1)

    primary_batch = take_batch_prefix(eval_batch, max_count)

    secondary_np = next(cross_cycle.secondary_iter)
    secondary_batch = Batch(**to_device(secondary_np, device))
    secondary_batch = trim_batch(secondary_batch, seq_len)
    secondary_batch = take_batch_prefix(secondary_batch, max_count)

    _, video_lat_primary = encode_lvp_video_latents(
        lvp,
        primary_batch,
        lvp_trainable_modules=lvp_trainable_modules,
    )
    conditioning_actions = resolve_conditioning_actions(
        action_source=args.lvp_action_source,
        batch=primary_batch,
        video_lat=video_lat_primary,
        idm=idm,
        target_frame_tokens=seq_len,
    )

    _, video_lat_secondary = encode_lvp_video_latents(
        lvp,
        secondary_batch,
        lvp_trainable_modules=lvp_trainable_modules,
    )

    sampling_batch = build_sampling_batch(
        lvp,
        secondary_batch,
        conditioning_actions.detach(),
        prompt_text=args.prompt,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
        negative_prompt_embed=negative_prompt_embed,
        negative_prompt_embed_len=negative_prompt_embed_len,
    )

    _, sampled_batch, _, sampled_video = rollout_lvp_latent_trainable(
        lvp,
        sampling_batch,
        build_lvp_condition_latents_fn=build_lvp_condition_latents,
        video_lat=video_lat_secondary.detach(),
        decode_video=True,
    )
    if sampled_video is None:
        return []

    eval_video_dir = output_dir / "eval_videos"
    eval_video_dir.mkdir(parents=True, exist_ok=True)
    hist_len = int(lvp_cfg.algorithm.hist_len)
    num_samples = min(
        max_count,
        primary_batch.observations.shape[0],
        sampled_video.shape[0],
    )

    primary_videos = select_rgb_channels(primary_batch.observations)
    primary_videos = to_lvp_range(primary_videos).contiguous().detach()

    saved_paths: list[Path] = []
    for sample_idx in range(num_samples):
        primary_np = video_tensor_to_uint8_numpy(primary_videos[sample_idx])
        secondary_np = video_tensor_to_uint8_numpy(sampled_batch["videos"][sample_idx])
        pred_np = video_tensor_to_uint8_numpy(sampled_video[sample_idx]).copy()

        if hist_len < pred_np.shape[0]:
            pred_np[hist_len:, :2, :, :] = 255
            pred_np[hist_len:, -2:, :, :] = 255
            pred_np[hist_len:, :, :2, :] = 255
            pred_np[hist_len:, :, -2:, :] = 255

        comparison = np.concatenate([primary_np, secondary_np, pred_np], axis=2)

        video_path = (
            eval_video_dir
            / f"step_{step:07d}_cross_cycle_sample_{sample_idx:02d}_compare.mp4"
        )
        write_numpy_to_mp4(comparison, str(video_path), fps=args.eval_video_fps)
        saved_paths.append(video_path)

        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        "eval/step": step,
                        f"eval/videos/cross_cycle/sample_{sample_idx:02d}": wandb.Video(
                            str(video_path),
                            fps=args.eval_video_fps,
                            format="mp4",
                        ),
                    }
                )
            except Exception:
                pass

    return saved_paths


def reduce_eval_metrics(
    *,
    eval_metrics_list: list[MetricBundle],
    eval_metric_weights: list[float] | None = None,
    dist_ctx,
    distributed: bool = True,
) -> MetricBundle:
    """Averages per-batch eval metrics locally and across ranks."""

    reduced_eval_metrics: dict[str, float] = {}
    if not eval_metrics_list:
        return MetricBundle()
    if eval_metric_weights is None:
        eval_metric_weights = [1.0] * len(eval_metrics_list)
    if len(eval_metric_weights) != len(eval_metrics_list):
        raise ValueError(
            "eval_metric_weights must have the same length as eval_metrics_list, "
            f"got {len(eval_metric_weights)} and {len(eval_metrics_list)}"
        )

    metric_keys = sorted({key for metrics in eval_metrics_list for key in metrics.keys()})
    for key in metric_keys:
        local_sum = sum(
            (metric.get(key, 0.0) or 0.0) * float(weight)
            for metric, weight in zip(eval_metrics_list, eval_metric_weights, strict=True)
        )
        local_count = sum(float(weight) for weight in eval_metric_weights)
        if distributed:
            reduced_eval_metrics[key] = reduce_pair_mean(
                local_sum,
                local_count,
                dist_ctx,
            )
        else:
            reduced_eval_metrics[key] = float(local_sum) / max(int(local_count), 1)
    return MetricBundle(reduced_eval_metrics)


def format_eval_summary(
    *,
    step: int,
    reduced_metrics: MetricBundle,
    prefix: str = "[eval]",
) -> str | None:
    """Formats the main eval scalar summary line for console logging."""

    summary = format_eval_metric_summary(step=step, metrics=reduced_metrics)
    if summary is None or prefix == "[eval]":
        return summary
    return re.sub(r"^\[eval\]", prefix, summary, count=1)


def build_eval_log_payload(
    *,
    step: int,
    result: EvalLoopResult,
) -> dict[str, Any]:
    """Builds the eval WandB payload from the reduced eval result."""

    merged_metrics = result.reduced_metrics.merge(result.video_metrics)
    payload = build_namespaced_log_payload(
        "eval",
        step=step,
        metrics=merged_metrics,
    )
    payload.update(
        build_namespaced_log_payload(
            "eval_table",
            step=step,
            metrics=build_eval_table_metric_bundle(merged_metrics),
        )
    )
    if result.saved_videos:
        try:
            import wandb

            for video_path in result.saved_videos:
                if "_cross_cycle_" in video_path.name:
                    continue
                match = re.search(r"sample_(\d+)", video_path.name)
                sample_idx = int(match.group(1)) if match is not None else 0
                payload[f"eval/videos/sample_{sample_idx:02d}"] = wandb.Video(
                    str(video_path),
                    format="mp4",
                )
        except Exception:
            pass
    aux_merged_metrics = result.right_target_reduced_metrics.merge(
        result.right_target_video_metrics
    )
    if aux_merged_metrics:
        payload.update(
            build_namespaced_log_payload(
                "eval_right_target",
                step=step,
                metrics=aux_merged_metrics,
            )
        )
        payload.update(
            build_namespaced_log_payload(
                "eval_table_right_target",
                step=step,
                metrics=build_eval_table_metric_bundle(aux_merged_metrics),
            )
        )
    if result.right_target_saved_videos:
        try:
            import wandb

            for video_path in result.right_target_saved_videos:
                match = re.search(r"sample_(\d+)", video_path.name)
                sample_idx = int(match.group(1)) if match is not None else 0
                payload[f"eval_right_target/videos/sample_{sample_idx:02d}"] = wandb.Video(
                    str(video_path),
                    format="mp4",
                )
        except Exception:
            pass
    return payload


def _make_eval_runtime_args(
    args,
    *,
    generate_video: bool,
    video_count: int | None = None,
):
    return SimpleNamespace(
        eval_generate_video=bool(generate_video),
        eval_video_count=int(args.eval_video_count if video_count is None else video_count),
        eval_video_fps=int(args.eval_video_fps),
        lvp_action_source=args.lvp_action_source,
        prompt=args.prompt,
    )


def _run_single_eval_suite(
    *,
    args,
    step: int,
    lvp,
    idm,
    gt_action_head,
    ema,
    cross_cycle,
    cross_cycle_eval_enabled: bool,
    eval_loader,
    seq_len: int,
    lvp_cfg,
    lvp_trainable_modules: set[str],
    lvp_trainable: list[Any],
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    dist_ctx,
    device,
    output_dir: Path,
    wandb_run,
    ctx: EvalLoopContext,
    enable_video: bool,
    allow_non_main_process: bool = False,
) -> EvalLoopResult:
    """Runs one finite eval suite over one loader."""

    from scar.dataloader import Batch, to_device
    from .objectives import compute_cycle_objectives

    if (not dist_ctx.is_main_process and not allow_non_main_process) or eval_loader is None:
        return _empty_eval_loop_result()

    eval_args = _make_eval_runtime_args(args, generate_video=enable_video)
    freeze_idm = bool(getattr(args, "freeze_idm", False))
    eval_idm = unwrap_module(idm)
    eval_gt_action_head = unwrap_module(gt_action_head)
    eval_lvp_trainable = [unwrap_module(module) for module in lvp_trainable]

    if eval_idm is not None:
        eval_idm.eval()
    if eval_gt_action_head is not None:
        eval_gt_action_head.eval()

    eval_metrics_list: list[MetricBundle] = []
    eval_metric_weights: list[float] = []
    eval_video_metrics = MetricBundle()
    saved_eval_videos: list[Path] = []
    last_eval_batch = None

    with temporarily_unwrap_lvp_ddp(lvp):
        set_lvp_mode(lvp, lvp_trainable_modules, training=False)
        with temporarily_disable_grads(eval_idm, eval_gt_action_head, *eval_lvp_trainable):
            with torch.no_grad():
                for eval_np in eval_loader:
                    eval_batch = Batch(**to_device(eval_np, device))
                    eval_batch = trim_batch(eval_batch, seq_len)
                    last_eval_batch = eval_batch
                    eval_metric_weights.append(float(eval_batch.observations.shape[0]))
                    eval_output = compute_cycle_objectives(
                        args=args,
                        step=step,
                        training=False,
                        lvp=lvp,
                        idm=eval_idm,
                        gt_action_head=eval_gt_action_head,
                        batch=eval_batch,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        lvp_trainable_modules=lvp_trainable_modules,
                        ema=ema,
                    )
                    eval_metrics_list.append(_build_eval_scalar_metrics(eval_output))

                cross_output = None
                if (
                    cross_cycle is not None
                    and cross_cycle_eval_enabled
                    and last_eval_batch is not None
                ):
                    cc_secondary_np = next(cross_cycle.secondary_iter)
                    cc_secondary_batch = Batch(**to_device(cc_secondary_np, device))
                    cc_secondary_batch = trim_batch(cc_secondary_batch, seq_len)
                    cross_output = compute_cross_cycle_objectives(
                        args=args,
                        step=step,
                        training=False,
                        lvp=lvp,
                        idm=eval_idm,
                        batch_primary=last_eval_batch,
                        batch_secondary=cc_secondary_batch,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        lvp_trainable_modules=lvp_trainable_modules,
                        ema=ema,
                    )
                eval_metrics_list = _apply_cross_cycle_eval_metrics(
                    eval_metrics_list=eval_metrics_list,
                    cross_output=cross_output,
                )

                needs_eval_video_rollout = (
                    eval_args.eval_generate_video or ctx.eval_video_metric is not None
                )
                if needs_eval_video_rollout:
                    video_result = run_eval_video_generation(
                        args=eval_args,
                        step=step,
                        lvp=lvp,
                        idm=eval_idm,
                        eval_loader=eval_loader,
                        seq_len=seq_len,
                        lvp_cfg=lvp_cfg,
                        lvp_trainable_modules=lvp_trainable_modules,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        negative_prompt_embed=negative_prompt_embed,
                        negative_prompt_embed_len=negative_prompt_embed_len,
                        dist_ctx=dist_ctx,
                        device=device,
                        output_dir=output_dir,
                        wandb_run=wandb_run,
                        ctx=ctx,
                        Batch=Batch,
                        to_device=to_device,
                        allow_non_main_process=allow_non_main_process,
                    )
                    eval_video_metrics = video_result.video_metrics
                    saved_eval_videos = video_result.saved_videos

                if (
                    cross_cycle is not None
                    and cross_cycle_eval_enabled
                    and eval_args.eval_generate_video
                    and last_eval_batch is not None
                ):
                    saved_cross_videos = run_cross_cycle_eval_video(
                        cross_cycle=cross_cycle,
                        args=eval_args,
                        step=step,
                        lvp=lvp,
                        idm=eval_idm,
                        seq_len=seq_len,
                        lvp_cfg=lvp_cfg,
                        lvp_trainable_modules=lvp_trainable_modules,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        negative_prompt_embed=negative_prompt_embed,
                        negative_prompt_embed_len=negative_prompt_embed_len,
                        device=device,
                        output_dir=output_dir,
                        wandb_run=wandb_run,
                        Batch=Batch,
                        to_device=to_device,
                        eval_batch=last_eval_batch,
                    )
                    saved_eval_videos.extend(saved_cross_videos)

    reduced_eval_metrics = reduce_eval_metrics(
        eval_metrics_list=eval_metrics_list,
        eval_metric_weights=eval_metric_weights,
        dist_ctx=dist_ctx,
        distributed=False,
    )

    if idm is not None:
        if freeze_idm:
            idm.eval()
        else:
            idm.train()
    if gt_action_head is not None:
        gt_action_head.train()
    set_lvp_mode(lvp, lvp_trainable_modules, training=True)

    return EvalLoopResult(
        reduced_metrics=reduced_eval_metrics,
        video_metrics=eval_video_metrics,
        saved_videos=saved_eval_videos,
    )


def run_periodic_eval(
    *,
    args,
    step: int,
    lvp,
    idm,
    gt_action_head,
    ema,
    cross_cycle,
    cross_cycle_eval_enabled: bool,
    eval_loader,
    right_target_eval_loader,
    seq_len: int,
    lvp_cfg,
    lvp_trainable_modules: set[str],
    lvp_trainable: list[Any],
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    dist_ctx,
    device,
    output_dir: Path,
    wandb_run,
    ctx: EvalLoopContext,
) -> EvalLoopResult:
    """Runs one periodic eval round and returns reduced scalars plus videos."""
    overall_eval_start = time.time()

    can_parallelize_two_suite_eval = (
        dist_ctx.enabled
        and dist_ctx.world_size >= 2
        and right_target_eval_loader is not None
    )

    if can_parallelize_two_suite_eval:
        main_result = _empty_eval_loop_result()
        right_target_result = _empty_eval_loop_result()
        main_eval_seconds = 0.0
        right_target_eval_seconds = 0.0

        if dist_ctx.rank == 0:
            main_eval_start = time.time()
            main_result = _run_single_eval_suite(
                args=args,
                step=step,
                lvp=lvp,
                idm=idm,
                gt_action_head=gt_action_head,
                ema=ema,
                cross_cycle=cross_cycle,
                cross_cycle_eval_enabled=cross_cycle_eval_enabled,
                eval_loader=eval_loader,
                seq_len=seq_len,
                lvp_cfg=lvp_cfg,
                lvp_trainable_modules=lvp_trainable_modules,
                lvp_trainable=lvp_trainable,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                negative_prompt_embed=negative_prompt_embed,
                negative_prompt_embed_len=negative_prompt_embed_len,
                dist_ctx=dist_ctx,
                device=device,
                output_dir=output_dir,
                wandb_run=wandb_run,
                ctx=ctx,
                enable_video=bool(args.eval_generate_video),
            )
            main_eval_seconds = time.time() - main_eval_start
        elif dist_ctx.rank == 1:
            right_target_eval_start = time.time()
            right_target_result = _run_single_eval_suite(
                args=args,
                step=step,
                lvp=lvp,
                idm=idm,
                gt_action_head=gt_action_head,
                ema=ema,
                cross_cycle=None,
                cross_cycle_eval_enabled=False,
                eval_loader=right_target_eval_loader,
                seq_len=seq_len,
                lvp_cfg=lvp_cfg,
                lvp_trainable_modules=lvp_trainable_modules,
                lvp_trainable=lvp_trainable,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                negative_prompt_embed=negative_prompt_embed,
                negative_prompt_embed_len=negative_prompt_embed_len,
                dist_ctx=dist_ctx,
                device=device,
                output_dir=output_dir,
                wandb_run=None,
                ctx=ctx,
                enable_video=False,
                allow_non_main_process=True,
            )
            right_target_eval_seconds = time.time() - right_target_eval_start

        gathered_payloads = all_gather_object_payload(
            {
                "rank": int(dist_ctx.rank),
                "main_result": main_result,
                "main_eval_seconds": float(main_eval_seconds),
                "right_target_result": right_target_result,
                "right_target_eval_seconds": float(right_target_eval_seconds),
            },
            dist_ctx,
        )

        if not dist_ctx.is_main_process:
            return _empty_eval_loop_result()

        merged_main_result = _empty_eval_loop_result()
        merged_right_target_result = _empty_eval_loop_result()
        merged_main_seconds = 0.0
        merged_right_target_seconds = 0.0
        for payload in gathered_payloads:
            if not isinstance(payload, dict):
                continue
            payload_main = payload.get("main_result")
            if isinstance(payload_main, EvalLoopResult) and _has_suite_outputs(payload_main):
                merged_main_result = payload_main
                merged_main_seconds = float(payload.get("main_eval_seconds", 0.0) or 0.0)
            payload_right = payload.get("right_target_result")
            if isinstance(payload_right, EvalLoopResult) and _has_suite_outputs(payload_right):
                merged_right_target_result = payload_right
                merged_right_target_seconds = float(
                    payload.get("right_target_eval_seconds", 0.0) or 0.0
                )

        return EvalLoopResult(
            reduced_metrics=merged_main_result.reduced_metrics,
            video_metrics=merged_main_result.video_metrics,
            saved_videos=merged_main_result.saved_videos,
            right_target_reduced_metrics=merged_right_target_result.reduced_metrics,
            right_target_video_metrics=merged_right_target_result.video_metrics,
            right_target_saved_videos=merged_right_target_result.saved_videos,
            main_eval_seconds=merged_main_seconds,
            right_target_eval_seconds=merged_right_target_seconds,
            total_eval_seconds=time.time() - overall_eval_start,
        )

    if not dist_ctx.is_main_process:
        return _empty_eval_loop_result()

    main_eval_start = time.time()
    main_result = _run_single_eval_suite(
        args=args,
        step=step,
        lvp=lvp,
        idm=idm,
        gt_action_head=gt_action_head,
        ema=ema,
        cross_cycle=cross_cycle,
        cross_cycle_eval_enabled=cross_cycle_eval_enabled,
        eval_loader=eval_loader,
        seq_len=seq_len,
        lvp_cfg=lvp_cfg,
        lvp_trainable_modules=lvp_trainable_modules,
        lvp_trainable=lvp_trainable,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
        negative_prompt_embed=negative_prompt_embed,
        negative_prompt_embed_len=negative_prompt_embed_len,
        dist_ctx=dist_ctx,
        device=device,
        output_dir=output_dir,
        wandb_run=wandb_run,
        ctx=ctx,
        enable_video=bool(args.eval_generate_video),
    )
    main_eval_seconds = time.time() - main_eval_start

    right_target_result = _empty_eval_loop_result()
    right_target_eval_seconds = 0.0
    if right_target_eval_loader is not None:
        right_target_eval_start = time.time()
        right_target_result = _run_single_eval_suite(
            args=args,
            step=step,
            lvp=lvp,
            idm=idm,
            gt_action_head=gt_action_head,
            ema=ema,
            cross_cycle=None,
            cross_cycle_eval_enabled=False,
            eval_loader=right_target_eval_loader,
            seq_len=seq_len,
            lvp_cfg=lvp_cfg,
            lvp_trainable_modules=lvp_trainable_modules,
            lvp_trainable=lvp_trainable,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            negative_prompt_embed=negative_prompt_embed,
            negative_prompt_embed_len=negative_prompt_embed_len,
            dist_ctx=dist_ctx,
            device=device,
            output_dir=output_dir,
            wandb_run=wandb_run,
            ctx=ctx,
            enable_video=False,
        )
        right_target_eval_seconds = time.time() - right_target_eval_start

    return EvalLoopResult(
        reduced_metrics=main_result.reduced_metrics,
        video_metrics=main_result.video_metrics,
        saved_videos=main_result.saved_videos,
        right_target_reduced_metrics=right_target_result.reduced_metrics,
        right_target_video_metrics=right_target_result.video_metrics,
        right_target_saved_videos=right_target_result.saved_videos,
        main_eval_seconds=main_eval_seconds,
        right_target_eval_seconds=right_target_eval_seconds,
        total_eval_seconds=time.time() - overall_eval_start,
    )
