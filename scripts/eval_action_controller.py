#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from infer_action_transfer import (
    build_split_records,
    filter_records,
    load_run_artifacts,
    maybe_rebase_project_path,
)
from scar.controller_fdm import (
    build_controller_fdm_cache,
    load_controller_checkpoint,
    load_controller_summary,
    resolve_backbone_run_and_ckpt,
    run_controller_fdm_eval_suite_from_cache,
)
from scar.cycle_api import (
    DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED,
    DEFAULT_LIBERO_PROMPT,
    DEFAULT_LIBERO_PROMPT_EMBED,
    align_idm_seq_len,
    align_lvp_action_dim,
    build_lvp_prior,
    get_lvp_target_seq_len,
    load_prompt_embedding,
    resolve_conditioning_action_dim,
    set_lvp_mode,
    set_seed,
)
from scar.evaluation import all_gather_object_payload
from scar.metrics import (
    MetricBundle,
    build_eval_table_metric_bundle,
    build_eval_video_metric,
    build_namespaced_log_payload,
)
from scar.models import LatentSpaceIDM
from scar.runtime import (
    cleanup_distributed,
    distributed_barrier,
    load_checkpoint,
    setup_distributed,
)


def log_step(message: str) -> None:
    print(f"[controller-eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a latent-action controller checkpoint on full eval and right_target_eval "
            "splits, then compute whole-split video metrics."
        )
    )
    parser.add_argument(
        "--controller-ckpt",
        required=True,
        help="Path to latent_action_controller_*.pt checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional SCAR training run dir. If omitted, infer from controller summary.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional SCAR checkpoint. If omitted, infer from controller summary.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "eval", "right_target_eval"],
        default="eval",
    )
    parser.add_argument(
        "--right-target-split",
        choices=["", "train", "eval", "right_target_eval"],
        default="right_target_eval",
    )
    parser.add_argument(
        "--dataset-filter",
        default="franka",
        help="Optional substring filter applied to the main eval split.",
    )
    parser.add_argument(
        "--right-target-dataset-filter",
        default="franka",
        help="Optional substring filter applied to the right-target eval split.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=0,
        help="Override controller context length in frames. <=0 uses the checkpoint setting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Eval batch size in windows. <=0 reuses the backbone run batch size.",
    )
    parser.add_argument(
        "--save-video-count",
        type=int,
        default=1,
        help="How many qualitative videos to save per split.",
    )
    parser.add_argument("--save-video-fps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <controller-run-dir>/controller_fdm_eval_<controller-ckpt-stem>/.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-embed-path", default=None)
    parser.add_argument("--negative-prompt-embed-path", default=None)
    return parser.parse_args()


def _maybe_init_wandb(
    *,
    cli_args: argparse.Namespace,
    controller_ckpt_path: Path,
    controller_summary: dict[str, Any],
    output_dir: Path,
) -> Any | None:
    if not bool(cli_args.wandb):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in the current environment. Install it or rerun without --wandb."
        ) from exc

    project = cli_args.wandb_project.strip() or "scar-robotwin"
    entity = cli_args.wandb_entity.strip() or None
    mode = cli_args.wandb_mode.strip() or "online"
    default_name = (
        f"{Path(controller_summary.get('run_dir', output_dir.parent)).name}"
        f"__controller_fdm_eval__{controller_ckpt_path.stem}"
    )
    name = cli_args.wandb_name.strip() or default_name

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        mode=mode,
        dir=str(output_dir),
        config={
            "controller_ckpt": str(controller_ckpt_path),
            "run_dir": controller_summary.get("run_dir", ""),
            "backbone_ckpt": controller_summary.get("ckpt_path", ""),
            "eval_split": cli_args.eval_split,
            "right_target_split": cli_args.right_target_split,
            "dataset_filter": cli_args.dataset_filter,
            "right_target_dataset_filter": cli_args.right_target_dataset_filter,
            "context_len": int(cli_args.context_len),
            "batch_size": int(cli_args.batch_size),
        },
    )
    wandb.define_metric("controller_fdm_eval/step")
    wandb.define_metric("controller_fdm_eval/*", step_metric="controller_fdm_eval/step")
    wandb.define_metric("controller_fdm_eval_table/step")
    wandb.define_metric(
        "controller_fdm_eval_table/*",
        step_metric="controller_fdm_eval_table/step",
    )
    wandb.define_metric("controller_fdm_eval_right_target/step")
    wandb.define_metric(
        "controller_fdm_eval_right_target/*",
        step_metric="controller_fdm_eval_right_target/step",
    )
    wandb.define_metric("controller_fdm_eval_table_right_target/step")
    wandb.define_metric(
        "controller_fdm_eval_table_right_target/*",
        step_metric="controller_fdm_eval_table_right_target/step",
    )
    return run


def _build_wandb_payload(
    *,
    step: int,
    main_result: dict[str, Any] | None,
    right_target_result: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if main_result is not None:
        main_metrics = MetricBundle(dict(main_result.get("metrics", {})))
        payload.update(
            build_namespaced_log_payload(
                "controller_fdm_eval",
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
                "controller_fdm_eval_table",
                step=step,
                metrics=build_eval_table_metric_bundle(main_metrics),
            )
        )
    if right_target_result is not None:
        right_metrics = MetricBundle(dict(right_target_result.get("metrics", {})))
        payload.update(
            build_namespaced_log_payload(
                "controller_fdm_eval_right_target",
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
                "controller_fdm_eval_table_right_target",
                step=step,
                metrics=build_eval_table_metric_bundle(right_metrics),
            )
        )
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    controller_ckpt_path = Path(args.controller_ckpt).resolve()
    if not controller_ckpt_path.is_file():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_ckpt_path}")

    controller_summary_path, controller_summary = load_controller_summary(controller_ckpt_path)
    run_dir, ckpt_path = resolve_backbone_run_and_ckpt(
        run_dir=args.run_dir,
        ckpt=args.ckpt,
        controller_summary=controller_summary,
    )
    artifacts = load_run_artifacts(run_dir, ckpt_path)

    if args.output_dir is None:
        controller_run_dir = controller_ckpt_path.parent.parent
        output_dir = controller_run_dir / f"controller_fdm_eval_{controller_ckpt_path.stem}"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_ctx, device = setup_distributed(args)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    controller_payload, controller = load_controller_checkpoint(controller_ckpt_path, device)
    controller_step = int(controller_payload.get("controller_step", -1))
    backbone_step = int(controller_payload.get("global_step", -1))

    default_context_len = int(controller_payload.get("controller_config", {}).get("context_len", 0) or 0)
    if default_context_len <= 0:
        default_context_len = int(controller_summary.get("context_len", 0) or 0)
    if default_context_len <= 0:
        default_context_len = int(artifacts.lvp_cfg.algorithm.hist_len)
    context_len = int(args.context_len) if int(args.context_len) > 0 else int(default_context_len)

    if dist_ctx.is_main_process:
        log_step(f"controller_ckpt={controller_ckpt_path}")
        if controller_summary_path is not None:
            log_step(f"controller_summary={controller_summary_path}")
        log_step(f"run_dir={run_dir}")
        log_step(f"ckpt={ckpt_path}")
        log_step(f"output_dir={output_dir}")
        log_step(f"device={device}")
        log_step(f"context_len={context_len}")
        log_step(f"controller_step={controller_step}, backbone_step={backbone_step}")
        log_step(
            "controller_architecture="
            f"{controller_payload.get('controller_config', {}).get('architecture', 'summary_add')}"
        )

    ckpt_probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt_probe.get("idm") is None:
        raise ValueError("Base checkpoint does not contain IDM weights; controller eval requires IDM.")

    conditioning_action_dim = resolve_conditioning_action_dim(
        artifacts.idm_cfg,
        action_source="idm",
    )
    align_lvp_action_dim(artifacts.lvp_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(artifacts.lvp_cfg)
    if int(artifacts.idm_cfg.data.seq_len) != target_seq_len:
        if dist_ctx.is_main_process:
            log_step(
                f"aligning IDM seq_len from {int(artifacts.idm_cfg.data.seq_len)} to {target_seq_len}"
            )
        align_idm_seq_len(artifacts.idm_cfg, target_seq_len)

    prompt_text = args.prompt or artifacts.args_dict.get("prompt") or DEFAULT_LIBERO_PROMPT
    prompt_embed_path = maybe_rebase_project_path(
        args.prompt_embed_path
        or artifacts.args_dict.get("prompt_embed_path")
        or str(DEFAULT_LIBERO_PROMPT_EMBED)
    )
    negative_prompt_embed_path = maybe_rebase_project_path(
        args.negative_prompt_embed_path
        or artifacts.args_dict.get("negative_prompt_embed_path")
        or str(DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED)
    )
    prompt_embed, prompt_embed_len = load_prompt_embedding(prompt_embed_path)
    negative_prompt_embed, negative_prompt_embed_len = load_prompt_embedding(
        negative_prompt_embed_path
    )

    ckpt_lvp_modules = set(ckpt_probe.get("lvp_trainable_modules", []))
    lvp = build_lvp_prior(
        artifacts.lvp_cfg,
        device=device,
        trainable_modules=ckpt_lvp_modules,
    )
    latent_input_dim = (int(lvp.lat_c), int(lvp.lat_h), int(lvp.lat_w))
    latent_temporal_stride = int(lvp.vae_stride[0])
    latent_idm_cfg = OmegaConf.create(
        OmegaConf.to_container(artifacts.idm_cfg.model.idm, resolve=True)
    )
    latent_idm_cfg.patch_size = 1
    idm = LatentSpaceIDM(
        latent_idm_cfg,
        input_dim=latent_input_dim,
        la_dim=int(artifacts.idm_cfg.model.la_dim),
        temporal_stride=latent_temporal_stride,
    ).to(device)

    load_checkpoint(
        ckpt_path,
        idm=idm,
        lvp=lvp,
        lvp_trainable_modules=ckpt_lvp_modules,
        map_location=device,
    )
    idm.eval()
    lvp.eval()
    set_lvp_mode(lvp, ckpt_lvp_modules, training=False)

    main_records = filter_records(
        build_split_records(artifacts, args.eval_split),
        args.dataset_filter,
        split_name=args.eval_split,
    )
    right_target_records = []
    if args.right_target_split:
        right_target_records = filter_records(
            build_split_records(artifacts, args.right_target_split),
            args.right_target_dataset_filter,
            split_name=args.right_target_split,
        )

    all_seq_len_records = main_records or right_target_records
    if not all_seq_len_records:
        raise ValueError("No records available for controller eval after filtering.")
    seq_len = int(all_seq_len_records[0].sample_np["observations"].shape[0])
    if seq_len != target_seq_len:
        raise RuntimeError(f"Expected seq_len={target_seq_len}, got {seq_len}.")
    if context_len <= 0 or context_len > seq_len:
        raise ValueError(f"context_len must be in [1, {seq_len}], got {context_len}")

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(
        artifacts.args_dict.get("batch_size", 8) or 8
    )
    metric_names = list(artifacts.args_dict.get("metric_video_names", ["mse", "psnr", "ssim"]))
    _, resolved_metric_names = build_eval_video_metric(
        metric_names,
        device=device,
        split_batch_size=int(artifacts.args_dict.get("metric_video_batch_size", 16) or 16),
    )
    metric_batch_size = int(artifacts.args_dict.get("metric_video_batch_size", 16) or 16)
    n_metrics_frames = getattr(artifacts.lvp_cfg.algorithm.logging, "n_metrics_frames", None)
    hist_len = int(artifacts.lvp_cfg.algorithm.hist_len)

    if dist_ctx.is_main_process:
        log_step(f"{args.eval_split}: building cache for {len(main_records)} windows")
    main_cache = build_controller_fdm_cache(
        split_name=args.eval_split,
        records=main_records,
        device=device,
        seq_len=seq_len,
        context_len=context_len,
        batch_size=batch_size,
        lvp=lvp,
        idm=idm,
        include_input_video=True,
        status_fn=log_step if dist_ctx.is_main_process else None,
    )
    right_target_cache = None
    if right_target_records:
        if dist_ctx.is_main_process:
            log_step(f"{args.right_target_split}: building cache for {len(right_target_records)} windows")
        right_target_cache = build_controller_fdm_cache(
            split_name=args.right_target_split,
            records=right_target_records,
            device=device,
            seq_len=seq_len,
            context_len=context_len,
            batch_size=batch_size,
            lvp=lvp,
            idm=idm,
            include_input_video=True,
            status_fn=log_step if dist_ctx.is_main_process else None,
        )
    del idm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    wandb_run = None
    if dist_ctx.is_main_process:
        wandb_run = _maybe_init_wandb(
            cli_args=args,
            controller_ckpt_path=controller_ckpt_path,
            controller_summary=controller_summary,
            output_dir=output_dir,
        )

    can_parallelize_two_suite_eval = (
        dist_ctx.enabled and dist_ctx.world_size >= 2 and right_target_cache is not None
    )
    main_result = None
    right_target_result = None

    try:
        if can_parallelize_two_suite_eval:
            if dist_ctx.rank == 0:
                main_result = run_controller_fdm_eval_suite_from_cache(
                    cache=main_cache,
                    controller=controller,
                    lvp=lvp,
                    device=device,
                    batch_size=batch_size,
                    metric_names=resolved_metric_names,
                    metric_batch_size=metric_batch_size,
                    hist_len=hist_len,
                    n_metrics_frames=n_metrics_frames,
                    prompt_text=prompt_text,
                    prompt_embed=prompt_embed,
                    prompt_embed_len=prompt_embed_len,
                    negative_prompt_embed=negative_prompt_embed,
                    negative_prompt_embed_len=negative_prompt_embed_len,
                    output_dir=output_dir / args.eval_split,
                    save_video_count=max(int(args.save_video_count), 0),
                    save_video_fps=int(args.save_video_fps),
                    seed=args.seed,
                    status_fn=log_step,
                )
            elif dist_ctx.rank == 1:
                right_target_result = run_controller_fdm_eval_suite_from_cache(
                    cache=right_target_cache,
                    controller=controller,
                    lvp=lvp,
                    device=device,
                    batch_size=batch_size,
                    metric_names=resolved_metric_names,
                    metric_batch_size=metric_batch_size,
                    hist_len=hist_len,
                    n_metrics_frames=n_metrics_frames,
                    prompt_text=prompt_text,
                    prompt_embed=prompt_embed,
                    prompt_embed_len=prompt_embed_len,
                    negative_prompt_embed=negative_prompt_embed,
                    negative_prompt_embed_len=negative_prompt_embed_len,
                    output_dir=output_dir / args.right_target_split,
                    save_video_count=max(int(args.save_video_count), 0),
                    save_video_fps=int(args.save_video_fps),
                    seed=args.seed + 100000,
                    status_fn=log_step,
                )
            gathered = all_gather_object_payload(
                {
                    "rank": int(dist_ctx.rank),
                    "main_result": main_result,
                    "right_target_result": right_target_result,
                },
                dist_ctx,
            )
            if dist_ctx.is_main_process:
                for payload in gathered:
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("main_result") is not None:
                        main_result = payload["main_result"]
                    if payload.get("right_target_result") is not None:
                        right_target_result = payload["right_target_result"]
        else:
            if dist_ctx.is_main_process:
                main_result = run_controller_fdm_eval_suite_from_cache(
                    cache=main_cache,
                    controller=controller,
                    lvp=lvp,
                    device=device,
                    batch_size=batch_size,
                    metric_names=resolved_metric_names,
                    metric_batch_size=metric_batch_size,
                    hist_len=hist_len,
                    n_metrics_frames=n_metrics_frames,
                    prompt_text=prompt_text,
                    prompt_embed=prompt_embed,
                    prompt_embed_len=prompt_embed_len,
                    negative_prompt_embed=negative_prompt_embed,
                    negative_prompt_embed_len=negative_prompt_embed_len,
                    output_dir=output_dir / args.eval_split,
                    save_video_count=max(int(args.save_video_count), 0),
                    save_video_fps=int(args.save_video_fps),
                    seed=args.seed,
                    status_fn=log_step,
                )
                if right_target_cache is not None:
                    right_target_result = run_controller_fdm_eval_suite_from_cache(
                        cache=right_target_cache,
                        controller=controller,
                        lvp=lvp,
                        device=device,
                        batch_size=batch_size,
                        metric_names=resolved_metric_names,
                        metric_batch_size=metric_batch_size,
                        hist_len=hist_len,
                        n_metrics_frames=n_metrics_frames,
                        prompt_text=prompt_text,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        negative_prompt_embed=negative_prompt_embed,
                        negative_prompt_embed_len=negative_prompt_embed_len,
                        output_dir=output_dir / args.right_target_split,
                        save_video_count=max(int(args.save_video_count), 0),
                        save_video_fps=int(args.save_video_fps),
                        seed=args.seed + 100000,
                        status_fn=log_step,
                    )

        if dist_ctx.is_main_process:
            summary = {
                "controller_ckpt": str(controller_ckpt_path),
                "controller_summary": str(controller_summary_path) if controller_summary_path is not None else None,
                "run_dir": str(run_dir),
                "backbone_ckpt": str(ckpt_path),
                "backbone_step": int(backbone_step),
                "controller_step": int(controller_step),
                "context_len": int(context_len),
                "metric_names": resolved_metric_names,
                "batch_size": int(batch_size),
                "eval_split": args.eval_split,
                "right_target_split": args.right_target_split,
                "eval_result": main_result,
                "right_target_result": right_target_result,
            }
            summary_path = output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log_step(f"summary={summary_path}")

            if main_result is not None:
                log_step(
                    f"{args.eval_split}: "
                    + " ".join(
                        f"{key}={value:.6f}" for key, value in main_result.get("metrics", {}).items()
                    )
                )
            if right_target_result is not None:
                log_step(
                    f"{args.right_target_split}: "
                    + " ".join(
                        f"{key}={value:.6f}"
                        for key, value in right_target_result.get("metrics", {}).items()
                    )
                )

            if wandb_run is not None:
                payload = _build_wandb_payload(
                    step=controller_step if controller_step >= 0 else backbone_step,
                    main_result=main_result,
                    right_target_result=right_target_result,
                )
                try:
                    import wandb

                    if main_result is not None:
                        for idx, video_path in enumerate(main_result.get("saved_videos", [])):
                            payload[f"controller_fdm_eval/videos/sample_{idx:02d}"] = wandb.Video(
                                str(video_path),
                                format="mp4",
                            )
                    if right_target_result is not None:
                        for idx, video_path in enumerate(right_target_result.get("saved_videos", [])):
                            payload[
                                f"controller_fdm_eval_right_target/videos/sample_{idx:02d}"
                            ] = wandb.Video(
                                str(video_path),
                                format="mp4",
                            )
                except Exception:
                    pass
                wandb_run.log(payload)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        distributed_barrier(dist_ctx)
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
