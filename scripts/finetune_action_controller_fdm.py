#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import datetime as dt
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from infer_action_transfer import (
    build_split_records,
    filter_records,
    load_run_artifacts,
    maybe_rebase_project_path,
)
from scar.config import load_yaml_config
from scar.controller_fdm import (
    ControllerFDMCache,
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
from scar.objectives import (
    compute_wrong_z_gate_sigma,
    forward_lvp_flow,
)
from scar.runtime import (
    build_joint_optimizer,
    build_lr_scheduler,
    cleanup_distributed,
    distributed_barrier,
    get_current_optimizer_lrs,
    get_lvp_module_map,
    iter_unique_trainable_params,
    load_checkpoint,
    maybe_ddp_no_sync,
    maybe_wrap_ddp,
    reduce_scalar,
    setup_distributed,
    unwrap_module,
)


def log_step(message: str) -> None:
    print(f"[controller-joint] {message}", flush=True)


@dataclass(frozen=True)
class JointTrainConfig:
    controller_ckpt: str
    run_dir: str | None
    ckpt: str | None
    output_dir: str
    train_split: str
    eval_split: str
    right_target_split: str
    train_dataset_filter: str
    eval_dataset_filter: str
    right_target_dataset_filter: str
    context_len: int
    batch_size: int
    eval_batch_size: int
    train_steps: int
    accumulate_grad_batches: int
    log_every: int
    eval_every: int
    save_every: int
    save_video_count: int
    save_video_fps: int
    controller_lr: float
    controller_weight_decay: float
    lvp_lr: float | None
    clip_grad_norm: float
    latent_anchor_weight: float
    low_noise_only: bool
    low_noise_weighted: bool
    low_noise_sigma_hi: float
    low_noise_weight_floor: float
    trainable_lvp_modules: tuple[str, ...]
    use_lr_scheduler: bool
    seed: int
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_name: str | None
    wandb_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly fine-tune a pretrained latent-action controller together with the "
            "pretrained FDM/LVP backbone using controller-predicted latent actions."
        )
    )
    parser.add_argument("--config", required=True, help="Joint controller-FDM YAML config.")
    parser.add_argument(
        "--controller-ckpt",
        default=None,
        help="Optional controller checkpoint override for controller_joint_finetune.controller_ckpt.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional SCAR backbone run directory override.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional SCAR backbone checkpoint override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-dir override. If omitted, use the config value or auto-generate one.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _cfg_value(raw_cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = raw_cfg
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _parse_joint_cfg(raw_cfg: dict[str, Any]) -> JointTrainConfig:
    joint_cfg = dict(raw_cfg.get("controller_joint_finetune", {}) or {})
    if not bool(joint_cfg.get("enabled", False)):
        raise ValueError(
            "controller_joint_finetune.enabled must be true in the provided YAML config."
        )

    wandb_cfg = dict(raw_cfg.get("wandb", {}) or {})
    optimizer_cfg = dict(raw_cfg.get("optimizer", {}) or {})
    train_cfg = dict(raw_cfg.get("train", {}) or {})

    parsed = JointTrainConfig(
        controller_ckpt=str(joint_cfg["controller_ckpt"]),
        run_dir=joint_cfg.get("run_dir"),
        ckpt=joint_cfg.get("ckpt"),
        output_dir=str(joint_cfg.get("output_dir", "") or ""),
        train_split=str(joint_cfg.get("train_split", "train")),
        eval_split=str(joint_cfg.get("eval_split", "eval")),
        right_target_split=str(joint_cfg.get("right_target_split", "right_target_eval") or ""),
        train_dataset_filter=str(joint_cfg.get("train_dataset_filter", "franka")),
        eval_dataset_filter=str(joint_cfg.get("eval_dataset_filter", "franka")),
        right_target_dataset_filter=str(
            joint_cfg.get("right_target_dataset_filter", "franka")
        ),
        context_len=int(joint_cfg.get("context_len", 17)),
        batch_size=int(joint_cfg.get("batch_size", _cfg_value(raw_cfg, "data", "batch_size", default=4))),
        eval_batch_size=int(joint_cfg.get("eval_batch_size", joint_cfg.get("batch_size", _cfg_value(raw_cfg, "data", "batch_size", default=4)))),
        train_steps=int(joint_cfg.get("train_steps", _cfg_value(raw_cfg, "train", "steps", default=5000))),
        accumulate_grad_batches=int(
            joint_cfg.get(
                "accumulate_grad_batches",
                train_cfg.get("accumulate_grad_batches", 1),
            )
        ),
        log_every=int(joint_cfg.get("log_every", train_cfg.get("log_every", 10))),
        eval_every=int(joint_cfg.get("eval_every", train_cfg.get("eval_every", 1000))),
        save_every=int(joint_cfg.get("save_every", train_cfg.get("save_every", 5000))),
        save_video_count=int(joint_cfg.get("save_video_count", 1)),
        save_video_fps=int(joint_cfg.get("save_video_fps", 20)),
        controller_lr=float(joint_cfg.get("controller_lr", optimizer_cfg.get("lr", 1.0e-4))),
        controller_weight_decay=float(
            joint_cfg.get("controller_weight_decay", optimizer_cfg.get("weight_decay", 1.0e-4))
        ),
        lvp_lr=(
            None
            if joint_cfg.get("lvp_lr", None) is None
            else float(joint_cfg.get("lvp_lr"))
        ),
        clip_grad_norm=float(
            joint_cfg.get("clip_grad_norm", optimizer_cfg.get("clip_grad_norm", 1.0))
        ),
        latent_anchor_weight=float(joint_cfg.get("latent_anchor_weight", 0.0)),
        low_noise_only=bool(joint_cfg.get("low_noise_only", False)),
        low_noise_weighted=bool(joint_cfg.get("low_noise_weighted", False)),
        low_noise_sigma_hi=float(joint_cfg.get("low_noise_sigma_hi", 1.0)),
        low_noise_weight_floor=float(joint_cfg.get("low_noise_weight_floor", 0.0)),
        trainable_lvp_modules=tuple(str(name) for name in list(joint_cfg.get("lvp_train_modules", []) or [])),
        use_lr_scheduler=bool(joint_cfg.get("use_lr_scheduler", True)),
        seed=int(joint_cfg.get("seed", train_cfg.get("seed", 0))),
        wandb_enabled=bool(joint_cfg.get("wandb", wandb_cfg.get("enabled", True))),
        wandb_project=str(wandb_cfg.get("project", "scar-robotwin")),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_name=wandb_cfg.get("name"),
        wandb_mode=str(wandb_cfg.get("mode", "online")),
    )
    if parsed.train_steps <= 0:
        raise ValueError("controller_joint_finetune.train_steps must be > 0.")
    if parsed.accumulate_grad_batches < 1:
        raise ValueError("controller_joint_finetune.accumulate_grad_batches must be >= 1.")
    if parsed.batch_size <= 0 or parsed.eval_batch_size <= 0:
        raise ValueError("controller_joint_finetune.batch_size and eval_batch_size must be > 0.")
    if parsed.eval_every < 0 or parsed.save_every < 0:
        raise ValueError("controller_joint_finetune.eval_every and save_every must be >= 0.")
    if parsed.low_noise_only and parsed.low_noise_weighted:
        raise ValueError(
            "controller_joint_finetune.low_noise_only and "
            "controller_joint_finetune.low_noise_weighted cannot both be true."
        )
    if not 0.0 <= parsed.low_noise_sigma_hi <= 1.0:
        raise ValueError("controller_joint_finetune.low_noise_sigma_hi must be in [0, 1].")
    if not 0.0 <= parsed.low_noise_weight_floor <= 1.0:
        raise ValueError("controller_joint_finetune.low_noise_weight_floor must be in [0, 1].")
    return parsed


def _compute_joint_recon_loss_from_video_lat(
    *,
    lvp,
    video_lat: torch.Tensor,
    conditioning_actions: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    low_noise_only: bool,
    low_noise_weighted: bool,
    low_noise_sigma_hi: float,
    low_noise_weight_floor: float,
) -> tuple[torch.Tensor, dict[str, float], Any]:
    pos_cache = forward_lvp_flow(
        lvp,
        video_lat,
        conditioning_actions,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )
    per_sample_recon = F.mse_loss(
        pos_cache.flow_pred.float(),
        pos_cache.target_flow.float(),
        reduction="none",
    ).reshape(pos_cache.flow_pred.shape[0], -1).mean(dim=1)

    sigma = compute_wrong_z_gate_sigma(
        lvp=lvp,
        t=pos_cache.t,
        history_tokens=pos_cache.hist_tokens,
    ).to(device=per_sample_recon.device, dtype=per_sample_recon.dtype)
    if low_noise_only:
        gate = (sigma <= float(low_noise_sigma_hi)).to(dtype=per_sample_recon.dtype)
        recon_loss = (gate * per_sample_recon).sum() / gate.sum().clamp_min(1.0)
        active_frac = float(gate.mean().detach().item())
        weight_mean = active_frac
    elif low_noise_weighted:
        sigma_hi = float(low_noise_sigma_hi)
        weight_floor = float(low_noise_weight_floor)
        if sigma_hi >= 1.0:
            gate = torch.ones_like(per_sample_recon)
        else:
            decay = ((sigma - sigma_hi) / max(1.0 - sigma_hi, 1.0e-6)).clamp(0.0, 1.0)
            gate = 1.0 - (1.0 - weight_floor) * decay
        recon_loss = (gate * per_sample_recon).sum() / gate.sum().clamp_min(1.0)
        active_frac = float((sigma <= sigma_hi).to(dtype=per_sample_recon.dtype).mean().detach().item())
        weight_mean = float(gate.mean().detach().item())
    else:
        gate = torch.ones_like(per_sample_recon)
        recon_loss = per_sample_recon.mean()
        active_frac = 1.0
        weight_mean = 1.0

    metrics = {
        "recon_loss": float(recon_loss.detach().item()),
        "sigma_mean": float(sigma.mean().detach().item()),
        "low_noise_active_frac": float(active_frac),
        "noise_weight_mean": float(weight_mean),
        "condition_action_abs": float(conditioning_actions.detach().abs().mean().item()),
    }
    return recon_loss, metrics, pos_cache


def _resolve_output_dir(
    *,
    cli_output_dir: str | None,
    cfg_output_dir: str,
    controller_ckpt_path: Path,
) -> Path:
    if cli_output_dir:
        return Path(cli_output_dir).resolve()
    if cfg_output_dir:
        candidate = Path(cfg_output_dir)
        return candidate.resolve() if candidate.is_absolute() else candidate.resolve()
    controller_run_dir = controller_ckpt_path.parent.parent
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        controller_run_dir
        / f"controller_fdm_joint_{controller_ckpt_path.stem}_{timestamp}"
    ).resolve()


def _maybe_init_wandb(
    *,
    joint_cfg: JointTrainConfig,
    output_dir: Path,
    config_path: Path,
    raw_cfg: dict[str, Any],
    controller_ckpt_path: Path,
    run_dir: Path,
    backbone_ckpt_path: Path,
) -> Any | None:
    if not joint_cfg.wandb_enabled:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in the current environment. Install it or disable wandb."
        ) from exc

    run = wandb.init(
        project=joint_cfg.wandb_project,
        entity=joint_cfg.wandb_entity or None,
        name=joint_cfg.wandb_name or output_dir.name,
        mode=joint_cfg.wandb_mode,
        dir=str(output_dir),
        config={
            "config_path": str(config_path),
            "raw_config": raw_cfg,
            "controller_ckpt": str(controller_ckpt_path),
            "run_dir": str(run_dir),
            "backbone_ckpt": str(backbone_ckpt_path),
        },
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("eval_table/step")
    wandb.define_metric("eval_table/*", step_metric="eval_table/step")
    wandb.define_metric("eval_right_target/step")
    wandb.define_metric("eval_right_target/*", step_metric="eval_right_target/step")
    wandb.define_metric("eval_table_right_target/step")
    wandb.define_metric("eval_table_right_target/*", step_metric="eval_table_right_target/step")
    return run


def _make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    sampler: DistributedSampler | None,
    seed: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        generator=generator if sampler is None else None,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _build_train_dataset(cache: ControllerFDMCache) -> TensorDataset:
    return TensorDataset(
        cache.full_video_lat,
        cache.context_tokens,
        cache.gt_actions,
        cache.teacher_latent_actions,
    )


def _save_joint_checkpoint(
    *,
    output_dir: Path,
    step: int,
    controller,
    optimizer: torch.optim.Optimizer,
    lvp,
    lvp_trainable_modules: set[str],
    controller_payload: dict[str, Any],
    run_dir: Path,
    backbone_ckpt_path: Path,
    lr_scheduler=None,
    write_step_checkpoint: bool = True,
    suffix: str = "",
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"controller_fdm_joint{suffix}"
    ckpt_path = ckpt_dir / f"{stem}_step_{step:07d}.pt"
    latest_path = ckpt_dir / f"{stem}_latest.pt"

    controller_module = unwrap_module(controller)
    state = {
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "controller_state_dict": controller_module.state_dict(),
        "controller_config": dict(controller_payload.get("controller_config", {})),
        "shape": dict(controller_payload.get("shape", {})),
        "controller_step": int(controller_payload.get("controller_step", -1)),
        "global_step": int(controller_payload.get("global_step", -1)),
        "run_dir": str(run_dir),
        "backbone_ckpt": str(backbone_ckpt_path),
        "lvp_trainable_modules": sorted(lvp_trainable_modules),
        "lvp": {
            name: unwrap_module(module).state_dict()
            for name, module in get_lvp_module_map(lvp).items()
            if name in lvp_trainable_modules and module is not None
        },
    }
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if write_step_checkpoint:
        torch.save(state, ckpt_path)
    torch.save(state, latest_path)
    return ckpt_path if write_step_checkpoint else latest_path


def _build_eval_payload(
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
                "eval",
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
                "eval_table",
                step=step,
                metrics=build_eval_table_metric_bundle(main_metrics),
            )
        )
    if right_target_result is not None:
        right_metrics = MetricBundle(dict(right_target_result.get("metrics", {})))
        payload.update(
            build_namespaced_log_payload(
                "eval_right_target",
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
                "eval_table_right_target",
                step=step,
                metrics=build_eval_table_metric_bundle(right_metrics),
            )
        )
    return payload


def _run_periodic_eval(
    *,
    step: int,
    dist_ctx,
    device: torch.device,
    controller,
    lvp,
    eval_cache: ControllerFDMCache,
    right_target_cache: ControllerFDMCache | None,
    batch_size: int,
    resolved_metric_names: list[str],
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
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    can_parallelize = dist_ctx.enabled and dist_ctx.world_size >= 2 and right_target_cache is not None
    main_result = None
    right_target_result = None

    if can_parallelize:
        if dist_ctx.rank == 0:
            main_result = run_controller_fdm_eval_suite_from_cache(
                cache=eval_cache,
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
                output_dir=output_dir / f"step_{step:07d}" / eval_cache.split,
                save_video_count=max(int(save_video_count), 0),
                save_video_fps=int(save_video_fps),
                seed=seed + step,
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
                output_dir=output_dir / f"step_{step:07d}" / right_target_cache.split,
                save_video_count=max(int(save_video_count), 0),
                save_video_fps=int(save_video_fps),
                seed=seed + step + 100000,
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
                cache=eval_cache,
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
                output_dir=output_dir / f"step_{step:07d}" / eval_cache.split,
                save_video_count=max(int(save_video_count), 0),
                save_video_fps=int(save_video_fps),
                seed=seed + step,
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
                    output_dir=output_dir / f"step_{step:07d}" / right_target_cache.split,
                    save_video_count=max(int(save_video_count), 0),
                    save_video_fps=int(save_video_fps),
                    seed=seed + step + 100000,
                    status_fn=log_step,
                )
        if dist_ctx.enabled:
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

    return main_result, right_target_result


def main() -> None:
    cli_args = parse_args()
    config_path = Path(cli_args.config).resolve()
    raw_cfg = load_yaml_config(config_path)
    joint_cfg = _parse_joint_cfg(raw_cfg)
    joint_cfg = replace(
        joint_cfg,
        controller_ckpt=cli_args.controller_ckpt or joint_cfg.controller_ckpt,
        run_dir=cli_args.run_dir or joint_cfg.run_dir,
        ckpt=cli_args.ckpt or joint_cfg.ckpt,
    )
    set_seed(joint_cfg.seed)

    if not joint_cfg.controller_ckpt:
        raise ValueError(
            "Set controller_joint_finetune.controller_ckpt in the config or pass "
            "--controller-ckpt."
        )
    controller_ckpt_path = Path(joint_cfg.controller_ckpt).resolve()
    if not controller_ckpt_path.is_file():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_ckpt_path}")

    controller_summary_path, controller_summary = load_controller_summary(controller_ckpt_path)
    run_dir, backbone_ckpt_path = resolve_backbone_run_and_ckpt(
        run_dir=joint_cfg.run_dir,
        ckpt=joint_cfg.ckpt,
        controller_summary=controller_summary,
    )
    artifacts = load_run_artifacts(run_dir, backbone_ckpt_path)
    output_dir = _resolve_output_dir(
        cli_output_dir=cli_args.output_dir,
        cfg_output_dir=joint_cfg.output_dir,
        controller_ckpt_path=controller_ckpt_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(raw_cfg, indent=2),
        encoding="utf-8",
    )

    dist_ctx, device = setup_distributed(SimpleNamespace(device=cli_args.device))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    controller_payload, controller = load_controller_checkpoint(controller_ckpt_path, device)
    controller_arch = str(
        controller_payload.get("controller_config", {}).get("architecture", "summary_add")
    )

    default_context_len = int(controller_payload.get("controller_config", {}).get("context_len", 0) or 0)
    if default_context_len <= 0:
        default_context_len = int(controller_summary.get("context_len", 0) or 0)
    context_len = int(joint_cfg.context_len) if int(joint_cfg.context_len) > 0 else int(default_context_len)

    if dist_ctx.is_main_process:
        log_step(f"config={config_path}")
        log_step(f"controller_ckpt={controller_ckpt_path}")
        if controller_summary_path is not None:
            log_step(f"controller_summary={controller_summary_path}")
        log_step(f"run_dir={run_dir}")
        log_step(f"backbone_ckpt={backbone_ckpt_path}")
        log_step(f"output_dir={output_dir}")
        log_step(f"device={device}")
        log_step(f"context_len={context_len}")
        log_step(f"controller_architecture={controller_arch}")
        log_step(
            f"latent_anchor_weight={joint_cfg.latent_anchor_weight:.4f} "
            f"low_noise_only={joint_cfg.low_noise_only} "
            f"low_noise_weighted={joint_cfg.low_noise_weighted} "
            f"low_noise_sigma_hi={joint_cfg.low_noise_sigma_hi:.3f} "
            f"low_noise_weight_floor={joint_cfg.low_noise_weight_floor:.3f}"
        )

    ckpt_probe = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
    if ckpt_probe.get("idm") is None:
        raise ValueError("Backbone checkpoint does not contain IDM weights.")

    conditioning_action_dim = resolve_conditioning_action_dim(artifacts.idm_cfg, action_source="idm")
    align_lvp_action_dim(artifacts.lvp_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(artifacts.lvp_cfg)
    if int(artifacts.idm_cfg.data.seq_len) != target_seq_len:
        if dist_ctx.is_main_process:
            log_step(
                f"aligning IDM seq_len from {int(artifacts.idm_cfg.data.seq_len)} to {target_seq_len}"
            )
        align_idm_seq_len(artifacts.idm_cfg, target_seq_len)

    prompt_text = artifacts.args_dict.get("prompt") or DEFAULT_LIBERO_PROMPT
    prompt_embed_path = maybe_rebase_project_path(
        artifacts.args_dict.get("prompt_embed_path") or str(DEFAULT_LIBERO_PROMPT_EMBED)
    )
    negative_prompt_embed_path = maybe_rebase_project_path(
        artifacts.args_dict.get("negative_prompt_embed_path")
        or str(DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED)
    )
    prompt_embed, prompt_embed_len = load_prompt_embedding(prompt_embed_path)
    negative_prompt_embed, negative_prompt_embed_len = load_prompt_embedding(
        negative_prompt_embed_path
    )

    ckpt_lvp_modules = set(ckpt_probe.get("lvp_trainable_modules", []))
    trainable_lvp_modules = (
        set(joint_cfg.trainable_lvp_modules)
        if joint_cfg.trainable_lvp_modules
        else set(ckpt_lvp_modules or {"model", "action_encoder"})
    )
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
        backbone_ckpt_path,
        idm=idm,
        lvp=lvp,
        lvp_trainable_modules=ckpt_lvp_modules,
        map_location=device,
    )
    valid_lvp_modules = {
        name for name, module in get_lvp_module_map(lvp).items() if module is not None
    }
    unknown_lvp_modules = sorted(trainable_lvp_modules - valid_lvp_modules)
    if unknown_lvp_modules:
        raise ValueError(
            "Unknown controller_joint_finetune.lvp_train_modules: "
            f"{unknown_lvp_modules}. Valid modules: {sorted(valid_lvp_modules)}"
        )

    train_records = filter_records(
        build_split_records(artifacts, joint_cfg.train_split),
        joint_cfg.train_dataset_filter,
        split_name=joint_cfg.train_split,
    )
    eval_records = filter_records(
        build_split_records(artifacts, joint_cfg.eval_split),
        joint_cfg.eval_dataset_filter,
        split_name=joint_cfg.eval_split,
    )
    right_target_records = []
    if joint_cfg.right_target_split:
        right_target_records = filter_records(
            build_split_records(artifacts, joint_cfg.right_target_split),
            joint_cfg.right_target_dataset_filter,
            split_name=joint_cfg.right_target_split,
        )
    if not train_records or not eval_records:
        raise ValueError("Joint controller-FDM training requires non-empty train and eval splits.")

    seq_len = int(train_records[0].sample_np["observations"].shape[0])
    if seq_len != target_seq_len:
        raise RuntimeError(f"Expected seq_len={target_seq_len}, got {seq_len}.")
    if context_len <= 0 or context_len > seq_len:
        raise ValueError(f"context_len must be in [1, {seq_len}], got {context_len}")

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
        log_step(f"{joint_cfg.train_split}: building cache for {len(train_records)} windows")
    train_cache = build_controller_fdm_cache(
        split_name=joint_cfg.train_split,
        records=train_records,
        device=device,
        seq_len=seq_len,
        context_len=context_len,
        batch_size=joint_cfg.batch_size,
        lvp=lvp,
        idm=idm,
        include_input_video=False,
        status_fn=log_step if dist_ctx.is_main_process else None,
    )
    if dist_ctx.is_main_process:
        log_step(f"{joint_cfg.eval_split}: building cache for {len(eval_records)} windows")
    eval_cache = build_controller_fdm_cache(
        split_name=joint_cfg.eval_split,
        records=eval_records,
        device=device,
        seq_len=seq_len,
        context_len=context_len,
        batch_size=joint_cfg.eval_batch_size,
        lvp=lvp,
        idm=idm,
        include_input_video=True,
        status_fn=log_step if dist_ctx.is_main_process else None,
    )
    right_target_cache = None
    if right_target_records:
        if dist_ctx.is_main_process:
            log_step(
                f"{joint_cfg.right_target_split}: building cache for {len(right_target_records)} windows"
            )
        right_target_cache = build_controller_fdm_cache(
            split_name=joint_cfg.right_target_split,
            records=right_target_records,
            device=device,
            seq_len=seq_len,
            context_len=context_len,
            batch_size=joint_cfg.eval_batch_size,
            lvp=lvp,
            idm=idm,
            include_input_video=True,
            status_fn=log_step if dist_ctx.is_main_process else None,
        )
    del idm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for param in controller.parameters():
        param.requires_grad_(True)
    controller.train()
    for name, module in get_lvp_module_map(lvp).items():
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(name in trainable_lvp_modules)
    set_lvp_mode(lvp, trainable_lvp_modules, training=True)

    lvp_trainable = [
        module
        for name, module in get_lvp_module_map(lvp).items()
        if name in trainable_lvp_modules and module is not None
    ]

    controller = maybe_wrap_ddp(
        controller,
        dist_ctx=dist_ctx,
        device=device,
    )
    for name in sorted(trainable_lvp_modules):
        module = getattr(lvp, name, None)
        wrapped_module = maybe_wrap_ddp(
            module,
            dist_ctx=dist_ctx,
            device=device,
        )
        if wrapped_module is not None:
            setattr(lvp, name, wrapped_module)
    lvp_trainable = [
        module
        for name, module in get_lvp_module_map(lvp).items()
        if name in trainable_lvp_modules and module is not None
    ]

    optimizer_args = SimpleNamespace(
        lr=joint_cfg.controller_lr,
        weight_decay=joint_cfg.controller_weight_decay,
        lvp_lr=joint_cfg.lvp_lr,
    )
    optimizer, optimizer_stats = build_joint_optimizer(
        controller,
        lvp_trainable,
        args=optimizer_args,
        lvp_cfg=artifacts.lvp_cfg,
    )
    lr_scheduler = build_lr_scheduler(optimizer, artifacts.lvp_cfg) if joint_cfg.use_lr_scheduler else None

    train_dataset = _build_train_dataset(train_cache)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            seed=joint_cfg.seed,
            drop_last=False,
        )
        if dist_ctx.enabled
        else None
    )
    train_epoch = 0

    def make_train_iter():
        nonlocal train_epoch
        if train_sampler is not None:
            train_sampler.set_epoch(joint_cfg.seed + train_epoch)
        loader = _make_loader(
            train_dataset,
            batch_size=joint_cfg.batch_size,
            sampler=train_sampler,
            seed=joint_cfg.seed + train_epoch,
        )
        train_epoch += 1
        return iter(loader)

    train_iter = make_train_iter()
    trainable_params = list(iter_unique_trainable_params(controller, *lvp_trainable))
    if not trainable_params:
        raise ValueError("No trainable parameters found for joint controller-FDM fine-tuning.")

    wandb_run = None
    if dist_ctx.is_main_process:
        wandb_run = _maybe_init_wandb(
            joint_cfg=joint_cfg,
            output_dir=output_dir,
            config_path=config_path,
            raw_cfg=raw_cfg,
            controller_ckpt_path=controller_ckpt_path,
            run_dir=run_dir,
            backbone_ckpt_path=backbone_ckpt_path,
        )
        log_step(
            "optimizer "
            f"controller_lr={optimizer_stats['idm_lr']:.2e}, "
            f"controller_weight_decay={optimizer_stats['idm_weight_decay']:.2e}, "
            f"controller_params={optimizer_stats['idm_param_count']}, "
            f"lvp_lr={optimizer_stats['lvp_lr']:.2e}, "
            f"lvp_weight_decay={optimizer_stats['lvp_weight_decay']:.2e}, "
            f"lvp_params={optimizer_stats['lvp_param_count']}, "
            f"lvp_trainable_modules={sorted(trainable_lvp_modules)}"
        )
        log_step(
            f"train_windows={train_cache.num_windows}, eval_windows={eval_cache.num_windows}, "
            f"right_target_windows={right_target_cache.num_windows if right_target_cache is not None else 0}"
        )

    eval_trace: list[dict[str, Any]] = []
    best_eval_video_mse = float("inf")
    best_eval_step: int | None = None
    best_checkpoint_path: str | None = None
    latest_checkpoint_path: str | None = None
    last_main_result: dict[str, Any] | None = None
    last_right_target_result: dict[str, Any] | None = None

    try:
        for step in range(1, joint_cfg.train_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_recon_loss = 0.0
            step_anchor_loss = 0.0
            step_total_loss = 0.0
            step_action_abs = 0.0
            step_sigma_mean = 0.0
            step_low_noise_active_frac = 0.0
            step_noise_weight_mean = 0.0

            for micro_step in range(joint_cfg.accumulate_grad_batches):
                try:
                    video_lat_batch, context_batch, gt_action_batch, teacher_latent_batch = next(train_iter)
                except StopIteration:
                    train_iter = make_train_iter()
                    video_lat_batch, context_batch, gt_action_batch, teacher_latent_batch = next(train_iter)

                video_lat_batch = video_lat_batch.to(device=device, non_blocking=True)
                context_batch = context_batch.to(device=device, non_blocking=True)
                gt_action_batch = gt_action_batch.to(device=device, non_blocking=True)
                if joint_cfg.latent_anchor_weight > 0:
                    teacher_latent_batch = teacher_latent_batch.to(device=device, non_blocking=True)

                use_no_sync = (
                    dist_ctx.enabled
                    and joint_cfg.accumulate_grad_batches > 1
                    and micro_step < joint_cfg.accumulate_grad_batches - 1
                )
                with maybe_ddp_no_sync([controller, *lvp_trainable], enabled=use_no_sync):
                    pred_latent_actions = controller(context_batch, gt_action_batch)
                    recon_loss, recon_metrics, _ = _compute_joint_recon_loss_from_video_lat(
                        lvp=lvp,
                        video_lat=video_lat_batch,
                        conditioning_actions=pred_latent_actions,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        low_noise_only=joint_cfg.low_noise_only,
                        low_noise_weighted=joint_cfg.low_noise_weighted,
                        low_noise_sigma_hi=joint_cfg.low_noise_sigma_hi,
                        low_noise_weight_floor=joint_cfg.low_noise_weight_floor,
                    )
                    latent_anchor_loss = torch.zeros((), device=device)
                    if joint_cfg.latent_anchor_weight > 0:
                        latent_anchor_loss = F.mse_loss(
                            pred_latent_actions,
                            teacher_latent_batch,
                        )
                    total_loss = recon_loss + joint_cfg.latent_anchor_weight * latent_anchor_loss
                    (total_loss / joint_cfg.accumulate_grad_batches).backward()

                step_recon_loss += float(recon_loss.detach().item())
                step_anchor_loss += float(latent_anchor_loss.detach().item())
                step_total_loss += float(total_loss.detach().item())
                step_action_abs += float(recon_metrics["condition_action_abs"])
                step_sigma_mean += float(recon_metrics["sigma_mean"])
                step_low_noise_active_frac += float(recon_metrics["low_noise_active_frac"])
                step_noise_weight_mean += float(recon_metrics["noise_weight_mean"])

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(joint_cfg.clip_grad_norm),
            )
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            step_recon_loss /= joint_cfg.accumulate_grad_batches
            step_anchor_loss /= joint_cfg.accumulate_grad_batches
            step_total_loss /= joint_cfg.accumulate_grad_batches
            step_action_abs /= joint_cfg.accumulate_grad_batches
            step_sigma_mean /= joint_cfg.accumulate_grad_batches
            step_low_noise_active_frac /= joint_cfg.accumulate_grad_batches
            step_noise_weight_mean /= joint_cfg.accumulate_grad_batches

            if step % joint_cfg.log_every == 0:
                mean_total_loss = reduce_scalar(step_total_loss, dist_ctx)
                mean_recon_loss = reduce_scalar(step_recon_loss, dist_ctx)
                mean_anchor_loss = reduce_scalar(step_anchor_loss, dist_ctx)
                mean_action_abs = reduce_scalar(step_action_abs, dist_ctx)
                mean_sigma_mean = reduce_scalar(step_sigma_mean, dist_ctx)
                mean_low_noise_active_frac = reduce_scalar(step_low_noise_active_frac, dist_ctx)
                mean_noise_weight_mean = reduce_scalar(step_noise_weight_mean, dist_ctx)
                mean_grad_norm = reduce_scalar(float(grad_norm), dist_ctx)
                if dist_ctx.is_main_process:
                    lrs = get_current_optimizer_lrs(optimizer)
                    log_step(
                        f"step={step:06d} total_loss={mean_total_loss:.6f} "
                        f"recon_loss={mean_recon_loss:.6f} "
                        f"latent_anchor_loss={mean_anchor_loss:.6f} "
                        f"action_abs={mean_action_abs:.6f} "
                        f"sigma_mean={mean_sigma_mean:.4f} "
                        f"low_noise_active_frac={mean_low_noise_active_frac:.4f} "
                        f"noise_weight_mean={mean_noise_weight_mean:.4f} "
                        f"grad_norm={mean_grad_norm:.4f} "
                        f"controller_lr={lrs.get('idm', 0.0):.2e} "
                        f"lvp_lr={lrs.get('lvp', 0.0):.2e}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/step": int(step),
                                "train/total_loss": float(mean_total_loss),
                                "train/recon_loss": float(mean_recon_loss),
                                "train/latent_anchor_loss": float(mean_anchor_loss),
                                "train/controller_action_abs": float(mean_action_abs),
                                "train/sigma_mean": float(mean_sigma_mean),
                                "train/low_noise_active_frac": float(mean_low_noise_active_frac),
                                "train/noise_weight_mean": float(mean_noise_weight_mean),
                                "train/grad_norm": float(mean_grad_norm),
                                "train/controller_lr": float(lrs.get("idm", 0.0)),
                                "train/lvp_lr": float(lrs.get("lvp", 0.0)),
                            }
                        )

            if joint_cfg.eval_every > 0 and step % joint_cfg.eval_every == 0:
                main_result, right_target_result = _run_periodic_eval(
                    step=step,
                    dist_ctx=dist_ctx,
                    device=device,
                    controller=controller,
                    lvp=lvp,
                    eval_cache=eval_cache,
                    right_target_cache=right_target_cache,
                    batch_size=joint_cfg.eval_batch_size,
                    resolved_metric_names=resolved_metric_names,
                    metric_batch_size=metric_batch_size,
                    hist_len=hist_len,
                    n_metrics_frames=n_metrics_frames,
                    prompt_text=prompt_text,
                    prompt_embed=prompt_embed,
                    prompt_embed_len=prompt_embed_len,
                    negative_prompt_embed=negative_prompt_embed,
                    negative_prompt_embed_len=negative_prompt_embed_len,
                    output_dir=output_dir / "periodic_eval",
                    save_video_count=joint_cfg.save_video_count,
                    save_video_fps=joint_cfg.save_video_fps,
                    seed=joint_cfg.seed,
                )
                if dist_ctx.is_main_process:
                    last_main_result = main_result
                    last_right_target_result = right_target_result
                    eval_entry = {
                        "step": int(step),
                        "eval": main_result,
                        "right_target_eval": right_target_result,
                    }
                    eval_trace.append(eval_entry)
                    if wandb_run is not None:
                        wandb_run.log(
                            _build_eval_payload(
                                step=step,
                                main_result=main_result,
                                right_target_result=right_target_result,
                            )
                        )
                    current_eval_video_mse = float(
                        (main_result or {}).get("metrics", {}).get("video_mse", float("inf"))
                    )
                    if current_eval_video_mse < best_eval_video_mse:
                        best_eval_video_mse = current_eval_video_mse
                        best_eval_step = int(step)
                        best_path = _save_joint_checkpoint(
                            output_dir=output_dir,
                            step=step,
                            controller=controller,
                            optimizer=optimizer,
                            lvp=lvp,
                            lvp_trainable_modules=trainable_lvp_modules,
                            controller_payload=controller_payload,
                            run_dir=run_dir,
                            backbone_ckpt_path=backbone_ckpt_path,
                            lr_scheduler=lr_scheduler,
                            write_step_checkpoint=True,
                            suffix="_best_eval",
                        )
                        best_checkpoint_path = str(best_path)
                        log_step(
                            f"saved best_eval checkpoint step={step} video_mse={current_eval_video_mse:.6f} "
                            f"path={best_path}"
                        )

            if joint_cfg.save_every > 0 and step % joint_cfg.save_every == 0 and dist_ctx.is_main_process:
                latest_path = _save_joint_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    controller=controller,
                    optimizer=optimizer,
                    lvp=lvp,
                    lvp_trainable_modules=trainable_lvp_modules,
                    controller_payload=controller_payload,
                    run_dir=run_dir,
                    backbone_ckpt_path=backbone_ckpt_path,
                    lr_scheduler=lr_scheduler,
                    write_step_checkpoint=True,
                )
                latest_checkpoint_path = str(latest_path)
                log_step(f"saved checkpoint step={step} path={latest_path}")

        if dist_ctx.is_main_process and (joint_cfg.save_every <= 0 or joint_cfg.train_steps % joint_cfg.save_every != 0):
            latest_path = _save_joint_checkpoint(
                output_dir=output_dir,
                step=joint_cfg.train_steps,
                controller=controller,
                optimizer=optimizer,
                lvp=lvp,
                lvp_trainable_modules=trainable_lvp_modules,
                controller_payload=controller_payload,
                run_dir=run_dir,
                backbone_ckpt_path=backbone_ckpt_path,
                lr_scheduler=lr_scheduler,
                write_step_checkpoint=True,
            )
            latest_checkpoint_path = str(latest_path)
            log_step(f"saved final checkpoint step={joint_cfg.train_steps} path={latest_path}")

        if joint_cfg.eval_every > 0 and joint_cfg.train_steps % joint_cfg.eval_every != 0:
            main_result, right_target_result = _run_periodic_eval(
                step=joint_cfg.train_steps,
                dist_ctx=dist_ctx,
                device=device,
                controller=controller,
                lvp=lvp,
                eval_cache=eval_cache,
                right_target_cache=right_target_cache,
                batch_size=joint_cfg.eval_batch_size,
                resolved_metric_names=resolved_metric_names,
                metric_batch_size=metric_batch_size,
                hist_len=hist_len,
                n_metrics_frames=n_metrics_frames,
                prompt_text=prompt_text,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                negative_prompt_embed=negative_prompt_embed,
                negative_prompt_embed_len=negative_prompt_embed_len,
                output_dir=output_dir / "periodic_eval",
                save_video_count=joint_cfg.save_video_count,
                save_video_fps=joint_cfg.save_video_fps,
                seed=joint_cfg.seed,
            )
            if dist_ctx.is_main_process:
                last_main_result = main_result
                last_right_target_result = right_target_result
                eval_trace.append(
                    {
                        "step": int(joint_cfg.train_steps),
                        "eval": main_result,
                        "right_target_eval": right_target_result,
                    }
                )
                if wandb_run is not None:
                    wandb_run.log(
                        _build_eval_payload(
                            step=joint_cfg.train_steps,
                            main_result=main_result,
                            right_target_result=right_target_result,
                        )
                    )

        if dist_ctx.is_main_process:
            summary = {
                "config_path": str(config_path),
                "controller_ckpt": str(controller_ckpt_path),
                "controller_summary": str(controller_summary_path) if controller_summary_path is not None else None,
                "run_dir": str(run_dir),
                "backbone_ckpt": str(backbone_ckpt_path),
                "output_dir": str(output_dir),
                "seed": int(joint_cfg.seed),
                "context_len": int(context_len),
                "controller_architecture": controller_arch,
                "trainable_lvp_modules": sorted(trainable_lvp_modules),
                "train_windows": int(train_cache.num_windows),
                "eval_windows": int(eval_cache.num_windows),
                "right_target_windows": int(right_target_cache.num_windows) if right_target_cache is not None else 0,
                "train_steps": int(joint_cfg.train_steps),
                "accumulate_grad_batches": int(joint_cfg.accumulate_grad_batches),
                "batch_size": int(joint_cfg.batch_size),
                "eval_batch_size": int(joint_cfg.eval_batch_size),
                "latent_anchor_weight": float(joint_cfg.latent_anchor_weight),
                "low_noise_only": bool(joint_cfg.low_noise_only),
                "low_noise_weighted": bool(joint_cfg.low_noise_weighted),
                "low_noise_sigma_hi": float(joint_cfg.low_noise_sigma_hi),
                "low_noise_weight_floor": float(joint_cfg.low_noise_weight_floor),
                "latest_checkpoint": latest_checkpoint_path,
                "best_eval_checkpoint": best_checkpoint_path,
                "best_eval_step": best_eval_step,
                "best_eval_video_mse": best_eval_video_mse if best_eval_step is not None else None,
                "last_eval_result": last_main_result,
                "last_right_target_result": last_right_target_result,
                "eval_trace": eval_trace,
            }
            summary_path = output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log_step(f"summary={summary_path}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        distributed_barrier(dist_ctx)
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
