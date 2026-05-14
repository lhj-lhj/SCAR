from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import OmegaConf

from .config import (
    align_lvp_action_dim,
    align_idm_seq_len,
    build_lvp_cfg,
    build_idm_cfg,
    get_lvp_target_seq_len,
    infer_idm_image_hw,
    parse_config_args,
)
from .evaluation import infer_obs_shape
from .eval_loop import (
    EvalLoopResult,
    build_eval_log_payload,
    build_eval_loop_context,
    format_eval_summary,
    run_periodic_eval,
)
from .gt_action_probe import train_and_save_gt_action_probe
from .cross_cycle_objectives import compute_cross_cycle_objectives
from .metrics import (
    MetricBundle,
    average_metric_bundles,
    build_metric_bundle,
    build_namespaced_log_payload,
    format_train_summary,
)
from .runtime import (
    EMAModel,
    LatentActionIDMWrapper,
    build_joint_optimizer,
    build_lr_scheduler,
    build_lvp_prior,
    cleanup_distributed,
    distributed_barrier,
    freeze_module,
    get_current_optimizer_lrs,
    get_lvp_module_map,
    init_wandb_run,
    iter_unique_trainable_params,
    load_checkpoint,
    load_prompt_embedding,
    maybe_ddp_no_sync,
    maybe_wrap_ddp,
    reduce_scalar,
    resolve_lvp_trainable_modules,
    save_checkpoint,
    set_lvp_mode,
    set_seed,
    setup_distributed,
    trim_batch,
    unwrap_idm_module,
    unwrap_module,
)


def _build_dataloaders(cfg, dist_ctx, args=None):
    from scar.dataloader import (
        build_explicit_eval_dataloader,
        build_primary_dataloaders,
    )

    dataset_paths = list(args.data.dataset_paths) if args else []
    if not dataset_paths:
        raise ValueError('data.dataset_paths must be provided in the IDM YAML config.')

    target_action_dim = int(args.data.action_dim) if args else 0
    if not target_action_dim:
        target_action_dim = int(getattr(cfg.env, "action_dim", 0) or 0)
    target_action_dim = target_action_dim or None

    train_loader, eval_loader, num_embodiments = build_primary_dataloaders(
        dataset_paths=dataset_paths,
        eval_dataset_paths=list(args.data.eval_dataset_paths) if args else [],
        dataset_episode_subsets=dict(args.data.dataset_episode_subsets) if args else {},
        eval_dataset_episode_subsets=(
            dict(args.data.eval_dataset_episode_subsets) if args else {}
        ),
        seq_len=int(args.data.seq_len if args else cfg.data.seq_len),
        batch_size=int(args.data.batch_size if args else cfg.data.batch_size),
        target_action_dim=target_action_dim,
        eval_num_trajs=int(
            args.data.eval_num_trajs if args else getattr(cfg.data, "eval_num_trajs", 16)
        ),
        shift=int(args.data.shift if args else getattr(cfg.data, "shift", 1)),
        num_workers=int(args.data.num_workers if args else 2),
        rank=dist_ctx.rank if dist_ctx.enabled else 0,
        world_size=dist_ctx.world_size if dist_ctx.enabled else 1,
    )
    right_target_eval_loader = build_explicit_eval_dataloader(
        dataset_paths=list(args.data.right_target_eval_dataset_paths) if args else [],
        dataset_episode_subsets=(
            dict(args.data.right_target_eval_dataset_episode_subsets) if args else {}
        ),
        seq_len=int(args.data.seq_len if args else cfg.data.seq_len),
        batch_size=int(args.data.batch_size if args else cfg.data.batch_size),
        target_action_dim=target_action_dim,
        num_workers=int(args.data.num_workers if args else 2),
        log_prefix="right-target eval",
    )
    return train_loader, eval_loader, right_target_eval_loader, num_embodiments


def _build_cross_cycle_dataloader(cfg, dataset_paths: list[str], args=None):
    import os
    from scar.dataloader import build_secondary_dataloader

    target_action_dim = int(args.data.action_dim) if args else 0
    if not target_action_dim:
        target_action_dim = int(getattr(cfg.env, "action_dim", 0) or 0)
    target_action_dim = target_action_dim or None

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    return build_secondary_dataloader(
        dataset_paths=dataset_paths,
        seq_len=int(args.data.seq_len if args else cfg.data.seq_len),
        batch_size=int(args.data.batch_size if args else cfg.data.batch_size),
        target_action_dim=target_action_dim,
        shift=int(args.data.shift if args else getattr(cfg.data, "shift", 1)),
        num_workers=int(args.data.num_workers if args else 0),
        rank=rank,
        world_size=world_size,
    )


@dataclass
class CrossCycleState:
    """Holds the secondary dataloader iterator for cross-embodiment cycle loss."""
    secondary_iter: Any  # InfiniteDataLoaderIter
    dataset_paths: list




def _run_startup_stage(rank0_print, prefix: str, stage_name: str, fn):
    rank0_print(f"{prefix} startup: {stage_name}...")
    start_time = time.time()
    result = fn()
    rank0_print(
        f"{prefix} startup: {stage_name} done in {time.time() - start_time:.2f}s"
    )
    return result


def _pre_dist_rank0_print(prefix: str, message: str) -> None:
    import os

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"{prefix} startup: {message}", flush=True)


def _make_rank0_print(is_main_process: bool):
    if not is_main_process:
        return lambda *unused_args, **unused_kwargs: None

    def _rank0_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

    return _rank0_print


def _is_cycle_burst_enabled(args) -> bool:
    return (
        bool(getattr(args, "enable_cycle_loss", False))
        and str(getattr(args, "lvp_action_source", "")) == "idm"
        and float(getattr(args, "cycle_loss_weight", 0.0)) > 0.0
        and int(getattr(args, "cycle_burst_every", 0)) > 0
        and int(getattr(args, "cycle_burst_steps", 0)) > 0
    )


def _should_run_cycle_burst(args, step: int) -> bool:
    if not _is_cycle_burst_enabled(args):
        return False
    burst_every = int(getattr(args, "cycle_burst_every", 0))
    if burst_every <= 0 or step % burst_every != 0:
        return False
    cycle_end_steps = int(getattr(args, "cycle_end_steps", 0))
    if cycle_end_steps > 0 and step > cycle_end_steps:
        return False
    return step > int(getattr(args, "cycle_warmup_steps", 0))


def _make_cycle_only_args(args):
    if hasattr(args, "to_flat_dict"):
        flat = dict(args.to_flat_dict())
    else:
        flat = dict(vars(args))
    flat.update(
        {
            "cycle_only_mode": True,
            "gt_action_loss_weight": 0.0,
            "latent_action_kl_weight": 0.0,
            "wrong_z_enabled": False,
            "wrong_z_weight": 0.0,
        }
    )
    return SimpleNamespace(**flat)


def _make_regular_args_without_cycle(args):
    if hasattr(args, "to_flat_dict"):
        flat = dict(args.to_flat_dict())
    else:
        flat = dict(vars(args))
    flat.update(
        {
            "enable_cycle_loss": False,
            "cycle_only_mode": False,
        }
    )
    return SimpleNamespace(**flat)

@dataclass
class _TrainStepResult:
    """Stores the outputs produced by one optimizer step in the train loop."""

    metrics: MetricBundle
    grad_norm: float
    current_lrs: dict[str, float]
    elapsed: float
    fetch_elapsed: float
    compute_elapsed: float
    optimize_elapsed: float


# This helper owns the "one train step" block: forward, optional regularizers,
# backward, optimizer update, scheduler update, and EMA update.
def _run_one_train_step(
    *,
    args,
    step: int,
    train_iter,
    Batch,
    to_device,
    compute_cycle_objectives_fn,
    device,
    seq_len: int,
    dist_ctx,
    ddp_sync_modules: list[Any],
    lvp,
    idm,
    gt_action_head,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    lvp_trainable_modules: set[str],
    ema,
    cross_cycle,
    cross_cycle_train_enabled: bool,
    grl_classifier,
    grl_weight: float,
    trainable_params: list[torch.nn.Parameter],
    optimizer,
    lr_scheduler,
    advance_lr_scheduler: bool = True,
    update_ema: bool = True,
):
    """Runs one optimizer step, including accumulation and optional regularizers."""

    step_start = time.time()
    micro_metrics: list[MetricBundle] = []
    fetch_elapsed = 0.0
    compute_elapsed = 0.0

    for accum_idx in range(args.accumulate_grad_batches):
        fetch_start = time.time()
        batch_np = next(train_iter)
        batch = Batch(**to_device(batch_np, device))
        batch = trim_batch(batch, seq_len)
        fetch_elapsed += time.time() - fetch_start

        should_sync = accum_idx == args.accumulate_grad_batches - 1
        compute_start = time.time()
        with maybe_ddp_no_sync(
            ddp_sync_modules,
            enabled=dist_ctx.enabled and not should_sync,
        ):
            output = compute_cycle_objectives_fn(
                args=args,
                step=step,
                training=True,
                lvp=lvp,
                idm=idm,
                gt_action_head=gt_action_head,
                batch=batch,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                lvp_trainable_modules=lvp_trainable_modules,
                ema=ema,
            )
            step_loss = output.total_loss
            step_metrics = output.metrics

            if cross_cycle is not None and cross_cycle_train_enabled:
                secondary_np = next(cross_cycle.secondary_iter)
                secondary_batch = Batch(**to_device(secondary_np, device))
                secondary_batch = trim_batch(secondary_batch, seq_len)
                cross_output = compute_cross_cycle_objectives(
                    args=args,
                    step=step,
                    training=True,
                    lvp=lvp,
                    idm=idm,
                    batch_primary=batch,
                    batch_secondary=secondary_batch,
                    prompt_embed=prompt_embed,
                    prompt_embed_len=prompt_embed_len,
                    lvp_trainable_modules=lvp_trainable_modules,
                    ema=ema,
                )
                step_loss = step_loss + cross_output.total_loss
                step_metrics = step_metrics.merge(cross_output.metrics)
            else:
                step_metrics = step_metrics.merge(build_metric_bundle("cross_cycle"))

            if grl_classifier is not None and grl_weight > 0:
                from scar.regularization import (
                    compute_embodiment_adversarial_loss,
                    compute_grl_alpha,
                )

                grl_alpha = compute_grl_alpha(
                    step,
                    warmup_steps=int(getattr(args, 'grl_warmup_steps', 200)),
                    max_alpha=float(getattr(args, 'grl_alpha', 1.0)),
                )
                grl_module = (
                    grl_classifier.module if hasattr(grl_classifier, 'module') else grl_classifier
                )
                grl_module.set_grl_alpha(grl_alpha)

                emb_ids = getattr(batch, 'embodiment_id', None)
                if emb_ids is not None:
                    if emb_ids.dim() == 0:
                        emb_ids = emb_ids.expand(batch.observations.shape[0])
                    grl_scale = grl_weight * min(
                        step / max(getattr(args, 'grl_warmup_steps', 200), 1),
                        1.0,
                    )
                    grl_loss, grl_metrics = compute_embodiment_adversarial_loss(
                        grl_classifier,
                        output.conditioning_actions,
                        emb_ids,
                        mask=getattr(batch, 'mask', None),
                    )
                    step_loss = step_loss + grl_scale * grl_loss
                    step_metrics = step_metrics.merge(
                        grl_metrics.with_updates(grl_scale=grl_scale)
                    )
                else:
                    step_metrics = step_metrics.merge(build_metric_bundle("grl"))
            else:
                step_metrics = step_metrics.merge(build_metric_bundle("grl"))

            step_metrics = step_metrics.with_updates(
                total_loss=float(step_loss.detach().cpu())
            )
            micro_metrics.append(step_metrics)
            (step_loss / args.accumulate_grad_batches).backward()
        compute_elapsed += time.time() - compute_start

    optimize_start = time.time()
    metrics = average_metric_bundles(micro_metrics)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params,
        max_norm=args.clip_grad_norm,
    )
    grad_norm_value = (
        float(grad_norm.detach().cpu())
        if isinstance(grad_norm, torch.Tensor)
        else float(grad_norm)
    )
    current_lrs = get_current_optimizer_lrs(optimizer)
    optimizer.step()
    if advance_lr_scheduler:
        lr_scheduler.step()
    if update_ema and ema is not None and idm is not None:
        ema.update(unwrap_idm_module(idm))
    optimizer.zero_grad(set_to_none=True)
    optimize_elapsed = time.time() - optimize_start

    return _TrainStepResult(
        metrics=metrics,
        grad_norm=grad_norm_value,
        current_lrs=current_lrs,
        elapsed=time.time() - step_start,
        fetch_elapsed=fetch_elapsed,
        compute_elapsed=compute_elapsed,
        optimize_elapsed=optimize_elapsed,
    )


# This helper owns the "train logging" block: distributed reduction,
# console formatting, and WandB scalar logging.
def _log_train_step(
    *,
    step: int,
    step_result: _TrainStepResult,
    dist_ctx,
    device,
    rank0_print,
    wandb_run,
):
    """Reduces train metrics and emits console / WandB logs for one step."""

    reduced_metrics = MetricBundle(
        {
            key: reduce_scalar(value, dist_ctx, op='mean')
            for key, value in step_result.metrics.items()
        }
    )
    grad_norm_value = reduce_scalar(step_result.grad_norm, dist_ctx, op='mean')
    elapsed_value = reduce_scalar(step_result.elapsed, dist_ctx, op='mean')
    fetch_elapsed_value = reduce_scalar(step_result.fetch_elapsed, dist_ctx, op='mean')
    compute_elapsed_value = reduce_scalar(step_result.compute_elapsed, dist_ctx, op='mean')
    optimize_elapsed_value = reduce_scalar(step_result.optimize_elapsed, dist_ctx, op='mean')
    gpu_mem_str = ''
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated(device) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3
        gpu_mem_str = f' gpu_alloc={gpu_alloc:.1f}G gpu_rsv={gpu_reserved:.1f}G'
    try:
        import psutil as _psutil

        rss = _psutil.Process().memory_info().rss / 1024**3
        gpu_mem_str += f' cpu_rss={rss:.1f}G'
    except Exception:
        pass

    rank0_print(
        format_train_summary(
            step=step,
            metrics=reduced_metrics,
            grad_norm=grad_norm_value,
            elapsed=elapsed_value,
            suffix=(
                f" fetch={fetch_elapsed_value:.2f}s"
                f" compute={compute_elapsed_value:.2f}s"
                f" optim={optimize_elapsed_value:.2f}s"
                f"{gpu_mem_str}"
            ),
        )
    )

    if wandb_run is None:
        return

    train_log = build_namespaced_log_payload(
        "train",
        step=step,
        metrics=reduced_metrics,
        extra_scalars={
            'grad_norm': grad_norm_value,
            'step_time_sec': elapsed_value,
            'fetch_time_sec': fetch_elapsed_value,
            'compute_time_sec': compute_elapsed_value,
            'optimize_time_sec': optimize_elapsed_value,
        },
    )
    if 'idm' in step_result.current_lrs:
        train_log['train/idm_lr'] = step_result.current_lrs['idm']
    if 'lvp' in step_result.current_lrs:
        train_log['train/lvp_lr'] = step_result.current_lrs['lvp']
    if torch.cuda.is_available():
        train_log['train/gpu_alloc_gb'] = torch.cuda.memory_allocated(device) / 1024**3
        train_log['train/gpu_reserved_gb'] = torch.cuda.memory_reserved(device) / 1024**3
    try:
        import psutil as _psutil

        train_log['train/cpu_rss_gb'] = _psutil.Process().memory_info().rss / 1024**3
    except Exception:
        pass
    wandb_run.log(train_log)


# This helper owns the "checkpoint/save" block: deciding when to save,
# serializing extra state, and synchronizing ranks afterwards.
def _maybe_save_checkpoint(
    *,
    step: int,
    args,
    dist_ctx,
    output_dir: Path,
    device,
    rank0_print,
    wandb_run,
    idm,
    optimizer,
    lvp,
    lvp_trainable_modules: set[str],
    gt_action_head,
    grl_classifier,
    lr_scheduler,
    ema,
) -> None:
    """Saves a checkpoint on save steps and synchronizes ranks afterwards."""

    if not bool(getattr(args, "save_checkpoint", True)):
        return

    save_every = int(getattr(args, "save_every", 0))
    if save_every <= 0:
        return

    should_save = step % save_every == 0 or step == args.steps
    if not should_save:
        return

    if dist_ctx.is_main_process:
        probe_result = train_and_save_gt_action_probe(
            args,
            global_step=step,
            output_dir=output_dir,
            device=device,
            idm=idm,
            lvp=lvp,
        )
        if probe_result is not None:
            rank0_print(
                f"[probe] step={step:07d} train_steps={probe_result.steps} "
                f"loss={probe_result.train_loss:.6f} mse={probe_result.train_mse:.6f} "
                f"l1={probe_result.train_l1:.6f} seq_len={probe_result.sequence_length} "
                f"windows={probe_result.num_windows} source={probe_result.source}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "probe/gt_action_train_loss": probe_result.train_loss,
                        "probe/gt_action_train_mse": probe_result.train_mse,
                        "probe/gt_action_train_l1": probe_result.train_l1,
                        "probe/gt_action_train_steps": probe_result.steps,
                        "probe/gt_action_seq_len": probe_result.sequence_length,
                        "probe/gt_action_num_windows": probe_result.num_windows,
                    },
                    step=step,
                )
        elif bool(getattr(args.gt_action_probe, 'enabled', False)):
            rank0_print(
                f"[probe] step={step:07d} skipped (no fixed probe subset available)"
            )
        extra_state = {}
        if gt_action_head is not None:
            extra_state['gt_action_head'] = (
                gt_action_head.module.state_dict()
                if hasattr(gt_action_head, 'module')
                else gt_action_head.state_dict()
            )
        if grl_classifier is not None:
            extra_state['grl_classifier'] = (
                grl_classifier.module.state_dict()
                if hasattr(grl_classifier, 'module')
                else grl_classifier.state_dict()
            )
        save_checkpoint(
            output_dir,
            step,
            idm,
            optimizer,
            lvp=lvp,
            lvp_trainable_modules=lvp_trainable_modules,
            extra_state=extra_state,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
    distributed_barrier(dist_ctx)


def train_cycle(args) -> None:
    _pre_dist_rank0_print("[cycle]", "entering train_cycle")
    import_start = time.time()
    _pre_dist_rank0_print("[cycle]", "importing train dependencies...")
    from scar.dataloader import Batch, to_device
    _pre_dist_rank0_print(
        "[cycle]",
        f"importing train dependencies done in {time.time() - import_start:.2f}s",
    )
    component_import_start = time.time()
    _pre_dist_rank0_print("[cycle]", "importing cycle model/objective components...")
    from .models import LatentSpaceIDM, build_gt_action_head, describe_gt_action_head
    from .objectives import (
        compute_cycle_objectives,
        get_cycle_loss_scale,
        get_gt_action_loss_scale,
        resolve_conditioning_action_dim,
        should_compute_eval_cycle_loss,
    )

    _pre_dist_rank0_print(
        "[cycle]",
        f"importing cycle model/objective components done in "
        f"{time.time() - component_import_start:.2f}s",
    )

    _pre_dist_rank0_print("[cycle]", "calling set_seed...")
    seed_start = time.time()
    set_seed(args.seed)
    _pre_dist_rank0_print(
        "[cycle]",
        f"set_seed done in {time.time() - seed_start:.2f}s",
    )

    _pre_dist_rank0_print("[cycle]", "calling setup_distributed...")
    dist_start = time.time()
    dist_ctx, device = setup_distributed(args)
    rank0_print = _make_rank0_print(dist_ctx.is_main_process)
    rank0_print(
        f"[cycle] startup: setup_distributed done in {time.time() - dist_start:.2f}s "
        f"(device={device}, rank={dist_ctx.rank}, world_size={dist_ctx.world_size})"
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    idm_cfg = build_idm_cfg(args)
    lvp_trainable_modules = resolve_lvp_trainable_modules(args)
    prompt_embed, prompt_embed_len = load_prompt_embedding(args.prompt_embed_path)
    negative_prompt_embed, negative_prompt_embed_len = load_prompt_embedding(
        args.negative_prompt_embed_path
    )
    conditioning_action_dim = resolve_conditioning_action_dim(
        idm_cfg,
        action_source=args.lvp_action_source,
    )
    idm_input_source = str(getattr(args, "idm_input_source", "vae_latent"))
    if args.lvp_action_source == "idm" and idm_input_source != "vae_latent":
        incompatible_features: list[str] = []
        if args.enable_cycle_loss and args.cycle_loss_weight > 0:
            incompatible_features.append("cycle loss")
        if args.cross_cycle.train_enabled or args.cross_cycle.eval_enabled:
            incompatible_features.append("cross-cycle")
        if args.wrong_z.enabled and args.wrong_z.weight > 0:
            incompatible_features.append("wrong-z")
        if incompatible_features:
            raise ValueError(
                "IDM raw-patch input currently supports the standard IDM->LVP path only. "
                f"Disable {', '.join(incompatible_features)} or use "
                "model.idm_input_source=vae_latent."
            )
    if (
        dist_ctx.is_main_process
        and args.enable_cycle_loss
        and args.cycle_loss_weight > 0
        and args.lvp_action_source != "idm"
    ):
        rank0_print(
            "[cycle] cycle loss requested, but it only applies to IDM-conditioned runs; "
            "disabling it for GT-action conditioning."
        )
    if (
        dist_ctx.is_main_process
        and args.gt_action_loss_weight > 0
        and args.lvp_action_source != "idm"
    ):
        rank0_print(
            "[cycle] GT-action auxiliary loss requested, but it only applies to "
            "IDM-conditioned runs; disabling it for GT-action conditioning."
        )
    image_h, image_w = infer_idm_image_hw(idm_cfg)
    lvp_probe_cfg = build_lvp_cfg(args, height=image_h, width=image_w)
    align_lvp_action_dim(lvp_probe_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(lvp_probe_cfg)
    original_seq_len = int(idm_cfg.data.seq_len)
    if original_seq_len != target_seq_len:
        rank0_print(
            f"[cycle] aligning IDM sequence length from {original_seq_len} to "
            f"{target_seq_len} to match the LVP default horizon."
        )
        align_idm_seq_len(idm_cfg, target_seq_len)

    train_loader, eval_loader, right_target_eval_loader, num_embodiments = _run_startup_stage(
        rank0_print,
        "[cycle]",
        "building dataloaders",
        lambda: _build_dataloaders(idm_cfg, dist_ctx, args=args),
    )
    from scar.dataloader import InfiniteDataLoaderIter
    _preview_iter = iter(train_loader)
    first_batch_np = _run_startup_stage(
        rank0_print,
        "[cycle]",
        "fetching first train batch",
        lambda: next(_preview_iter),
    )
    obs_shape = infer_obs_shape(first_batch_np, idm_cfg)
    raw_seq_len = int(first_batch_np["observations"].shape[1])
    if raw_seq_len != target_seq_len:
        raise RuntimeError(
            f"Expected dataloader seq_len={target_seq_len} after alignment, "
            f"but got raw_seq_len={raw_seq_len}."
        )
    seq_len = raw_seq_len
    lvp_cfg = build_lvp_cfg(args, height=obs_shape[1], width=obs_shape[2])
    align_lvp_action_dim(lvp_cfg, conditioning_action_dim)

    # ---- Cross-embodiment cycle (optional) ----
    cross_cycle_train_enabled = args.cross_cycle.train_enabled
    cross_cycle_eval_enabled = args.cross_cycle.eval_enabled
    cross_cycle_paths = [p for p in args.data.cross_cycle_datasets if p]
    cross_cycle: CrossCycleState | None = None
    # Load secondary dataloader if needed for either training or eval
    cross_cycle_needs_data = (cross_cycle_train_enabled or cross_cycle_eval_enabled) and cross_cycle_paths and args.lvp_action_source == "idm"
    if cross_cycle_needs_data:
        secondary_loader = _run_startup_stage(
            rank0_print,
            "[cycle]",
            f"building cross-cycle secondary dataloader ({len(cross_cycle_paths)} datasets)",
            lambda: _build_cross_cycle_dataloader(idm_cfg, cross_cycle_paths, args=args),
        )
        cross_cycle = CrossCycleState(
            secondary_iter=InfiniteDataLoaderIter(secondary_loader),
            dataset_paths=cross_cycle_paths,
        )
        for p in cross_cycle_paths:
            rank0_print(f"[cycle]   cross-cycle dataset: {p}")
        rank0_print(f"[cycle]   cross-cycle train_enabled={cross_cycle_train_enabled}, eval_enabled={cross_cycle_eval_enabled}")
    elif cross_cycle_paths:
        rank0_print(
            "[cycle] --cross-cycle-dataset ignored (requires IDM action source)"
        )

    if dist_ctx.is_main_process:
        OmegaConf.save(idm_cfg, output_dir / "idm_config.yaml")
        OmegaConf.save(lvp_cfg, output_dir / "lvp_config.yaml")
        args_dict = args.to_flat_dict()
        (output_dir / "args.json").write_text(
            json.dumps(args_dict, indent=2, ensure_ascii=False)
        )
    distributed_barrier(dist_ctx)
    wandb_run = None

    la_dim = int(idm_cfg.model.la_dim)
    action_dim = int(lvp_cfg.algorithm.action_dim)
    rank0_print(f"[cycle] startup: building LVP model (this may take a while)...")
    _lvp_start = time.time()
    lvp = build_lvp_prior(
        lvp_cfg,
        device=device,
        trainable_modules=lvp_trainable_modules,
    )
    rank0_print(f"[cycle] startup: LVP built in {time.time() - _lvp_start:.2f}s")
    latent_input_dim = (int(lvp.lat_c), int(lvp.lat_h), int(lvp.lat_w))
    latent_temporal_stride = int(lvp.vae_stride[0])
    latent_idm_cfg = None
    idm_input_dim = latent_input_dim
    idm_temporal_stride = latent_temporal_stride
    idm = None
    gt_action_head = None
    gt_action_dim = int(getattr(idm_cfg.env, "action_dim", 0) or 0)
    freeze_idm = bool(getattr(args, "freeze_idm", False))
    if args.lvp_action_source == "idm":
        latent_idm_cfg = OmegaConf.create(
            OmegaConf.to_container(idm_cfg.model.idm, resolve=True)
        )
        if idm_input_source == "vae_latent":
            latent_idm_cfg.patch_size = 1
        elif idm_input_source == "rgb_patch":
            idm_input_dim = (3, int(obs_shape[1]), int(obs_shape[2]))
            idm_temporal_stride = 1
        else:
            raise ValueError(f"Unsupported IDM input source: {idm_input_source}")
        rank0_print(f"[cycle] startup: building IDM on {device}...")
        idm = LatentSpaceIDM(
            latent_idm_cfg,
            input_dim=idm_input_dim,
            la_dim=la_dim,
            temporal_stride=idm_temporal_stride,
        ).to(device)
        idm.input_source = idm_input_source
        rank0_print(
            f"[cycle] startup: IDM built, params={sum(p.numel() for p in idm.parameters())}, "
            f"input_source={idm_input_source}, input_dim={idm_input_dim}, "
            f"patch_size={latent_idm_cfg.patch_size}, temporal_stride={idm_temporal_stride}"
        )
        if freeze_idm:
            freeze_module(idm)
            rank0_print("[cycle] IDM finetune mode: parameters frozen, eval mode forced")
        if dist_ctx.enabled:
            idm = LatentActionIDMWrapper(idm).to(device)
        if args.gt_action_loss_weight > 0:
            if gt_action_dim <= 0:
                raise ValueError(
                    "GT-action auxiliary loss requested, but env.action_dim is unset."
                )
            gt_action_head = build_gt_action_head(
                head_type=args.gt_action_head_type,
                latent_action_dim=la_dim,
                gt_action_dim=gt_action_dim,
            ).to(device)
    lvp_trainable = [
        module
        for name, module in get_lvp_module_map(lvp).items()
        if name in lvp_trainable_modules and module is not None
    ]

    rank0_print(f"[cycle] startup: wrapping models with DDP...")
    _ddp_start = time.time()
    idm = maybe_wrap_ddp(
        idm,
        dist_ctx=dist_ctx,
        device=device,
        find_unused_parameters=True,
    )
    rank0_print(f"[cycle] startup: IDM DDP wrapped")
    gt_action_head = maybe_wrap_ddp(
        gt_action_head,
        dist_ctx=dist_ctx,
        device=device,
    )
    rank0_print(f"[cycle] startup: gt_action_head DDP wrapped")
    for name in sorted(lvp_trainable_modules):
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
        if name in lvp_trainable_modules and module is not None
    ]
    rank0_print(
        f"[cycle] startup: DDP wrapping done in {time.time() - _ddp_start:.2f}s, "
        f"lvp_trainable_modules={sorted(lvp_trainable_modules)}"
    )

    # ---- EMA teacher for cycle loss ----
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema: EMAModel | None = None
    if (
        ema_decay > 0
        and idm is not None
        and args.lvp_action_source == "idm"
        and args.enable_cycle_loss
    ):
        ema = EMAModel(unwrap_idm_module(idm), decay=ema_decay, device=device)
        rank0_print(f"[cycle] EMA teacher enabled: decay={ema_decay}")
    elif ema_decay > 0:
        rank0_print(
            f"[cycle] --ema-decay={ema_decay} ignored (requires IDM action source "
            "with cycle loss enabled)"
        )

    # ---- Gradient Reversal Embodiment Classifier ----
    from scar.regularization import EmbodimentClassifier
    grl_classifier: EmbodimentClassifier | None = None
    grl_enabled = args.grl.enabled
    grl_weight = float(args.grl.weight)
    if grl_enabled and grl_weight > 0 and num_embodiments >= 2 and idm is not None:
        grl_classifier = EmbodimentClassifier(
            input_dim=idm_cfg.model.la_dim,
            num_embodiments=num_embodiments,
            hidden_dim=int(args.grl.hidden_dim),
            grl_alpha=float(args.grl.alpha),
        ).to(device)
        grl_classifier = maybe_wrap_ddp(grl_classifier, dist_ctx=dist_ctx, device=device)
        rank0_print(
            f"[cycle] GRL embodiment classifier enabled: "
            f"num_embodiments={num_embodiments}, weight={grl_weight}, "
            f"alpha={args.grl.alpha}, "
            f"warmup={args.grl.warmup_steps}, "
            f"params={sum(p.numel() for p in grl_classifier.parameters())}"
        )
    elif grl_weight > 0:
        rank0_print(
            f"[cycle] --grl-weight={grl_weight} ignored "
            f"(need IDM + >=2 embodiments, got num_embodiments={num_embodiments})"
        )

    extra_idm = [m for m in [gt_action_head, grl_classifier] if m is not None]
    optimizer, optimizer_stats = build_joint_optimizer(
        idm,
        lvp_trainable,
        args=args,
        lvp_cfg=lvp_cfg,
        extra_idm_modules=extra_idm or None,
    )
    lr_scheduler = build_lr_scheduler(optimizer, lvp_cfg)
    if args.accumulate_grad_batches < 1:
        raise ValueError(
            f"--accumulate-grad-batches must be >= 1, got {args.accumulate_grad_batches}"
        )

    # ---- Resume from checkpoint (partial-match) ----
    start_step = 0
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        rank0_print(f"[cycle] resuming from checkpoint: {resume_from}")
        resume_state = load_checkpoint(
            resume_from,
            idm=idm,
            gt_action_head=gt_action_head,
            grl_classifier=grl_classifier,
            lvp=lvp,
            lvp_trainable_modules=lvp_trainable_modules,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            map_location=device,
        )
        start_step = resume_state.start_step
        rank0_print(
            f"[cycle] resumed: start_step={start_step}, "
            f"optimizer_restored={resume_state.optimizer_restored}, "
            f"idm_keys={len(resume_state.idm_loaded_keys)} loaded / "
            f"{len(resume_state.idm_skipped_keys)} skipped, "
            f"gt_head_keys={len(resume_state.gt_action_head_loaded_keys)} loaded / "
            f"{len(resume_state.gt_action_head_skipped_keys)} skipped, "
            f"grl_keys={len(resume_state.grl_classifier_loaded_keys)} loaded / "
            f"{len(resume_state.grl_classifier_skipped_keys)} skipped, "
            f"lvp_modules={resume_state.lvp_loaded_modules}"
        )
        if resume_state.idm_skipped_keys:
            rank0_print(
                f"[cycle] idm skipped keys: {resume_state.idm_skipped_keys[:10]}"
            )
        if resume_state.gt_action_head_skipped_keys:
            rank0_print(
                f"[cycle] gt_action_head skipped keys: "
                f"{resume_state.gt_action_head_skipped_keys[:10]}"
            )
        if resume_state.grl_classifier_skipped_keys:
            rank0_print(
                f"[cycle] grl_classifier skipped keys: "
                f"{resume_state.grl_classifier_skipped_keys[:10]}"
            )

    if dist_ctx.is_main_process:
        wandb_run = _run_startup_stage(
            rank0_print,
            "[cycle]",
            "initializing wandb",
            lambda: init_wandb_run(args, output_dir, idm_cfg, lvp_cfg),
        )
    distributed_barrier(dist_ctx)

    per_rank_batch_size = int(first_batch_np["observations"].shape[0])
    global_micro_batch_size = per_rank_batch_size * max(dist_ctx.world_size, 1)
    effective_global_batch_size = global_micro_batch_size * args.accumulate_grad_batches
    eval_video_ctx = build_eval_loop_context(
        args=args,
        device=device,
    )
    eval_metric_video_count = eval_video_ctx.video_metric_config.max_video_count
    eval_video_metric_names = list(eval_video_ctx.video_metric_config.names)

    train_iter = InfiniteDataLoaderIter(train_loader)
    rank0_print(
        f"[cycle] device={device}, obs_shape={obs_shape}, seq_len={seq_len}, "
        f"la_dim={la_dim}, action_dim={action_dim}, conditioning_source={args.lvp_action_source}, "
        f"rank={dist_ctx.rank}, world_size={dist_ctx.world_size}"
    )
    if latent_idm_cfg is not None:
        rank0_print(
            f"[cycle] idm_space={idm_input_source}, input_dim={idm_input_dim}, "
            f"patch_size={latent_idm_cfg.patch_size}, "
            f"temporal_stride={idm_temporal_stride}, "
            f"causal={bool(getattr(latent_idm_cfg, 'causal', False))}"
        )
    else:
        rank0_print(
            f"[cycle] idm_space=disabled, using dataset GT actions with dim={action_dim}"
        )
    rank0_print("[cycle] forward_model=LVP")
    cycle_burst_enabled = _is_cycle_burst_enabled(args)
    rank0_print(
        f"[cycle] cycle_loss_enabled={args.enable_cycle_loss and args.lvp_action_source == 'idm' and args.cycle_loss_weight > 0}, "
        f"cycle_loss_weight={args.cycle_loss_weight:.3f}, "
        f"cycle_warmup_steps={args.cycle_warmup_steps}, "
        f"cycle_end_steps={int(getattr(args, 'cycle_end_steps', 0))}, "
        f"cycle_burst_every={int(getattr(args, 'cycle_burst_every', 0))}, "
        f"cycle_burst_steps={int(getattr(args, 'cycle_burst_steps', 0))}"
    )
    if cycle_burst_enabled:
        rank0_print(
            "[cycle] cycle training mode=burst_only "
            "(regular train steps run recon/aux losses without cycle; "
            "cycle-only updates are inserted periodically)."
        )
    rank0_print(
        f"[cycle] eval_cycle_metric_enabled={should_compute_eval_cycle_loss(args=args, idm=idm)}"
    )
    rank0_print(
        f"[cycle] gt_action_aux_enabled={get_gt_action_loss_scale(args) > 0}, "
        f"gt_action_loss_weight={args.gt_action_loss_weight:.3f}, "
        f"gt_action_head={describe_gt_action_head(gt_action_head, unwrap_module_fn=unwrap_module)}"
    )
    rank0_print(
        f"[cycle] reparameterized_la={bool(getattr(args, 'use_reparameterized_la', False))}, "
        f"latent_action_kl_weight={float(getattr(args, 'latent_action_kl_weight', 0.0)):.6f}"
    )
    rank0_print(
        f"[cycle] gt_action_probe_enabled={bool(getattr(args.gt_action_probe, 'enabled', False))}, "
        f"gt_action_probe_train_steps={int(getattr(args.gt_action_probe, 'train_steps', 0))}, "
        f"gt_action_probe_lr={float(getattr(args.gt_action_probe, 'lr', 1e-4)):.2e}"
    )
    if bool(getattr(args.gt_action_probe, 'enabled', False)) and args.lvp_action_source != 'idm':
        rank0_print(
            "[cycle] GT-action probe is enabled, but latent-action probing only works when model.lvp_action_source=idm; it will be skipped."
        )
    rank0_print(
        f"[cycle] wrong_z_enabled={bool(getattr(args, 'wrong_z_enabled', False)) and args.lvp_action_source == 'idm'}, "
        f"wrong_z_weight={float(getattr(args, 'wrong_z_weight', 0.0)):.3f}, "
        f"wrong_z_warmup_steps={int(getattr(args, 'wrong_z_warmup_steps', 0))}, "
        f"wrong_z_sigma_hi={float(getattr(args, 'wrong_z_sigma_hi', 0.6)):.3f}, "
        "wrong_z_distance=flow_mse"
    )
    rank0_print(
        f"[cycle] cross_cycle_enabled={cross_cycle is not None}, "
        f"cross_cycle_loss_weight={args.cross_cycle_loss_weight:.3f}, "
        f"cross_cycle_warmup_steps={args.cross_cycle_warmup_steps}"
    )
    if cross_cycle is not None:
        for p in cross_cycle.dataset_paths:
            rank0_print(f"[cycle]   cross-cycle dataset: {p}")
    if optimizer_stats["idm_param_count"] > 0:
        rank0_print(
            f"[cycle] optimizer idm_lr={optimizer_stats['idm_lr']:.2e}, "
            f"idm_weight_decay={optimizer_stats['idm_weight_decay']:.2e}, "
            f"idm_params={optimizer_stats['idm_param_count']}"
        )
    else:
        idm_zero_reason = "IDM frozen" if freeze_idm and idm is not None else "IDM disabled"
        rank0_print(f"[cycle] optimizer idm_params=0 ({idm_zero_reason})")
    if optimizer_stats["lvp_param_count"] > 0:
        rank0_print(
            f"[cycle] optimizer lvp_lr={optimizer_stats['lvp_lr']:.2e}, "
            f"lvp_weight_decay={optimizer_stats['lvp_weight_decay']:.2e}, "
            f"lvp_betas={optimizer_stats['lvp_betas']}, "
            f"lvp_params={optimizer_stats['lvp_param_count']}"
        )
    rank0_print(
        f"[cycle] accumulate_grad_batches={args.accumulate_grad_batches}, "
        f"global_micro_batch={global_micro_batch_size}, "
        f"effective_global_batch={effective_global_batch_size}"
    )
    rank0_print(
        f"[cycle] save_checkpoint={bool(getattr(args, 'save_checkpoint', True))}, "
        f"save_every={int(getattr(args, 'save_every', 0))}"
    )
    rank0_print(
        "[cycle] metric_video_names="
        + (",".join(eval_video_metric_names) if eval_video_metric_names else "disabled")
        + f", eval_metric_video_count={eval_metric_video_count}"
    )

    # ---- Memory diagnostics before training ----
    if torch.cuda.is_available():
        _gpu_alloc = torch.cuda.memory_allocated(device) / 1024**3
        _gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3
        rank0_print(
            f"[cycle] GPU memory before training: "
            f"allocated={_gpu_alloc:.2f}GB, reserved={_gpu_reserved:.2f}GB"
        )
    try:
        import psutil
        _proc = psutil.Process()
        _rss = _proc.memory_info().rss / 1024**3
        rank0_print(f"[cycle] CPU RSS before training: {_rss:.2f}GB")
    except ImportError:
        pass

    if idm is not None:
        if freeze_idm:
            idm.eval()
        else:
            idm.train()
    if gt_action_head is not None:
        gt_action_head.train()
    if grl_classifier is not None:
        grl_classifier.train()
    set_lvp_mode(lvp, lvp_trainable_modules, training=True)
    trainable_params = list(
        iter_unique_trainable_params(idm, gt_action_head, grl_classifier, *lvp_trainable)
    )
    ddp_sync_modules = [
        module for module in [idm, gt_action_head, grl_classifier, *lvp_trainable] if module is not None
    ]
    regular_train_args = (
        _make_regular_args_without_cycle(args) if cycle_burst_enabled else args
    )
    cycle_only_args = _make_cycle_only_args(args) if cycle_burst_enabled else None

    # ---- Main train / eval / save loop ----
    try:
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, args.steps + 1):
            step_result = _run_one_train_step(
                args=regular_train_args,
                step=step,
                train_iter=train_iter,
                Batch=Batch,
                to_device=to_device,
                compute_cycle_objectives_fn=compute_cycle_objectives,
                device=device,
                seq_len=seq_len,
                dist_ctx=dist_ctx,
                ddp_sync_modules=ddp_sync_modules,
                lvp=lvp,
                idm=idm,
                gt_action_head=gt_action_head,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                lvp_trainable_modules=lvp_trainable_modules,
                ema=ema,
                cross_cycle=cross_cycle,
                cross_cycle_train_enabled=cross_cycle_train_enabled,
                grl_classifier=grl_classifier,
                grl_weight=grl_weight,
                trainable_params=trainable_params,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

            if step % args.log_every == 0 or step == start_step + 1:
                _log_train_step(
                    step=step,
                    step_result=step_result,
                    dist_ctx=dist_ctx,
                    device=device,
                    rank0_print=rank0_print,
                    wandb_run=wandb_run,
                )

            if eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                eval_start_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                rank0_print(f"[eval] step={step:06d} starting full eval at {eval_start_utc}")
                eval_result = run_periodic_eval(
                    args=args,
                    step=step,
                    lvp=lvp,
                    idm=idm,
                    gt_action_head=gt_action_head,
                    ema=ema,
                    cross_cycle=cross_cycle,
                    cross_cycle_eval_enabled=cross_cycle_eval_enabled,
                    eval_loader=eval_loader,
                    right_target_eval_loader=right_target_eval_loader,
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
                    ctx=eval_video_ctx,
                )
                eval_end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                rank0_print(
                    f"[eval] step={step:06d} finished at {eval_end_utc} "
                    f"elapsed={eval_result.total_eval_seconds:.2f}s "
                    f"(main={eval_result.main_eval_seconds:.2f}s, "
                    f"right_target={eval_result.right_target_eval_seconds:.2f}s)"
                )
                eval_summary = format_eval_summary(
                    step=step,
                    reduced_metrics=eval_result.reduced_metrics,
                )
                if eval_summary is not None:
                    rank0_print(eval_summary)
                for metric_name, metric_value in eval_result.video_metrics.items():
                    rank0_print(f'[eval] {metric_name}={metric_value:.6f}')
                for video_path in eval_result.saved_videos:
                    rank0_print(f'[eval] saved_video={video_path}')
                right_target_summary = format_eval_summary(
                    step=step,
                    reduced_metrics=eval_result.right_target_reduced_metrics,
                    prefix="[eval_right_target]",
                )
                if right_target_summary is not None:
                    rank0_print(right_target_summary)
                for metric_name, metric_value in eval_result.right_target_video_metrics.items():
                    rank0_print(f'[eval_right_target] {metric_name}={metric_value:.6f}')
                for video_path in eval_result.right_target_saved_videos:
                    rank0_print(f'[eval_right_target] saved_video={video_path}')
                if wandb_run is not None:
                    wandb_run.log(
                        build_eval_log_payload(step=step, result=eval_result)
                    )
                distributed_barrier(dist_ctx)
                save_every = int(getattr(args, "save_every", 0))
                is_save_step = save_every > 0 and (step % save_every == 0 or step == args.steps)
                if bool(getattr(args, "save_checkpoint", True)) and not is_save_step:
                    if dist_ctx.is_main_process:
                        extra_state = {}
                        if gt_action_head is not None:
                            extra_state["gt_action_head"] = (
                                gt_action_head.module.state_dict()
                                if hasattr(gt_action_head, "module")
                                else gt_action_head.state_dict()
                            )
                        if grl_classifier is not None:
                            extra_state["grl_classifier"] = (
                                grl_classifier.module.state_dict()
                                if hasattr(grl_classifier, "module")
                                else grl_classifier.state_dict()
                            )
                        save_checkpoint(
                            output_dir,
                            step,
                            idm,
                            optimizer,
                            lvp=lvp,
                            lvp_trainable_modules=lvp_trainable_modules,
                            extra_state=extra_state,
                            lr_scheduler=lr_scheduler,
                            ema=ema,
                            write_step_checkpoint=False,
                        )
                        rank0_print(
                            f"[cycle] refreshed latest checkpoint after eval at step={step:07d}"
                        )
                    distributed_barrier(dist_ctx)

            _maybe_save_checkpoint(
                step=step,
                args=args,
                dist_ctx=dist_ctx,
                output_dir=output_dir,
                device=device,
                rank0_print=rank0_print,
                wandb_run=wandb_run,
                idm=idm,
                optimizer=optimizer,
                lvp=lvp,
                lvp_trainable_modules=lvp_trainable_modules,
                gt_action_head=gt_action_head,
                grl_classifier=grl_classifier,
                lr_scheduler=lr_scheduler,
                ema=ema,
            )

            if cycle_burst_enabled and _should_run_cycle_burst(args, step):
                burst_steps = int(getattr(args, "cycle_burst_steps", 0))
                cycle_scale = get_cycle_loss_scale(step, args)
                rank0_print(
                    f"[cycle_burst] main_step={step:06d} starting burst_steps={burst_steps} "
                    f"cycle_loss_scale={cycle_scale:.6f}"
                )
                burst_start = time.time()
                last_burst_result = None
                for burst_idx in range(1, burst_steps + 1):
                    last_burst_result = _run_one_train_step(
                        args=cycle_only_args,
                        step=step,
                        train_iter=train_iter,
                        Batch=Batch,
                        to_device=to_device,
                        compute_cycle_objectives_fn=compute_cycle_objectives,
                        device=device,
                        seq_len=seq_len,
                        dist_ctx=dist_ctx,
                        ddp_sync_modules=ddp_sync_modules,
                        lvp=lvp,
                        idm=idm,
                        gt_action_head=None,
                        prompt_embed=prompt_embed,
                        prompt_embed_len=prompt_embed_len,
                        lvp_trainable_modules=lvp_trainable_modules,
                        ema=None,
                        cross_cycle=None,
                        cross_cycle_train_enabled=False,
                        grl_classifier=None,
                        grl_weight=0.0,
                        trainable_params=trainable_params,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        advance_lr_scheduler=False,
                        update_ema=False,
                    )
                    if (
                        burst_idx == 1
                        or burst_idx == burst_steps
                        or (
                            int(getattr(args, "log_every", 0)) > 0
                            and burst_idx % int(getattr(args, "log_every", 0)) == 0
                        )
                    ):
                        reduced_burst_metrics = MetricBundle(
                            {
                                key: reduce_scalar(value, dist_ctx, op='mean')
                                for key, value in last_burst_result.metrics.items()
                            }
                        )
                        rank0_print(
                            format_train_summary(
                                step=step,
                                metrics=reduced_burst_metrics,
                                grad_norm=reduce_scalar(last_burst_result.grad_norm, dist_ctx, op='mean'),
                                elapsed=reduce_scalar(last_burst_result.elapsed, dist_ctx, op='mean'),
                                suffix=(
                                    f" fetch={reduce_scalar(last_burst_result.fetch_elapsed, dist_ctx, op='mean'):.2f}s "
                                    f"compute={reduce_scalar(last_burst_result.compute_elapsed, dist_ctx, op='mean'):.2f}s "
                                    f"optim={reduce_scalar(last_burst_result.optimize_elapsed, dist_ctx, op='mean'):.2f}s "
                                    f"[cycle_burst {burst_idx}/{burst_steps}]"
                                ),
                            ).replace("[train]", "[cycle_burst]", 1)
                        )
                rank0_print(
                    f"[cycle_burst] main_step={step:06d} finished in {time.time() - burst_start:.2f}s"
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        distributed_barrier(dist_ctx)
        cleanup_distributed(dist_ctx)


def eval_cycle(args) -> EvalLoopResult:
    _pre_dist_rank0_print("[cycle-eval]", "entering eval_cycle")
    component_import_start = time.time()
    _pre_dist_rank0_print("[cycle-eval]", "importing eval model/objective components...")
    from scar.dataloader import InfiniteDataLoaderIter
    from .models import LatentSpaceIDM, build_gt_action_head, describe_gt_action_head
    from .objectives import (
        resolve_conditioning_action_dim,
        should_compute_eval_cycle_loss,
    )

    _pre_dist_rank0_print(
        "[cycle-eval]",
        "importing eval model/objective components done in "
        f"{time.time() - component_import_start:.2f}s",
    )

    _pre_dist_rank0_print("[cycle-eval]", "calling set_seed...")
    seed_start = time.time()
    set_seed(args.seed)
    _pre_dist_rank0_print(
        "[cycle-eval]",
        f"set_seed done in {time.time() - seed_start:.2f}s",
    )

    _pre_dist_rank0_print("[cycle-eval]", "calling setup_distributed...")
    dist_start = time.time()
    dist_ctx, device = setup_distributed(args)
    rank0_print = _make_rank0_print(dist_ctx.is_main_process)
    rank0_print(
        f"[cycle-eval] startup: setup_distributed done in {time.time() - dist_start:.2f}s "
        f"(device={device}, rank={dist_ctx.rank}, world_size={dist_ctx.world_size})"
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    idm_cfg = build_idm_cfg(args)
    lvp_trainable_modules = resolve_lvp_trainable_modules(args)
    prompt_embed, prompt_embed_len = load_prompt_embedding(args.prompt_embed_path)
    negative_prompt_embed, negative_prompt_embed_len = load_prompt_embedding(
        args.negative_prompt_embed_path
    )
    conditioning_action_dim = resolve_conditioning_action_dim(
        idm_cfg,
        action_source=args.lvp_action_source,
    )
    idm_input_source = str(getattr(args, "idm_input_source", "vae_latent"))
    if args.lvp_action_source == "idm" and idm_input_source != "vae_latent":
        incompatible_features: list[str] = []
        if args.cross_cycle.eval_enabled:
            incompatible_features.append("cross-cycle eval")
        if args.wrong_z.enabled and args.wrong_z.weight > 0:
            incompatible_features.append("wrong-z")
        if incompatible_features:
            raise ValueError(
                "IDM raw-patch input currently supports the standard IDM->LVP path only. "
                f"Disable {', '.join(incompatible_features)} or use "
                "model.idm_input_source=vae_latent."
            )
    image_h, image_w = infer_idm_image_hw(idm_cfg)
    lvp_probe_cfg = build_lvp_cfg(args, height=image_h, width=image_w)
    align_lvp_action_dim(lvp_probe_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(lvp_probe_cfg)
    original_seq_len = int(idm_cfg.data.seq_len)
    if original_seq_len != target_seq_len:
        rank0_print(
            f"[cycle-eval] aligning IDM sequence length from {original_seq_len} to "
            f"{target_seq_len} to match the LVP default horizon."
        )
        align_idm_seq_len(idm_cfg, target_seq_len)

    train_loader, eval_loader, right_target_eval_loader, num_embodiments = _run_startup_stage(
        rank0_print,
        "[cycle-eval]",
        "building dataloaders",
        lambda: _build_dataloaders(idm_cfg, dist_ctx, args=args),
    )
    if eval_loader is None and right_target_eval_loader is None:
        raise RuntimeError(
            "Eval-only run requires at least one eval loader "
            "(main eval or right_target_eval)."
        )

    _preview_iter = iter(train_loader)
    first_batch_np = _run_startup_stage(
        rank0_print,
        "[cycle-eval]",
        "fetching first train batch",
        lambda: next(_preview_iter),
    )
    obs_shape = infer_obs_shape(first_batch_np, idm_cfg)
    raw_seq_len = int(first_batch_np["observations"].shape[1])
    if raw_seq_len != target_seq_len:
        raise RuntimeError(
            f"Expected dataloader seq_len={target_seq_len} after alignment, "
            f"but got raw_seq_len={raw_seq_len}."
        )
    seq_len = raw_seq_len
    lvp_cfg = build_lvp_cfg(args, height=obs_shape[1], width=obs_shape[2])
    align_lvp_action_dim(lvp_cfg, conditioning_action_dim)

    cross_cycle_eval_enabled = args.cross_cycle.eval_enabled
    cross_cycle_paths = [p for p in args.data.cross_cycle_datasets if p]
    cross_cycle: CrossCycleState | None = None
    if cross_cycle_eval_enabled and cross_cycle_paths and args.lvp_action_source == "idm":
        secondary_loader = _run_startup_stage(
            rank0_print,
            "[cycle-eval]",
            f"building cross-cycle secondary dataloader ({len(cross_cycle_paths)} datasets)",
            lambda: _build_cross_cycle_dataloader(idm_cfg, cross_cycle_paths, args=args),
        )
        cross_cycle = CrossCycleState(
            secondary_iter=InfiniteDataLoaderIter(secondary_loader),
            dataset_paths=cross_cycle_paths,
        )
        for p in cross_cycle_paths:
            rank0_print(f"[cycle-eval]   cross-cycle dataset: {p}")
    elif cross_cycle_paths and args.lvp_action_source != "idm":
        rank0_print(
            "[cycle-eval] cross-cycle eval datasets ignored "
            "(requires IDM action source)."
        )

    if dist_ctx.is_main_process:
        OmegaConf.save(idm_cfg, output_dir / "idm_config.yaml")
        OmegaConf.save(lvp_cfg, output_dir / "lvp_config.yaml")
        (output_dir / "args.json").write_text(
            json.dumps(args.to_flat_dict(), indent=2, ensure_ascii=False)
        )
    distributed_barrier(dist_ctx)

    wandb_run = None
    la_dim = int(idm_cfg.model.la_dim)
    action_dim = int(lvp_cfg.algorithm.action_dim)

    rank0_print("[cycle-eval] startup: building LVP model (this may take a while)...")
    lvp_start = time.time()
    lvp = build_lvp_prior(
        lvp_cfg,
        device=device,
        trainable_modules=lvp_trainable_modules,
    )
    rank0_print(
        f"[cycle-eval] startup: LVP built in {time.time() - lvp_start:.2f}s"
    )

    latent_input_dim = (int(lvp.lat_c), int(lvp.lat_h), int(lvp.lat_w))
    latent_temporal_stride = int(lvp.vae_stride[0])
    latent_idm_cfg = None
    idm_input_dim = latent_input_dim
    idm_temporal_stride = latent_temporal_stride
    idm = None
    gt_action_head = None
    gt_action_dim = int(getattr(idm_cfg.env, "action_dim", 0) or 0)
    if args.lvp_action_source == "idm":
        latent_idm_cfg = OmegaConf.create(
            OmegaConf.to_container(idm_cfg.model.idm, resolve=True)
        )
        if idm_input_source == "vae_latent":
            latent_idm_cfg.patch_size = 1
        elif idm_input_source == "rgb_patch":
            idm_input_dim = (3, int(obs_shape[1]), int(obs_shape[2]))
            idm_temporal_stride = 1
        else:
            raise ValueError(f"Unsupported IDM input source: {idm_input_source}")
        rank0_print(f"[cycle-eval] startup: building IDM on {device}...")
        idm = LatentSpaceIDM(
            latent_idm_cfg,
            input_dim=idm_input_dim,
            la_dim=la_dim,
            temporal_stride=idm_temporal_stride,
        ).to(device)
        idm.input_source = idm_input_source
        rank0_print(
            f"[cycle-eval] startup: IDM built, params={sum(p.numel() for p in idm.parameters())}, "
            f"input_source={idm_input_source}, input_dim={idm_input_dim}, "
            f"patch_size={latent_idm_cfg.patch_size}, temporal_stride={idm_temporal_stride}"
        )
        if args.gt_action_loss_weight > 0:
            if gt_action_dim <= 0:
                raise ValueError(
                    "GT-action auxiliary loss requested, but env.action_dim is unset."
                )
            gt_action_head = build_gt_action_head(
                head_type=args.gt_action_head_type,
                latent_action_dim=la_dim,
                gt_action_dim=gt_action_dim,
            ).to(device)

    lvp_trainable = [
        module
        for name, module in get_lvp_module_map(lvp).items()
        if name in lvp_trainable_modules and module is not None
    ]

    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema: EMAModel | None = None
    if (
        ema_decay > 0
        and idm is not None
        and args.lvp_action_source == "idm"
        and args.enable_cycle_loss
    ):
        ema = EMAModel(idm, decay=ema_decay, device=device)
        rank0_print(f"[cycle-eval] EMA teacher enabled: decay={ema_decay}")

    start_step = 0
    resume_from = getattr(args, "resume_from", None)
    if resume_from is None:
        raise ValueError("Eval-only run requires resume_from / --ckpt.")
    rank0_print(f"[cycle-eval] loading checkpoint: {resume_from}")
    resume_state = load_checkpoint(
        resume_from,
        idm=idm,
        gt_action_head=gt_action_head,
        lvp=lvp,
        lvp_trainable_modules=lvp_trainable_modules,
        ema=ema,
        map_location=device,
    )
    start_step = resume_state.start_step
    rank0_print(
        f"[cycle-eval] loaded checkpoint: step={start_step}, "
        f"idm_keys={len(resume_state.idm_loaded_keys)} loaded / "
        f"{len(resume_state.idm_skipped_keys)} skipped, "
        f"gt_head_keys={len(resume_state.gt_action_head_loaded_keys)} loaded / "
        f"{len(resume_state.gt_action_head_skipped_keys)} skipped, "
        f"lvp_modules={resume_state.lvp_loaded_modules}"
    )

    if dist_ctx.is_main_process:
        wandb_run = _run_startup_stage(
            rank0_print,
            "[cycle-eval]",
            "initializing wandb",
            lambda: init_wandb_run(args, output_dir, idm_cfg, lvp_cfg),
        )
    distributed_barrier(dist_ctx)

    eval_video_ctx = build_eval_loop_context(
        args=args,
        device=device,
    )
    eval_metric_video_count = eval_video_ctx.video_metric_config.max_video_count
    eval_video_metric_names = list(eval_video_ctx.video_metric_config.names)
    main_eval_windows = (
        len(eval_loader.dataset) if eval_loader is not None and hasattr(eval_loader, "dataset") else 0
    )
    right_target_eval_windows = (
        len(right_target_eval_loader.dataset)
        if right_target_eval_loader is not None and hasattr(right_target_eval_loader, "dataset")
        else 0
    )

    rank0_print(
        f"[cycle-eval] device={device}, obs_shape={obs_shape}, seq_len={seq_len}, "
        f"la_dim={la_dim}, action_dim={action_dim}, conditioning_source={args.lvp_action_source}, "
        f"rank={dist_ctx.rank}, world_size={dist_ctx.world_size}"
    )
    if latent_idm_cfg is not None:
        rank0_print(
            f"[cycle-eval] idm_space={idm_input_source}, input_dim={idm_input_dim}, "
            f"patch_size={latent_idm_cfg.patch_size}, "
            f"temporal_stride={idm_temporal_stride}, "
            f"causal={bool(getattr(latent_idm_cfg, 'causal', False))}"
        )
    else:
        rank0_print(
            f"[cycle-eval] idm_space=disabled, using dataset GT actions with dim={action_dim}"
        )
    rank0_print(
        f"[cycle-eval] eval_windows={main_eval_windows}, "
        f"right_target_eval_windows={right_target_eval_windows}, "
        f"num_embodiments={num_embodiments}"
    )
    rank0_print(
        f"[cycle-eval] eval_cycle_metric_enabled={should_compute_eval_cycle_loss(args=args, idm=idm)}"
    )
    rank0_print(
        f"[cycle-eval] gt_action_head={describe_gt_action_head(gt_action_head, unwrap_module_fn=unwrap_module)}"
    )
    rank0_print(
        "[cycle-eval] metric_video_names="
        + (",".join(eval_video_metric_names) if eval_video_metric_names else "disabled")
        + f", eval_metric_video_count={eval_metric_video_count}"
    )

    try:
        eval_start_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        rank0_print(
            f"[cycle-eval] step={start_step:06d} starting eval at {eval_start_utc}"
        )
        eval_result = run_periodic_eval(
            args=args,
            step=start_step,
            lvp=lvp,
            idm=idm,
            gt_action_head=gt_action_head,
            ema=ema,
            cross_cycle=cross_cycle,
            cross_cycle_eval_enabled=cross_cycle_eval_enabled,
            eval_loader=eval_loader,
            right_target_eval_loader=right_target_eval_loader,
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
            ctx=eval_video_ctx,
        )
        if dist_ctx.is_main_process:
            eval_end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            rank0_print(
                f"[cycle-eval] step={start_step:06d} finished at {eval_end_utc} "
                f"elapsed={eval_result.total_eval_seconds:.2f}s "
                f"(main={eval_result.main_eval_seconds:.2f}s, "
                f"right_target={eval_result.right_target_eval_seconds:.2f}s)"
            )
            eval_summary = format_eval_summary(
                step=start_step,
                reduced_metrics=eval_result.reduced_metrics,
            )
            if eval_summary is not None:
                rank0_print(eval_summary)
            for metric_name, metric_value in eval_result.video_metrics.items():
                rank0_print(f"[eval] {metric_name}={metric_value:.6f}")
            for video_path in eval_result.saved_videos:
                rank0_print(f"[eval] saved_video={video_path}")
            right_target_summary = format_eval_summary(
                step=start_step,
                reduced_metrics=eval_result.right_target_reduced_metrics,
                prefix="[eval_right_target]",
            )
            if right_target_summary is not None:
                rank0_print(right_target_summary)
            for metric_name, metric_value in eval_result.right_target_video_metrics.items():
                rank0_print(f"[eval_right_target] {metric_name}={metric_value:.6f}")
            for video_path in eval_result.right_target_saved_videos:
                rank0_print(f"[eval_right_target] saved_video={video_path}")
            if wandb_run is not None:
                wandb_run.log(build_eval_log_payload(step=start_step, result=eval_result))
        return eval_result
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        distributed_barrier(dist_ctx)
        cleanup_distributed(dist_ctx)


def main_cycle() -> None:
    args = parse_config_args()
    train_cycle(args)


__all__ = [
    "eval_cycle",
    "main_cycle",
    "train_cycle",
]
