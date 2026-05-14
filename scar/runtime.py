from __future__ import annotations

import contextlib
import datetime as dt
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

from .environment import LVP_ROOT


_DEBUG_PRINTED_TAGS: set[str] = set()


def debug_log_once(tag: str, message: str) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    if tag in _DEBUG_PRINTED_TAGS:
        return
    print(f"[log:{tag}] {message}")
    _DEBUG_PRINTED_TAGS.add(tag)


class LatentActionIDMWrapper(nn.Module):
    """DDP-friendly wrapper that preserves the full IDM output object."""

    def __init__(self, idm: nn.Module):
        super().__init__()
        self.idm = idm

    def forward(self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs):
        return self.idm(
            observations,
            timesteps=timesteps,
            states=states,
            **kwargs,
        )


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def set_seed(seed: int) -> None:
    import os
    import sys

    print(f"[set_seed] pid={os.getpid()} seed={seed}", file=sys.stderr, flush=True)
    random.seed(seed)
    print(f"[set_seed] random.seed done", file=sys.stderr, flush=True)
    torch.manual_seed(seed)
    print(f"[set_seed] torch.manual_seed done", file=sys.stderr, flush=True)

    np.random.seed(seed)
    print(f"[set_seed] all done", file=sys.stderr, flush=True)


def setup_distributed(args) -> tuple[DistributedContext, torch.device]:
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_minutes = int(os.environ.get("IDM_LVP_DDP_TIMEOUT_MINUTES", "120"))
        dist.init_process_group(
            backend=backend,
            timeout=dt.timedelta(minutes=max(timeout_minutes, 1)),
        )

    if enabled and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    ), device


def cleanup_distributed(dist_ctx: DistributedContext) -> None:
    if dist_ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(dist_ctx: DistributedContext) -> None:
    if dist_ctx.enabled:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[dist_ctx.local_rank])
        else:
            dist.barrier()


def reduce_scalar(value: float, dist_ctx: DistributedContext, *, op: str = "mean") -> float:
    if not dist_ctx.enabled:
        return float(value)
    device = (
        torch.device("cuda", dist_ctx.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor /= dist_ctx.world_size
    elif op != "sum":
        raise ValueError(f"Unsupported reduce op: {op}")
    return float(tensor.item())


def reduce_pair_mean(value: float, count: int, dist_ctx: DistributedContext) -> float:
    if not dist_ctx.enabled:
        return float(value) / max(int(count), 1)
    device = (
        torch.device("cuda", dist_ctx.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tensor = torch.tensor([float(value), float(count)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(int(tensor[1].item()), 1)
    return float(tensor[0].item() / total_count)


def unwrap_module(module: nn.Module | None):
    if module is None:
        return None
    return module.module if isinstance(module, DDP) else module


def unwrap_idm_module(module: nn.Module | None):
    module = unwrap_module(module)
    if module is None:
        return None
    return module.idm if isinstance(module, LatentActionIDMWrapper) else module


def maybe_wrap_ddp(
    module: nn.Module | None,
    *,
    dist_ctx: DistributedContext,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module | None:
    if module is None or not dist_ctx.enabled:
        return module
    if not any(param.requires_grad for param in module.parameters()):
        return module
    if device.type == "cuda":
        return DDP(
            module,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=find_unused_parameters,
        )
    return DDP(module, find_unused_parameters=find_unused_parameters)


@contextlib.contextmanager
def maybe_ddp_no_sync(modules: list[nn.Module], enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.ExitStack() as stack:
        for module in modules:
            if isinstance(module, DDP):
                stack.enter_context(module.no_sync())
        yield


@contextlib.contextmanager
def temporarily_unwrap_lvp_ddp(lvp):
    original_modules = {}
    for name, module in get_lvp_module_map(lvp).items():
        base_module = unwrap_module(module)
        if base_module is not module:
            original_modules[name] = module
            setattr(lvp, name, base_module)
    try:
        yield
    finally:
        for name, module in original_modules.items():
            setattr(lvp, name, module)


@contextlib.contextmanager
def temporarily_disable_grads(*modules: nn.Module | None):
    previous_flags: list[tuple[torch.nn.Parameter, bool]] = []
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            previous_flags.append((param, param.requires_grad))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in previous_flags:
            param.requires_grad_(requires_grad)


@contextlib.contextmanager
def temporarily_frozen_eval(module: nn.Module):
    was_training = module.training
    with temporarily_disable_grads(module):
        module.eval()
        try:
            yield
        finally:
            module.train(was_training)


def freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def unfreeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.train()
    for param in module.parameters():
        param.requires_grad_(True)


def get_lvp_module_map(lvp) -> dict[str, nn.Module | None]:
    return {
        "model": getattr(lvp, "model", None),
        "vae": getattr(lvp, "vae", None),
        "text_encoder": getattr(lvp, "text_encoder", None),
        "clip": getattr(lvp, "clip", None),
        "action_encoder": getattr(lvp, "action_encoder", None),
    }


def set_lvp_mode(lvp, trainable_modules: set[str], training: bool) -> None:
    for name, module in get_lvp_module_map(lvp).items():
        if module is None:
            continue
        if name in trainable_modules:
            module.train(training)
        else:
            module.eval()


def resolve_lvp_trainable_modules(args) -> set[str]:
    modules = set(args.lvp_train_module)
    if args.joint_tune_lvp:
        modules.update({"model", "action_encoder"})
    return modules


def iter_unique_trainable_params(*modules: nn.Module | None):
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield param


def get_lvp_optimizer_hparams(args, lvp_cfg) -> tuple[float, float, tuple[float, float]]:
    lvp_lr = float(args.lvp_lr) if args.lvp_lr is not None else float(lvp_cfg.algorithm.lr)
    lvp_weight_decay = float(lvp_cfg.algorithm.weight_decay)
    lvp_betas = tuple(float(beta) for beta in lvp_cfg.algorithm.betas)
    return lvp_lr, lvp_weight_decay, lvp_betas


def build_joint_optimizer(
    idm: nn.Module | None,
    lvp_trainable: list[nn.Module],
    *,
    args,
    lvp_cfg,
    extra_idm_modules: list[nn.Module] | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    idm_params = list(iter_unique_trainable_params(idm, *(extra_idm_modules or [])))
    lvp_params = list(iter_unique_trainable_params(*lvp_trainable))
    lvp_lr, lvp_weight_decay, lvp_betas = get_lvp_optimizer_hparams(args, lvp_cfg)

    param_groups = []
    if idm_params:
        param_groups.append(
            {
                "name": "idm",
                "params": idm_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        )
    if lvp_params:
        param_groups.append(
            {
                "name": "lvp",
                "params": lvp_params,
                "lr": lvp_lr,
                "weight_decay": lvp_weight_decay,
                "betas": lvp_betas,
            }
        )
    if not param_groups:
        raise ValueError(
            "No trainable parameters were found for the optimizer. "
            "Enable IDM training or mark at least one LVP module as trainable."
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_stats = {
        "idm_lr": float(args.lr),
        "idm_weight_decay": float(args.weight_decay),
        "idm_param_count": sum(param.numel() for param in idm_params),
        "lvp_lr": float(lvp_lr),
        "lvp_weight_decay": float(lvp_weight_decay),
        "lvp_betas": lvp_betas,
        "lvp_param_count": sum(param.numel() for param in lvp_params),
    }
    return optimizer, optimizer_stats


def build_lr_scheduler(optimizer: torch.optim.Optimizer, lvp_cfg):
    scheduler_cfg = OmegaConf.to_container(
        lvp_cfg.algorithm.lr_scheduler,
        resolve=True,
    )
    if not isinstance(scheduler_cfg, dict):
        raise TypeError(
            f"Expected algorithm.lr_scheduler to resolve to a dict, got {type(scheduler_cfg)}"
        )
    return get_scheduler(
        optimizer=optimizer,
        **scheduler_cfg,
    )


def get_current_optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    lr_by_group: dict[str, float] = {}
    for group_idx, param_group in enumerate(optimizer.param_groups):
        group_name = str(param_group.get("name", f"group_{group_idx}"))
        lr_by_group[group_name] = float(param_group["lr"])
    return lr_by_group


@contextlib.contextmanager
def maybe_stub_wan_video_metrics(lvp_cfg):
    """Provide a no-op WAN video metric module for the SCAR bridge path.

    WAN's text/video base class eagerly imports ``algorithms.common.metrics.video``
    even though this bridge pipeline computes eval metrics on its own. When bridge
    config disables WAN internal metrics, temporarily inject a lightweight stub so
    the WAN import path does not pull external metric backbones such as ``clip``.
    """

    metric_types = list(getattr(lvp_cfg.algorithm.logging, "metrics", []) or [])
    if metric_types:
        yield
        return

    import importlib
    import sys
    import types

    module_name = "algorithms.common.metrics.video"
    previous_module = sys.modules.get(module_name)

    stub_module = types.ModuleType(module_name)

    class SharedVideoMetricModelRegistry:
        def __call__(self, *args, **kwargs):
            return None

    class VideoMetric(nn.Module):
        def __init__(self, registry, metric_types, split_batch_size=None):
            super().__init__()
            self.registry = registry
            self.metric_types = list(metric_types)
            self.split_batch_size = split_batch_size

        def forward(self, *args, **kwargs):
            return {}

        def log(self, prefix):
            return {}

    stub_module.SharedVideoMetricModelRegistry = SharedVideoMetricModelRegistry
    stub_module.VideoMetric = VideoMetric
    stub_module.__all__ = ["VideoMetric", "SharedVideoMetricModelRegistry"]

    sys.modules[module_name] = stub_module
    metrics_parent = importlib.import_module("algorithms.common.metrics")
    previous_attr = getattr(metrics_parent, "video", None)
    setattr(metrics_parent, "video", stub_module)
    try:
        yield
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if previous_attr is None:
            if hasattr(metrics_parent, "video"):
                delattr(metrics_parent, "video")
        else:
            setattr(metrics_parent, "video", previous_attr)


def build_lvp_prior(
    lvp_cfg,
    device: torch.device,
    trainable_modules: set[str] | None = None,
):
    try:
        with maybe_stub_wan_video_metrics(lvp_cfg):
            from algorithms.wan.wan_v2a2v import WanVideoToActionToVideo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Importing large-video-planner failed. Make sure the training environment "
            "has the LVP dependencies installed, especially 'transformers'."
        ) from exc

    algo = WanVideoToActionToVideo(lvp_cfg.algorithm)
    algo.configure_model()
    algo = algo.to(device)
    algo.vae_scale = [algo.vae_mean, algo.vae_inv_std]
    algo.eval()
    trainable_modules = set(trainable_modules or [])
    for name, module in get_lvp_module_map(algo).items():
        if name in trainable_modules:
            unfreeze_module(module)
        else:
            freeze_module(module)
    return algo


def trim_batch(batch, seq_len: int):
    from scar.dataloader import Batch

    trimmed = {}
    for key, value in batch.__dict__.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] >= seq_len:
            trimmed[key] = value[:, :seq_len]
        else:
            trimmed[key] = value
    return Batch(**trimmed)


def take_batch_prefix(batch, count: int):
    from scar.dataloader import Batch

    sliced = {}
    for key, value in batch.__dict__.items():
        if isinstance(value, torch.Tensor) and value.shape[0] >= count:
            sliced[key] = value[:count]
        else:
            sliced[key] = value
    return Batch(**sliced)


def select_rgb_channels(observations: torch.Tensor) -> torch.Tensor:
    channels = observations.shape[2]
    if channels == 3:
        return observations
    if channels > 3 and channels % 3 == 0:
        return observations[:, :, -3:]
    raise ValueError(
        f"LVP prior currently expects RGB frames, got channel count {channels}"
    )


def to_lvp_range(videos: torch.Tensor) -> torch.Tensor:
    video_min = videos.min().item()
    video_max = videos.max().item()
    if -1.1 <= video_min and video_max <= 1.1 and video_min < 0.0:
        return videos
    if -1e-6 <= video_min and video_max <= 1.0 + 1e-6:
        return videos * 2.0 - 1.0
    if 0.0 <= video_min and video_max <= 255.0 + 1e-6:
        return videos / 127.5 - 1.0
    raise ValueError(f"Unsupported observation range for LVP prior: min={video_min}, max={video_max}")


def cast_tensor_tree(value, *, device: torch.device, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, tuple):
        return tuple(cast_tensor_tree(v, device=device, dtype=dtype) for v in value)
    if isinstance(value, list):
        return [cast_tensor_tree(v, device=device, dtype=dtype) for v in value]
    return value


def load_prompt_embedding(path_value: str | None) -> tuple[torch.Tensor | None, int | None]:
    if not path_value:
        return None, None
    path = Path(path_value)
    if not path.is_absolute():
        path = (LVP_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt embedding file not found: {path}")
    prompt_embed = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(prompt_embed, torch.Tensor):
        raise TypeError(f"Prompt embedding at {path} must be a torch.Tensor")
    return prompt_embed, int(prompt_embed.shape[0])


def init_wandb_run(
    args,
    output_dir: Path,
    idm_cfg,
    lvp_cfg,
):
    if not args.wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in the current environment. "
            "Install it or rerun without --wandb."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or output_dir.name,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config={
            "args": args.to_flat_dict(),
            "idm_cfg": OmegaConf.to_container(idm_cfg, resolve=True),
            "lvp_cfg": OmegaConf.to_container(lvp_cfg, resolve=True),
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
    wandb.define_metric(
        "eval_table_right_target/*",
        step_metric="eval_table_right_target/step",
    )
    return run


def build_prompt_context(
    *,
    batch_size: int,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
):
    return [prompt_embed[:prompt_embed_len]] * batch_size


def make_iter(dataset: Any) -> Iterator[dict]:
    return dataset.repeat().as_numpy_iterator()


class EMAModel:
    """Exponential Moving Average shadow of a module's parameters.

    Maintains a separate copy of the model weights that are updated as:
        shadow = decay * shadow + (1 - decay) * online
    after each optimizer step.  The shadow model is used for inference only
    (e.g. as the re-encoder in cycle-consistency loss) and is never trained
    directly.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = decay
        # Deep-copy parameters to shadow buffers (detached, no grad).
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            shadow_param = param.data.clone()
            if device is not None:
                shadow_param = shadow_param.to(device)
            self.shadow[name] = shadow_param

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from the online model after an optimizer step."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data.to(self.shadow[name].device), 1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Overwrite *model* weights with shadow weights.

        Returns the original weights so they can be restored via
        :meth:`restore`.
        """
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore *model* weights from a backup produced by :meth:`apply_shadow`."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, value in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(value)


def save_checkpoint(
    output_dir: Path,
    step: int,
    idm: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    lvp=None,
    lvp_trainable_modules: set[str] | None = None,
    extra_state: dict[str, Any] | None = None,
    lr_scheduler=None,
    ema: EMAModel | None = None,
    write_step_checkpoint: bool = True,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
    latest_path = ckpt_dir / "latest.pt"
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
    }
    if idm is not None:
        state["idm"] = unwrap_idm_module(idm).state_dict()
    lvp_trainable_modules = set(lvp_trainable_modules or [])
    if lvp is not None and lvp_trainable_modules:
        state["lvp_trainable_modules"] = sorted(lvp_trainable_modules)
        state["lvp"] = {
            name: unwrap_module(module).state_dict()
            for name, module in get_lvp_module_map(lvp).items()
            if name in lvp_trainable_modules and module is not None
        }
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    if extra_state:
        state.update(extra_state)
    if write_step_checkpoint:
        torch.save(state, ckpt_path)
    torch.save(state, latest_path)


_log = logging.getLogger(__name__)


def _resolve_checkpoint_path(path_or_dir: str | Path) -> Path:
    """Accept either a .pt file or a directory containing ``checkpoints/latest.pt``."""
    p = Path(path_or_dir)
    if p.is_file():
        return p
    candidate = p / "checkpoints" / "latest.pt"
    if candidate.is_file():
        return candidate
    candidate = p / "latest.pt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot find a checkpoint at '{path_or_dir}'. "
        "Pass a .pt file or a directory containing checkpoints/latest.pt."
    )


def _load_state_dict_partial(
    module: nn.Module,
    saved_state: dict[str, Any],
    label: str,
) -> tuple[list[str], list[str]]:
    """Load *matching* keys into *module*, skip mismatched ones.

    Returns (loaded_keys, skipped_keys) so callers can log what happened.
    """
    current_state = module.state_dict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    filtered: dict[str, Any] = {}

    for key, value in saved_state.items():
        if key not in current_state:
            skipped_keys.append(key)
            continue
        if isinstance(value, torch.Tensor) and value.shape != current_state[key].shape:
            skipped_keys.append(key)
            continue
        filtered[key] = value
        loaded_keys.append(key)

    missing_in_ckpt = [k for k in current_state if k not in saved_state]
    if missing_in_ckpt:
        skipped_keys.extend(f"(new) {k}" for k in missing_in_ckpt)

    module.load_state_dict(filtered, strict=False)

    if skipped_keys:
        _log.warning(
            "[resume] %s: loaded %d/%d keys, skipped %d: %s",
            label,
            len(loaded_keys),
            len(current_state),
            len(skipped_keys),
            skipped_keys[:20],
        )
    else:
        _log.info("[resume] %s: loaded all %d keys", label, len(loaded_keys))
    return loaded_keys, skipped_keys


@dataclass
class ResumeState:
    """Information recovered from a checkpoint."""

    start_step: int
    optimizer_restored: bool
    idm_loaded_keys: list[str]
    idm_skipped_keys: list[str]
    gt_action_head_loaded_keys: list[str]
    gt_action_head_skipped_keys: list[str]
    grl_classifier_loaded_keys: list[str]
    grl_classifier_skipped_keys: list[str]
    lvp_loaded_modules: list[str]
    lvp_skipped_modules: list[str]
    extra_state: dict[str, Any]


def load_checkpoint(
    path_or_dir: str | Path,
    *,
    idm: nn.Module | None = None,
    gt_action_head: nn.Module | None = None,
    grl_classifier: nn.Module | None = None,
    lvp=None,
    lvp_trainable_modules: set[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    ema: EMAModel | None = None,
    map_location: str | torch.device = "cpu",
) -> ResumeState:
    """Load a checkpoint with flexible partial matching.

    - Model weights are loaded with ``strict=False``: keys that exist in both
      the checkpoint and the current model are restored; extra or missing keys
      are skipped and logged.  This lets you resume after adding a new head.
    - Optimizer / LR-scheduler states are restored only when their structure
      matches (same number of param groups).  On mismatch they are silently
      skipped so the optimizer re-initializes fresh — the right default for
      fine-tuning with architecture changes.
    """
    ckpt_path = _resolve_checkpoint_path(path_or_dir)
    _log.info("[resume] loading checkpoint from %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    start_step = int(state.get("step", 0))

    # ---- IDM ----
    idm_loaded: list[str] = []
    idm_skipped: list[str] = []
    if idm is not None and "idm" in state:
        idm_module = unwrap_idm_module(idm)
        idm_loaded, idm_skipped = _load_state_dict_partial(
            idm_module, state["idm"], label="idm"
        )

    # ---- GT-action head ----
    gt_head_loaded: list[str] = []
    gt_head_skipped: list[str] = []
    if gt_action_head is not None and "gt_action_head" in state:
        gt_head_module = unwrap_module(gt_action_head)
        gt_head_loaded, gt_head_skipped = _load_state_dict_partial(
            gt_head_module, state["gt_action_head"], label="gt_action_head"
        )

    # ---- GRL classifier ----
    grl_loaded: list[str] = []
    grl_skipped: list[str] = []
    if grl_classifier is not None and "grl_classifier" in state:
        grl_module = unwrap_module(grl_classifier)
        grl_loaded, grl_skipped = _load_state_dict_partial(
            grl_module, state["grl_classifier"], label="grl_classifier"
        )

    # ---- LVP trainable modules ----
    lvp_trainable_modules = set(lvp_trainable_modules or [])
    lvp_loaded_mods: list[str] = []
    lvp_skipped_mods: list[str] = []
    if lvp is not None and "lvp" in state:
        for mod_name, mod_state in state["lvp"].items():
            module = getattr(lvp, mod_name, None)
            if module is None:
                lvp_skipped_mods.append(mod_name)
                continue
            mod_module = unwrap_module(module)
            loaded, skipped = _load_state_dict_partial(
                mod_module, mod_state, label=f"lvp.{mod_name}"
            )
            if loaded:
                lvp_loaded_mods.append(mod_name)
            if skipped:
                lvp_skipped_mods.append(mod_name)

    # ---- Optimizer ----
    optimizer_restored = False
    if optimizer is not None and "optimizer" in state:
        saved_opt = state["optimizer"]
        try:
            if len(saved_opt["param_groups"]) == len(optimizer.param_groups):
                optimizer.load_state_dict(saved_opt)
                optimizer_restored = True
                _log.info("[resume] optimizer state restored")
            else:
                _log.warning(
                    "[resume] optimizer param_groups mismatch "
                    "(ckpt=%d, current=%d) — starting fresh optimizer",
                    len(saved_opt["param_groups"]),
                    len(optimizer.param_groups),
                )
        except Exception as exc:
            _log.warning("[resume] failed to restore optimizer: %s", exc)

    # ---- LR scheduler ----
    if lr_scheduler is not None and "lr_scheduler" in state and optimizer_restored:
        try:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            _log.info("[resume] lr_scheduler state restored")
        except Exception as exc:
            _log.warning("[resume] failed to restore lr_scheduler: %s", exc)

    # ---- EMA ----
    if ema is not None and "ema" in state:
        try:
            ema.load_state_dict(state["ema"])
            _log.info("[resume] EMA shadow weights restored")
        except Exception as exc:
            _log.warning("[resume] failed to restore EMA: %s", exc)

    # ---- Extra state ----
    known_top_keys = {
        "step", "idm", "optimizer", "lr_scheduler",
        "lvp", "lvp_trainable_modules", "gt_action_head", "grl_classifier", "ema",
    }
    extra_state = {k: v for k, v in state.items() if k not in known_top_keys}

    _log.info(
        "[resume] checkpoint loaded: start_step=%d, optimizer_restored=%s",
        start_step,
        optimizer_restored,
    )
    return ResumeState(
        start_step=start_step,
        optimizer_restored=optimizer_restored,
        idm_loaded_keys=idm_loaded,
        idm_skipped_keys=idm_skipped,
        gt_action_head_loaded_keys=gt_head_loaded,
        gt_action_head_skipped_keys=gt_head_skipped,
        grl_classifier_loaded_keys=grl_loaded,
        grl_classifier_skipped_keys=grl_skipped,
        lvp_loaded_modules=lvp_loaded_mods,
        lvp_skipped_modules=lvp_skipped_mods,
        extra_state=extra_state,
    )


__all__ = [
    "DistributedContext",
    "EMAModel",
    "LatentActionIDMWrapper",
    "ResumeState",
    "build_joint_optimizer",
    "build_lvp_prior",
    "build_lr_scheduler",
    "build_prompt_context",
    "cast_tensor_tree",
    "cleanup_distributed",
    "debug_log_once",
    "distributed_barrier",
    "freeze_module",
    "get_current_optimizer_lrs",
    "get_lvp_module_map",
    "get_lvp_optimizer_hparams",
    "init_wandb_run",
    "iter_unique_trainable_params",
    "load_checkpoint",
    "load_prompt_embedding",
    "make_iter",
    "maybe_ddp_no_sync",
    "maybe_wrap_ddp",
    "reduce_pair_mean",
    "reduce_scalar",
    "resolve_lvp_trainable_modules",
    "save_checkpoint",
    "select_rgb_channels",
    "set_lvp_mode",
    "set_seed",
    "setup_distributed",
    "take_batch_prefix",
    "temporarily_disable_grads",
    "temporarily_frozen_eval",
    "temporarily_unwrap_lvp_ddp",
    "to_lvp_range",
    "trim_batch",
    "unfreeze_module",
    "unwrap_idm_module",
    "unwrap_module",
]
