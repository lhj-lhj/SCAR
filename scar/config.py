from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf, open_dict

import yaml

from .environment import (
    CONFIG_ROOT,
    DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED,
    DEFAULT_LIBERO_PROMPT,
    DEFAULT_LIBERO_PROMPT_EMBED,
    LVP_ROOT,
    PACKAGE_ROOT,
    SCAR_ROOT,
)


PRESET_ROOT = CONFIG_ROOT / "presets"
DEFAULT_VIDEO_METRIC_NAMES = "mse,psnr,ssim,lpips"
DEFAULT_VIDEO_METRIC_BATCH_SIZE = 16
DEFAULT_VIDEO_METRIC_MAX_VIDEO_COUNT = 16


@dataclass(frozen=True)
class DataConfig:
    dataset_paths: tuple[str, ...]
    eval_dataset_paths: tuple[str, ...]
    dataset_episode_subsets: dict[str, tuple[int, ...]]
    eval_dataset_episode_subsets: dict[str, tuple[int, ...]]
    right_target_eval_dataset_paths: tuple[str, ...]
    right_target_eval_dataset_episode_subsets: dict[str, tuple[int, ...]]
    cross_cycle_datasets: tuple[str, ...]
    seq_len: int
    batch_size: int
    eval_num_trajs: int
    shift: int
    action_dim: int
    num_workers: int


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    accumulate_grad_batches: int
    seed: int
    log_every: int
    save_checkpoint: bool
    save_every: int
    eval_every: int
    output_dir: str


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    lvp_lr: float
    weight_decay: float
    clip_grad_norm: float


@dataclass(frozen=True)
class ModelConfig:
    la_dim: int
    lvp_action_source: str
    idm_input_source: str
    freeze_idm: bool
    joint_tune_lvp: bool
    lvp_train_modules: tuple[str, ...]
    lvp_gradient_checkpointing_rate: float
    gt_action_head_type: str
    gt_action_loss_weight: float
    use_reparameterized_la: bool
    latent_action_kl_weight: float


@dataclass(frozen=True)
class CycleConfig:
    enabled: bool
    weight: float
    warmup_steps: int
    end_steps: int
    burst_every: int
    burst_steps: int


@dataclass(frozen=True)
class CrossCycleConfig:
    train_enabled: bool
    eval_enabled: bool
    weight: float
    warmup_steps: int


@dataclass(frozen=True)
class GRLConfig:
    enabled: bool
    weight: float
    alpha: float
    warmup_steps: int
    hidden_dim: int


@dataclass(frozen=True)
class WrongZConfig:
    enabled: bool
    weight: float
    warmup_steps: int
    sigma_hi: float


@dataclass(frozen=True)
class GTActionProbeConfig:
    enabled: bool
    subset_manifest: str | None
    train_steps: int
    lr: float
    weight_decay: float
    batch_size: int
    num_workers: int
    seed: int
    dim_model: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    dropout: float


@dataclass(frozen=True)
class EvalConfig:
    generate_video: bool
    video_count: int
    video_fps: int


@dataclass(frozen=True)
class VideoMetricsConfig:
    names: tuple[str, ...]
    batch_size: int
    max_video_count: int


@dataclass(frozen=True)
class MetricsConfig:
    video: VideoMetricsConfig


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    name: str | None
    mode: str


@dataclass(frozen=True)
class BridgeConfig:
    config_path: str
    raw_config: dict[str, object]
    idm_cfg: dict[str, object]
    lvp_preset: str
    lvp_patch: dict[str, object]
    data: DataConfig
    train: TrainConfig
    optimizer: OptimizerConfig
    model: ModelConfig
    ema_decay: float
    cycle: CycleConfig
    cross_cycle: CrossCycleConfig
    grl: GRLConfig
    wrong_z: WrongZConfig
    gt_action_probe: GTActionProbeConfig
    eval: EvalConfig
    metrics: MetricsConfig
    wandb_cfg: WandbConfig
    resume_from: str | None
    prompt: str
    prompt_embed_path: str
    negative_prompt_embed_path: str
    device: str

    def __getattr__(self, name: str):
        getter = _BRIDGE_CONFIG_FLAT_GETTERS.get(name)
        if getter is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getter(self)

    def to_flat_dict(self) -> dict[str, object]:
        return {
            key: _to_serializable(getter(self))
            for key, getter in _BRIDGE_CONFIG_FLAT_GETTERS.items()
        }


def _to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    return value


_BRIDGE_CONFIG_FLAT_GETTERS: dict[str, Callable[[BridgeConfig], object]] = {
    "idm_cfg": lambda cfg: copy.deepcopy(cfg.idm_cfg),
    "lvp_preset": lambda cfg: cfg.lvp_preset,
    "lvp_patch": lambda cfg: copy.deepcopy(cfg.lvp_patch),
    "dataset_paths": lambda cfg: list(cfg.data.dataset_paths),
    "eval_dataset_paths": lambda cfg: list(cfg.data.eval_dataset_paths),
    "dataset_episode_subsets": lambda cfg: {
        path: list(indices) for path, indices in cfg.data.dataset_episode_subsets.items()
    },
    "eval_dataset_episode_subsets": lambda cfg: {
        path: list(indices)
        for path, indices in cfg.data.eval_dataset_episode_subsets.items()
    },
    "right_target_eval_dataset_paths": lambda cfg: list(cfg.data.right_target_eval_dataset_paths),
    "right_target_eval_dataset_episode_subsets": lambda cfg: {
        path: list(indices)
        for path, indices in cfg.data.right_target_eval_dataset_episode_subsets.items()
    },
    "cross_cycle_dataset": lambda cfg: ",".join(cfg.data.cross_cycle_datasets),
    "seq_len": lambda cfg: cfg.data.seq_len,
    "batch_size": lambda cfg: cfg.data.batch_size,
    "eval_num_trajs": lambda cfg: cfg.data.eval_num_trajs,
    "shift": lambda cfg: cfg.data.shift,
    "action_dim": lambda cfg: cfg.data.action_dim,
    "num_workers": lambda cfg: cfg.data.num_workers,
    "steps": lambda cfg: cfg.train.steps,
    "accumulate_grad_batches": lambda cfg: cfg.train.accumulate_grad_batches,
    "seed": lambda cfg: cfg.train.seed,
    "log_every": lambda cfg: cfg.train.log_every,
    "save_checkpoint": lambda cfg: cfg.train.save_checkpoint,
    "save_every": lambda cfg: cfg.train.save_every,
    "eval_every": lambda cfg: cfg.train.eval_every,
    "output_dir": lambda cfg: cfg.train.output_dir,
    "lr": lambda cfg: cfg.optimizer.lr,
    "lvp_lr": lambda cfg: cfg.optimizer.lvp_lr,
    "weight_decay": lambda cfg: cfg.optimizer.weight_decay,
    "clip_grad_norm": lambda cfg: cfg.optimizer.clip_grad_norm,
    "la_dim": lambda cfg: cfg.model.la_dim,
    "lvp_action_source": lambda cfg: cfg.model.lvp_action_source,
    "idm_input_source": lambda cfg: cfg.model.idm_input_source,
    "freeze_idm": lambda cfg: cfg.model.freeze_idm,
    "joint_tune_lvp": lambda cfg: cfg.model.joint_tune_lvp,
    "lvp_train_module": lambda cfg: list(cfg.model.lvp_train_modules),
    "lvp_gradient_checkpointing_rate": lambda cfg: cfg.model.lvp_gradient_checkpointing_rate,
    "gt_action_head_type": lambda cfg: cfg.model.gt_action_head_type,
    "gt_action_loss_weight": lambda cfg: cfg.model.gt_action_loss_weight,
    "use_reparameterized_la": lambda cfg: cfg.model.use_reparameterized_la,
    "latent_action_kl_weight": lambda cfg: cfg.model.latent_action_kl_weight,
    "ema_decay": lambda cfg: cfg.ema_decay,
    "enable_cycle_loss": lambda cfg: cfg.cycle.enabled,
    "cycle_loss_weight": lambda cfg: cfg.cycle.weight,
    "cycle_warmup_steps": lambda cfg: cfg.cycle.warmup_steps,
    "cycle_end_steps": lambda cfg: cfg.cycle.end_steps,
    "cycle_burst_every": lambda cfg: cfg.cycle.burst_every,
    "cycle_burst_steps": lambda cfg: cfg.cycle.burst_steps,
    "cross_cycle_train_enabled": lambda cfg: cfg.cross_cycle.train_enabled,
    "cross_cycle_eval_enabled": lambda cfg: cfg.cross_cycle.eval_enabled,
    "cross_cycle_enabled": lambda cfg: cfg.cross_cycle.train_enabled,
    "cross_cycle_loss_weight": lambda cfg: cfg.cross_cycle.weight,
    "cross_cycle_warmup_steps": lambda cfg: cfg.cross_cycle.warmup_steps,
    "grl_enabled": lambda cfg: cfg.grl.enabled,
    "grl_weight": lambda cfg: cfg.grl.weight,
    "grl_alpha": lambda cfg: cfg.grl.alpha,
    "grl_warmup_steps": lambda cfg: cfg.grl.warmup_steps,
    "grl_hidden_dim": lambda cfg: cfg.grl.hidden_dim,
    "wrong_z_enabled": lambda cfg: cfg.wrong_z.enabled,
    "wrong_z_weight": lambda cfg: cfg.wrong_z.weight,
    "wrong_z_warmup_steps": lambda cfg: cfg.wrong_z.warmup_steps,
    "wrong_z_sigma_hi": lambda cfg: cfg.wrong_z.sigma_hi,
    "gt_action_probe_enabled": lambda cfg: cfg.gt_action_probe.enabled,
    "gt_action_probe_subset_manifest": lambda cfg: cfg.gt_action_probe.subset_manifest,
    "gt_action_probe_train_steps": lambda cfg: cfg.gt_action_probe.train_steps,
    "gt_action_probe_lr": lambda cfg: cfg.gt_action_probe.lr,
    "gt_action_probe_batch_size": lambda cfg: cfg.gt_action_probe.batch_size,
    "gt_action_probe_num_workers": lambda cfg: cfg.gt_action_probe.num_workers,
    "gt_action_probe_seed": lambda cfg: cfg.gt_action_probe.seed,
    "eval_generate_video": lambda cfg: cfg.eval.generate_video,
    "eval_video_count": lambda cfg: cfg.eval.video_count,
    "eval_video_fps": lambda cfg: cfg.eval.video_fps,
    "metric_video_names": lambda cfg: list(cfg.metrics.video.names),
    "metric_video_batch_size": lambda cfg: cfg.metrics.video.batch_size,
    "metric_video_count": lambda cfg: cfg.metrics.video.max_video_count,
    "wandb": lambda cfg: cfg.wandb_cfg.enabled,
    "wandb_project": lambda cfg: cfg.wandb_cfg.project,
    "wandb_entity": lambda cfg: cfg.wandb_cfg.entity,
    "wandb_name": lambda cfg: cfg.wandb_cfg.name,
    "wandb_mode": lambda cfg: cfg.wandb_cfg.mode,
    "resume_from": lambda cfg: cfg.resume_from,
    "prompt": lambda cfg: cfg.prompt,
    "prompt_embed_path": lambda cfg: cfg.prompt_embed_path,
    "negative_prompt_embed_path": lambda cfg: cfg.negative_prompt_embed_path,
    "device": lambda cfg: cfg.device,
}


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path: str | Path) -> dict:
    """Load YAML config with _base_ inheritance."""
    config_path = Path(config_path)
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle) or {}

    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = config_path.parent / base_name
        base_cfg = load_yaml_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _validate_idm_section(section_cfg: dict, errors: list[str]) -> None:
    if not isinstance(section_cfg, dict):
        errors.append("idm must be a mapping")
        return

    for legacy_key in ("preset", "patch", "config_name", "overrides"):
        if legacy_key in section_cfg:
            errors.append(
                f"idm.{legacy_key} is no longer supported; define the IDM structure inline under idm."
            )


def _validate_lvp_preset_section(section_cfg: dict, errors: list[str]) -> None:
    preset = section_cfg.get("preset")
    if preset is not None and not isinstance(preset, str):
        errors.append("lvp.preset must be a string")

    patch = section_cfg.get("patch")
    if patch is not None and not isinstance(patch, dict):
        errors.append("lvp.patch must be a dict")

    if "config_name" in section_cfg:
        errors.append(
            "lvp.config_name is no longer supported after Hydra removal; use lvp.preset instead"
        )

    overrides = section_cfg.get("overrides")
    if overrides not in (None, []):
        errors.append(
            "lvp.overrides is no longer supported after Hydra removal; use lvp.patch instead"
        )


def _validate_yaml_config(cfg: dict) -> None:
    """Validate critical fields in the parsed YAML config. Raises SystemExit on bad values."""
    errors: list[str] = []

    paths_cfg = cfg.get("paths", {})
    if paths_cfg is not None and not isinstance(paths_cfg, dict):
        errors.append("paths must be a mapping")
        paths_cfg = {}
    root_value = (paths_cfg or {}).get("root", None)
    if root_value is not None and not isinstance(root_value, str):
        errors.append("paths.root must be a string or null")

    d = cfg.get("data", {})
    dataset_paths = d.get("dataset_paths", [])
    if not isinstance(dataset_paths, list) or len(dataset_paths) == 0:
        errors.append("data.dataset_paths must be a non-empty list")
    eval_dataset_paths = d.get("eval_dataset_paths", [])
    if not isinstance(eval_dataset_paths, list):
        errors.append("data.eval_dataset_paths must be a list")
    right_target_eval_dataset_paths = d.get("right_target_eval_dataset_paths", [])
    if not isinstance(right_target_eval_dataset_paths, list):
        errors.append("data.right_target_eval_dataset_paths must be a list")
    dataset_episode_subsets = d.get("dataset_episode_subsets", {})
    if dataset_episode_subsets is not None and not isinstance(dataset_episode_subsets, dict):
        errors.append("data.dataset_episode_subsets must be a mapping from dataset path to index list")
        dataset_episode_subsets = {}
    eval_dataset_episode_subsets = d.get("eval_dataset_episode_subsets", {})
    if eval_dataset_episode_subsets is not None and not isinstance(eval_dataset_episode_subsets, dict):
        errors.append(
            "data.eval_dataset_episode_subsets must be a mapping from dataset path to index list"
        )
        eval_dataset_episode_subsets = {}
    right_target_eval_dataset_episode_subsets = d.get(
        "right_target_eval_dataset_episode_subsets",
        {},
    )
    if (
        right_target_eval_dataset_episode_subsets is not None
        and not isinstance(right_target_eval_dataset_episode_subsets, dict)
    ):
        errors.append(
            "data.right_target_eval_dataset_episode_subsets must be a mapping from dataset path to index list"
        )
        right_target_eval_dataset_episode_subsets = {}
    cross_cycle_datasets = d.get("cross_cycle_datasets", [])
    if not isinstance(cross_cycle_datasets, list):
        errors.append("data.cross_cycle_datasets must be a list")
    for field_name, subset_mapping in (
        ("data.dataset_episode_subsets", dataset_episode_subsets),
        ("data.eval_dataset_episode_subsets", eval_dataset_episode_subsets),
        (
            "data.right_target_eval_dataset_episode_subsets",
            right_target_eval_dataset_episode_subsets,
        ),
    ):
        for dataset_path, indices in (subset_mapping or {}).items():
            if not isinstance(dataset_path, str) or not dataset_path:
                errors.append(f"{field_name} keys must be non-empty dataset path strings")
                continue
            if not isinstance(indices, list):
                errors.append(f"{field_name}[{dataset_path!r}] must be a list of episode indices")
                continue
            for raw_index in indices:
                if not isinstance(raw_index, int):
                    errors.append(
                        f"{field_name}[{dataset_path!r}] must contain only integer episode indices"
                    )
                    break
    if d.get("seq_len", 49) <= 0:
        errors.append("data.seq_len must be > 0")
    if d.get("batch_size", 2) <= 0:
        errors.append("data.batch_size must be > 0")

    t = cfg.get("train", {})
    if t.get("steps", 20000) <= 0:
        errors.append("train.steps must be > 0")
    if bool(t.get("save_checkpoint", True)) and t.get("save_every", 1000) <= 0:
        errors.append("train.save_every must be > 0 when train.save_checkpoint is true")
    if "eval_batches" in t:
        errors.append(
            "train.eval_batches has been removed; periodic eval now exhausts the full explicit eval loader on rank 0"
        )

    m = cfg.get("model", {})
    idm_input_source = str(m.get("idm_input_source", "vae_latent"))
    if idm_input_source not in {"vae_latent", "rgb_patch"}:
        errors.append(
            "model.idm_input_source must be one of ['vae_latent', 'rgb_patch']"
        )
    if m.get("latent_action_kl_weight", 0.0) < 0:
        errors.append("model.latent_action_kl_weight must be >= 0")

    cycle_cfg = cfg.get("cycle", {})
    if int(cycle_cfg.get("end_steps", 0)) < 0:
        errors.append("cycle.end_steps must be >= 0")
    if int(cycle_cfg.get("burst_every", 0)) < 0:
        errors.append("cycle.burst_every must be >= 0")
    if int(cycle_cfg.get("burst_steps", 0)) < 0:
        errors.append("cycle.burst_steps must be >= 0")

    e = cfg.get("eval", {})
    if e.get("video_count", 1) < 0:
        errors.append("eval.video_count must be >= 0")
    if "video_metrics" in e:
        errors.append("eval.video_metrics is no longer supported; use metrics.video.names")
    if "video_metrics_batch_size" in e:
        errors.append(
            "eval.video_metrics_batch_size is no longer supported; use metrics.video.batch_size"
        )

    metrics_cfg = cfg.get("metrics", {})
    if metrics_cfg is not None and not isinstance(metrics_cfg, dict):
        errors.append("metrics must be a mapping")
        metrics_cfg = {}
    video_metrics_cfg = metrics_cfg.get("video", {})
    if video_metrics_cfg is not None and not isinstance(video_metrics_cfg, dict):
        errors.append("metrics.video must be a mapping")
        video_metrics_cfg = {}
    video_metric_names = video_metrics_cfg.get("names", DEFAULT_VIDEO_METRIC_NAMES)
    if video_metric_names is not None and not isinstance(video_metric_names, (list, str)):
        errors.append("metrics.video.names must be a list or comma-separated string")
    if video_metrics_cfg.get("batch_size", DEFAULT_VIDEO_METRIC_BATCH_SIZE) <= 0:
        errors.append("metrics.video.batch_size must be > 0")
    if video_metrics_cfg.get("max_video_count", DEFAULT_VIDEO_METRIC_MAX_VIDEO_COUNT) < 0:
        errors.append("metrics.video.max_video_count must be >= 0")

    wz = cfg.get("wrong_z", {})
    if wz.get("weight", 1.0) < 0:
        errors.append("wrong_z.weight must be >= 0")
    if wz.get("warmup_steps", 0) < 0:
        errors.append("wrong_z.warmup_steps must be >= 0")
    sigma_hi = float(wz.get("sigma_hi", 0.6))
    if sigma_hi < 0.0 or sigma_hi > 1.0:
        errors.append("wrong_z.sigma_hi must be in [0, 1]")

    probe_cfg = cfg.get("gt_action_probe", {})
    if probe_cfg is not None and not isinstance(probe_cfg, dict):
        errors.append("gt_action_probe must be a mapping")
        probe_cfg = {}
    if bool(probe_cfg.get("enabled", False)) and not probe_cfg.get("subset_manifest"):
        errors.append(
            "gt_action_probe.subset_manifest must be set when gt_action_probe.enabled is true"
        )
    if int(probe_cfg.get("train_steps", 0)) < 0:
        errors.append("gt_action_probe.train_steps must be >= 0")
    if float(probe_cfg.get("lr", 1e-4)) <= 0:
        errors.append("gt_action_probe.lr must be > 0")
    if float(probe_cfg.get("weight_decay", 1e-4)) < 0:
        errors.append("gt_action_probe.weight_decay must be >= 0")
    if int(probe_cfg.get("batch_size", 16)) <= 0:
        errors.append("gt_action_probe.batch_size must be > 0")
    if int(probe_cfg.get("num_workers", 0)) < 0:
        errors.append("gt_action_probe.num_workers must be >= 0")
    if int(probe_cfg.get("dim_model", 256)) <= 0:
        errors.append("gt_action_probe.dim_model must be > 0")
    if int(probe_cfg.get("n_heads", 4)) <= 0:
        errors.append("gt_action_probe.n_heads must be > 0")
    if int(probe_cfg.get("dim_model", 256)) % int(probe_cfg.get("n_heads", 4) or 1) != 0:
        errors.append("gt_action_probe.dim_model must be divisible by gt_action_probe.n_heads")
    if int(probe_cfg.get("n_layers", 2)) <= 0:
        errors.append("gt_action_probe.n_layers must be > 0")
    if int(probe_cfg.get("dim_feedforward", 1024)) <= 0:
        errors.append("gt_action_probe.dim_feedforward must be > 0")
    if float(probe_cfg.get("dropout", 0.1)) < 0:
        errors.append("gt_action_probe.dropout must be >= 0")

    _validate_idm_section(cfg.get("idm", {}), errors)
    _validate_lvp_preset_section(cfg.get("lvp", {}), errors)

    if errors:
        raise SystemExit("Config validation errors:\n  - " + "\n  - ".join(errors))

def _build_bridge_config(
    cfg: dict,
    *,
    config_path: str | Path,
) -> BridgeConfig:
    config_path = str(config_path)
    config_file_path = Path(config_path).resolve()
    config_dir = config_file_path.parent

    paths_cfg = cfg.get("paths", {})
    d = cfg.get("data", {})
    t = cfg.get("train", {})
    o = cfg.get("optimizer", {})
    m = cfg.get("model", {})
    c = cfg.get("cycle", {})
    cc = cfg.get("cross_cycle", {})
    g = cfg.get("grl", {})
    wz = cfg.get("wrong_z", {})
    probe_cfg = cfg.get("gt_action_probe", {})
    e = cfg.get("eval", {})
    metrics_cfg = cfg.get("metrics", {})
    video_metrics_cfg = metrics_cfg.get("video", {})
    w = cfg.get("wandb", {})
    lv = cfg.get("lvp", {})

    raw_video_metric_names = video_metrics_cfg.get("names", DEFAULT_VIDEO_METRIC_NAMES)
    if isinstance(raw_video_metric_names, str) or raw_video_metric_names is None:
        metric_video_names = tuple(parse_comma_separated_list(raw_video_metric_names))
    else:
        metric_video_names = tuple(
            str(name).strip()
            for name in raw_video_metric_names
            if str(name).strip()
        )

    path_root = resolve_paths_root((paths_cfg or {}).get("root", None), config_dir=config_dir)

    output_dir = t.get("output_dir", "")
    if not output_dir:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(config_path).stem
        output_dir = str(
            path_root
            / "outputs"
            / "scar"
            / f"{config_name}_{timestamp}"
        )
    else:
        output_dir = str(
            resolve_runtime_path(
                str(output_dir),
                path_root=path_root,
                config_dir=config_dir,
            )
        )

    wandb_name = w.get("name", "") or None
    if not wandb_name:
        wandb_name = Path(output_dir).name

    probe_subset_manifest = probe_cfg.get("subset_manifest", None)
    if probe_subset_manifest:
        probe_subset_manifest = resolve_runtime_path(
            str(probe_subset_manifest),
            path_root=path_root,
            config_dir=config_dir,
        )

    def _resolve_episode_subset_mapping(
        raw_mapping: dict[str, object] | None,
    ) -> dict[str, tuple[int, ...]]:
        resolved: dict[str, tuple[int, ...]] = {}
        for dataset_path, raw_indices in (raw_mapping or {}).items():
            resolved_dataset_path = resolve_runtime_path(
                str(dataset_path),
                path_root=path_root,
                config_dir=config_dir,
            )
            resolved[str(Path(resolved_dataset_path).resolve())] = tuple(
                int(index) for index in list(raw_indices or [])
            )
        return resolved

    def _resolve_path_seq(raw_paths: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        return tuple(
            str(
                resolve_runtime_path(
                    str(path_value),
                    path_root=path_root,
                    config_dir=config_dir,
                )
            )
            for path_value in (raw_paths or [])
        )

    return BridgeConfig(
        config_path=config_path,
        raw_config=copy.deepcopy(cfg),
        idm_cfg=copy.deepcopy(cfg.get("idm", {})),
        lvp_preset=lv.get("preset") or "config_wan_v2a2v",
        lvp_patch=dict(lv.get("patch", {})),
        data=DataConfig(
            dataset_paths=_resolve_path_seq(d.get("dataset_paths", [])),
            eval_dataset_paths=_resolve_path_seq(d.get("eval_dataset_paths", [])),
            dataset_episode_subsets=_resolve_episode_subset_mapping(
                d.get("dataset_episode_subsets", {})
            ),
            eval_dataset_episode_subsets=_resolve_episode_subset_mapping(
                d.get("eval_dataset_episode_subsets", {})
            ),
            right_target_eval_dataset_paths=_resolve_path_seq(
                d.get("right_target_eval_dataset_paths", [])
            ),
            right_target_eval_dataset_episode_subsets=_resolve_episode_subset_mapping(
                d.get("right_target_eval_dataset_episode_subsets", {})
            ),
            cross_cycle_datasets=_resolve_path_seq(d.get("cross_cycle_datasets", [])),
            seq_len=int(d.get("seq_len", 49)),
            batch_size=int(d.get("batch_size", 2)),
            eval_num_trajs=int(d.get("eval_num_trajs", 16)),
            shift=int(d.get("shift", 1)),
            action_dim=int(d.get("action_dim", 16)),
            num_workers=int(d.get("num_workers", 2)),
        ),
        train=TrainConfig(
            steps=int(t.get("steps", 20000)),
            accumulate_grad_batches=int(t.get("accumulate_grad_batches", 2)),
            seed=int(t.get("seed", 0)),
            log_every=int(t.get("log_every", 10)),
            save_checkpoint=bool(t.get("save_checkpoint", True)),
            save_every=int(t.get("save_every", 1000)),
            eval_every=int(t.get("eval_every", 200)),
            output_dir=str(output_dir),
        ),
        optimizer=OptimizerConfig(
            lr=float(o.get("lr", 5e-5)),
            lvp_lr=float(o.get("lvp_lr", 5e-6)),
            weight_decay=float(o.get("weight_decay", 1e-4)),
            clip_grad_norm=float(o.get("clip_grad_norm", 1.0)),
        ),
        model=ModelConfig(
            la_dim=int(m.get("la_dim", 8)),
            lvp_action_source=str(m.get("lvp_action_source", "idm")),
            idm_input_source=str(m.get("idm_input_source", "vae_latent")),
            freeze_idm=bool(m.get("freeze_idm", False)),
            joint_tune_lvp=bool(m.get("joint_tune_lvp", True)),
            lvp_train_modules=tuple(m.get("lvp_train_modules", [])),
            lvp_gradient_checkpointing_rate=float(
                m.get("lvp_gradient_checkpointing_rate", 1.0)
            ),
            gt_action_head_type=str(m.get("gt_action_head_type", "linear")),
            gt_action_loss_weight=float(m.get("gt_action_loss_weight", 0.0)),
            use_reparameterized_la=bool(m.get("use_reparameterized_la", False)),
            latent_action_kl_weight=float(m.get("latent_action_kl_weight", 0.0)),
        ),
        ema_decay=float(cfg.get("ema", {}).get("decay", 0.0)),
        cycle=CycleConfig(
            enabled=bool(c.get("enabled", True)),
            weight=float(c.get("weight", 0.01)),
            warmup_steps=int(c.get("warmup_steps", 200)),
            end_steps=int(c.get("end_steps", 0)),
            burst_every=int(c.get("burst_every", 0)),
            burst_steps=int(c.get("burst_steps", 0)),
        ),
        cross_cycle=CrossCycleConfig(
            train_enabled=bool(cc.get("train_enabled", True)),
            eval_enabled=bool(cc.get("eval_enabled", True)),
            weight=float(cc.get("weight", 0.01)),
            warmup_steps=int(cc.get("warmup_steps", 200)),
        ),
        grl=GRLConfig(
            enabled=bool(g.get("enabled", False)),
            weight=float(g.get("weight", 0.0)),
            alpha=float(g.get("alpha", 1.0)),
            warmup_steps=int(g.get("warmup_steps", 200)),
            hidden_dim=int(g.get("hidden_dim", 256)),
        ),
        wrong_z=WrongZConfig(
            enabled=bool(wz.get("enabled", False)),
            weight=float(wz.get("weight", 1.0)),
            warmup_steps=int(wz.get("warmup_steps", 0)),
            sigma_hi=float(wz.get("sigma_hi", 0.6)),
        ),
        gt_action_probe=GTActionProbeConfig(
            enabled=bool(probe_cfg.get("enabled", False)),
            subset_manifest=probe_subset_manifest,
            train_steps=int(probe_cfg.get("train_steps", 0)),
            lr=float(probe_cfg.get("lr", 1e-4)),
            weight_decay=float(probe_cfg.get("weight_decay", 1e-4)),
            batch_size=int(probe_cfg.get("batch_size", 16)),
            num_workers=int(probe_cfg.get("num_workers", 0)),
            seed=int(probe_cfg.get("seed", 0)),
            dim_model=int(probe_cfg.get("dim_model", 256)),
            n_heads=int(probe_cfg.get("n_heads", 4)),
            n_layers=int(probe_cfg.get("n_layers", 2)),
            dim_feedforward=int(probe_cfg.get("dim_feedforward", 1024)),
            dropout=float(probe_cfg.get("dropout", 0.1)),
        ),
        eval=EvalConfig(
            generate_video=bool(e.get("generate_video", True)),
            video_count=int(e.get("video_count", 1)),
            video_fps=int(e.get("video_fps", 20)),
        ),
        metrics=MetricsConfig(
            video=VideoMetricsConfig(
                names=metric_video_names,
                batch_size=int(
                    video_metrics_cfg.get("batch_size", DEFAULT_VIDEO_METRIC_BATCH_SIZE)
                ),
                max_video_count=int(
                    video_metrics_cfg.get(
                        "max_video_count",
                        DEFAULT_VIDEO_METRIC_MAX_VIDEO_COUNT,
                    )
                ),
            )
        ),
        wandb_cfg=WandbConfig(
            enabled=bool(w.get("enabled", True)),
            project=str(w.get("project", "scar-robotwin")),
            entity=w.get("entity", "") or None,
            name=wandb_name,
            mode=str(w.get("mode", "online")),
        ),
        resume_from=resolve_runtime_path(
            cfg.get("resume_from", None),
            path_root=path_root,
            config_dir=config_dir,
        ),
        prompt=DEFAULT_LIBERO_PROMPT,
        prompt_embed_path=str(DEFAULT_LIBERO_PROMPT_EMBED),
        negative_prompt_embed_path=str(DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def load_bridge_config(config_path: str | Path) -> BridgeConfig:
    cfg = load_yaml_config(config_path)
    _validate_yaml_config(cfg)
    return _build_bridge_config(cfg, config_path=config_path)


def parse_config_args() -> BridgeConfig:
    """Parse --config path and return the structured bridge config."""
    argv = sys.argv[1:]
    config_path = None
    unknown: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--config":
            if index + 1 < len(argv):
                config_path = argv[index + 1]
                index += 2
                continue
            raise SystemExit("--config requires a value")
        if argv[index].startswith("--config="):
            config_path = argv[index].split("=", 1)[1]
            index += 1
            continue
        unknown.append(argv[index])
        index += 1

    if unknown:
        raise SystemExit(
            f"Unknown CLI arguments: {' '.join(unknown)}\n"
            "All configuration must be set in the YAML file. Only --config <path> is accepted."
        )

    if config_path is None:
        config_path = str(CONFIG_ROOT / "default.yaml")

    return load_bridge_config(config_path)


def _load_local_preset(category: str, preset_name: str) -> dict:
    preset_path = PRESET_ROOT / category / f"{preset_name}.yaml"
    if not preset_path.is_file():
        raise FileNotFoundError(
            f"Local preset not found: {preset_path}. "
            f"Expected preset '{preset_name}' under {PRESET_ROOT / category}."
        )
    return load_yaml_config(preset_path)


def _build_local_cfg(category: str, preset_name: str, patch: dict | None):
    cfg_dict = _load_local_preset(category, preset_name)
    if patch:
        cfg_dict = _deep_merge(cfg_dict, patch)
    return OmegaConf.create(cfg_dict)


def resolve_path(path_value: str | None, repo_root: Path) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def resolve_paths_root(
    root_value: str | None,
    *,
    config_dir: Path,
) -> Path:
    if not root_value:
        return SCAR_ROOT.resolve()
    root_path = Path(root_value).expanduser()
    if not root_path.is_absolute():
        root_path = config_dir / root_path
    return root_path.resolve()


def resolve_runtime_path(
    path_value: str | None,
    *,
    path_root: Path,
    config_dir: Path,
) -> str | None:
    if not path_value:
        return path_value
    path_str = str(path_value)
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if path_str.startswith("./") or path_str.startswith("../"):
        path = config_dir / path
    else:
        path = path_root / path
    return str(path.resolve())


def build_idm_cfg(args: BridgeConfig):
    cfg = OmegaConf.create(copy.deepcopy(args.idm_cfg))
    with open_dict(cfg):
        if not hasattr(cfg, "data"):
            cfg.data = OmegaConf.create({})
        if not hasattr(cfg, "env"):
            cfg.env = OmegaConf.create({})
        if not hasattr(cfg, "model"):
            cfg.model = OmegaConf.create({})
        if not hasattr(cfg.model, "idm"):
            cfg.model.idm = OmegaConf.create({})
        cfg.data.seq_len = int(args.data.seq_len)
        cfg.data.batch_size = int(args.data.batch_size)
        cfg.data.eval_num_trajs = int(args.data.eval_num_trajs)
        cfg.data.shift = int(args.data.shift)
        cfg.env.action_dim = int(args.data.action_dim)
        cfg.model.la_dim = int(args.model.la_dim)
        cfg.model.idm_input_source = str(args.model.idm_input_source)
        cfg.model.idm.la_dim = int(args.model.la_dim)
        cfg.model.idm.input_source = str(args.model.idm_input_source)
        cfg.model.idm.use_reparameterized_la = bool(args.model.use_reparameterized_la)
        cfg.model.idm.latent_action_kl_weight = float(args.model.latent_action_kl_weight)
    return cfg


def build_lvp_cfg(
    args: BridgeConfig,
    height: int,
    width: int,
):
    cfg = _build_local_cfg("lvp", args.lvp_preset, args.lvp_patch)

    with open_dict(cfg):
        if not hasattr(cfg, "load"):
            cfg.load = None
        if not hasattr(cfg, "dataset"):
            cfg.dataset = OmegaConf.create({})

        cfg.load = None
        cfg.dataset.load_prompt_embed = True
        cfg.dataset.load_video_latent = False
        cfg.dataset.height = int(height)
        cfg.dataset.width = int(width)

        cfg.algorithm.load_prompt_embed = True
        cfg.algorithm.load_video_latent = False
        cfg.algorithm.height = int(height)
        cfg.algorithm.width = int(width)
        cfg.algorithm.action_dim = int(args.data.action_dim)
        cfg.algorithm.gradient_checkpointing_rate = float(
            args.model.lvp_gradient_checkpointing_rate
        )
        # The SCAR bridge computes eval metrics itself, so keep WAN internal video metrics disabled
        # to avoid pulling in heavyweight metric backbones during model construction.
        cfg.algorithm.logging.metrics = []
        cfg.algorithm.model.compile = False
        cfg.algorithm.vae.compile = False
        cfg.algorithm.text_encoder.compile = False
        if "clip" in cfg.algorithm:
            cfg.algorithm.clip.compile = False

        if not hasattr(cfg.algorithm, "max_frames") or cfg.algorithm.max_frames is None:
            cfg.algorithm.max_frames = cfg.algorithm.n_frames
        if not hasattr(cfg.algorithm.model, "tuned_ckpt_path"):
            cfg.algorithm.model.tuned_ckpt_path = None

        cfg.algorithm.model.ckpt_path = resolve_path(cfg.algorithm.model.ckpt_path, LVP_ROOT)
        cfg.algorithm.model.tuned_ckpt_path = resolve_path(
            cfg.algorithm.model.tuned_ckpt_path, LVP_ROOT
        )
        cfg.algorithm.vae.ckpt_path = resolve_path(cfg.algorithm.vae.ckpt_path, LVP_ROOT)
        cfg.algorithm.text_encoder.ckpt_path = resolve_path(
            cfg.algorithm.text_encoder.ckpt_path, LVP_ROOT
        )
        if "clip" in cfg.algorithm:
            cfg.algorithm.clip.ckpt_path = resolve_path(
                cfg.algorithm.clip.ckpt_path, LVP_ROOT
            )

    return cfg


def get_lvp_target_seq_len(lvp_cfg) -> int:
    return int(getattr(lvp_cfg.algorithm, "max_frames", lvp_cfg.algorithm.n_frames))


def infer_idm_image_hw(cfg) -> tuple[int, int]:
    image_shape = getattr(cfg.env, "image_shape", None)
    if image_shape is None or len(image_shape) != 2:
        raise ValueError(f"Expected env.image_shape=[H,W], got {image_shape}")
    return int(image_shape[0]), int(image_shape[1])


def align_idm_seq_len(idm_cfg, seq_len: int) -> None:
    with open_dict(idm_cfg):
        idm_cfg.data.seq_len = int(seq_len)


def align_lvp_action_dim(lvp_cfg, action_dim: int) -> None:
    with open_dict(lvp_cfg):
        lvp_cfg.algorithm.action_dim = int(action_dim)


def parse_comma_separated_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


__all__ = [
    "BridgeConfig",
    "DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED",
    "DEFAULT_LIBERO_PROMPT",
    "DEFAULT_LIBERO_PROMPT_EMBED",
    "CONFIG_ROOT",
    "LVP_ROOT",
    "SCAR_ROOT",
    "align_lvp_action_dim",
    "align_idm_seq_len",
    "build_lvp_cfg",
    "build_idm_cfg",
    "get_lvp_target_seq_len",
    "infer_idm_image_hw",
    "load_yaml_config",
    "parse_comma_separated_list",
    "parse_config_args",
    "resolve_path",
]
