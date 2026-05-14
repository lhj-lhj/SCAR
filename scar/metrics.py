from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


METRIC_GROUP_DEFAULTS: dict[str, dict[str, float]] = {
    "objective": {
        "recon_loss": 0.0,
        "cycle_loss": 0.0,
        "cycle_idm_grad": 0.0,
        "cycle_loss_scale": 0.0,
        "gt_action_aux_mse": 0.0,
        "gt_action_loss_scale": 0.0,
        "latent_action_kl_loss": 0.0,
        "latent_action_kl_loss_scale": 0.0,
        "wrong_z_loss_scale": 0.0,
        "wrong_z_train_loss": 0.0,
        "wrong_z_active_loss": 0.0,
        "wrong_z_active_frac": 0.0,
        "wrong_z_same_emb_hit_rate": 0.0,
        "wrong_z_neighbor_similarity_mean": 0.0,
        "wrong_z_rank_loss": 0.0,
        "condition_action_abs": 0.0,
        "recycled_latent_action_abs": 0.0,
        "total_loss": 0.0,
    },
    "cross_cycle": {
        "cross_cycle_loss": 0.0,
        "cross_cycle_loss_scale": 0.0,
        "cross_latent_action_kl_loss": 0.0,
        "cross_recycled_latent_action_abs": 0.0,
        "cross_condition_action_abs": 0.0,
    },
    "grl": {
        "grl_loss": 0.0,
        "grl_accuracy": 0.0,
        "grl_scale": 0.0,
    },
}

EVAL_TABLE_METRIC_KEYS: tuple[str, ...] = (
    "video_mse",
    "video_psnr",
    "video_ssim",
    "video_lpips",
    "video_mse_last",
    "video_psnr_last",
    "video_ssim_last",
    "video_lpips_last",
)


@dataclass(frozen=True)
class MetricBundle:
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            {str(key): float(value) for key, value in self.values.items()},
        )

    def __bool__(self) -> bool:
        return bool(self.values)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.values.get(key, default)

    def items(self):
        return self.values.items()

    def keys(self):
        return self.values.keys()

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)

    def merge(self, *others: MetricBundle | None) -> MetricBundle:
        merged = self.to_dict()
        for other in others:
            if other is None:
                continue
            merged.update(other.values)
        return MetricBundle(merged)

    def with_updates(self, **updates: float) -> MetricBundle:
        merged = self.to_dict()
        merged.update({key: float(value) for key, value in updates.items()})
        return MetricBundle(merged)

    def namespaced(self, namespace: str) -> dict[str, float]:
        return {
            f"{namespace}/{key}": float(value)
            for key, value in self.values.items()
        }


def build_metric_bundle(*groups: str, **updates: float) -> MetricBundle:
    values: dict[str, float] = {}
    for group in groups:
        group_defaults = METRIC_GROUP_DEFAULTS.get(group)
        if group_defaults is None:
            raise KeyError(f"Unknown metric group: {group}")
        values.update(group_defaults)
    values.update({key: float(value) for key, value in updates.items()})
    return MetricBundle(values)


def average_metric_bundles(bundles: Iterable[MetricBundle]) -> MetricBundle:
    bundles = list(bundles)
    if not bundles:
        return MetricBundle()

    metric_keys = sorted({key for bundle in bundles for key in bundle.keys()})
    averaged = {
        key: sum(bundle.get(key, 0.0) or 0.0 for bundle in bundles) / len(bundles)
        for key in metric_keys
    }
    return MetricBundle(averaged)


def build_namespaced_log_payload(
    namespace: str,
    *,
    step: int,
    metrics: MetricBundle,
    extra_scalars: Mapping[str, float] | None = None,
) -> dict[str, float | int]:
    payload = {f"{namespace}/step": step}
    payload.update(metrics.namespaced(namespace))
    if extra_scalars:
        payload.update(
            {
                f"{namespace}/{key}": float(value)
                for key, value in extra_scalars.items()
            }
        )
    return payload


def select_metric_keys(
    metrics: MetricBundle,
    *,
    keys: Iterable[str],
) -> MetricBundle:
    selected = {
        key: metrics[key]
        for key in keys
        if key in metrics.keys()
    }
    return MetricBundle(selected)


def build_eval_table_metric_bundle(metrics: MetricBundle) -> MetricBundle:
    return select_metric_keys(metrics, keys=EVAL_TABLE_METRIC_KEYS)


def format_train_summary(
    *,
    step: int,
    metrics: MetricBundle,
    grad_norm: float,
    elapsed: float,
    suffix: str = "",
) -> str:
    cross_cycle_str = (
        f"cross_cycle={metrics['cross_cycle_loss']:.6f} "
        if metrics.get("cross_cycle_loss", 0.0) > 0
        else ""
    )
    summary = (
        f"[train] step={step:06d} "
        f"total_loss={metrics['total_loss']:.6f} "
        f"recon_loss={metrics['recon_loss']:.6f} "
        f"cycle_loss={metrics['cycle_loss']:.6f} "
        f"cycle_idm_grad={'on' if metrics.get('cycle_idm_grad', 0.0) > 0.5 else 'off'} "
        f"{cross_cycle_str}"
        f"gt_action_aux_mse={metrics['gt_action_aux_mse']:.6f} "
        f"latent_kl={metrics['latent_action_kl_loss']:.6f} "
        f"wrong_z={metrics['wrong_z_train_loss']:.6f} "
        f"wrong_z_active={metrics['wrong_z_active_loss']:.6f} "
        f"wrong_z_frac={metrics['wrong_z_active_frac']:.2f} "
        f"wrong_z_same={metrics['wrong_z_same_emb_hit_rate']:.2f} "
        f"wrong_z_sim={metrics['wrong_z_neighbor_similarity_mean']:.3f} "
        f"grl={metrics['grl_loss']:.4f}({metrics['grl_accuracy']:.2f}) "
        f"action_abs={metrics['condition_action_abs']:.6f} "
        f"grad_norm={grad_norm:.4f} "
        f"time={elapsed:.2f}s"
    )
    return summary + suffix


def format_eval_summary(
    *,
    step: int,
    metrics: MetricBundle,
) -> str | None:
    if not metrics:
        return None

    cross_cycle_str = (
        f"cross_cycle={metrics['cross_cycle_loss']:.6f} "
        if metrics.get("cross_cycle_loss", 0.0) > 0
        else ""
    )
    return (
        f"[eval] step={step:06d} "
        f"total_loss={metrics['total_loss']:.6f} "
        f"recon_loss={metrics['recon_loss']:.6f} "
        f"cycle_loss={metrics['cycle_loss']:.6f} "
        f"cycle_idm_grad={'on' if metrics.get('cycle_idm_grad', 0.0) > 0.5 else 'off'} "
        f"{cross_cycle_str}"
        f"gt_action_aux_mse={metrics['gt_action_aux_mse']:.6f} "
        f"latent_kl={metrics['latent_action_kl_loss']:.6f} "
        f"wrong_z={metrics['wrong_z_train_loss']:.6f} "
        f"wrong_z_active={metrics['wrong_z_active_loss']:.6f} "
        f"wrong_z_frac={metrics['wrong_z_active_frac']:.2f} "
        f"wrong_z_same={metrics['wrong_z_same_emb_hit_rate']:.2f} "
        f"wrong_z_sim={metrics['wrong_z_neighbor_similarity_mean']:.3f}"
    )


@dataclass(frozen=True)
class VideoMetricConfig:
    names: tuple[str, ...] = ()
    batch_size: int = 16
    max_video_count: int = 16


def compute_local_ssim(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected preds and target to have the same shape, got {preds.shape} and {target.shape}"
        )
    if preds.ndim != 4:
        raise ValueError(f"Expected image tensors [N,C,H,W], got {preds.shape}")

    channels = preds.shape[1]
    kernel = torch.ones(
        (channels, 1, window_size, window_size),
        device=preds.device,
        dtype=preds.dtype,
    ) / float(window_size * window_size)
    padding = window_size // 2

    mu_x = F.conv2d(preds, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(preds * preds, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(preds * target, kernel, padding=padding, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-8)
    return ssim_map.mean()


def _compute_video_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    split_batch_size: int | None = None,
) -> torch.Tensor:
    del split_batch_size
    return F.mse_loss(preds, target)


def _compute_video_psnr(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    split_batch_size: int | None = None,
) -> torch.Tensor:
    mse_value = _compute_video_mse(preds, target)
    del split_batch_size
    return -10.0 * torch.log10(mse_value.clamp_min(1e-8))


def _compute_video_ssim(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    split_batch_size: int | None = None,
) -> torch.Tensor:
    del split_batch_size
    return compute_local_ssim(preds, target)


_LPIPS_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _get_lpips_model(device: torch.device) -> torch.nn.Module:
    cache_key = ("alex", str(device))
    model = _LPIPS_MODEL_CACHE.get(cache_key)
    if model is not None:
        return model

    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS metric requested but the 'lpips' package is not installed. "
            "Install it or remove 'lpips' from metric_video_names."
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*pretrained.*deprecated.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Arguments other than a weight enum.*deprecated.*",
                category=UserWarning,
            )
            try:
                model = lpips.LPIPS(net="alex", verbose=False)
            except TypeError:
                model = lpips.LPIPS(net="alex")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize LPIPS. The lpips package requires the AlexNet "
            "backbone weights in the PyTorch cache; if this machine cannot download "
            "them automatically, prefetch alexnet-owt-7be5be79.pth into "
            "~/.cache/torch/hub/checkpoints/."
        ) from exc
    model = model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _LPIPS_MODEL_CACHE[cache_key] = model
    return model


def _compute_video_lpips(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    split_batch_size: int | None = None,
) -> torch.Tensor:
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected preds and target to have the same shape, got {preds.shape} and {target.shape}"
        )
    if preds.ndim != 4:
        raise ValueError(f"Expected image tensors [N,C,H,W], got {preds.shape}")
    if preds.shape[1] != 3:
        raise ValueError(f"LPIPS expects RGB tensors with 3 channels, got {preds.shape}")

    device = preds.device
    if device.type != "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    model = _get_lpips_model(device)
    chunk_size = max(int(split_batch_size or 0), 1) if split_batch_size else preds.shape[0]
    distances: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, preds.shape[0], chunk_size):
            end = min(start + chunk_size, preds.shape[0])
            pred_chunk = preds[start:end].to(device=device, dtype=torch.float32) * 2.0 - 1.0
            target_chunk = target[start:end].to(device=device, dtype=torch.float32) * 2.0 - 1.0
            distances.append(model(pred_chunk, target_chunk).reshape(-1))
    return torch.cat(distances, dim=0).mean()


VIDEO_METRIC_REGISTRY = {
    "mse": _compute_video_mse,
    "psnr": _compute_video_psnr,
    "ssim": _compute_video_ssim,
    "lpips": _compute_video_lpips,
}


def resolve_video_metric_names(metric_names: Iterable[str] | None) -> list[str]:
    names = [str(name).strip() for name in (metric_names or []) if str(name).strip()]
    unknown_metrics = [name for name in names if name not in VIDEO_METRIC_REGISTRY]
    if unknown_metrics:
        raise ValueError(
            f"Unsupported eval video metrics: {unknown_metrics}. "
            f"Supported metrics are: {sorted(VIDEO_METRIC_REGISTRY)}"
        )
    return names


def build_eval_video_metric(
    metric_names: Iterable[str] | None,
    *,
    device: torch.device,
    split_batch_size: int,
) -> tuple[list[str] | None, list[str]]:
    del device, split_batch_size
    resolved = resolve_video_metric_names(metric_names)
    if not resolved:
        return None, []
    return resolved, resolved


def _prepare_eval_video_metric_tensors(
    *,
    video_pred: torch.Tensor,
    video_gt: torch.Tensor,
    hist_len: int,
    n_metrics_frames: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = ((video_pred.detach().clamp(-1.0, 1.0) + 1.0) * 0.5).to(torch.float32)
    target = ((video_gt.detach().clamp(-1.0, 1.0) + 1.0) * 0.5).to(torch.float32)
    if n_metrics_frames is not None:
        preds = preds[:, :n_metrics_frames]
        target = target[:, :n_metrics_frames]

    context_len = min(hist_len, preds.shape[1])
    context_mask = torch.zeros(preds.shape[1], dtype=torch.bool, device=preds.device)
    context_mask[:context_len] = True

    preds = preds[:, ~context_mask]
    target = target[:, ~context_mask]
    return preds, target


def compute_eval_video_metrics(
    metric_names: Iterable[str] | None,
    *,
    video_pred: torch.Tensor,
    video_gt: torch.Tensor,
    hist_len: int,
    n_metrics_frames: int | None,
    split_batch_size: int | None = None,
) -> MetricBundle:
    if metric_names is None:
        return MetricBundle()

    preds, target = _prepare_eval_video_metric_tensors(
        video_pred=video_pred,
        video_gt=video_gt,
        hist_len=hist_len,
        n_metrics_frames=n_metrics_frames,
    )
    if preds.numel() == 0 or target.numel() == 0:
        return MetricBundle()

    full_metrics = compute_video_metric_bundle(
        metric_names,
        preds=preds.reshape(-1, *preds.shape[-3:]),
        target=target.reshape(-1, *target.shape[-3:]),
        split_batch_size=split_batch_size,
    )
    last_metrics = compute_video_metric_bundle(
        metric_names,
        preds=preds[:, -1],
        target=target[:, -1],
        suffix="_last",
        split_batch_size=split_batch_size,
    )
    return full_metrics.merge(last_metrics)


def compute_lvp_style_eval_video_metrics_from_batches(
    metric_names: Iterable[str] | None,
    *,
    video_pred_batches: list[torch.Tensor],
    video_gt_batches: list[torch.Tensor],
    hist_len: int,
    n_metrics_frames: int | None,
    split_batch_size: int,
    max_video_count: int | None,
) -> MetricBundle:
    if not metric_names:
        return MetricBundle()

    video_pred_batches = [
        batch for batch in video_pred_batches if batch is not None and batch.numel() > 0
    ]
    video_gt_batches = [
        batch for batch in video_gt_batches if batch is not None and batch.numel() > 0
    ]
    if not video_pred_batches or not video_gt_batches:
        return MetricBundle()

    preds = torch.cat(video_pred_batches, dim=0)
    target = torch.cat(video_gt_batches, dim=0)
    if max_video_count is not None and max_video_count > 0:
        preds = preds[:max_video_count]
        target = target[:max_video_count]

    if preds.numel() == 0 or target.numel() == 0:
        return MetricBundle()
    return compute_eval_video_metrics(
        metric_names,
        video_pred=preds,
        video_gt=target,
        hist_len=hist_len,
        n_metrics_frames=n_metrics_frames,
        split_batch_size=split_batch_size,
    )


def compute_video_metric_bundle(
    metric_names: Iterable[str] | None,
    *,
    preds: torch.Tensor,
    target: torch.Tensor,
    suffix: str = "",
    split_batch_size: int | None = None,
) -> MetricBundle:
    resolved = resolve_video_metric_names(metric_names)
    if not resolved:
        return MetricBundle()

    metric_values: dict[str, float] = {}
    for metric_name in resolved:
        value = VIDEO_METRIC_REGISTRY[metric_name](
            preds,
            target,
            split_batch_size=split_batch_size,
        )
        metric_values[f"video_{metric_name}{suffix}"] = float(value.detach().cpu().item())
    return MetricBundle(metric_values)


__all__ = [
    "MetricBundle",
    "VideoMetricConfig",
    "average_metric_bundles",
    "build_eval_table_metric_bundle",
    "build_eval_video_metric",
    "build_metric_bundle",
    "build_namespaced_log_payload",
    "compute_eval_video_metrics",
    "compute_local_ssim",
    "compute_lvp_style_eval_video_metrics_from_batches",
    "compute_video_metric_bundle",
    "format_eval_summary",
    "format_train_summary",
    "resolve_video_metric_names",
    "select_metric_keys",
]
