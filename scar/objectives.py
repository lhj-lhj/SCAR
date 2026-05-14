from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import MetricBundle, build_metric_bundle
from .models import (
    build_gt_action_padding_mask,
    prepare_latent_idm_inputs,
    prepare_rgb_patch_idm_inputs,
)
from .runtime import (
    build_prompt_context,
    cast_tensor_tree,
    debug_log_once,
    select_rgb_channels,
    temporarily_frozen_eval,
    to_lvp_range,
    unwrap_idm_module,
    unwrap_module,
)


@dataclass
class ObjectiveOutput:
    total_loss: torch.Tensor
    metrics: MetricBundle
    conditioning_actions: torch.Tensor | None = None


def run_idm_on_video_lat(
    idm: nn.Module,
    video_lat: torch.Tensor,
    *,
    target_frame_tokens: int,
    return_output: bool = False,
):
    return run_idm_with_input_source(
        idm,
        target_frame_tokens=target_frame_tokens,
        video_lat=video_lat,
        idm_input_source="vae_latent",
        return_output=return_output,
    )


def resolve_idm_input_source(
    idm: nn.Module | None,
    *,
    idm_input_source: str | None = None,
) -> str:
    if idm_input_source is not None:
        return str(idm_input_source)
    idm_module = unwrap_idm_module(idm)
    return str(getattr(idm_module, "input_source", "vae_latent"))


def run_idm_with_input_source(
    idm: nn.Module,
    *,
    target_frame_tokens: int,
    video_lat: torch.Tensor | None = None,
    batch_observations: torch.Tensor | None = None,
    idm_input_source: str | None = None,
    return_output: bool = False,
):
    idm_module = unwrap_idm_module(idm)
    input_source = resolve_idm_input_source(idm, idm_input_source=idm_input_source)
    if input_source == "vae_latent":
        if video_lat is None:
            raise ValueError("VAE-latent IDM input requested, but video_lat is missing.")
        observations, timesteps = prepare_latent_idm_inputs(idm_module, video_lat)
    elif input_source == "rgb_patch":
        if batch_observations is None:
            raise ValueError(
                "RGB-patch IDM input requested, but batch observations are missing."
            )
        observations, timesteps = prepare_rgb_patch_idm_inputs(
            idm_module,
            batch_observations,
        )
    else:
        raise ValueError(f"Unsupported IDM input source: {input_source}")
    idm_output = idm(
        observations,
        timesteps=timesteps,
        states=None,
        target_frame_tokens=target_frame_tokens,
    )
    if return_output:
        return idm_output
    if hasattr(idm_output, "la"):
        return idm_output.la
    return idm_output


def resolve_conditioning_action_dim(
    idm_cfg,
    *,
    action_source: str,
) -> int:
    if action_source == "gt":
        action_dim = getattr(idm_cfg.env, "action_dim", None)
        if action_dim is None:
            raise ValueError(
                "GT-action conditioning requested, but idm_cfg.env.action_dim is unset."
            )
        return int(action_dim)
    return int(idm_cfg.model.la_dim)


def resolve_conditioning_actions(
    *,
    action_source: str,
    batch,
    video_lat: torch.Tensor,
    idm: nn.Module | None,
    target_frame_tokens: int,
    idm_input_source: str | None = None,
    return_idm_output: bool = False,
):
    if action_source == "gt":
        gt_actions = getattr(batch, "actions", None)
        if gt_actions is None:
            raise ValueError("GT-action conditioning requested, but batch.actions is missing.")
        if gt_actions.ndim != 3:
            raise ValueError(
                f"Expected batch.actions to have shape [B, T, D], got {tuple(gt_actions.shape)}"
            )
        if gt_actions.shape[1] < target_frame_tokens:
            raise ValueError(
                "GT actions contain fewer timesteps than required for LVP conditioning: "
                f"have {gt_actions.shape[1]}, need {target_frame_tokens}"
            )
        conditioning_actions = gt_actions[:, :target_frame_tokens].detach()
        if return_idm_output:
            return conditioning_actions, None
        return conditioning_actions

    if idm is None:
        raise ValueError("IDM action source requested, but no IDM module was constructed.")
    batch_observations = getattr(batch, "observations", None)
    if isinstance(batch_observations, torch.Tensor):
        batch_observations = batch_observations.detach()
    idm_output = run_idm_with_input_source(
        idm,
        target_frame_tokens=target_frame_tokens,
        video_lat=video_lat.detach(),
        batch_observations=batch_observations,
        idm_input_source=idm_input_source,
        return_output=return_idm_output,
    )
    if return_idm_output:
        conditioning_actions = idm_output.la if hasattr(idm_output, "la") else idm_output
        return conditioning_actions, idm_output
    return idm_output


def get_gt_action_loss_scale(args) -> float:
    if args.gt_action_loss_weight <= 0:
        return 0.0
    if args.lvp_action_source != "idm":
        return 0.0
    return float(args.gt_action_loss_weight)


def get_latent_action_kl_loss_scale(args) -> float:
    if not bool(getattr(args, "use_reparameterized_la", False)):
        return 0.0
    if getattr(args, "lvp_action_source", "idm") != "idm":
        return 0.0
    return float(getattr(args, "latent_action_kl_weight", 0.0))


def _get_idm_model_cfg(args) -> dict:
    idm_cfg = getattr(args, "idm_cfg", {}) or {}
    if not isinstance(idm_cfg, dict):
        return {}
    model_cfg = idm_cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        return {}
    idm_model_cfg = model_cfg.get("idm", {})
    if not isinstance(idm_model_cfg, dict):
        return {}
    return idm_model_cfg


def get_latent_action_vq_loss_scale(args) -> float:
    if getattr(args, "lvp_action_source", "idm") != "idm":
        return 0.0
    idm_model_cfg = _get_idm_model_cfg(args)
    if not bool(idm_model_cfg.get("quantize_la", False)):
        return 0.0
    return max(0.0, float(idm_model_cfg.get("vq_loss_weight", 1.0)))


def compute_gt_action_aux_loss(
    action_head: nn.Module,
    latent_actions: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, MetricBundle]:
    if latent_actions.shape[1] <= 1:
        zero = latent_actions.new_zeros(())
        return zero, build_metric_bundle("objective")

    gt_actions = getattr(batch, "actions", None)
    if gt_actions is None:
        raise ValueError("GT-action auxiliary loss requested, but batch.actions is missing.")

    pred_latent_actions = latent_actions[:, 1:]
    seq_padding_mask = build_gt_action_padding_mask(
        getattr(batch, "mask", None),
        target_seq_len=pred_latent_actions.shape[1],
    )
    pred_actions = action_head(
        pred_latent_actions,
        padding_mask=seq_padding_mask,
    )
    target_actions = gt_actions[:, : pred_actions.shape[1]].to(
        device=pred_actions.device,
        dtype=pred_actions.dtype,
    )
    if pred_actions.shape != target_actions.shape:
        raise ValueError(
            "GT-action auxiliary supervision shape mismatch: "
            f"pred={tuple(pred_actions.shape)}, target={tuple(target_actions.shape)}"
        )

    mask = getattr(batch, "mask", None)
    raw_mask_shape = tuple(mask.shape) if mask is not None else None
    if mask is not None:
        mask = mask[:, : pred_actions.shape[1]]
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
            debug_log_once(
                "gt_action_aux_mask_broadcast",
                "broadcasting sequence batch.mask from "
                f"{raw_mask_shape} to per-action shape {tuple(pred_actions.shape)}",
            )
        if mask.ndim != 3:
            raise ValueError(
                "GT-action auxiliary mask must have shape [B, T], [B, T, 1], or [B, T, D]; "
                f"got raw_mask_shape={raw_mask_shape}, sliced_mask_shape={tuple(mask.shape)}"
            )
        if mask.shape[:2] != pred_actions.shape[:2]:
            raise ValueError(
                "GT-action auxiliary mask shape mismatch: "
                f"raw_mask_shape={raw_mask_shape}, sliced_mask_shape={tuple(mask.shape)}, "
                f"pred={tuple(pred_actions.shape)}"
            )
        if mask.shape[2] == 1:
            mask = mask.expand(-1, -1, pred_actions.shape[2])
        elif mask.shape[2] != pred_actions.shape[2]:
            raise ValueError(
                "GT-action auxiliary mask channel mismatch: "
                f"raw_mask_shape={raw_mask_shape}, sliced_mask_shape={tuple(mask.shape)}, "
                f"pred={tuple(pred_actions.shape)}"
            )
        mask = mask.to(device=pred_actions.device, dtype=pred_actions.dtype)
    debug_log_once(
        "gt_action_aux_shapes",
        f"head_type={getattr(unwrap_module(action_head), 'head_type', 'unknown')} "
        f"latent_actions={tuple(latent_actions.shape)} "
        f"pred_latent_actions={tuple(pred_latent_actions.shape)} "
        f"pred_actions={tuple(pred_actions.shape)} "
        f"target_actions={tuple(target_actions.shape)} "
        f"batch_actions={tuple(gt_actions.shape)} "
        f"raw_mask_shape={raw_mask_shape} "
        f"seq_padding_mask_shape={tuple(seq_padding_mask.shape) if seq_padding_mask is not None else None} "
        f"effective_mask_shape={tuple(mask.shape) if mask is not None else None}",
    )
    diff = pred_actions - target_actions
    per_elem_loss = diff.square()

    if mask is not None:
        normalizer = mask.sum().clamp_min(1.0)
        loss = (per_elem_loss * mask).sum() / normalizer
    else:
        loss = per_elem_loss.mean()

    metrics = build_metric_bundle(
        "objective",
        gt_action_aux_mse=float(loss.detach().cpu()),
    )
    return loss, metrics


def encode_lvp_video_latents(
    lvp,
    batch,
    *,
    lvp_trainable_modules: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    videos = select_rgb_channels(batch.observations)
    videos = to_lvp_range(videos).contiguous()
    lvp_trainable_modules = set(lvp_trainable_modules or [])
    video_ctx = (
        torch.enable_grad()
        if torch.is_grad_enabled() and "vae" in lvp_trainable_modules
        else torch.no_grad()
    )
    with video_ctx:
        video_lat = lvp.encode_video(videos.permute(0, 2, 1, 3, 4))
    return videos, video_lat


def unpack_lvp_action_encoder_output(
    action_encoder_output,
) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, ...]]:
    if isinstance(action_encoder_output, tuple):
        if len(action_encoder_output) != 2:
            raise ValueError(
                "Expected lvp.action_encoder to return a 2-tuple, got "
                f"{len(action_encoder_output)} elements."
            )
        action_tokens = action_encoder_output[0]
    else:
        action_tokens = action_encoder_output

    if not isinstance(action_tokens, torch.Tensor) or action_tokens.ndim != 3:
        raise ValueError(
            "Expected LVP action encoder features to have shape [B, T, D], got "
            f"{type(action_tokens)} with shape "
            f"{tuple(action_tokens.shape) if isinstance(action_tokens, torch.Tensor) else None}"
        )
    return action_tokens, action_encoder_output


def project_action_tokens_to_latent_bias(
    action_tokens: torch.Tensor,
    *,
    latent_channels: int,
) -> torch.Tensor:
    if latent_channels <= 0:
        raise ValueError(f"latent_channels must be positive, got {latent_channels}")

    action_tokens = F.layer_norm(action_tokens.float(), (action_tokens.shape[-1],))
    hidden_dim = action_tokens.shape[-1]
    pad_channels = (-hidden_dim) % latent_channels
    if pad_channels:
        action_tokens = F.pad(action_tokens, (0, pad_channels))

    grouped = action_tokens.reshape(
        action_tokens.shape[0],
        action_tokens.shape[1],
        latent_channels,
        -1,
    ).mean(dim=-1)
    return grouped.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)


def build_lvp_condition_latents(
    lvp,
    conditioning_actions: torch.Tensor,
    *,
    model_device: torch.device,
    model_dtype: torch.dtype,
    video_lat: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | tuple[torch.Tensor, ...],
    torch.Tensor | None,
]:
    action_encoder_module = unwrap_module(lvp.action_encoder)
    action_encoder_param = next(action_encoder_module.parameters())
    conditioning_actions = conditioning_actions.to(
        device=action_encoder_param.device,
        dtype=action_encoder_param.dtype,
    )
    raw_cond_lat = lvp.action_encoder(conditioning_actions)
    _action_tokens, raw_cond_lat = unpack_lvp_action_encoder_output(raw_cond_lat)

    cond_lat = cast_tensor_tree(
        raw_cond_lat,
        device=model_device,
        dtype=model_dtype,
    )
    return conditioning_actions, cond_lat, video_lat


@dataclass
class LVPFlowCache:
    flow_pred: torch.Tensor
    target_flow: torch.Tensor
    noise: torch.Tensor
    t: torch.Tensor
    hist_tokens: int
    conditioning_actions: torch.Tensor


def forward_lvp_flow(
    lvp,
    video_lat: torch.Tensor,
    conditioning_actions: torch.Tensor,
    *,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    noise: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> LVPFlowCache:
    batch_size = video_lat.shape[0]
    prompt_embeds = build_prompt_context(
        batch_size=batch_size,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )

    model_param = next(lvp.model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype
    prompt_embeds = cast_tensor_tree(prompt_embeds, device=model_device, dtype=model_dtype)

    conditioning_actions, cond_lat, video_lat = build_lvp_condition_latents(
        lvp,
        conditioning_actions,
        model_device=model_device,
        model_dtype=model_dtype,
        video_lat=video_lat,
    )
    if video_lat is None:
        raise RuntimeError("Expected build_lvp_condition_latents to return video latents.")

    if noise is None or t is None:
        noisy_lat, noise, t = lvp.add_training_noise(video_lat)
    else:
        noisy_lat = _build_continuous_noisy_lat(
            video_lat,
            noise,
            t,
            num_train_timesteps=int(lvp.num_train_timesteps),
        )

    noisy_lat = noisy_lat.to(device=model_device, dtype=model_dtype)
    t = t.to(device=model_device)
    target_flow = (noise - video_lat).to(device=model_device, dtype=model_dtype)

    flow_pred = lvp.model(
        noisy_lat,
        t=t,
        context=prompt_embeds,
        seq_len=lvp.max_tokens,
        clip_fea=None,
        y=None,
        cond=cond_lat,
    )
    return LVPFlowCache(
        flow_pred=flow_pred,
        target_flow=target_flow,
        noise=noise,
        t=t,
        hist_tokens=int(lvp.hist_tokens),
        conditioning_actions=conditioning_actions,
    )


def compute_lvp_recon_loss_from_video_lat(
    lvp,
    video_lat: torch.Tensor,
    conditioning_actions: torch.Tensor,
    *,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
) -> tuple[torch.Tensor, MetricBundle, LVPFlowCache]:
    pos_cache = forward_lvp_flow(
        lvp,
        video_lat,
        conditioning_actions,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )
    recon_loss = F.mse_loss(pos_cache.flow_pred.float(), pos_cache.target_flow.float())
    metrics = build_metric_bundle(
        "objective",
        recon_loss=float(recon_loss.detach().cpu()),
        condition_action_abs=float(conditioning_actions.detach().abs().mean().cpu()),
    )
    return recon_loss, metrics, pos_cache


def predict_lvp_clean_video_lat_single_step(
    lvp,
    video_lat: torch.Tensor,
    conditioning_actions: torch.Tensor,
    *,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    noise: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    pred_clean_lat, history_frames, _hist_tokens = _predict_lvp_clean_video_lat_single_step_impl(
        lvp,
        video_lat,
        conditioning_actions,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
        noise=noise,
        t=t,
    )
    return pred_clean_lat, history_frames


def predict_lvp_clean_video_lat_from_flow_cache(
    lvp,
    video_lat: torch.Tensor,
    pos_cache: LVPFlowCache,
) -> tuple[torch.Tensor, int]:
    # Only valid for cycle variants that explicitly share the recon forward cache.
    # Do not use this helper for frozen-target dual-forward cycle designs.
    t = pos_cache.t
    flow_pred = pos_cache.flow_pred
    noisy_lat = _build_continuous_noisy_lat(
        video_lat,
        pos_cache.noise,
        t,
        num_train_timesteps=int(lvp.num_train_timesteps),
    )
    if t.ndim == 1:
        sigma = (t / lvp.num_train_timesteps).view(-1, 1, 1, 1, 1)
    elif t.ndim == 2:
        sigma = (t / lvp.num_train_timesteps).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported timestep shape for cached single-step cycle: {t.shape}")
    pred_clean_lat = noisy_lat.float() - sigma.float() * flow_pred.float()
    hist_tokens = int(pos_cache.hist_tokens)
    pred_clean_lat[:, :, :hist_tokens] = video_lat[:, :, :hist_tokens].float()
    history_frames = int(lvp.max_frames - lvp.pred_len)
    return pred_clean_lat, history_frames


def _build_continuous_noisy_lat(
    video_lat: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    *,
    num_train_timesteps: int,
) -> torch.Tensor:
    if t.ndim == 1:
        sigma = (t / num_train_timesteps).view(-1, 1, 1, 1, 1)
    elif t.ndim == 2:
        sigma = (t / num_train_timesteps).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported timestep shape for single-step cycle: {t.shape}")
    sigma = sigma.to(device=video_lat.device, dtype=video_lat.dtype)
    noise = noise.to(device=video_lat.device, dtype=video_lat.dtype)
    return video_lat * (1.0 - sigma) + noise * sigma


def _predict_lvp_clean_video_lat_single_step_impl(
    lvp,
    video_lat: torch.Tensor,
    conditioning_actions: torch.Tensor,
    *,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    noise: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    if lvp.diffusion_type != "continuous":
        raise NotImplementedError(
            "The single-step cycle helper currently supports only continuous diffusion."
        )

    batch_size = video_lat.shape[0]
    prompt_embeds = build_prompt_context(
        batch_size=batch_size,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )

    model_param = next(lvp.model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    prompt_embeds = cast_tensor_tree(
        prompt_embeds,
        device=model_device,
        dtype=model_dtype,
    )
    conditioning_actions, cond_lat, video_lat = build_lvp_condition_latents(
        lvp,
        conditioning_actions,
        model_device=model_device,
        model_dtype=model_dtype,
        video_lat=video_lat,
    )
    if noise is None or t is None:
        noisy_lat, _noise, t = lvp.add_training_noise(video_lat)
    else:
        noisy_lat = _build_continuous_noisy_lat(
            video_lat,
            noise,
            t,
            num_train_timesteps=int(lvp.num_train_timesteps),
        )
    noisy_lat = noisy_lat.to(device=model_device, dtype=model_dtype)
    t = t.to(device=model_device)

    flow_pred = lvp.model(
        noisy_lat,
        t=t,
        context=prompt_embeds,
        seq_len=lvp.max_tokens,
        clip_fea=None,
        y=None,
        cond=cond_lat,
    )

    if t.ndim == 1:
        sigma = (t / lvp.num_train_timesteps).view(-1, 1, 1, 1, 1)
    elif t.ndim == 2:
        sigma = (t / lvp.num_train_timesteps).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported timestep shape for single-step cycle: {t.shape}")
    pred_clean_lat = noisy_lat.float() - sigma.float() * flow_pred.float()
    hist_tokens = int(lvp.hist_tokens)
    pred_clean_lat[:, :, :hist_tokens] = video_lat[:, :, :hist_tokens].float()
    history_frames = int(lvp.max_frames - lvp.pred_len)
    return pred_clean_lat, history_frames, hist_tokens


def build_wrong_z_permutation(
    batch_size: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if batch_size <= 1:
        return None
    base = torch.arange(batch_size, device=device)
    for _ in range(8):
        perm = torch.randperm(batch_size, device=device)
        if not torch.any(perm == base):
            return perm
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return torch.roll(base, shifts=shift)


def build_wrong_z_reference_features(
    video_lat: torch.Tensor,
    *,
    history_tokens: int,
) -> torch.Tensor:
    if video_lat.ndim != 5:
        raise ValueError(
            f"Expected video_lat to have shape [B, C, T, H, W], got {tuple(video_lat.shape)}"
        )
    total_tokens = int(video_lat.shape[2])
    ref_tokens = min(max(int(history_tokens), 1), total_tokens)
    ref_lat = video_lat[:, :, :ref_tokens].detach().float()
    ref_feat = ref_lat.reshape(ref_lat.shape[0], -1)
    return F.normalize(ref_feat, dim=1, eps=1e-6)


def build_wrong_z_nearest_neighbor_indices(
    video_lat: torch.Tensor,
    *,
    history_tokens: int,
    embodiment_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    batch_size = int(video_lat.shape[0])
    if batch_size <= 1:
        return None, None, None

    ref_feat = build_wrong_z_reference_features(
        video_lat,
        history_tokens=history_tokens,
    )
    similarity = ref_feat @ ref_feat.transpose(0, 1)
    similarity.fill_diagonal_(float("-inf"))

    if embodiment_ids is not None:
        emb = embodiment_ids.to(device=video_lat.device).reshape(-1)
        if emb.numel() == batch_size:
            same_emb = emb[:, None] == emb[None, :]
            same_emb.fill_diagonal_(False)
            row_has_same_emb = same_emb.any(dim=1)
            selected_same_emb = same_emb.new_zeros((batch_size,))
            if torch.any(row_has_same_emb):
                same_emb_similarity = similarity.masked_fill(~same_emb, float("-inf"))
                indices = similarity.argmax(dim=1)
                indices[row_has_same_emb] = same_emb_similarity[row_has_same_emb].argmax(dim=1)
                selected_same_emb[row_has_same_emb] = True
                selected_similarity = similarity.gather(1, indices[:, None]).squeeze(1)
                return indices, selected_similarity, selected_same_emb

    indices = similarity.argmax(dim=1)
    selected_similarity = similarity.gather(1, indices[:, None]).squeeze(1)
    selected_same_emb = None
    if embodiment_ids is not None:
        emb = embodiment_ids.to(device=video_lat.device).reshape(-1)
        if emb.numel() == batch_size:
            selected_same_emb = emb == emb[indices]
    return indices, selected_similarity, selected_same_emb


def compute_future_flow_mse_distance(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    *,
    history_tokens: int,
) -> torch.Tensor:
    if pred_flow.shape != target_flow.shape:
        raise ValueError(
            "Expected predicted and target flow tensors to share shape, got "
            f"{tuple(pred_flow.shape)} vs {tuple(target_flow.shape)}"
        )
    history_tokens = max(0, min(int(history_tokens), int(pred_flow.shape[2])))
    pred_future = pred_flow[:, :, history_tokens:]
    target_future = target_flow[:, :, history_tokens:]
    if pred_future.numel() == 0:
        return pred_flow.new_zeros((pred_flow.shape[0],), dtype=torch.float32)
    per_elem_loss = F.mse_loss(
        pred_future.float(),
        target_future.float(),
        reduction="none",
    )
    return per_elem_loss.reshape(per_elem_loss.shape[0], -1).mean(dim=1)


def compute_wrong_z_gate_sigma(
    *,
    lvp,
    t: torch.Tensor,
    history_tokens: int,
) -> torch.Tensor:
    num_train_timesteps = float(getattr(lvp, "num_train_timesteps", 0))
    if num_train_timesteps <= 0:
        raise ValueError(
            f"Expected positive lvp.num_train_timesteps, got {num_train_timesteps}"
        )

    if t.ndim == 1:
        sigma = t.float() / num_train_timesteps
    elif t.ndim == 2:
        future_t = t[:, history_tokens:]
        if future_t.shape[1] == 0:
            sigma = t.new_zeros((t.shape[0],), dtype=torch.float32)
        else:
            sigma = future_t.float().mean(dim=1) / num_train_timesteps
    else:
        raise ValueError(f"Unsupported timestep shape for wrong-z sigma gate: {t.shape}")
    return sigma.clamp_(0.0, 1.0)


def should_compute_wrong_z_loss(
    *,
    args,
    idm: nn.Module | None,
) -> bool:
    return (
        idm is not None
        and args.lvp_action_source == "idm"
        and bool(getattr(args, "wrong_z_enabled", False))
    )


def compute_wrong_z_loss(
    *,
    args,
    training: bool,
    lvp,
    idm: nn.Module | None,
    video_lat: torch.Tensor,
    conditioning_actions: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    pos_cache: LVPFlowCache,
    embodiment_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, MetricBundle]:

    neg_indices, neighbor_similarity, same_emb_hits = build_wrong_z_nearest_neighbor_indices(
        video_lat,
        history_tokens=pos_cache.hist_tokens,
        embodiment_ids=embodiment_ids,
    )
    if neg_indices is None:
        zero = video_lat.new_zeros(())
        return zero, build_metric_bundle("objective")
    wrong_actions = conditioning_actions[neg_indices].detach()

    d_pos = compute_future_flow_mse_distance(
        pos_cache.flow_pred,
        pos_cache.target_flow,
        history_tokens=pos_cache.hist_tokens,
    )
    with torch.no_grad():
        neg_cache = forward_lvp_flow(
            lvp,
            video_lat,
            wrong_actions,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            noise=pos_cache.noise,
            t=pos_cache.t,
        )
        d_neg = compute_future_flow_mse_distance(
            neg_cache.flow_pred,
            neg_cache.target_flow,
            history_tokens=neg_cache.hist_tokens,
        )

    sigma = compute_wrong_z_gate_sigma(
        lvp=lvp,
        t=pos_cache.t,
        history_tokens=pos_cache.hist_tokens,
    )
    sigma_hi = float(getattr(args, "wrong_z_sigma_hi", 0.6))
    gate = (sigma <= sigma_hi).to(d_pos.dtype)

    # Use the mismatched branch only as a stop-gradient baseline:
    # improve the positive match without explicitly pushing the negative worse.
    loss_per_sample = F.softplus(d_pos - d_neg)
    train_loss = (gate * loss_per_sample).mean()
    active_loss = (gate * loss_per_sample).sum() / gate.sum().clamp_min(1.0)
    active_frac = gate.mean()
    same_emb_hit_rate = (
        float(same_emb_hits.float().mean().cpu())
        if same_emb_hits is not None
        else 0.0
    )
    neighbor_similarity_mean = (
        float(neighbor_similarity.mean().cpu())
        if neighbor_similarity is not None
        else 0.0
    )

    return train_loss, build_metric_bundle(
        "objective",
        wrong_z_train_loss=float(train_loss.detach().cpu()),
        wrong_z_active_loss=float(active_loss.detach().cpu()),
        wrong_z_active_frac=float(active_frac.detach().cpu()),
        wrong_z_same_emb_hit_rate=same_emb_hit_rate,
        wrong_z_neighbor_similarity_mean=neighbor_similarity_mean,
        wrong_z_rank_loss=float(train_loss.detach().cpu()),
    )


def compute_lvp_cycle_loss(
    *,
    idm: nn.Module,
    latent_actions: torch.Tensor,
    video_pred_lat: torch.Tensor,
    history_frames: int,
    target_frame_tokens: int,
) -> tuple[torch.Tensor, MetricBundle]:
    if history_frames >= target_frame_tokens:
        zero = latent_actions.new_zeros(())
        return zero, build_metric_bundle("objective")

    reidm_module = unwrap_idm_module(idm)
    observations, latent_timesteps = prepare_latent_idm_inputs(reidm_module, video_pred_lat)

    with temporarily_frozen_eval(reidm_module):
        recycled_output = reidm_module(
            observations,
            timesteps=latent_timesteps,
            states=None,
            target_frame_tokens=target_frame_tokens,
        )

    recycled_latent_actions = recycled_output.la
    target_latent_actions = latent_actions.detach()[:, history_frames:target_frame_tokens]
    recycled_future_actions = recycled_latent_actions[:, history_frames:target_frame_tokens]
    cycle_loss = F.mse_loss(
        recycled_future_actions.float(),
        target_latent_actions.float(),
    )
    metrics = build_metric_bundle(
        "objective",
        cycle_loss=float(cycle_loss.detach().cpu()),
        recycled_latent_action_abs=float(
            recycled_future_actions.detach().abs().mean().cpu()
        ),
    )
    return cycle_loss, metrics


def resolve_cycle_target_actions(
    *,
    batch,
    video_lat: torch.Tensor,
    idm: nn.Module,
    target_frame_tokens: int,
) -> torch.Tensor:
    frozen_idm = unwrap_idm_module(idm)
    with temporarily_frozen_eval(frozen_idm):
        cycle_target_actions = run_idm_on_video_lat(
            frozen_idm,
            video_lat.detach(),
            target_frame_tokens=target_frame_tokens,
        )
    return cycle_target_actions.detach()


def get_cycle_loss_scale(step: int, args) -> float:
    if not args.enable_cycle_loss:
        return 0.0
    if args.lvp_action_source != "idm":
        return 0.0
    if args.cycle_loss_weight <= 0:
        return 0.0
    cycle_end_steps = int(getattr(args, "cycle_end_steps", 0))
    if cycle_end_steps > 0 and step > cycle_end_steps:
        return 0.0
    return float(args.cycle_loss_weight) if step > args.cycle_warmup_steps else 0.0


def should_compute_eval_cycle_loss(
    *,
    args,
    idm: nn.Module | None,
) -> bool:
    return (
        idm is not None
        and args.lvp_action_source == "idm"
        and resolve_idm_input_source(idm) == "vae_latent"
    )


def get_wrong_z_loss_scale(step: int, args) -> float:
    if not bool(getattr(args, "wrong_z_enabled", False)):
        return 0.0
    if args.lvp_action_source != "idm":
        return 0.0
    wrong_z_weight = float(getattr(args, "wrong_z_weight", 0.0))
    if wrong_z_weight <= 0:
        return 0.0
    warmup = int(getattr(args, "wrong_z_warmup_steps", 0))
    return wrong_z_weight if step > warmup else 0.0


def compute_cycle_objectives(
    *,
    args,
    step: int,
    training: bool,
    lvp,
    idm: nn.Module | None,
    gt_action_head: nn.Module | None,
    batch,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    lvp_trainable_modules: set[str] | None = None,
    ema: "EMAModel | None" = None,
) -> ObjectiveOutput:
    cycle_only_mode = bool(getattr(args, "cycle_only_mode", False))
    cycle_loss_scale = get_cycle_loss_scale(step, args)
    gt_action_loss_scale = get_gt_action_loss_scale(args)
    wrong_z_loss_scale = get_wrong_z_loss_scale(step, args) if training else (
        float(getattr(args, "wrong_z_weight", 0.0))
        if should_compute_wrong_z_loss(args=args, idm=idm)
        else 0.0
    )
    _, video_lat = encode_lvp_video_latents(
        lvp,
        batch,
        lvp_trainable_modules=lvp_trainable_modules,
    )

    if cycle_only_mode:
        cycle_target_actions = resolve_cycle_target_actions(
            batch=batch,
            video_lat=video_lat,
            idm=idm,
            target_frame_tokens=int(batch.observations.shape[1]),
        )
        if cycle_loss_scale > 0:
            video_pred_lat, history_frames = predict_lvp_clean_video_lat_single_step(
                lvp,
                video_lat,
                cycle_target_actions,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
            )
            cycle_loss, cycle_metrics = compute_lvp_cycle_loss(
                idm=idm,
                latent_actions=cycle_target_actions,
                video_pred_lat=video_pred_lat,
                history_frames=history_frames,
                target_frame_tokens=batch.observations.shape[1],
            )
            cycle_metrics = cycle_metrics.with_updates(cycle_idm_grad=0.0)
        else:
            cycle_loss = video_lat.new_zeros(())
            cycle_metrics = build_metric_bundle(
                "objective",
                cycle_idm_grad=0.0,
            )

        total_loss = cycle_loss_scale * cycle_loss
        metrics = build_metric_bundle(
            "objective",
            recon_loss=0.0,
            condition_action_abs=float(cycle_target_actions.detach().abs().mean().cpu()),
            cycle_loss=cycle_metrics["cycle_loss"],
            recycled_latent_action_abs=cycle_metrics["recycled_latent_action_abs"],
            cycle_idm_grad=cycle_metrics["cycle_idm_grad"],
            gt_action_aux_mse=0.0,
            latent_action_kl_loss=0.0,
            latent_action_vq_loss=0.0,
            latent_action_vq_loss_scale=0.0,
            latent_action_vq_perplexity=0.0,
            wrong_z_loss_scale=0.0,
            wrong_z_train_loss=0.0,
            wrong_z_active_loss=0.0,
            wrong_z_active_frac=0.0,
            wrong_z_same_emb_hit_rate=0.0,
            wrong_z_neighbor_similarity_mean=0.0,
            total_loss=float(total_loss.detach().cpu()),
        )
        return ObjectiveOutput(
            total_loss=total_loss,
            metrics=metrics,
            conditioning_actions=cycle_target_actions,
        )

    conditioning_actions, idm_output = resolve_conditioning_actions(
        action_source=args.lvp_action_source,
        batch=batch,
        video_lat=video_lat,
        idm=idm,
        target_frame_tokens=int(batch.observations.shape[1]),
        return_idm_output=True,
    )

    latent_action_kl_loss_scale = get_latent_action_kl_loss_scale(args)
    latent_action_vq_loss_scale = get_latent_action_vq_loss_scale(args)

    recon_loss, prior_metrics, pos_cache = compute_lvp_recon_loss_from_video_lat(
        lvp,
        video_lat,
        conditioning_actions,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )

    if (
        idm_output is not None
        and getattr(idm_output, "kl_loss", None) is not None
        and latent_action_kl_loss_scale > 0
    ):
        latent_action_kl_loss = idm_output.kl_loss
    else:
        latent_action_kl_loss = recon_loss.new_zeros(())

    if (
        idm_output is not None
        and getattr(idm_output, "vq_loss", None) is not None
        and latent_action_vq_loss_scale > 0
    ):
        latent_action_vq_loss = idm_output.vq_loss
    else:
        latent_action_vq_loss = recon_loss.new_zeros(())

    latent_action_vq_perplexity = 0.0
    if idm_output is not None:
        vq_metrics = getattr(idm_output, "vq_metrics", None)
        if isinstance(vq_metrics, dict):
            latent_action_vq_perplexity = float(vq_metrics.get("perplexity", 0.0) or 0.0)

    if gt_action_loss_scale > 0:
        gt_action_aux_loss, gt_action_aux_metrics = compute_gt_action_aux_loss(
            gt_action_head,
            conditioning_actions,
            batch,
        )
    else:
        gt_action_aux_loss = recon_loss.new_zeros(())
        gt_action_aux_metrics = build_metric_bundle("objective")
    if wrong_z_loss_scale > 0 if training else should_compute_wrong_z_loss(args=args, idm=idm):
        wrong_z_loss, wrong_z_metrics = compute_wrong_z_loss(
            args=args,
            training=training,
            lvp=lvp,
            idm=idm,
            video_lat=video_lat,
            conditioning_actions=conditioning_actions,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            pos_cache=pos_cache,
            embodiment_ids=getattr(batch, "embodiment_id", None),
        )
    else:
        wrong_z_loss = recon_loss.new_zeros(())
        wrong_z_metrics = build_metric_bundle("objective")

    compute_cycle_metric = cycle_loss_scale > 0 if training else should_compute_eval_cycle_loss(
        args=args,
        idm=idm,
    )
    if compute_cycle_metric:
        cycle_target_actions = resolve_cycle_target_actions(
            batch=batch,
            video_lat=video_lat,
            idm=idm,
            target_frame_tokens=int(batch.observations.shape[1]),
        )
        video_pred_lat, history_frames = predict_lvp_clean_video_lat_single_step(
            lvp,
            video_lat,
            cycle_target_actions,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            noise=pos_cache.noise,
            t=pos_cache.t,
        )
        cycle_loss, cycle_metrics = compute_lvp_cycle_loss(
            idm=idm,
            latent_actions=cycle_target_actions,
            video_pred_lat=video_pred_lat,
            history_frames=history_frames,
            target_frame_tokens=batch.observations.shape[1],
        )
        cycle_metrics = cycle_metrics.with_updates(cycle_idm_grad=0.0)
    else:
        cycle_loss = recon_loss.new_zeros(())
        cycle_metrics = build_metric_bundle(
            "objective",
            cycle_idm_grad=0.0,
        )
    total_loss = (
        recon_loss
        + cycle_loss_scale * cycle_loss
        + gt_action_loss_scale * gt_action_aux_loss
        + wrong_z_loss_scale * wrong_z_loss
        + latent_action_kl_loss_scale * latent_action_kl_loss
        + latent_action_vq_loss_scale * latent_action_vq_loss
    )
    metrics = build_metric_bundle(
        "objective",
        recon_loss=prior_metrics["recon_loss"],
        condition_action_abs=prior_metrics["condition_action_abs"],
        cycle_loss=cycle_metrics["cycle_loss"],
        recycled_latent_action_abs=cycle_metrics["recycled_latent_action_abs"],
        cycle_idm_grad=cycle_metrics["cycle_idm_grad"],
        cycle_loss_scale=cycle_loss_scale,
        gt_action_aux_mse=gt_action_aux_metrics["gt_action_aux_mse"],
        gt_action_loss_scale=gt_action_loss_scale,
        latent_action_kl_loss=float(latent_action_kl_loss.detach().cpu()),
        latent_action_kl_loss_scale=latent_action_kl_loss_scale,
        latent_action_vq_loss=float(latent_action_vq_loss.detach().cpu()),
        latent_action_vq_loss_scale=latent_action_vq_loss_scale,
        latent_action_vq_perplexity=latent_action_vq_perplexity,
        wrong_z_loss_scale=wrong_z_loss_scale,
        wrong_z_train_loss=wrong_z_metrics["wrong_z_train_loss"],
        wrong_z_active_loss=wrong_z_metrics["wrong_z_active_loss"],
        wrong_z_active_frac=wrong_z_metrics["wrong_z_active_frac"],
        wrong_z_same_emb_hit_rate=wrong_z_metrics["wrong_z_same_emb_hit_rate"],
        wrong_z_neighbor_similarity_mean=wrong_z_metrics["wrong_z_neighbor_similarity_mean"],
        wrong_z_rank_loss=wrong_z_metrics["wrong_z_rank_loss"],
        total_loss=float(total_loss.detach().cpu()),
    )
    return ObjectiveOutput(
        total_loss=total_loss,
        metrics=metrics,
        conditioning_actions=conditioning_actions,
    )


__all__ = [
    "ObjectiveOutput",
    "build_lvp_condition_latents",
    "compute_cycle_objectives",
    "compute_gt_action_aux_loss",
    "compute_lvp_cycle_loss",
    "compute_lvp_recon_loss_from_video_lat",
    "compute_wrong_z_loss",
    "encode_lvp_video_latents",
    "get_cycle_loss_scale",
    "get_wrong_z_loss_scale",
    "get_gt_action_loss_scale",
    "build_wrong_z_permutation",
    "predict_lvp_clean_video_lat_single_step",
    "project_action_tokens_to_latent_bias",
    "resolve_conditioning_action_dim",
    "resolve_conditioning_actions",
    "resolve_idm_input_source",
    "run_idm_on_video_lat",
    "run_idm_with_input_source",
    "should_compute_eval_cycle_loss",
    "should_compute_wrong_z_loss",
    "unpack_lvp_action_encoder_output",
]
