"""Cross-embodiment cycle consistency loss.

Data flow
---------
1.  video_primary  → VAE encode → video_lat_primary
2.  IDM_online(video_lat_primary) → z                        [grad]
3.  video_secondary → VAE encode → video_lat_secondary
4.  LVP single-step denoise(video_lat_secondary, z) → video_pred_lat   [grad through LVP]
5.  IDM_ema(video_pred_lat) → z_hat                          [no grad, EMA weights]
6.  Loss = MSE(z_hat, z.detach())

The loss encourages the IDM to learn latent actions that transfer across
embodiments: actions extracted from a Franka video should, when applied to an
Aloha video via the LVP, produce a result from which the (EMA) IDM can recover
the same actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import MetricBundle, build_metric_bundle
from .models import prepare_latent_idm_inputs
from .objectives import (
    ObjectiveOutput,
    build_lvp_condition_latents,
    encode_lvp_video_latents,
    get_latent_action_kl_loss_scale,
    run_idm_on_video_lat,
)
from .runtime import (
    build_prompt_context,
    cast_tensor_tree,
    debug_log_once,
    temporarily_frozen_eval,
    unwrap_idm_module,
)


def predict_cross_clean_video_lat_single_step(
    lvp,
    video_lat_secondary: torch.Tensor,
    conditioning_actions: torch.Tensor,
    *,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
) -> tuple[torch.Tensor, int]:
    """One-step LVP denoising on the *secondary* video conditioned on actions
    from the *primary* video.

    Same maths as ``predict_lvp_clean_video_lat_single_step`` but makes the
    cross-embodiment intent explicit in the name and docstring.
    """
    if lvp.diffusion_type != "continuous":
        raise NotImplementedError(
            "Cross-embodiment cycle helper currently supports only continuous diffusion."
        )

    batch_size = video_lat_secondary.shape[0]
    prompt_embeds = build_prompt_context(
        batch_size=batch_size,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )

    model_param = next(lvp.model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    prompt_embeds = cast_tensor_tree(
        prompt_embeds, device=model_device, dtype=model_dtype,
    )
    conditioning_actions, cond_lat, video_lat_secondary = build_lvp_condition_latents(
        lvp,
        conditioning_actions,
        model_device=model_device,
        model_dtype=model_dtype,
        video_lat=video_lat_secondary,
    )
    noisy_lat, _noise, t = lvp.add_training_noise(video_lat_secondary)
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
        raise ValueError(f"Unsupported timestep shape: {t.shape}")

    pred_clean_lat = noisy_lat.float() - sigma.float() * flow_pred.float()
    hist_tokens = int(lvp.hist_tokens)
    pred_clean_lat[:, :, :hist_tokens] = video_lat_secondary[:, :, :hist_tokens].float()
    history_frames = int(lvp.max_frames - lvp.pred_len)
    return pred_clean_lat, history_frames


def compute_cross_cycle_loss(
    *,
    idm: nn.Module,
    latent_actions: torch.Tensor,
    video_pred_lat: torch.Tensor,
    history_frames: int,
    target_frame_tokens: int,
    ema: "EMAModel | None" = None,
) -> tuple[torch.Tensor, MetricBundle]:
    """MSE between original latent actions and re-encoded actions from the
    cross-embodiment predicted video.

    If *ema* is provided, the re-encoding uses the EMA shadow weights
    (momentum teacher).  Otherwise the online IDM weights are used in
    frozen-eval mode, same as the standard cycle loss.
    """
    if history_frames >= target_frame_tokens:
        zero = latent_actions.new_zeros(())
        return zero, build_metric_bundle("cross_cycle")

    reidm_module = unwrap_idm_module(idm)
    observations, latent_timesteps = prepare_latent_idm_inputs(
        reidm_module, video_pred_lat,
    )

    if ema is not None:
        backup = ema.apply_shadow(reidm_module)
    try:
        with temporarily_frozen_eval(reidm_module):
            recycled_output = reidm_module(
                observations,
                timesteps=latent_timesteps,
                states=None,
                target_frame_tokens=target_frame_tokens,
            )
    finally:
        if ema is not None:
            ema.restore(reidm_module, backup)

    recycled_latent_actions = recycled_output.la
    target_latent_actions = latent_actions.detach()[:, history_frames:target_frame_tokens]
    recycled_future_actions = recycled_latent_actions[:, history_frames:target_frame_tokens]

    loss = F.mse_loss(
        recycled_future_actions.float(),
        target_latent_actions.float(),
    )
    metrics = build_metric_bundle(
        "cross_cycle",
        cross_cycle_loss=float(loss.detach().cpu()),
        cross_recycled_latent_action_abs=float(
            recycled_future_actions.detach().abs().mean().cpu()
        ),
    )
    return loss, metrics


def get_cross_cycle_loss_scale(step: int, args) -> float:
    """Return the effective weight for the cross-embodiment cycle loss at
    the given training *step*, or 0.0 if it should be disabled."""
    if args.lvp_action_source != "idm":
        return 0.0
    weight = float(args.cross_cycle_loss_weight)
    if weight <= 0:
        return 0.0
    warmup = int(args.cross_cycle_warmup_steps)
    return weight if step > warmup else 0.0


def compute_cross_cycle_objectives(
    *,
    args,
    step: int,
    training: bool,
    lvp,
    idm: nn.Module,
    batch_primary,
    batch_secondary,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    lvp_trainable_modules: set[str] | None = None,
    ema: "EMAModel | None" = None,
) -> ObjectiveOutput:
    """Compute the cross-embodiment cycle consistency loss.

    Parameters
    ----------
    batch_primary : Batch
        A batch from the primary (source) dataset.  The IDM extracts latent
        actions *z* from this batch's video.
    batch_secondary : Batch
        A batch from the secondary (target-embodiment) dataset.  The LVP
        conditions on this batch's video to produce predicted future frames,
        which are then re-encoded to recover *z_hat*.
    """
    cross_scale = get_cross_cycle_loss_scale(step, args) if training else (
        float(args.cross_cycle_loss_weight)
        if args.lvp_action_source == "idm" and args.cross_cycle_loss_weight > 0
        else 0.0
    )

    if cross_scale <= 0:
        zero = torch.tensor(0.0, device=next(lvp.model.parameters()).device)
        return ObjectiveOutput(
            total_loss=zero,
            metrics=build_metric_bundle("cross_cycle"),
        )

    # 0. Validate seq_len match between primary and secondary
    primary_seq = int(batch_primary.observations.shape[1])
    secondary_seq = int(batch_secondary.observations.shape[1])
    if primary_seq != secondary_seq:
        raise ValueError(
            f"Cross-cycle seq_len mismatch: primary={primary_seq}, "
            f"secondary={secondary_seq}. Both datasets must use the same seq_len."
        )

    primary_batch_size = int(batch_primary.observations.shape[0])
    secondary_batch_size = int(batch_secondary.observations.shape[0])
    if primary_batch_size != secondary_batch_size:
        raise ValueError(
            "Cross-cycle batch size mismatch: "
            f"primary={primary_batch_size}, secondary={secondary_batch_size}. "
            "Primary and secondary loaders must use the same per-rank batch_size."
        )

    target_frame_tokens = primary_seq

    # 1. Encode primary video → get latent actions from online IDM
    _, video_lat_primary = encode_lvp_video_latents(
        lvp, batch_primary, lvp_trainable_modules=lvp_trainable_modules,
    )
    idm_output = run_idm_on_video_lat(
        idm,
        video_lat_primary.detach(),
        target_frame_tokens=target_frame_tokens,
        return_output=True,
    )
    latent_actions = idm_output.la if hasattr(idm_output, "la") else idm_output

    # 2. Encode secondary video
    _, video_lat_secondary = encode_lvp_video_latents(
        lvp, batch_secondary, lvp_trainable_modules=lvp_trainable_modules,
    )

    # 3. LVP single-step denoise on secondary video conditioned on primary z
    #    NOTE: do NOT detach latent_actions here — gradient must flow
    #    loss → z_hat → video_pred → LVP → cond_lat → action_encoder → z → IDM
    video_pred_lat, history_frames = predict_cross_clean_video_lat_single_step(
        lvp,
        video_lat_secondary,
        latent_actions,
        prompt_embed=prompt_embed,
        prompt_embed_len=prompt_embed_len,
    )

    # 4. Re-encode with EMA IDM → z_hat, compute MSE(z_hat, z.detach())
    #    target side (z) is detached to prevent trivial collapse;
    #    gradient flows through z_hat side only.
    cross_loss, cross_metrics = compute_cross_cycle_loss(
        idm=idm,
        latent_actions=latent_actions,
        video_pred_lat=video_pred_lat,
        history_frames=history_frames,
        target_frame_tokens=target_frame_tokens,
        ema=ema,
    )

    debug_log_once(
        "cross_cycle_shapes",
        f"video_lat_primary={tuple(video_lat_primary.shape)} "
        f"video_lat_secondary={tuple(video_lat_secondary.shape)} "
        f"latent_actions={tuple(latent_actions.shape)} "
        f"video_pred_lat={tuple(video_pred_lat.shape)} "
        f"history_frames={history_frames} "
        f"target_frame_tokens={target_frame_tokens}",
    )

    latent_action_kl_loss_scale = get_latent_action_kl_loss_scale(args)
    if (
        getattr(idm_output, "kl_loss", None) is not None
        and latent_action_kl_loss_scale > 0
    ):
        latent_action_kl_loss = idm_output.kl_loss
    else:
        latent_action_kl_loss = cross_loss.new_zeros(())

    total_loss = cross_scale * cross_loss + latent_action_kl_loss_scale * latent_action_kl_loss
    metrics = build_metric_bundle(
        "cross_cycle",
        cross_cycle_loss=cross_metrics["cross_cycle_loss"],
        cross_cycle_loss_scale=cross_scale,
        cross_latent_action_kl_loss=float(latent_action_kl_loss.detach().cpu()),
        cross_recycled_latent_action_abs=cross_metrics["cross_recycled_latent_action_abs"],
        cross_condition_action_abs=float(latent_actions.detach().abs().mean().cpu()),
    )
    return ObjectiveOutput(total_loss=total_loss, metrics=metrics)


__all__ = [
    "compute_cross_cycle_loss",
    "compute_cross_cycle_objectives",
    "get_cross_cycle_loss_scale",
    "predict_cross_clean_video_lat_single_step",
]
