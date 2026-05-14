from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange

from .runtime import cast_tensor_tree, select_rgb_channels, to_lvp_range, unwrap_module


def infer_obs_shape(batch_np: dict, cfg) -> tuple[int, int, int]:
    observations = batch_np["observations"]
    if observations.ndim < 5:
        raise ValueError(
            f"Expected image observations with shape [B,T,C,H,W], got {observations.shape}. "
            "This SCAR script currently expects image observations."
        )
    return tuple(int(x) for x in observations.shape[-3:])


def build_sampling_batch(
    lvp,
    batch,
    latent_actions: torch.Tensor,
    *,
    prompt_text: str,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor | None,
    negative_prompt_embed_len: int | None,
) -> dict[str, Any]:
    videos = select_rgb_channels(batch.observations)
    videos = to_lvp_range(videos).contiguous().detach()
    conds = latent_actions.detach()
    batch_size = videos.shape[0]
    sampling_batch: dict[str, Any] = {
        "videos": videos,
        "conds": conds,
        "prompts": [prompt_text] * batch_size,
    }
    if negative_prompt_embed is None or negative_prompt_embed_len is None:
        raise ValueError("Negative prompt embeddings are required for eval video sampling.")
    model_param = next(lvp.model.parameters())
    prompt_device = model_param.device
    sampling_batch["prompt_embeds"] = (
        prompt_embed[:prompt_embed_len]
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .to(device=prompt_device)
    )
    sampling_batch["prompt_embed_len"] = torch.full(
        (batch_size,),
        prompt_embed_len,
        dtype=torch.long,
        device=prompt_device,
    )
    sampling_batch["negative_prompt_embeds"] = (
        negative_prompt_embed[:negative_prompt_embed_len]
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .to(device=prompt_device)
    )
    sampling_batch["negative_prompt_embed_len"] = torch.full(
        (batch_size,),
        negative_prompt_embed_len,
        dtype=torch.long,
        device=prompt_device,
    )
    return sampling_batch


def sample_lvp_video(
    lvp,
    sampling_batch: dict[str, Any],
    *,
    seed: int | None = None,
) -> torch.Tensor:
    model_param = next(lvp.model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype
    autocast_enabled = model_device.type in {"cuda", "cpu"} and model_dtype in {
        torch.float16,
        torch.bfloat16,
    }
    autocast_ctx = (
        torch.autocast(device_type=model_device.type, dtype=model_dtype)
        if autocast_enabled
        else contextlib.nullcontext()
    )
    if seed is not None:
        fork_devices: list[int] = []
        if model_device.type == "cuda":
            fork_devices = [model_device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            if model_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with autocast_ctx:
                return lvp.sample_seq(sampling_batch)

    with autocast_ctx:
        return lvp.sample_seq(sampling_batch)


def video_tensor_to_uint8_numpy(video: torch.Tensor) -> Any:
    video = video.detach().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().to(torch.uint8)
    return video.permute(0, 2, 3, 1).cpu().numpy()


def save_eval_videos(
    output_dir: Path,
    step: int,
    video_gt: torch.Tensor,
    video_pred: torch.Tensor,
    *,
    fps: int,
    hist_len: int,
    max_count: int,
    wandb_run=None,
    tag: str = "",
) -> list[Path]:
    from utils.video_utils import write_numpy_to_mp4
    del wandb_run

    eval_video_dir = output_dir / "eval_videos"
    eval_video_dir.mkdir(parents=True, exist_ok=True)

    tag_prefix = f"{tag}_" if tag else ""

    saved_paths: list[Path] = []
    num_samples = min(max_count, video_gt.shape[0], video_pred.shape[0])
    for sample_idx in range(num_samples):
        gt_np = video_tensor_to_uint8_numpy(video_gt[sample_idx])
        pred_np = video_tensor_to_uint8_numpy(video_pred[sample_idx])
        pred_np = pred_np.copy()

        if hist_len < pred_np.shape[0]:
            pred_np[hist_len:, :2, :, :] = 255
            pred_np[hist_len:, -2:, :, :] = 255
            pred_np[hist_len:, :, :2, :] = 255
            pred_np[hist_len:, :, -2:, :] = 255

        comparison = np.concatenate([gt_np, pred_np], axis=2)

        video_path = (
            eval_video_dir
            / f"step_{step:07d}_{tag_prefix}sample_{sample_idx:02d}_compare.mp4"
        )
        write_numpy_to_mp4(comparison, str(video_path), fps=fps)
        saved_paths.append(video_path)

    return saved_paths


def all_gather_object_payload(payload, dist_ctx) -> list[Any]:
    if not dist_ctx.enabled:
        return [payload]
    gathered = [None for _ in range(dist_ctx.world_size)]
    torch.distributed.all_gather_object(gathered, payload)
    return gathered


def rollout_lvp_latent_trainable(
    lvp,
    sampling_batch: dict[str, Any],
    *,
    build_lvp_condition_latents_fn,
    video_lat: torch.Tensor | None = None,
    decode_video: bool = False,
) -> tuple[torch.Tensor, dict[str, Any], int, torch.Tensor | None]:
    working_batch = lvp.clone_batch(sampling_batch)
    lang_guidance = lvp.lang_guidance if lvp.lang_guidance else 0
    hist_guidance = lvp.hist_guidance if lvp.hist_guidance else 0
    lvp.inference_scheduler, lvp.inference_timesteps = lvp.build_scheduler(False)
    model_param = next(lvp.model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    if video_lat is None:
        videos = working_batch["videos"]
        with torch.no_grad():
            video_lat = lvp.encode_video(rearrange(videos, "b t c h w -> b c t h w"))

    prepared_batch = working_batch
    prepared_batch["video_lat"] = video_lat.to(device=model_device, dtype=model_dtype)
    prepared_batch["prompt_embeds"] = cast_tensor_tree(
        prepared_batch["prompt_embeds"],
        device=model_device,
        dtype=model_dtype,
    )
    if "negative_prompt_embeds" in prepared_batch:
        prepared_batch["negative_prompt_embeds"] = cast_tensor_tree(
            prepared_batch["negative_prompt_embeds"],
            device=model_device,
            dtype=model_dtype,
        )
    prepared_batch["clip_embeds"] = None
    prepared_batch["image_embeds"] = None
    conds, cond_lat, conditioned_video_lat = build_lvp_condition_latents_fn(
        lvp,
        prepared_batch["conds"],
        model_device=model_device,
        model_dtype=model_dtype,
        video_lat=prepared_batch["video_lat"],
    )
    prepared_batch["conds"] = conds
    prepared_batch["cond_lat"] = cond_lat
    if conditioned_video_lat is not None:
        prepared_batch["video_lat"] = conditioned_video_lat.to(
            device=model_device,
            dtype=model_dtype,
        )

    clip_embeds = prepared_batch["clip_embeds"]
    image_embeds = prepared_batch["image_embeds"]
    prompt_embeds = prepared_batch["prompt_embeds"]
    video_lat = prepared_batch["video_lat"]
    cond_lat = prepared_batch["cond_lat"]

    batch_size = video_lat.shape[0]
    hist_tokens = int(lvp.hist_tokens)
    video_pred_lat = torch.randn_like(video_lat)
    if lang_guidance:
        neg_prompt_embeds = prepared_batch["negative_prompt_embeds"]

    for t in lvp.inference_timesteps:
        if lvp.diffusion_forcing.enabled:
            video_pred_lat[:, :, :hist_tokens] = video_lat[:, :, :hist_tokens]
            t_expanded = torch.full((batch_size, lvp.lat_t), t, device=lvp.device)
            t_expanded[:, :hist_tokens] = lvp.inference_timesteps[-1]
        else:
            t_expanded = torch.full((batch_size,), t, device=lvp.device)

        pred_cond = lvp.model(
            video_pred_lat,
            t=t_expanded,
            context=prompt_embeds,
            seq_len=lvp.max_tokens,
            clip_fea=clip_embeds,
            y=image_embeds,
            cond=cond_lat,
        )
        if lang_guidance:
            pred_uncond = lvp.model(
                video_pred_lat,
                t=t_expanded,
                context=neg_prompt_embeds,
                seq_len=lvp.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
                cond=cond_lat,
            )
            pred = pred_uncond + lang_guidance * (pred_cond - pred_uncond)
        else:
            pred = pred_cond

        if hist_guidance:
            no_hist_video_pred_lat = video_pred_lat.clone()
            no_hist_video_pred_lat[:, :, :hist_tokens] = torch.randn_like(
                no_hist_video_pred_lat[:, :, :hist_tokens]
            )
            t_hist = t_expanded.clone()
            if t_hist.ndim == 2:
                t_hist[:, :hist_tokens] = lvp.inference_timesteps[0]
            pred_no_hist = lvp.model(
                no_hist_video_pred_lat,
                t=t_hist,
                context=prompt_embeds,
                seq_len=lvp.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
                cond=cond_lat,
            )
            pred = pred_no_hist + hist_guidance * (pred - pred_no_hist)

        video_pred_lat = lvp.inference_scheduler.step(
            pred.float(), t, video_pred_lat.float(), return_dict=False
        )[0]

    video_pred_lat[:, :, :hist_tokens] = video_lat[:, :, :hist_tokens]
    history_frames = int(prepared_batch["videos"].shape[1] - lvp.pred_len)
    decoded_video = None
    if decode_video:
        decoded_video = compose_lvp_video_prediction(
            lvp,
            prepared_batch=prepared_batch,
            video_pred_lat=video_pred_lat,
        )
    return video_pred_lat, prepared_batch, history_frames, decoded_video


def compose_lvp_video_prediction(
    lvp,
    *,
    prepared_batch: dict[str, Any],
    video_pred_lat: torch.Tensor,
) -> torch.Tensor:
    decoded_video = rearrange(lvp.decode_video(video_pred_lat), "b c t h w -> b t c h w")
    video_gt = prepared_batch["videos"]
    return torch.concat(
        [video_gt[:, :-lvp.pred_len], decoded_video[:, -lvp.pred_len:]],
        dim=1,
    )


__all__ = [
    "all_gather_object_payload",
    "build_sampling_batch",
    "compose_lvp_video_prediction",
    "infer_obs_shape",
    "rollout_lvp_latent_trainable",
    "sample_lvp_video",
    "save_eval_videos",
    "video_tensor_to_uint8_numpy",
]
