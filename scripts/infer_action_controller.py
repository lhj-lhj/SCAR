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

from export_split_videos import save_split_outputs
from infer_action_transfer import (
    batchify_window_sample,
    build_split_records,
    fetch_record,
    filter_records,
    load_run_artifacts,
    maybe_rebase_project_path,
    resolve_run_dir_and_ckpt,
    sample_random_records_by_trajectory,
)
from scar.cycle_api import (
    DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED,
    DEFAULT_LIBERO_PROMPT,
    DEFAULT_LIBERO_PROMPT_EMBED,
    align_idm_seq_len,
    align_lvp_action_dim,
    build_lvp_prior,
    build_sampling_batch,
    encode_lvp_video_latents,
    get_lvp_target_seq_len,
    load_prompt_embedding,
    resolve_conditioning_action_dim,
    run_idm_on_video_lat,
    sample_lvp_video,
    select_rgb_channels,
    set_lvp_mode,
    set_seed,
    to_lvp_range,
    trim_batch,
)
from scar.dataloader import Batch, to_device
from scar.gt_action_probe import extract_gt_action_sequence
from scar.models import ContextGTActionToLatentActionTransformer, LatentSpaceIDM
from scar.runtime import load_checkpoint


def log_step(message: str) -> None:
    print(f"[controller-fdm] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a latent-action controller checkpoint on a target split and feed the predicted "
            "latent actions into LVP/FDM to export GT-vs-pred videos."
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
        "--split",
        choices=["train", "eval", "right_target_eval"],
        default="right_target_eval",
    )
    parser.add_argument(
        "--dataset-filter",
        default="franka",
        help="Optional substring filter applied to target dataset path/name.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random-samples", action="store_true")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--random-log-every-records", type=int, default=5000)
    parser.add_argument(
        "--context-len",
        type=int,
        default=0,
        help="Override controller context length in frames. <=0 uses the checkpoint setting.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <controller-run-dir>/controller_fdm_<controller-ckpt-stem>_<split>/.",
    )
    parser.add_argument(
        "--latent-source",
        choices=["controller", "noise"],
        default="controller",
        help="Use controller-predicted latent actions or a pure-noise latent baseline.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Standard deviation for Gaussian noise when --latent-source noise.",
    )
    parser.add_argument(
        "--noise-match-teacher-scale",
        action="store_true",
        help="If set, scale noise per sample to match the teacher latent-action std.",
    )
    parser.add_argument("--save-video-fps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-embed-path", default=None)
    parser.add_argument("--negative-prompt-embed-path", default=None)
    return parser.parse_args()


def _select_deterministic_latent_actions(idm_output: Any) -> torch.Tensor:
    la_mean = getattr(idm_output, "la_mean", None)
    if torch.is_tensor(la_mean):
        return la_mean
    latent_actions = getattr(idm_output, "la", None)
    if torch.is_tensor(latent_actions):
        return latent_actions
    raise ValueError("IDM output does not contain a latent action tensor.")


def _align_sequences(
    gt_actions: torch.Tensor,
    latent_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_actions.ndim == 2:
        gt_actions = gt_actions.unsqueeze(1)
    target_len = min(gt_actions.shape[1], latent_targets.shape[1])
    if target_len <= 0:
        raise ValueError("Controller inference requires a non-empty action trajectory.")
    return gt_actions[:, :target_len], latent_targets[:, :target_len]


def _spatial_pool_context_tokens(video_lat: torch.Tensor) -> torch.Tensor:
    if video_lat.ndim != 5:
        raise ValueError(
            "Expected context video latents to have shape [B, C, T, H, W], got "
            f"{tuple(video_lat.shape)}"
        )
    return video_lat.mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()


def _load_controller_checkpoint(controller_ckpt_path: Path, device: torch.device) -> tuple[dict[str, Any], ContextGTActionToLatentActionTransformer]:
    payload = torch.load(controller_ckpt_path, map_location="cpu", weights_only=False)
    controller_config = dict(payload.get("controller_config", {}))
    shape = dict(payload.get("shape", {}))
    controller = ContextGTActionToLatentActionTransformer(
        context_latent_dim=int(shape["context_latent_dim"]),
        gt_action_dim=int(shape["gt_action_dim"]),
        latent_action_dim=int(shape["latent_action_dim"]),
        dim_model=int(controller_config.get("dim_model", 128)),
        n_heads=int(controller_config.get("n_heads", 4)),
        n_layers=int(controller_config.get("n_layers", 1)),
        dim_feedforward=int(controller_config.get("dim_feedforward", 256)),
        dropout=float(controller_config.get("dropout", 0.1)),
        architecture=str(
            controller_config.get(
                "architecture",
                ContextGTActionToLatentActionTransformer.ARCH_SUMMARY_ADD,
            )
        ),
    ).to(device=device)
    controller.load_state_dict(payload["state_dict"])
    controller.eval()
    return payload, controller


def _load_controller_summary(controller_ckpt_path: Path) -> tuple[Path | None, dict[str, Any]]:
    controller_run_dir = controller_ckpt_path.parent.parent
    summary_path = controller_run_dir / "summary.json"
    if not summary_path.is_file():
        return None, {}
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_backbone_run_and_ckpt(
    *,
    args: argparse.Namespace,
    controller_summary: dict[str, Any],
) -> tuple[Path, Path]:
    if args.run_dir is not None or args.ckpt is not None:
        return resolve_run_dir_and_ckpt(args)
    run_dir_value = controller_summary.get("run_dir")
    ckpt_value = controller_summary.get("ckpt_path")
    if not run_dir_value or not ckpt_value:
        raise ValueError(
            "Unable to infer SCAR run/ckpt from controller summary. "
            "Provide --run-dir and/or --ckpt explicitly."
        )
    run_dir = Path(str(run_dir_value)).resolve()
    ckpt_path = Path(str(ckpt_value)).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Inferred run_dir not found: {run_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Inferred ckpt not found: {ckpt_path}")
    return run_dir, ckpt_path


def _save_latent_payload(
    sample_output_dir: Path,
    *,
    pred_latent_actions: torch.Tensor,
    teacher_latent_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    context_len: int,
) -> str:
    latent_path = sample_output_dir / "latent_payload.pt"
    torch.save(
        {
            "pred_latent_actions": pred_latent_actions.detach().cpu(),
            "teacher_latent_actions": teacher_latent_actions.detach().cpu(),
            "gt_actions": gt_actions.detach().cpu(),
            "context_len": int(context_len),
        },
        latent_path,
    )
    return str(latent_path)


def _build_noise_latent_actions(
    *,
    teacher_latent_actions: torch.Tensor,
    noise_std: float,
    match_teacher_scale: bool,
) -> torch.Tensor:
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")
    noise = torch.randn_like(teacher_latent_actions)
    if match_teacher_scale:
        teacher_scale = teacher_latent_actions.detach().float().std(dim=(1, 2), keepdim=True)
        teacher_scale = teacher_scale.clamp_min(1e-6).to(
            device=noise.device,
            dtype=noise.dtype,
        )
        noise = noise * teacher_scale
    if noise_std != 1.0:
        noise = noise * float(noise_std)
    return noise


def run_single_record(
    *,
    sample_idx: int,
    num_samples: int,
    record,
    sample_output_dir: Path,
    device: torch.device,
    seq_len: int,
    context_len: int,
    controller: ContextGTActionToLatentActionTransformer | None,
    lvp,
    idm: torch.nn.Module,
    lvp_cfg,
    latent_source: str,
    noise_std: float,
    noise_match_teacher_scale: bool,
    prompt_embed: torch.Tensor,
    prompt_embed_len: int,
    negative_prompt_embed: torch.Tensor,
    negative_prompt_embed_len: int,
    prompt_text: str,
    fps: int,
    seed: int,
) -> dict[str, Any]:
    log_step(
        f"sample {sample_idx + 1}/{num_samples}: split={record.split}, idx={record.index}, "
        f"dataset={record.dataset_name}, traj={record.trajectory_index}"
    )

    batch = Batch(**to_device(batchify_window_sample(record.sample_np), device))
    batch = trim_batch(batch, seq_len)

    with torch.no_grad():
        _, full_video_lat = encode_lvp_video_latents(lvp, batch)
        videos = select_rgb_channels(batch.observations)
        videos = to_lvp_range(videos).contiguous()
        context_video_lat = lvp.encode_video(videos[:, :context_len].permute(0, 2, 1, 3, 4))
        context_tokens = _spatial_pool_context_tokens(context_video_lat)

        idm_output = run_idm_on_video_lat(
            idm,
            full_video_lat,
            target_frame_tokens=int(batch.observations.shape[1]),
            return_output=True,
        )
        teacher_latent_actions = _select_deterministic_latent_actions(idm_output)
        gt_actions = extract_gt_action_sequence(batch.__dict__)
        gt_actions, teacher_latent_actions = _align_sequences(gt_actions, teacher_latent_actions)
        if latent_source == "controller":
            if controller is None:
                raise ValueError("controller model is required when latent_source='controller'.")
            pred_latent_actions = controller(context_tokens, gt_actions)
        elif latent_source == "noise":
            pred_latent_actions = _build_noise_latent_actions(
                teacher_latent_actions=teacher_latent_actions,
                noise_std=noise_std,
                match_teacher_scale=noise_match_teacher_scale,
            )
        else:
            raise ValueError(f"Unsupported latent_source: {latent_source}")

        sampling_batch = build_sampling_batch(
            lvp,
            batch,
            pred_latent_actions,
            prompt_text=prompt_text,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            negative_prompt_embed=negative_prompt_embed,
            negative_prompt_embed_len=negative_prompt_embed_len,
        )
        pred_video = sample_lvp_video(lvp, sampling_batch, seed=seed)

    input_video = to_lvp_range(select_rgb_channels(batch.observations))
    saved_paths = save_split_outputs(
        output_dir=sample_output_dir,
        input_video=input_video,
        pred_video=pred_video,
        hist_len=int(lvp_cfg.algorithm.hist_len),
        fps=fps,
    )
    latent_payload_path = _save_latent_payload(
        sample_output_dir,
        pred_latent_actions=pred_latent_actions,
        teacher_latent_actions=teacher_latent_actions,
        gt_actions=gt_actions,
        context_len=context_len,
    )

    latent_mse = float(
        torch.nn.functional.mse_loss(pred_latent_actions, teacher_latent_actions).detach().cpu()
    )
    future_latent_mse = 0.0
    if teacher_latent_actions.shape[1] > context_len:
        future_latent_mse = float(
            torch.nn.functional.mse_loss(
                pred_latent_actions[:, context_len:],
                teacher_latent_actions[:, context_len:],
            ).detach().cpu()
        )

    for name, path in saved_paths.items():
        log_step(f"sample {sample_idx + 1}/{num_samples}: {name}={path}")
    log_step(
        f"sample {sample_idx + 1}/{num_samples}: latent_mse={latent_mse:.6f}, "
        f"future_latent_mse={future_latent_mse:.6f}, latent_payload={latent_payload_path}"
    )

    return {
        "sample_index": int(sample_idx),
        "split": record.split,
        "index": int(record.index),
        "dataset_path": record.dataset_path,
        "dataset_name": record.dataset_name,
        "embodiment_id": int(record.embodiment_id),
        "trajectory_index": int(record.trajectory_index),
        "episode_index": int(record.episode_index),
        "window_start": int(record.window_start),
        "context_len": int(context_len),
        "latent_source": str(latent_source),
        "pred_latent_action_abs_mean": float(pred_latent_actions.detach().abs().mean().cpu()),
        "teacher_latent_action_abs_mean": float(teacher_latent_actions.detach().abs().mean().cpu()),
        "latent_mse": latent_mse,
        "future_latent_mse": future_latent_mse,
        "saved_paths": saved_paths,
        "latent_payload_path": latent_payload_path,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    controller_ckpt_path = Path(args.controller_ckpt).resolve()
    if not controller_ckpt_path.is_file():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_ckpt_path}")

    controller_summary_path, controller_summary = _load_controller_summary(controller_ckpt_path)
    run_dir, ckpt_path = _resolve_backbone_run_and_ckpt(
        args=args,
        controller_summary=controller_summary,
    )
    artifacts = load_run_artifacts(run_dir, ckpt_path)

    if args.output_dir is None:
        controller_run_dir = controller_ckpt_path.parent.parent
        output_dir = controller_run_dir / (
            f"controller_fdm_{controller_ckpt_path.stem}_{args.latent_source}_{args.split}"
        )
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    controller_payload, controller = _load_controller_checkpoint(controller_ckpt_path, device)
    default_context_len = int(controller_payload.get("controller_config", {}).get("context_len", 0) or 0)
    if default_context_len <= 0:
        default_context_len = int(controller_summary.get("context_len", 0) or 0)
    if default_context_len <= 0:
        default_context_len = int(artifacts.lvp_cfg.algorithm.hist_len)
    context_len = int(args.context_len) if int(args.context_len) > 0 else int(default_context_len)

    log_step(f"controller_ckpt={controller_ckpt_path}")
    if controller_summary_path is not None:
        log_step(f"controller_summary={controller_summary_path}")
    log_step(f"run_dir={run_dir}")
    log_step(f"ckpt={ckpt_path}")
    log_step(f"output_dir={output_dir}")
    log_step(f"device={device}")
    log_step(f"context_len={context_len}")
    log_step(
        "controller_architecture="
        f"{controller_payload.get('controller_config', {}).get('architecture', 'summary_add')}"
    )
    log_step(
        f"latent_source={args.latent_source}, noise_std={float(args.noise_std):.4f}, "
        f"noise_match_teacher_scale={bool(args.noise_match_teacher_scale)}"
    )

    ckpt_probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt_probe.get("idm") is None:
        raise ValueError("Base checkpoint does not contain IDM weights; controller inference requires IDM.")

    conditioning_action_dim = resolve_conditioning_action_dim(
        artifacts.idm_cfg,
        action_source="idm",
    )
    align_lvp_action_dim(artifacts.lvp_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(artifacts.lvp_cfg)
    if int(artifacts.idm_cfg.data.seq_len) != target_seq_len:
        log_step(f"aligning IDM seq_len from {int(artifacts.idm_cfg.data.seq_len)} to {target_seq_len}")
        align_idm_seq_len(artifacts.idm_cfg, target_seq_len)

    records = build_split_records(artifacts, args.split)
    records = filter_records(records, args.dataset_filter, split_name=args.split)

    if args.random_samples:
        selected_records = sample_random_records_by_trajectory(
            records,
            count=args.num_samples,
            seed=args.seed,
            split_name=args.split,
            log_every_records=args.random_log_every_records,
        )
    else:
        if args.num_samples != 1:
            raise ValueError("--num-samples > 1 requires --random-samples.")
        selected_records = [fetch_record(records, args.index, split_name=args.split)]

    first_sample_np = selected_records[0].sample_np
    seq_len = int(first_sample_np["observations"].shape[0])
    if seq_len != target_seq_len:
        raise RuntimeError(f"Expected seq_len={target_seq_len}, got {seq_len}.")
    if context_len <= 0 or context_len > seq_len:
        raise ValueError(f"context_len must be in [1, {seq_len}], got {context_len}")

    prompt_text = (
        args.prompt
        or artifacts.args_dict.get("prompt")
        or DEFAULT_LIBERO_PROMPT
    )
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

    sample_summaries = []
    for sample_idx, record in enumerate(selected_records):
        sample_output_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_summaries.append(
            run_single_record(
                sample_idx=sample_idx,
                num_samples=len(selected_records),
                record=record,
                sample_output_dir=sample_output_dir,
                device=device,
                seq_len=seq_len,
                context_len=context_len,
                controller=controller,
                lvp=lvp,
                idm=idm,
                lvp_cfg=artifacts.lvp_cfg,
                latent_source=args.latent_source,
                noise_std=float(args.noise_std),
                noise_match_teacher_scale=bool(args.noise_match_teacher_scale),
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                negative_prompt_embed=negative_prompt_embed,
                negative_prompt_embed_len=negative_prompt_embed_len,
                prompt_text=prompt_text,
                fps=args.save_video_fps,
                seed=args.seed + sample_idx,
            )
        )

    summary = {
        "controller_ckpt": str(controller_ckpt_path),
        "controller_summary": str(controller_summary_path) if controller_summary_path is not None else None,
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(ckpt_probe.get("step", -1)),
        "split": args.split,
        "dataset_filter": args.dataset_filter,
        "random_samples": bool(args.random_samples),
        "num_samples": int(len(sample_summaries)),
        "seq_len": int(seq_len),
        "context_len": int(context_len),
        "latent_source": str(args.latent_source),
        "noise_std": float(args.noise_std),
        "noise_match_teacher_scale": bool(args.noise_match_teacher_scale),
        "hist_len": int(artifacts.lvp_cfg.algorithm.hist_len),
        "la_dim": int(artifacts.idm_cfg.model.la_dim),
        "lvp_action_dim": int(artifacts.lvp_cfg.algorithm.action_dim),
        "samples": sample_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_step(f"summary={summary_path}")


if __name__ == "__main__":
    main()
