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

import numpy as np
import torch
from omegaconf import OmegaConf

from infer_action_transfer import (
    batchify_window_sample,
    build_split_records,
    fetch_record,
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
    resolve_conditioning_actions,
    sample_lvp_video,
    select_rgb_channels,
    set_lvp_mode,
    set_seed,
    to_lvp_range,
    trim_batch,
    video_tensor_to_uint8_numpy,
)
from scar.dataloader import Batch, to_device
from scar.models import LatentSpaceIDM
from scar.runtime import load_checkpoint


def log_step(message: str) -> None:
    print(f"[split-export] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export GT-vs-pred videos for samples drawn from a run-defined split "
            "(train/eval/right_target_eval) using an SCAR checkpoint."
        )
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--split",
        choices=["train", "eval", "right_target_eval"],
        default="right_target_eval",
    )
    parser.add_argument(
        "--dataset-filter",
        default="",
        help="Optional substring filter applied to dataset path/name within the selected split.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random-samples", action="store_true")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--random-log-every-records", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run-dir>/split_videos_<ckpt-stem>_<split>/.",
    )
    parser.add_argument("--save-video-fps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-embed-path", default=None)
    parser.add_argument("--negative-prompt-embed-path", default=None)
    return parser.parse_args()


def save_split_outputs(
    *,
    output_dir: Path,
    input_video: torch.Tensor,
    pred_video: torch.Tensor,
    hist_len: int,
    fps: int,
) -> dict[str, str]:
    from utils.video_utils import write_numpy_to_mp4

    output_dir.mkdir(parents=True, exist_ok=True)

    input_np = video_tensor_to_uint8_numpy(input_video[0])
    pred_np = video_tensor_to_uint8_numpy(pred_video[0]).copy()

    # Keep the optional predicted-region white border code for debugging, but
    # disable it for cleaner qualitative exports.
    # if hist_len < pred_np.shape[0]:
    #     pred_np[hist_len:, :2, :, :] = 255
    #     pred_np[hist_len:, -2:, :, :] = 255
    #     pred_np[hist_len:, :, :2, :] = 255
    #     pred_np[hist_len:, :, -2:, :] = 255

    input_vs_pred = np.concatenate([input_np, pred_np], axis=2)

    input_path = output_dir / "input_video.mp4"
    pred_path = output_dir / "pred_video.mp4"
    compare_path = output_dir / "input_vs_pred.mp4"

    write_numpy_to_mp4(input_np, str(input_path), fps=fps)
    write_numpy_to_mp4(pred_np, str(pred_path), fps=fps)
    write_numpy_to_mp4(input_vs_pred, str(compare_path), fps=fps)

    return {
        "input_video": str(input_path),
        "pred_video": str(pred_path),
        "input_vs_pred": str(compare_path),
    }


def run_single_record(
    *,
    sample_idx: int,
    num_samples: int,
    record,
    sample_output_dir: Path,
    device: torch.device,
    seq_len: int,
    lvp,
    idm: torch.nn.Module | None,
    action_source: str,
    lvp_cfg,
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

    if action_source == "idm":
        with torch.no_grad():
            _, video_lat = encode_lvp_video_latents(lvp, batch)
            conditioning_actions = resolve_conditioning_actions(
                action_source=action_source,
                batch=batch,
                video_lat=video_lat.detach(),
                idm=idm,
                target_frame_tokens=seq_len,
            )
    else:
        with torch.no_grad():
            conditioning_actions = resolve_conditioning_actions(
                action_source=action_source,
                batch=batch,
                video_lat=batch.observations,
                idm=None,
                target_frame_tokens=seq_len,
            )

    with torch.no_grad():
        sampling_batch = build_sampling_batch(
            lvp,
            batch,
            conditioning_actions.detach(),
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
    for name, path in saved_paths.items():
        log_step(f"sample {sample_idx + 1}/{num_samples}: {name}={path}")

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
        "conditioning_source": action_source,
        "conditioning_action_abs_mean": float(conditioning_actions.detach().abs().mean().cpu()),
        "saved_paths": saved_paths,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir, ckpt_path = resolve_run_dir_and_ckpt(args)
    artifacts = load_run_artifacts(run_dir, ckpt_path)

    if args.output_dir is None:
        output_dir = run_dir / f"split_videos_{ckpt_path.stem}_{args.split}"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    log_step(f"run_dir={run_dir}")
    log_step(f"ckpt={ckpt_path}")
    log_step(f"output_dir={output_dir}")
    log_step(f"device={device}")

    ckpt_probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    action_source = "idm" if ckpt_probe.get("idm") is not None else "gt"
    conditioning_action_dim = resolve_conditioning_action_dim(
        artifacts.idm_cfg,
        action_source=action_source,
    )
    align_lvp_action_dim(artifacts.lvp_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(artifacts.lvp_cfg)
    if int(artifacts.idm_cfg.data.seq_len) != target_seq_len:
        log_step(
            f"aligning IDM seq_len from {int(artifacts.idm_cfg.data.seq_len)} to {target_seq_len}"
        )
        align_idm_seq_len(artifacts.idm_cfg, target_seq_len)

    records = build_split_records(artifacts, args.split)
    if args.dataset_filter.strip():
        dataset_filter = args.dataset_filter.strip().lower()
        records = [
            record
            for record in records
            if dataset_filter in record.dataset_path.lower()
            or dataset_filter in record.dataset_name.lower()
        ]
        if not records:
            raise ValueError(
                f"No records matched dataset filter {args.dataset_filter!r} in split={args.split!r}."
            )

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
    log_step(
        f"split={args.split}, num_samples={len(selected_records)}, conditioning_source={action_source}, "
        f"action_dim={conditioning_action_dim}, lvp_modules="
        + (",".join(sorted(ckpt_lvp_modules)) if ckpt_lvp_modules else "none")
    )

    lvp = build_lvp_prior(
        artifacts.lvp_cfg,
        device=device,
        trainable_modules=ckpt_lvp_modules,
    )
    idm: torch.nn.Module | None = None
    if action_source == "idm":
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
    if idm is not None:
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
                lvp=lvp,
                idm=idm,
                action_source=action_source,
                lvp_cfg=artifacts.lvp_cfg,
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
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(ckpt_probe.get("step", -1)),
        "split": args.split,
        "dataset_filter": args.dataset_filter,
        "random_samples": bool(args.random_samples),
        "num_samples": int(len(sample_summaries)),
        "seq_len": int(seq_len),
        "hist_len": int(artifacts.lvp_cfg.algorithm.hist_len),
        "la_dim": int(artifacts.idm_cfg.model.la_dim),
        "conditioning_source": action_source,
        "lvp_action_dim": int(artifacts.lvp_cfg.algorithm.action_dim),
        "samples": sample_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_step(f"summary={summary_path}")


if __name__ == "__main__":
    main()
