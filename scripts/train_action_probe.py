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
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import OmegaConf

from infer_action_transfer import load_run_artifacts, resolve_run_dir_and_ckpt
from scar.cycle_api import (
    align_idm_seq_len,
    align_lvp_action_dim,
    build_lvp_prior,
    get_lvp_target_seq_len,
    resolve_conditioning_action_dim,
    set_lvp_mode,
    set_seed,
)
from scar.fixed_subset import build_fixed_window_subset_manifest
from scar.gt_action_probe import (
    GroundTruthActionTransformerProbe,
    _align_probe_sequences,
    _evaluate_probe_dataset,
    _extract_probe_subset_tensors,
    train_and_save_gt_action_probe,
)
from scar.models import LatentSpaceIDM
from scar.runtime import load_checkpoint


def log(message: str) -> None:
    print(f"[gt-probe] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen latent->GT-action probe from an SCAR checkpoint "
            "on a deterministic subset of Robotwin windows."
        )
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--train-split",
        choices=["train", "eval", "right_target_eval"],
        default="train",
        help="Which split to sample the probe training subset from.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "eval", "right_target_eval"],
        default="eval",
        help="Which split to sample the probe evaluation subset from.",
    )
    parser.add_argument(
        "--train-dataset-filter",
        default="",
        help="Optional substring filter on dataset path/name for the probe training subset.",
    )
    parser.add_argument(
        "--eval-dataset-filter",
        default="",
        help="Optional substring filter on dataset path/name for the probe evaluation subset.",
    )
    parser.add_argument(
        "--train-windows-per-dataset",
        type=int,
        default=128,
        help="How many windows to sample from each selected dataset for probe training (<=0 means all).",
    )
    parser.add_argument(
        "--eval-windows-per-dataset",
        type=int,
        default=64,
        help="How many windows to sample from each selected dataset for probe evaluation (<=0 means all).",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed controlling deterministic window sampling.",
    )
    parser.add_argument("--probe-train-steps", type=int, default=400)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-dim-model", type=int, default=256)
    parser.add_argument("--probe-n-heads", type=int, default=4)
    parser.add_argument("--probe-n-layers", type=int, default=2)
    parser.add_argument("--probe-dim-feedforward", type=int, default=1024)
    parser.add_argument("--probe-dropout", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true", help="Log probe metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="", help="Optional wandb project override.")
    parser.add_argument("--wandb-entity", default="", help="Optional wandb entity override.")
    parser.add_argument("--wandb-name", default="", help="Optional wandb run name override.")
    parser.add_argument("--wandb-mode", default="", help="Optional wandb mode override, e.g. online/offline/disabled.")
    parser.add_argument("--wandb-log-every", type=int, default=10, help="Log probe training loss to wandb every N probe steps.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run-dir>/gt_action_probe_<ckpt-stem>_<train-split>_to_<eval-split>/.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_probe_args(
    *,
    args_dict: dict[str, Any],
    subset_manifest_path: Path,
    cli_args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        lvp_action_source="idm",
        data=SimpleNamespace(
            seq_len=int(args_dict.get("seq_len", 49)),
            shift=int(args_dict.get("shift", 1)),
            action_dim=int(args_dict.get("action_dim", 0)),
        ),
        gt_action_probe=SimpleNamespace(
            enabled=True,
            subset_manifest=str(subset_manifest_path.resolve()),
            train_steps=int(cli_args.probe_train_steps),
            lr=float(cli_args.probe_lr),
            weight_decay=float(cli_args.probe_weight_decay),
            batch_size=int(cli_args.probe_batch_size),
            num_workers=0,
            seed=int(cli_args.seed),
            dim_model=int(cli_args.probe_dim_model),
            n_heads=int(cli_args.probe_n_heads),
            n_layers=int(cli_args.probe_n_layers),
            dim_feedforward=int(cli_args.probe_dim_feedforward),
            dropout=float(cli_args.probe_dropout),
        ),
    )


def _build_probe_model_from_checkpoint(
    *,
    probe_checkpoint_path: Path,
    device: torch.device,
) -> GroundTruthActionTransformerProbe:
    payload = torch.load(probe_checkpoint_path, map_location="cpu", weights_only=False)
    probe_config = dict(payload.get("probe_config", {}))
    shape = dict(payload.get("shape", {}))
    probe = GroundTruthActionTransformerProbe(
        latent_action_dim=int(shape["latent_action_dim"]),
        gt_action_dim=int(shape["gt_action_dim"]),
        dim_model=int(probe_config.get("dim_model", 256)),
        n_heads=int(probe_config.get("n_heads", 4)),
        n_layers=int(probe_config.get("n_layers", 2)),
        dim_feedforward=int(probe_config.get("dim_feedforward", 1024)),
        dropout=float(probe_config.get("dropout", 0.1)),
    ).to(device=device)
    probe.load_state_dict(payload["state_dict"])
    probe.eval()
    return probe


def _maybe_init_wandb(
    *,
    cli_args: argparse.Namespace,
    artifacts,
    output_dir: Path,
    ckpt_path: Path,
):
    if not bool(cli_args.wandb):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in the current environment. Install it or rerun without --wandb."
        ) from exc

    args_dict = artifacts.args_dict
    project = cli_args.wandb_project.strip() or str(args_dict.get("wandb_project") or "scar")
    entity = cli_args.wandb_entity.strip() or str(args_dict.get("wandb_entity") or "")
    default_name = f"{Path(artifacts.run_dir).name}__gt_probe__{ckpt_path.stem}__{cli_args.train_split}_to_{cli_args.eval_split}"
    name = cli_args.wandb_name.strip() or default_name
    mode = cli_args.wandb_mode.strip() or str(args_dict.get("wandb_mode") or "online")

    run = wandb.init(
        project=project,
        entity=entity or None,
        name=name,
        mode=mode,
        dir=str(output_dir),
        config={
            "run_dir": str(artifacts.run_dir),
            "ckpt_path": str(ckpt_path),
            "train_split": cli_args.train_split,
            "eval_split": cli_args.eval_split,
            "train_dataset_filter": cli_args.train_dataset_filter,
            "eval_dataset_filter": cli_args.eval_dataset_filter,
            "train_windows_per_dataset": int(cli_args.train_windows_per_dataset),
            "eval_windows_per_dataset": int(cli_args.eval_windows_per_dataset),
            "probe_train_steps": int(cli_args.probe_train_steps),
            "probe_batch_size": int(cli_args.probe_batch_size),
            "probe_lr": float(cli_args.probe_lr),
            "probe_weight_decay": float(cli_args.probe_weight_decay),
            "probe_dim_model": int(cli_args.probe_dim_model),
            "probe_n_heads": int(cli_args.probe_n_heads),
            "probe_n_layers": int(cli_args.probe_n_layers),
            "probe_dim_feedforward": int(cli_args.probe_dim_feedforward),
            "probe_dropout": float(cli_args.probe_dropout),
        },
    )
    wandb.define_metric("probe/train_step")
    wandb.define_metric("probe/train_step_loss", step_metric="probe/train_step")
    wandb.define_metric("probe/ckpt_step")
    return run


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir, ckpt_path = resolve_run_dir_and_ckpt(args)
    artifacts = load_run_artifacts(run_dir, ckpt_path)

    if args.output_dir is None:
        output_dir = run_dir / f"gt_action_probe_{ckpt_path.stem}_{args.train_split}_to_{args.eval_split}"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    log(f"run_dir={run_dir}")
    log(f"ckpt={ckpt_path}")
    log(f"output_dir={output_dir}")
    log(f"device={device}")
    log(
        "probe_splits="
        f"train:{args.train_split}({args.train_dataset_filter or '*'}) -> "
        f"eval:{args.eval_split}({args.eval_dataset_filter or '*'})"
    )

    ckpt_probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt_probe.get("idm") is None:
        raise ValueError("This checkpoint does not contain IDM weights; latent->GT probe requires IDM.")

    conditioning_action_dim = resolve_conditioning_action_dim(
        artifacts.idm_cfg,
        action_source="idm",
    )
    align_lvp_action_dim(artifacts.lvp_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(artifacts.lvp_cfg)
    if int(artifacts.idm_cfg.data.seq_len) != target_seq_len:
        log(f"aligning IDM seq_len from {int(artifacts.idm_cfg.data.seq_len)} to {target_seq_len}")
        align_idm_seq_len(artifacts.idm_cfg, target_seq_len)

    train_manifest_path = build_fixed_window_subset_manifest(
        args_dict=artifacts.args_dict,
        split_name=args.train_split,
        dataset_filter=args.train_dataset_filter,
        windows_per_dataset=args.train_windows_per_dataset,
        subset_seed=args.subset_seed,
        output_manifest_path=output_dir / "train_subset_manifest.json",
    )
    eval_manifest_path = build_fixed_window_subset_manifest(
        args_dict=artifacts.args_dict,
        split_name=args.eval_split,
        dataset_filter=args.eval_dataset_filter,
        windows_per_dataset=args.eval_windows_per_dataset,
        subset_seed=args.subset_seed,
        output_manifest_path=output_dir / "eval_subset_manifest.json",
    )
    log(f"train_subset_manifest={train_manifest_path}")
    log(f"eval_subset_manifest={eval_manifest_path}")

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

    probe_args = build_probe_args(
        args_dict=artifacts.args_dict,
        subset_manifest_path=train_manifest_path,
        cli_args=args,
    )
    wandb_run = _maybe_init_wandb(
        cli_args=args,
        artifacts=artifacts,
        output_dir=output_dir,
        ckpt_path=ckpt_path,
    )

    probe_result = train_and_save_gt_action_probe(
        probe_args,
        global_step=int(ckpt_probe.get("step", -1)),
        output_dir=output_dir,
        device=device,
        idm=idm,
        lvp=lvp,
        log_fn=wandb_run.log if wandb_run is not None else None,
        log_every=max(int(args.wandb_log_every), 0),
    )
    if probe_result is None:
        raise RuntimeError("GT-action probe returned no result.")

    probe_checkpoint_path = output_dir / "checkpoints" / "gt_action_probe_latest.pt"
    probe = _build_probe_model_from_checkpoint(
        probe_checkpoint_path=probe_checkpoint_path,
        device=device,
    )
    eval_probe_args = build_probe_args(
        args_dict=artifacts.args_dict,
        subset_manifest_path=eval_manifest_path,
        cli_args=args,
    )
    eval_latent_actions, eval_gt_actions, _ = _extract_probe_subset_tensors(
        eval_probe_args,
        subset_manifest_path=eval_manifest_path,
        idm=idm,
        lvp=lvp,
        device=device,
    )
    eval_latent_actions, eval_gt_actions = _align_probe_sequences(eval_latent_actions, eval_gt_actions)
    eval_loss, eval_mse, eval_l1 = _evaluate_probe_dataset(
        probe,
        eval_latent_actions,
        eval_gt_actions,
        batch_size=int(args.probe_batch_size),
    )

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(ckpt_probe.get("step", -1)),
        "train_split": str(args.train_split),
        "eval_split": str(args.eval_split),
        "train_dataset_filter": str(args.train_dataset_filter),
        "eval_dataset_filter": str(args.eval_dataset_filter),
        "train_windows_per_dataset": int(args.train_windows_per_dataset),
        "eval_windows_per_dataset": int(args.eval_windows_per_dataset),
        "subset_seed": int(args.subset_seed),
        "probe_train_steps": int(args.probe_train_steps),
        "probe_batch_size": int(args.probe_batch_size),
        "train_metrics": {
            "loss": float(probe_result.train_loss),
            "mse": float(probe_result.train_mse),
            "l1": float(probe_result.train_l1),
        },
        "eval_metrics": {
            "loss": float(eval_loss),
            "mse": float(eval_mse),
            "l1": float(eval_l1),
        },
        "shape": {
            "sequence_length": int(probe_result.sequence_length),
            "latent_action_dim": int(probe_result.latent_action_dim),
            "gt_action_dim": int(probe_result.gt_action_dim),
            "num_train_windows": int(probe_result.num_windows),
            "num_eval_windows": int(eval_latent_actions.shape[0]),
        },
        "train_subset_manifest": str(train_manifest_path),
        "eval_subset_manifest": str(eval_manifest_path),
        "probe_checkpoint": str(probe_checkpoint_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.log(
            {
                "probe/ckpt_step": int(ckpt_probe.get("step", -1)),
                "probe/train_loss": float(probe_result.train_loss),
                "probe/train_mse": float(probe_result.train_mse),
                "probe/train_l1": float(probe_result.train_l1),
                "probe/eval_loss": float(eval_loss),
                "probe/eval_mse": float(eval_mse),
                "probe/eval_l1": float(eval_l1),
                "probe/num_train_windows": int(probe_result.num_windows),
                "probe/num_eval_windows": int(eval_latent_actions.shape[0]),
            }
        )
        wandb_run.finish()

    log(
        "done: "
        f"train_mse={probe_result.train_mse:.6f}, "
        f"train_l1={probe_result.train_l1:.6f}, "
        f"eval_mse={eval_mse:.6f}, "
        f"eval_l1={eval_l1:.6f}, "
        f"train_windows={probe_result.num_windows}, "
        f"eval_windows={int(eval_latent_actions.shape[0])}"
    )
    log(f"summary={summary_path}")


if __name__ == "__main__":
    main()
