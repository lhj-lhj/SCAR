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
from scar.latent_action_controller import (
    evaluate_controller_dataset,
    extract_controller_subset_tensors,
    train_and_save_latent_action_controller,
)
from scar.models import ContextGTActionToLatentActionTransformer, LatentSpaceIDM
from scar.runtime import load_checkpoint


def log(message: str) -> None:
    print(f"[latent-controller] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-teacher controller that predicts latent actions from "
            "visual context and ground-truth actions."
        )
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--train-split",
        choices=["train", "eval", "right_target_eval"],
        default="train",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "eval", "right_target_eval"],
        default="eval",
    )
    parser.add_argument("--train-dataset-filter", default="")
    parser.add_argument("--eval-dataset-filter", default="")
    parser.add_argument(
        "--extra-train-split",
        choices=["", "train", "eval", "right_target_eval"],
        default="",
        help="Optional extra split to append to controller training data.",
    )
    parser.add_argument(
        "--extra-train-dataset-filter",
        default="",
        help="Dataset substring filter for --extra-train-split.",
    )
    parser.add_argument(
        "--extra-train-windows-per-dataset",
        type=int,
        default=0,
        help="Windows per selected dataset for the extra train split (<=0 means all).",
    )
    parser.add_argument(
        "--train-windows-per-dataset",
        type=int,
        default=128,
        help="How many windows to sample from each selected dataset for controller training (<=0 means all).",
    )
    parser.add_argument(
        "--eval-windows-per-dataset",
        type=int,
        default=64,
        help="How many windows to sample from each selected dataset for controller evaluation (<=0 means all).",
    )
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument(
        "--context-len",
        type=int,
        default=0,
        help="Visual context length in frames. <=0 uses the run's LVP hist_len.",
    )
    parser.add_argument("--controller-train-steps", type=int, default=20000)
    parser.add_argument("--controller-eval-every", type=int, default=1000)
    parser.add_argument("--controller-ckpt-every", type=int, default=5000)
    parser.add_argument("--controller-batch-size", type=int, default=16)
    parser.add_argument("--controller-lr", type=float, default=1e-4)
    parser.add_argument("--controller-weight-decay", type=float, default=1e-4)
    parser.add_argument("--controller-dim-model", type=int, default=128)
    parser.add_argument("--controller-n-heads", type=int, default=4)
    parser.add_argument("--controller-n-layers", type=int, default=1)
    parser.add_argument("--controller-dim-feedforward", type=int, default=256)
    parser.add_argument("--controller-dropout", type=float, default=0.1)
    parser.add_argument(
        "--controller-architecture",
        choices=[
            ContextGTActionToLatentActionTransformer.ARCH_LATENT_RESIDUAL_CROSS_ATTENTION,
            ContextGTActionToLatentActionTransformer.ARCH_POINTWISE_MLP,
            ContextGTActionToLatentActionTransformer.ARCH_SUMMARY_ADD,
        ],
        default=ContextGTActionToLatentActionTransformer.ARCH_LATENT_RESIDUAL_CROSS_ATTENTION,
        help=(
            "Controller fusion design. latent_residual_cross_attention maps GT actions "
            "to latent-action space first, then refines them with visual context; "
            "pointwise_mlp predicts each latent action independently from the "
            "corresponding GT action."
        ),
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", default="")
    parser.add_argument("--wandb-log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run-dir>/latent_action_controller_<ckpt-stem>_<train-split>_to_<eval-split>/.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _maybe_init_wandb(
    *,
    cli_args: argparse.Namespace,
    artifacts,
    output_dir: Path,
    ckpt_path: Path,
    context_len: int,
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
    train_label = cli_args.train_split
    if cli_args.extra_train_split:
        train_label = f"{train_label}_plus_{cli_args.extra_train_split}"
    if cli_args.controller_architecture != ContextGTActionToLatentActionTransformer.ARCH_SUMMARY_ADD:
        train_label = f"{train_label}_{cli_args.controller_architecture}"
    default_name = (
        f"{Path(artifacts.run_dir).name}__latent_controller__{ckpt_path.stem}__"
        f"{train_label}_to_{cli_args.eval_split}"
    )
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
            "extra_train_split": cli_args.extra_train_split,
            "eval_split": cli_args.eval_split,
            "train_dataset_filter": cli_args.train_dataset_filter,
            "extra_train_dataset_filter": cli_args.extra_train_dataset_filter,
            "eval_dataset_filter": cli_args.eval_dataset_filter,
            "train_windows_per_dataset": int(cli_args.train_windows_per_dataset),
            "extra_train_windows_per_dataset": int(cli_args.extra_train_windows_per_dataset),
            "eval_windows_per_dataset": int(cli_args.eval_windows_per_dataset),
            "subset_seed": int(cli_args.subset_seed),
            "context_len": int(context_len),
            "controller_train_steps": int(cli_args.controller_train_steps),
            "controller_eval_every": int(cli_args.controller_eval_every),
            "controller_ckpt_every": int(cli_args.controller_ckpt_every),
            "controller_batch_size": int(cli_args.controller_batch_size),
            "controller_lr": float(cli_args.controller_lr),
            "controller_weight_decay": float(cli_args.controller_weight_decay),
            "controller_dim_model": int(cli_args.controller_dim_model),
            "controller_n_heads": int(cli_args.controller_n_heads),
            "controller_n_layers": int(cli_args.controller_n_layers),
            "controller_dim_feedforward": int(cli_args.controller_dim_feedforward),
            "controller_dropout": float(cli_args.controller_dropout),
            "controller_architecture": str(cli_args.controller_architecture),
        },
    )
    wandb.define_metric("controller/train_step")
    wandb.define_metric("controller/train_step_loss", step_metric="controller/train_step")
    wandb.define_metric("controller/eval_step")
    wandb.define_metric("controller/eval_step_*", step_metric="controller/eval_step")
    wandb.define_metric("controller/ckpt_step")
    return run


def build_controller_cfg(
    *,
    args_dict: dict[str, Any],
    subset_manifest_path: Path,
    cli_args: argparse.Namespace,
    context_len: int,
    extra_subset_manifest_paths: list[Path] | None = None,
) -> SimpleNamespace:
    subset_manifests = [str(subset_manifest_path.resolve())]
    subset_manifests.extend(
        str(path.resolve())
        for path in list(extra_subset_manifest_paths or [])
    )
    return SimpleNamespace(
        enabled=True,
        subset_manifest=str(subset_manifest_path.resolve()),
        subset_manifests=subset_manifests,
        train_steps=int(cli_args.controller_train_steps),
        eval_every=int(cli_args.controller_eval_every),
        ckpt_every=int(cli_args.controller_ckpt_every),
        lr=float(cli_args.controller_lr),
        weight_decay=float(cli_args.controller_weight_decay),
        batch_size=int(cli_args.controller_batch_size),
        num_workers=0,
        seed=int(cli_args.seed),
        dim_model=int(cli_args.controller_dim_model),
        n_heads=int(cli_args.controller_n_heads),
        n_layers=int(cli_args.controller_n_layers),
        dim_feedforward=int(cli_args.controller_dim_feedforward),
        dropout=float(cli_args.controller_dropout),
        architecture=str(cli_args.controller_architecture),
        context_len=int(context_len),
        action_dim=int(args_dict.get("action_dim", 0)),
    )


def _build_controller_model_from_checkpoint(
    *,
    controller_checkpoint_path: Path,
    device: torch.device,
) -> ContextGTActionToLatentActionTransformer:
    payload = torch.load(controller_checkpoint_path, map_location="cpu", weights_only=False)
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
    return controller


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir, ckpt_path = resolve_run_dir_and_ckpt(args)
    artifacts = load_run_artifacts(run_dir, ckpt_path)

    default_context_len = int(artifacts.lvp_cfg.algorithm.hist_len)
    context_len = int(args.context_len) if int(args.context_len) > 0 else default_context_len

    train_label = args.train_split
    if args.extra_train_split:
        train_label = f"{train_label}_plus_{args.extra_train_split}"
    if args.controller_architecture != ContextGTActionToLatentActionTransformer.ARCH_SUMMARY_ADD:
        train_label = f"{train_label}_{args.controller_architecture}"
    if args.output_dir is None:
        output_dir = run_dir / (
            f"latent_action_controller_{ckpt_path.stem}_{train_label}_to_{args.eval_split}"
        )
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
    log(f"context_len={context_len}")
    log(f"controller_architecture={args.controller_architecture}")
    log(
        "controller_splits="
        f"train:{args.train_split}({args.train_dataset_filter or '*'}) -> "
        f"eval:{args.eval_split}({args.eval_dataset_filter or '*'})"
    )
    if args.extra_train_split:
        log(
            "controller_extra_train="
            f"{args.extra_train_split}({args.extra_train_dataset_filter or '*'})"
        )

    ckpt_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt_payload.get("idm") is None:
        raise ValueError(
            "This checkpoint does not contain IDM weights; latent-action controller requires IDM."
        )

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
    extra_train_manifest_paths: list[Path] = []
    if args.extra_train_split:
        extra_train_manifest_paths.append(
            build_fixed_window_subset_manifest(
                args_dict=artifacts.args_dict,
                split_name=args.extra_train_split,
                dataset_filter=args.extra_train_dataset_filter,
                windows_per_dataset=args.extra_train_windows_per_dataset,
                subset_seed=args.subset_seed,
                output_manifest_path=output_dir
                / f"extra_train_{args.extra_train_split}_subset_manifest.json",
            )
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
    for extra_manifest_path in extra_train_manifest_paths:
        log(f"extra_train_subset_manifest={extra_manifest_path}")
    log(f"eval_subset_manifest={eval_manifest_path}")

    ckpt_lvp_modules = set(ckpt_payload.get("lvp_trainable_modules", []))
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

    train_controller_cfg = build_controller_cfg(
        args_dict=artifacts.args_dict,
        subset_manifest_path=train_manifest_path,
        extra_subset_manifest_paths=extra_train_manifest_paths,
        cli_args=args,
        context_len=context_len,
    )
    eval_controller_cfg = build_controller_cfg(
        args_dict=artifacts.args_dict,
        subset_manifest_path=eval_manifest_path,
        extra_subset_manifest_paths=[],
        cli_args=args,
        context_len=context_len,
    )
    eval_context_tokens, eval_gt_actions, eval_latent_targets, _ = extract_controller_subset_tensors(
        eval_controller_cfg,
        subset_manifest_path=eval_manifest_path,
        idm=idm,
        lvp=lvp,
        device=device,
    )
    wandb_run = _maybe_init_wandb(
        cli_args=args,
        artifacts=artifacts,
        output_dir=output_dir,
        ckpt_path=ckpt_path,
        context_len=context_len,
    )

    train_result = train_and_save_latent_action_controller(
        train_controller_cfg,
        global_step=int(ckpt_payload.get("step", -1)),
        output_dir=output_dir,
        device=device,
        idm=idm,
        lvp=lvp,
        eval_context_tokens=eval_context_tokens,
        eval_gt_actions=eval_gt_actions,
        eval_latent_targets=eval_latent_targets,
        log_fn=wandb_run.log if wandb_run is not None else None,
        status_fn=log,
        log_every=max(int(args.wandb_log_every), 0),
        ckpt_every=max(int(args.controller_ckpt_every), 0),
    )
    if train_result is None:
        raise RuntimeError("Latent-action controller returned no result.")

    controller_checkpoint_path = output_dir / "checkpoints" / "latent_action_controller_latest.pt"
    controller = _build_controller_model_from_checkpoint(
        controller_checkpoint_path=controller_checkpoint_path,
        device=device,
    )
    eval_mse, eval_future_mse = evaluate_controller_dataset(
        controller,
        eval_context_tokens,
        eval_gt_actions,
        eval_latent_targets,
        batch_size=int(args.controller_batch_size),
        context_len=context_len,
        device=device,
    )

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(ckpt_payload.get("step", -1)),
        "train_split": str(args.train_split),
        "extra_train_split": str(args.extra_train_split),
        "eval_split": str(args.eval_split),
        "train_dataset_filter": str(args.train_dataset_filter),
        "extra_train_dataset_filter": str(args.extra_train_dataset_filter),
        "eval_dataset_filter": str(args.eval_dataset_filter),
        "train_windows_per_dataset": int(args.train_windows_per_dataset),
        "extra_train_windows_per_dataset": int(args.extra_train_windows_per_dataset),
        "eval_windows_per_dataset": int(args.eval_windows_per_dataset),
        "subset_seed": int(args.subset_seed),
        "context_len": int(context_len),
        "controller_train_steps": int(args.controller_train_steps),
        "controller_eval_every": int(args.controller_eval_every),
        "controller_ckpt_every": int(args.controller_ckpt_every),
        "controller_batch_size": int(args.controller_batch_size),
        "controller_architecture": str(args.controller_architecture),
        "train_metrics": {
            "loss": float(train_result.train_loss),
            "mse": float(train_result.train_mse),
            "future_mse": float(train_result.train_future_mse),
        },
        "eval_metrics": {
            "mse": float(eval_mse),
            "future_mse": float(eval_future_mse),
        },
        "shape": {
            "sequence_length": int(train_result.sequence_length),
            "context_sequence_length": int(train_result.context_sequence_length),
            "context_latent_dim": int(train_result.context_latent_dim),
            "latent_action_dim": int(train_result.latent_action_dim),
            "gt_action_dim": int(train_result.gt_action_dim),
            "num_train_windows": int(train_result.num_windows),
            "num_eval_windows": int(eval_latent_targets.shape[0]),
        },
        "train_subset_manifest": str(train_manifest_path),
        "extra_train_subset_manifests": [str(path) for path in extra_train_manifest_paths],
        "eval_subset_manifest": str(eval_manifest_path),
        "controller_checkpoint": str(controller_checkpoint_path),
        "controller_latest_checkpoint": str(train_result.latest_checkpoint),
        "controller_best_eval_checkpoint": str(train_result.best_eval_checkpoint)
        if train_result.best_eval_checkpoint
        else None,
        "controller_best_eval_step": int(train_result.best_eval_step)
        if train_result.best_eval_step is not None
        else None,
        "controller_best_eval_mse": float(train_result.best_eval_mse)
        if train_result.best_eval_mse is not None
        else None,
        "controller_saved_checkpoints": list(train_result.saved_checkpoints),
        "eval_trace": train_result.eval_trace,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.log(
            {
                "controller/ckpt_step": int(ckpt_payload.get("step", -1)),
                "controller/train_loss": float(train_result.train_loss),
                "controller/train_mse": float(train_result.train_mse),
                "controller/train_future_mse": float(train_result.train_future_mse),
                "controller/eval_mse": float(eval_mse),
                "controller/eval_future_mse": float(eval_future_mse),
                "controller/num_train_windows": int(train_result.num_windows),
                "controller/num_eval_windows": int(eval_latent_targets.shape[0]),
                "controller/best_eval_step": int(train_result.best_eval_step)
                if train_result.best_eval_step is not None
                else -1,
                "controller/best_eval_mse": float(train_result.best_eval_mse)
                if train_result.best_eval_mse is not None
                else float("nan"),
            }
        )
        wandb_run.finish()

    log(
        "done: "
        f"train_mse={train_result.train_mse:.6f}, "
        f"train_future_mse={train_result.train_future_mse:.6f}, "
        f"eval_mse={eval_mse:.6f}, "
        f"eval_future_mse={eval_future_mse:.6f}, "
        f"train_windows={train_result.num_windows}, "
        f"eval_windows={int(eval_latent_targets.shape[0])}"
    )
    log(f"summary={summary_path}")


if __name__ == "__main__":
    main()
