#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SCAR eval repeatedly for one checkpoint, aggregate mean/std for "
            "all scalar and video metrics, and optionally log the aggregate to WandB."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path to evaluate.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Defaults to a new posthoc_eval_repeat_* "
            "directory under the checkpoint run dir."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of repeated eval runs to aggregate.",
    )
    parser.add_argument(
        "--metric-names",
        default="mse,psnr,ssim,lpips",
        help="Comma-separated video metrics to compute. Defaults to mse,psnr,ssim,lpips.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1000,
        help="Seed offset between repeated eval runs.",
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        default=1,
        help="Delta degrees of freedom for std. Use 1 for sample std over repeated runs.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Optional aggregate WandB run name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        help="Optional WandB mode override (e.g. online, offline, disabled).",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable aggregate WandB logging even if it is enabled in the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for single-process runs (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--cuda-linalg-library",
        default=os.environ.get("CUDA_LINALG_LIBRARY", "magma"),
        choices=("default", "magma", "cusolver"),
        help=(
            "Preferred CUDA linalg backend for torch.linalg operations. "
            "Defaults to magma to avoid intermittent cuSOLVER initialization "
            "errors in the Wan UniPC scheduler."
        ),
    )
    return parser.parse_args()


def _default_output_dir(ckpt_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    return (run_dir / f"posthoc_eval_repeat_{ckpt_path.stem}_{timestamp}").resolve()


def _parse_metric_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("--metric-names must contain at least one metric name.")
    return names


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _configure_cuda_linalg_library(preferred: str) -> None:
    if preferred == "default":
        return
    try:
        import torch
    except ModuleNotFoundError:
        return
    if not torch.cuda.is_available():
        return
    setter = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    if setter is None:
        if _is_main_process():
            print(
                "[repeat-eval] warning: torch.backends.cuda.preferred_linalg_library "
                "is unavailable in this PyTorch build",
                flush=True,
            )
        return
    setter(preferred)
    if _is_main_process():
        print(
            f"[repeat-eval] cuda_linalg_library={torch.backends.cuda.preferred_linalg_library()}",
            flush=True,
        )


def _flatten_result(result) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in result.reduced_metrics.items():
        flat[f"eval/{key}"] = float(value)
    for key, value in result.video_metrics.items():
        flat[f"eval/{key}"] = float(value)
    for key, value in result.right_target_reduced_metrics.items():
        flat[f"eval_right_target/{key}"] = float(value)
    for key, value in result.right_target_video_metrics.items():
        flat[f"eval_right_target/{key}"] = float(value)
    flat["timing/main_eval_seconds"] = float(result.main_eval_seconds)
    flat["timing/right_target_eval_seconds"] = float(result.right_target_eval_seconds)
    flat["timing/total_eval_seconds"] = float(result.total_eval_seconds)
    return flat


def _mean_std(values: list[float], *, ddof: int) -> tuple[float, float]:
    count = len(values)
    if count == 0:
        raise ValueError("Cannot aggregate an empty value list.")
    mean = sum(values) / count
    if count <= ddof:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - ddof)
    return mean, variance ** 0.5


def _aggregate_runs(
    run_metrics: list[dict[str, float]],
    *,
    ddof: int,
) -> dict[str, dict[str, Any]]:
    metric_keys = sorted({key for metrics in run_metrics for key in metrics})
    aggregate: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        values = [metrics[key] for metrics in run_metrics if key in metrics]
        mean, std = _mean_std(values, ddof=ddof)
        aggregate[key] = {
            "mean": mean,
            "std": std,
            "count": len(values),
            "values": values,
        }
    return aggregate


def _log_aggregate_to_wandb(
    *,
    bridge_cfg,
    output_dir: Path,
    ckpt_path: Path,
    cli_args,
    metric_names: tuple[str, ...],
    aggregate: dict[str, dict[str, Any]],
    summary_path: Path,
) -> None:
    wandb_enabled = bool(bridge_cfg.wandb_cfg.enabled) and not cli_args.disable_wandb
    if not wandb_enabled:
        return

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in the current environment. "
            "Install it or rerun with --disable-wandb."
        ) from exc

    wandb_name = cli_args.wandb_name or output_dir.name
    wandb_mode = cli_args.wandb_mode or bridge_cfg.wandb_cfg.mode
    run = wandb.init(
        project=bridge_cfg.wandb_cfg.project,
        entity=bridge_cfg.wandb_cfg.entity,
        name=wandb_name,
        mode=wandb_mode,
        dir=str(output_dir),
        config={
            "config_path": bridge_cfg.config_path,
            "ckpt": str(ckpt_path),
            "num_runs": int(cli_args.num_runs),
            "metric_names": list(metric_names),
            "std_ddof": int(cli_args.std_ddof),
            "summary_path": str(summary_path),
            "args": bridge_cfg.to_flat_dict(),
        },
    )
    payload: dict[str, float] = {
        "repeat_eval/num_runs": float(cli_args.num_runs),
        "repeat_eval/std_ddof": float(cli_args.std_ddof),
    }
    table = wandb.Table(columns=["metric", "mean", "std", "count"])
    for key, stats in aggregate.items():
        payload[f"repeat_eval/{key}_mean"] = float(stats["mean"])
        payload[f"repeat_eval/{key}_std"] = float(stats["std"])
        table.add_data(key, float(stats["mean"]), float(stats["std"]), int(stats["count"]))
    run.log(payload)
    run.log({"repeat_eval/summary_table": table})
    run.summary.update(payload)
    run.finish()


def main() -> None:
    cli_args = _parse_args()
    if cli_args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")
    if cli_args.std_ddof < 0:
        raise ValueError("--std-ddof must be non-negative.")
    _configure_cuda_linalg_library(str(cli_args.cuda_linalg_library))

    from scar.config import load_bridge_config
    from scar.engines import eval_cycle

    ckpt_path = Path(cli_args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    bridge_cfg = load_bridge_config(cli_args.config)
    metric_names = _parse_metric_names(cli_args.metric_names)
    output_dir = (
        Path(cli_args.output_dir).expanduser().resolve()
        if cli_args.output_dir
        else _default_output_dir(ckpt_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(bridge_cfg.train.seed)
    base_cfg = replace(
        bridge_cfg,
        metrics=replace(
            bridge_cfg.metrics,
            video=replace(bridge_cfg.metrics.video, names=metric_names),
        ),
        wandb_cfg=replace(bridge_cfg.wandb_cfg, enabled=False),
        resume_from=str(ckpt_path),
        device=cli_args.device or bridge_cfg.device,
    )

    if _is_main_process():
        print(f"[repeat-eval] config={cli_args.config}", flush=True)
        print(f"[repeat-eval] ckpt={ckpt_path}", flush=True)
        print(f"[repeat-eval] output_dir={output_dir}", flush=True)
        print(f"[repeat-eval] metric_names={','.join(metric_names)}", flush=True)
        print(
            f"[repeat-eval] num_runs={cli_args.num_runs}, "
            f"base_seed={base_seed}, seed_stride={cli_args.seed_stride}",
            flush=True,
        )

    runs: list[dict[str, Any]] = []
    run_metric_dicts: list[dict[str, float]] = []
    for run_index in range(cli_args.num_runs):
        run_seed = base_seed + run_index * int(cli_args.seed_stride)
        run_output_dir = output_dir / f"run_{run_index:02d}"
        run_cfg = replace(
            base_cfg,
            train=replace(
                base_cfg.train,
                output_dir=str(run_output_dir),
                seed=run_seed,
            ),
        )
        if _is_main_process():
            print(
                f"[repeat-eval] starting run {run_index + 1}/{cli_args.num_runs} "
                f"seed={run_seed} output_dir={run_output_dir}",
                flush=True,
            )
        result = eval_cycle(run_cfg)
        flat_metrics = _flatten_result(result)
        run_metric_dicts.append(flat_metrics)
        runs.append(
            {
                "index": run_index,
                "seed": run_seed,
                "output_dir": str(run_output_dir),
                "metrics": flat_metrics,
            }
        )

    if not _is_main_process():
        return

    aggregate = _aggregate_runs(run_metric_dicts, ddof=int(cli_args.std_ddof))
    summary = {
        "config": str(Path(cli_args.config).expanduser().resolve()),
        "ckpt": str(ckpt_path),
        "output_dir": str(output_dir),
        "num_runs": int(cli_args.num_runs),
        "metric_names": list(metric_names),
        "std_ddof": int(cli_args.std_ddof),
        "runs": runs,
        "aggregate": aggregate,
    }
    summary_path = output_dir / "repeat_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[repeat-eval] wrote summary={summary_path}", flush=True)
    for key, stats in aggregate.items():
        print(
            f"[repeat-eval] {key}: mean={stats['mean']:.6f}, "
            f"std={stats['std']:.6f}, count={stats['count']}",
            flush=True,
        )

    _log_aggregate_to_wandb(
        bridge_cfg=replace(
            bridge_cfg,
            metrics=replace(
                bridge_cfg.metrics,
                video=replace(bridge_cfg.metrics.video, names=metric_names),
            ),
        ),
        output_dir=output_dir,
        ckpt_path=ckpt_path,
        cli_args=cli_args,
        metric_names=metric_names,
        aggregate=aggregate,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()
