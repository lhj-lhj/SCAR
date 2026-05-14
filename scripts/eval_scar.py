#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Load an SCAR checkpoint and run eval-only (including right_target_eval)."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path to evaluate.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to a new posthoc_eval_* directory under the checkpoint run dir.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Optional wandb run name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        help="Optional wandb mode override (e.g. online, offline, disabled).",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb even if it is enabled in the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for single-process runs (e.g. cuda, cuda:0, cpu).",
    )
    return parser.parse_args()


def _default_output_dir(ckpt_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    return (run_dir / f"posthoc_eval_{ckpt_path.stem}_{timestamp}").resolve()


def main() -> None:
    cli_args = _parse_args()

    from scar.config import load_bridge_config
    from scar.engines import eval_cycle

    ckpt_path = Path(cli_args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    bridge_cfg = load_bridge_config(cli_args.config)
    output_dir = (
        Path(cli_args.output_dir).expanduser().resolve()
        if cli_args.output_dir
        else _default_output_dir(ckpt_path)
    )
    wandb_name = cli_args.wandb_name or output_dir.name
    wandb_mode = cli_args.wandb_mode or bridge_cfg.wandb_cfg.mode
    wandb_enabled = bool(bridge_cfg.wandb_cfg.enabled) and not cli_args.disable_wandb

    bridge_cfg = replace(
        bridge_cfg,
        train=replace(bridge_cfg.train, output_dir=str(output_dir)),
        wandb_cfg=replace(
            bridge_cfg.wandb_cfg,
            enabled=wandb_enabled,
            name=wandb_name,
            mode=wandb_mode,
        ),
        resume_from=str(ckpt_path),
        device=cli_args.device or bridge_cfg.device,
    )

    print(f"[cycle-eval-script] config={cli_args.config}", flush=True)
    print(f"[cycle-eval-script] ckpt={ckpt_path}", flush=True)
    print(f"[cycle-eval-script] output_dir={output_dir}", flush=True)
    print(
        f"[cycle-eval-script] wandb_enabled={wandb_enabled}, "
        f"wandb_name={wandb_name}, wandb_mode={wandb_mode}",
        flush=True,
    )

    eval_cycle(bridge_cfg)


if __name__ == "__main__":
    main()
