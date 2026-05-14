from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PACKAGE_ROOT = Path(__file__).resolve().parent
SCAR_ROOT = PACKAGE_ROOT.parent
SCRIPTS_ROOT = SCAR_ROOT / "scripts"
CONFIG_ROOT = SCAR_ROOT / "configs"

_default_lvp_root = SCAR_ROOT.parent / "large-video-planner"
LVP_ROOT = Path(os.environ.get("SCAR_LVP_ROOT", str(_default_lvp_root))).expanduser()

CYCLE_TRAIN_SCRIPT_PATH = SCRIPTS_ROOT / "train_scar.py"

DEFAULT_LIBERO_PROMPT = "Generate a video of the robot"
DEFAULT_LIBERO_PROMPT_EMBED = (
    LVP_ROOT / "data" / "meta_data" / "robosuite_default_prompt.pt"
)
DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED = (
    LVP_ROOT / "data" / "meta_data" / "robosuite_default_neg_prompt.pt"
)


def ensure_repo_paths() -> None:
    for repo_root in (str(LVP_ROOT), str(SCAR_ROOT), str(SCRIPTS_ROOT)):
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)


ensure_repo_paths()


__all__ = [
    "CYCLE_TRAIN_SCRIPT_PATH",
    "DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED",
    "DEFAULT_LIBERO_PROMPT",
    "DEFAULT_LIBERO_PROMPT_EMBED",
    "CONFIG_ROOT",
    "LVP_ROOT",
    "PACKAGE_ROOT",
    "SCRIPTS_ROOT",
    "SCAR_ROOT",
    "ensure_repo_paths",
]
