#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib
import os
import time

_CYCLE_API_MODULE = None


def _load_cycle_api():
    global _CYCLE_API_MODULE
    if _CYCLE_API_MODULE is None:
        _CYCLE_API_MODULE = importlib.import_module("scar.cycle_api")
    return _CYCLE_API_MODULE


def __getattr__(name: str):
    return getattr(_load_cycle_api(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_load_cycle_api())))


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print("[cycle-script] startup: importing scar.engines...", flush=True)
    import_start = time.time()
    from scar.engines import main_cycle

    if rank == 0:
        print(
            "[cycle-script] startup: importing scar.engines done in "
            f"{time.time() - import_start:.2f}s",
            flush=True,
        )
    main_cycle()


if __name__ == "__main__":
    main()
