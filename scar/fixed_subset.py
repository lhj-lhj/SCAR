from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .dataloader import (
    EpisodeWindowDataset,
    _preprocess_episode,
    load_dataset_metadata,
    load_episodes,
)


def _stable_seed(*parts: str | int) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _normalize_subset_mapping(mapping: dict[str, Any] | None) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for dataset_path, raw_indices in (mapping or {}).items():
        normalized[str(Path(dataset_path).resolve())] = [
            int(index) for index in list(raw_indices or [])
        ]
    return normalized


def select_split_dataset_paths(
    args_dict: dict[str, Any],
    split_name: str,
) -> tuple[list[str], dict[str, list[int]], bool]:
    dataset_paths = [str(Path(path).resolve()) for path in list(args_dict.get("dataset_paths", []))]
    eval_dataset_paths = [
        str(Path(path).resolve()) for path in list(args_dict.get("eval_dataset_paths", []))
    ]
    right_target_paths = [
        str(Path(path).resolve())
        for path in list(args_dict.get("right_target_eval_dataset_paths", []) or [])
    ]

    dataset_episode_subsets = _normalize_subset_mapping(args_dict.get("dataset_episode_subsets", {}))
    eval_episode_subsets = _normalize_subset_mapping(args_dict.get("eval_dataset_episode_subsets", {}))
    right_target_subsets = _normalize_subset_mapping(
        args_dict.get("right_target_eval_dataset_episode_subsets", {})
    )
    explicit_eval = bool(eval_dataset_paths)

    if split_name == "right_target_eval":
        return right_target_paths, right_target_subsets, True
    if split_name == "eval" and explicit_eval:
        return eval_dataset_paths, eval_episode_subsets, True
    return dataset_paths, dataset_episode_subsets, explicit_eval


def filter_dataset_paths(dataset_paths: list[str], dataset_filter: str) -> list[str]:
    dataset_filter = dataset_filter.strip().lower()
    if not dataset_filter:
        return dataset_paths
    filtered = [
        path
        for path in dataset_paths
        if dataset_filter in path.lower() or dataset_filter in Path(path).name.lower()
    ]
    if not filtered:
        raise ValueError(
            f"No datasets matched filter {dataset_filter!r}. "
            f"Available datasets: {sorted(Path(path).name for path in dataset_paths)}"
        )
    return filtered


def select_episodes_for_split(
    *,
    split_name: str,
    explicit_eval: bool,
    dataset_paths: list[str],
    dataset_path: str,
    subset_mapping: dict[str, list[int]],
    args_dict: dict[str, Any],
    target_action_dim: int | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    selected_indices = subset_mapping.get(dataset_path)
    all_episode_indices = (
        [int(index) for index in selected_indices] if selected_indices is not None else None
    )
    episodes = load_episodes(dataset_path, episode_indices=all_episode_indices)
    processed = [_preprocess_episode(ep, target_action_dim=target_action_dim) for ep in episodes]

    if explicit_eval:
        selected_episode_indices = (
            all_episode_indices if all_episode_indices is not None else list(range(len(processed)))
        )
        if split_name in {"train", "eval", "right_target_eval"}:
            return processed, selected_episode_indices

    eval_num_trajs = int(args_dict.get("eval_num_trajs", 0))
    n_eval = min(eval_num_trajs // max(len(dataset_paths), 1), len(processed))
    if eval_num_trajs <= 0:
        n_eval = 0
    train_eps = processed[:-n_eval] if n_eval > 0 else processed
    eval_eps = processed[-n_eval:] if n_eval > 0 else []
    full_episode_indices = (
        all_episode_indices if all_episode_indices is not None else list(range(len(processed)))
    )
    train_episode_indices = full_episode_indices[:-n_eval] if n_eval > 0 else full_episode_indices
    eval_episode_indices = full_episode_indices[-n_eval:] if n_eval > 0 else []
    if split_name == "train":
        return train_eps, train_episode_indices
    return eval_eps, eval_episode_indices


def load_fixed_window_subset_manifest(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fixed subset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("tasks"), list):
        raise ValueError(f"Invalid fixed subset manifest: missing tasks list in {path}")
    return payload


def build_fixed_window_subset_manifest(
    *,
    args_dict: dict[str, Any],
    split_name: str,
    dataset_filter: str,
    windows_per_dataset: int,
    subset_seed: int,
    output_manifest_path: Path,
) -> Path:
    dataset_paths, subset_mapping, explicit_eval = select_split_dataset_paths(args_dict, split_name)
    dataset_paths = filter_dataset_paths(dataset_paths, dataset_filter)
    if not dataset_paths:
        raise ValueError(f"No dataset paths available for split={split_name!r}")

    seq_len = int(args_dict.get("seq_len", 49))
    shift = int(args_dict.get("shift", 1))
    target_action_dim = int(args_dict.get("action_dim", 0)) or None
    cover_tail = split_name != "train"
    window_shift = shift if split_name == "train" else seq_len

    task_entries: list[dict[str, Any]] = []
    total_selected_windows = 0
    for dataset_path in dataset_paths:
        metadata = load_dataset_metadata(dataset_path)
        processed, selected_episode_indices = select_episodes_for_split(
            split_name=split_name,
            explicit_eval=explicit_eval,
            dataset_paths=dataset_paths,
            dataset_path=dataset_path,
            subset_mapping=subset_mapping,
            args_dict=args_dict,
            target_action_dim=target_action_dim,
        )
        dataset = EpisodeWindowDataset(
            processed,
            seq_len=seq_len,
            shift=window_shift,
            embodiment_id=int(metadata.get("embodiment_id", 0)),
            cover_tail=cover_tail,
        )
        num_total_windows = int(len(dataset))
        requested = int(windows_per_dataset)
        count = num_total_windows if requested <= 0 else min(requested, num_total_windows)
        rng = np.random.default_rng(
            _stable_seed(
                "robotwin_gt_probe_subset",
                subset_seed,
                split_name,
                dataset_path,
                seq_len,
                shift,
            )
        )
        if count > 0:
            selected_window_indices = sorted(
                int(index)
                for index in rng.choice(num_total_windows, size=count, replace=False).tolist()
            )
        else:
            selected_window_indices = []
        total_selected_windows += len(selected_window_indices)
        task_entries.append(
            {
                "task_name": str(metadata.get("task_name") or Path(dataset_path).name),
                "dataset_path": str(Path(dataset_path).resolve()),
                "embodiment_id": int(metadata.get("embodiment_id", 0)),
                "selected_episode_indices": [int(index) for index in selected_episode_indices],
                "num_total_windows": num_total_windows,
                "num_selected_windows": int(len(selected_window_indices)),
                "selected_window_indices": selected_window_indices,
            }
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_output_dir": str(args_dict.get("output_dir", "")),
        "split": split_name,
        "dataset_filter": dataset_filter,
        "subset_seed": int(subset_seed),
        "seq_len": int(seq_len),
        "shift": int(window_shift),
        "cover_tail": bool(cover_tail),
        "windows_per_task": int(windows_per_dataset),
        "num_tasks": int(len(task_entries)),
        "num_selected_windows_total": int(total_selected_windows),
        "tasks": task_entries,
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_manifest_path


__all__ = [
    "build_fixed_window_subset_manifest",
    "filter_dataset_paths",
    "load_fixed_window_subset_manifest",
    "select_episodes_for_split",
    "select_split_dataset_paths",
]
