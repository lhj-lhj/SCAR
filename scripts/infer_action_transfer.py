#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from scar.cycle_api import (
    DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED,
    DEFAULT_LIBERO_PROMPT,
    DEFAULT_LIBERO_PROMPT_EMBED,
    LVP_ROOT,
    SCAR_ROOT,
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
from scar.dataloader import Batch, EpisodeWindowDataset, load_dataset_metadata, load_episodes, to_device
from scar.models import LatentSpaceIDM
from scar.runtime import load_checkpoint


def log_step(message: str) -> None:
    print(f"[transfer] {message}", flush=True)


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    ckpt_path: Path
    args_dict: dict[str, Any]
    idm_cfg: Any
    lvp_cfg: Any


@dataclass(frozen=True)
class WindowRecord:
    split: str
    index: int
    dataset_path: str
    dataset_name: str
    embodiment_id: int
    trajectory_index: int
    episode_index: int
    window_start: int
    sample_np: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer latent actions from a source sample to a target sample using a "
            "SCAR checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Training run directory containing args.json, idm_config.yaml, and lvp_config.yaml.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoints/latest.pt if --run-dir is given.",
    )
    parser.add_argument(
        "--source-split",
        choices=["train", "eval", "right_target_eval"],
        default="eval",
    )
    parser.add_argument(
        "--target-split",
        choices=["train", "eval", "right_target_eval"],
        default="eval",
    )
    parser.add_argument(
        "--source-dataset-filter",
        default="",
        help="Optional substring filter applied to source dataset paths.",
    )
    parser.add_argument(
        "--target-dataset-filter",
        default="",
        help="Optional substring filter applied to target dataset paths.",
    )
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument("--random-pairs", action="store_true")
    parser.add_argument("--num-pairs", type=int, default=1)
    parser.add_argument("--random-log-every-records", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run-dir>/action_transfer_<checkpoint-stem>/.",
    )
    parser.add_argument("--save-video-fps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-embed-path", default=None)
    parser.add_argument("--negative-prompt-embed-path", default=None)
    return parser.parse_args()


def resolve_run_dir_and_ckpt(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir is None and args.ckpt is None:
        raise ValueError("Provide at least one of --run-dir or --ckpt.")

    ckpt_path: Path | None = None
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        assert ckpt_path is not None
        if ckpt_path.parent.name == "checkpoints":
            run_dir = ckpt_path.parent.parent
        else:
            run_dir = ckpt_path.parent

    if ckpt_path is None:
        ckpt_dir = (run_dir / "checkpoints").resolve()
        latest_ckpt = ckpt_dir / "latest.pt"
        if latest_ckpt.is_file():
            ckpt_path = latest_ckpt
        else:
            step_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
            if step_ckpts:
                ckpt_path = step_ckpts[-1].resolve()
                log_step(
                    f"latest.pt not found under {ckpt_dir}; falling back to {ckpt_path.name}"
                )
            else:
                raise FileNotFoundError(
                    f"No checkpoints found under: {ckpt_dir}. "
                    "Expected latest.pt or at least one step_*.pt checkpoint."
                )

    return run_dir, ckpt_path


def maybe_rebase_project_path(path_value: str | None) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value).expanduser()
    if path.exists():
        return str(path.resolve())

    path_str = str(path).replace("\\", "/")
    for repo_name, repo_root in (("SCAR", SCAR_ROOT), ("large-video-planner", LVP_ROOT)):
        needle = f"/{repo_name}/"
        if needle in path_str:
            suffix = path_str.split(needle, 1)[1]
            return str((repo_root / suffix).resolve())
        if path_str.endswith(f"/{repo_name}"):
            return str(repo_root.resolve())
    return str(path)


def rebase_path_list(values: list[str] | None) -> list[str]:
    return [
        str(maybe_rebase_project_path(str(value)))
        for value in list(values or [])
        if str(value).strip()
    ]


def rebase_path_key_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    rebased: dict[str, Any] = {}
    for key, value in (mapping or {}).items():
        rebased[str(maybe_rebase_project_path(str(key)))] = value
    return rebased


def patch_saved_lvp_cfg_paths(lvp_cfg) -> None:
    for attr_chain in (
        ("algorithm", "model", "ckpt_path"),
        ("algorithm", "model", "tuned_ckpt_path"),
        ("algorithm", "vae", "ckpt_path"),
        ("algorithm", "text_encoder", "ckpt_path"),
        ("algorithm", "clip", "ckpt_path"),
    ):
        current = lvp_cfg
        missing = False
        for attr_name in attr_chain[:-1]:
            if attr_name not in current:
                missing = True
                break
            current = current[attr_name]
        if missing:
            continue
        leaf = attr_chain[-1]
        if leaf not in current:
            continue
        current[leaf] = maybe_rebase_project_path(current[leaf])


def load_run_artifacts(run_dir: Path, ckpt_path: Path) -> RunArtifacts:
    args_path = run_dir / "args.json"
    if not args_path.is_file():
        bridge_args_path = run_dir / "bridge_args.json"
        if bridge_args_path.is_file():
            args_path = bridge_args_path
    idm_cfg_path = run_dir / "idm_config.yaml"
    lvp_cfg_path = run_dir / "lvp_config.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(
            f"Neither args.json nor bridge_args.json found under run_dir: {run_dir}"
        )
    if not idm_cfg_path.is_file():
        raise FileNotFoundError(f"idm_config.yaml not found: {idm_cfg_path}")
    if not lvp_cfg_path.is_file():
        raise FileNotFoundError(f"lvp_config.yaml not found: {lvp_cfg_path}")

    args_dict = json.loads(args_path.read_text(encoding="utf-8"))
    args_dict["dataset_paths"] = rebase_path_list(args_dict.get("dataset_paths", []))
    args_dict["eval_dataset_paths"] = rebase_path_list(args_dict.get("eval_dataset_paths", []))
    args_dict["right_target_eval_dataset_paths"] = rebase_path_list(
        args_dict.get("right_target_eval_dataset_paths", [])
    )
    args_dict["dataset_episode_subsets"] = rebase_path_key_mapping(
        args_dict.get("dataset_episode_subsets", {})
    )
    args_dict["eval_dataset_episode_subsets"] = rebase_path_key_mapping(
        args_dict.get("eval_dataset_episode_subsets", {})
    )
    args_dict["right_target_eval_dataset_episode_subsets"] = rebase_path_key_mapping(
        args_dict.get("right_target_eval_dataset_episode_subsets", {})
    )
    args_dict["prompt_embed_path"] = maybe_rebase_project_path(args_dict.get("prompt_embed_path"))
    args_dict["negative_prompt_embed_path"] = maybe_rebase_project_path(
        args_dict.get("negative_prompt_embed_path")
    )

    idm_cfg = OmegaConf.load(idm_cfg_path)
    lvp_cfg = OmegaConf.load(lvp_cfg_path)
    patch_saved_lvp_cfg_paths(lvp_cfg)

    return RunArtifacts(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        args_dict=args_dict,
        idm_cfg=idm_cfg,
        lvp_cfg=lvp_cfg,
    )


def resolve_embodiment_key(metadata: dict[str, Any], dataset_path: str) -> str:
    embodiment_name = str(metadata.get("embodiment", "") or "").strip()
    if embodiment_name:
        return embodiment_name
    return str(Path(dataset_path).resolve())


def build_embodiment_id_mapping(dataset_paths: list[str]) -> dict[str, int]:
    explicit_ids_by_key: dict[str, int] = {}
    unresolved_keys: set[str] = set()
    for dataset_path in dataset_paths:
        metadata = load_dataset_metadata(dataset_path)
        embodiment_key = resolve_embodiment_key(metadata, dataset_path)
        raw_embodiment_id = metadata.get("embodiment_id", None)
        if raw_embodiment_id is None:
            unresolved_keys.add(embodiment_key)
        else:
            explicit_ids_by_key[embodiment_key] = int(raw_embodiment_id)

    embodiment_id_by_key = dict(explicit_ids_by_key)
    next_auto_id = 0
    for embodiment_key in sorted(unresolved_keys):
        if embodiment_key in embodiment_id_by_key:
            continue
        while next_auto_id in embodiment_id_by_key.values():
            next_auto_id += 1
        embodiment_id_by_key[embodiment_key] = next_auto_id
        next_auto_id += 1
    return embodiment_id_by_key


def preprocess_episode(ep: dict[str, np.ndarray], *, target_action_dim: int | None) -> dict[str, np.ndarray]:
    ep = dict(ep)
    if target_action_dim is not None:
        actions = ep["actions"]
        current_dim = actions.shape[-1]
        if current_dim < target_action_dim:
            pad_width = [(0, 0)] * (actions.ndim - 1) + [(0, target_action_dim - current_dim)]
            ep["actions"] = np.pad(actions, pad_width, mode="constant")

    length = ep["actions"].shape[0]
    ep["mask"] = np.ones(length, dtype=np.float32)
    ep["timestep"] = np.arange(length, dtype=np.int64)
    return ep


def batchify_window_sample(sample_np: dict[str, Any]) -> dict[str, Any]:
    """Match DataLoader collation for a single sampled window."""
    batched: dict[str, Any] = {}
    for key, value in sample_np.items():
        if isinstance(value, np.ndarray):
            batched[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            batched[key] = np.asarray([value])
        else:
            batched[key] = value
    return batched


def normalize_subset_mapping(mapping: dict[str, Any] | None) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for dataset_path, raw_indices in (mapping or {}).items():
        normalized[str(Path(dataset_path).resolve())] = [int(index) for index in list(raw_indices or [])]
    return normalized


def build_window_records(
    *,
    dataset_paths: list[str],
    subset_mapping: dict[str, list[int]],
    seq_len: int,
    shift: int,
    target_action_dim: int,
    split_name: str,
    cover_tail: bool = False,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    dataset_paths = [str(Path(path).resolve()) for path in dataset_paths]
    embodiment_id_by_key = build_embodiment_id_mapping(dataset_paths)
    next_trajectory_offset = 0
    next_index = 0

    for dataset_path in dataset_paths:
        metadata = load_dataset_metadata(dataset_path)
        embodiment_key = resolve_embodiment_key(metadata, dataset_path)
        embodiment_id = int(embodiment_id_by_key[embodiment_key])
        selected_indices = subset_mapping.get(dataset_path)
        episodes = load_episodes(dataset_path, episode_indices=selected_indices)
        processed = [
            preprocess_episode(ep, target_action_dim=target_action_dim)
            for ep in episodes
        ]
        dataset = EpisodeWindowDataset(
            processed,
            seq_len=seq_len,
            shift=shift,
            embodiment_id=embodiment_id,
            cover_tail=cover_tail,
        )
        dataset_name = Path(dataset_path).name

        for local_index, (episode_index, window_start) in enumerate(dataset._windows):
            records.append(
                WindowRecord(
                    split=split_name,
                    index=next_index,
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    embodiment_id=embodiment_id,
                    trajectory_index=next_trajectory_offset + int(episode_index),
                    episode_index=int(episode_index),
                    window_start=int(window_start),
                    sample_np=dataset[local_index],
                )
            )
            next_index += 1

        next_trajectory_offset += len(processed)

    return records


def build_split_records(artifacts: RunArtifacts, split_name: str) -> list[WindowRecord]:
    args_dict = artifacts.args_dict
    dataset_paths = list(args_dict.get("dataset_paths", []))
    eval_dataset_paths = list(args_dict.get("eval_dataset_paths", []))
    right_target_paths = list(args_dict.get("right_target_eval_dataset_paths", []))
    target_action_dim = int(args_dict.get("action_dim", 0) or 0)
    seq_len = int(args_dict["seq_len"])
    shift = int(args_dict.get("shift", 1))
    eval_num_trajs = int(args_dict.get("eval_num_trajs", 0))
    dataset_episode_subsets = normalize_subset_mapping(args_dict.get("dataset_episode_subsets", {}))
    eval_episode_subsets = normalize_subset_mapping(args_dict.get("eval_dataset_episode_subsets", {}))
    right_target_subsets = normalize_subset_mapping(
        args_dict.get("right_target_eval_dataset_episode_subsets", {})
    )

    if split_name == "right_target_eval":
        if not right_target_paths:
            raise ValueError("This run has no right_target_eval_dataset_paths.")
        return build_window_records(
            dataset_paths=right_target_paths,
            subset_mapping=right_target_subsets,
            seq_len=seq_len,
            shift=seq_len,
            target_action_dim=target_action_dim,
            split_name=split_name,
            cover_tail=True,
        )

    explicit_eval = bool(eval_dataset_paths)
    if split_name == "train":
        return build_records_from_selected_episodes(
            split_name=split_name,
            dataset_paths=dataset_paths,
            dataset_episode_subsets=dataset_episode_subsets,
            seq_len=seq_len,
            shift=shift,
            target_action_dim=target_action_dim,
            eval_num_trajs=eval_num_trajs,
            explicit_eval=explicit_eval,
            select_train=True,
        )

    if explicit_eval:
        return build_window_records(
            dataset_paths=eval_dataset_paths,
            subset_mapping=eval_episode_subsets,
            seq_len=seq_len,
            shift=seq_len,
            target_action_dim=target_action_dim,
            split_name=split_name,
            cover_tail=True,
        )

    return build_records_from_selected_episodes(
        split_name=split_name,
        dataset_paths=dataset_paths,
        dataset_episode_subsets=dataset_episode_subsets,
        seq_len=seq_len,
        shift=seq_len,
        target_action_dim=target_action_dim,
        eval_num_trajs=eval_num_trajs,
        explicit_eval=False,
        select_train=False,
    )


def build_records_from_selected_episodes(
    *,
    split_name: str,
    dataset_paths: list[str],
    dataset_episode_subsets: dict[str, list[int]],
    seq_len: int,
    shift: int,
    target_action_dim: int,
    eval_num_trajs: int,
    explicit_eval: bool,
    select_train: bool,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    dataset_paths = [str(Path(path).resolve()) for path in dataset_paths]
    embodiment_id_by_key = build_embodiment_id_mapping(dataset_paths)
    next_index = 0
    next_trajectory_offset = 0

    for dataset_path in dataset_paths:
        selected_indices = dataset_episode_subsets.get(dataset_path)
        episodes = load_episodes(dataset_path, episode_indices=selected_indices)
        processed = [preprocess_episode(ep, target_action_dim=target_action_dim) for ep in episodes]
        if explicit_eval:
            selected_eps = processed if select_train else []
        else:
            n_eval = min(eval_num_trajs // max(len(dataset_paths), 1), len(processed))
            if eval_num_trajs <= 0:
                n_eval = 0
            train_eps = processed[:-n_eval] if n_eval > 0 else processed
            eval_eps = processed[-n_eval:] if n_eval > 0 else []
            selected_eps = train_eps if select_train else eval_eps
        metadata = load_dataset_metadata(dataset_path)
        embodiment_key = resolve_embodiment_key(metadata, dataset_path)
        embodiment_id = int(embodiment_id_by_key[embodiment_key])
        dataset = EpisodeWindowDataset(
            selected_eps,
            seq_len=seq_len,
            shift=shift,
            embodiment_id=embodiment_id,
            cover_tail=not select_train,
        )
        dataset_name = Path(dataset_path).name
        for local_index, (episode_index, window_start) in enumerate(dataset._windows):
            records.append(
                WindowRecord(
                    split=split_name,
                    index=next_index,
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    embodiment_id=embodiment_id,
                    trajectory_index=next_trajectory_offset + int(episode_index),
                    episode_index=int(episode_index),
                    window_start=int(window_start),
                    sample_np=dataset[local_index],
                )
            )
            next_index += 1
        next_trajectory_offset += len(selected_eps)

    return records


def filter_records(
    records: list[WindowRecord],
    dataset_filter: str,
    *,
    split_name: str | None = None,
) -> list[WindowRecord]:
    dataset_filter = dataset_filter.strip()
    if not dataset_filter:
        return records
    dataset_filter = dataset_filter.lower()
    split_label = split_name or (records[0].split if records else "unknown")
    filtered = [
        record
        for record in records
        if dataset_filter in record.dataset_path.lower()
        or dataset_filter in record.dataset_name.lower()
    ]
    if not filtered:
        available_datasets = sorted({record.dataset_name for record in records})
        hint = ""
        if split_label in {"eval", "right_target_eval"}:
            hint = (
                " This split usually only contains target-side evaluation datasets. "
                "If you want source samples from a non-target embodiment such as aloha/arx/ur5, "
                "use --source-split train."
            )
        raise ValueError(
            f"No records matched dataset filter {dataset_filter!r}. "
            f"Available datasets in split='{split_label}': {available_datasets}.{hint}"
        )
    return [
        WindowRecord(
            split=record.split,
            index=new_index,
            dataset_path=record.dataset_path,
            dataset_name=record.dataset_name,
            embodiment_id=record.embodiment_id,
            trajectory_index=record.trajectory_index,
            episode_index=record.episode_index,
            window_start=record.window_start,
            sample_np=record.sample_np,
        )
        for new_index, record in enumerate(filtered)
    ]


def fetch_record(
    records: list[WindowRecord],
    index: int,
    *,
    exclude_trajectory_indices: set[int] | None = None,
    split_name: str,
) -> WindowRecord:
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    excluded = set(exclude_trajectory_indices or [])
    kept_index = 0
    for record in records:
        if record.trajectory_index in excluded:
            continue
        if kept_index == index:
            if record.index != index:
                log_step(
                    f"adjusted split='{split_name}' requested index {index} to {record.index} "
                    f"to avoid trajectory ids {sorted(excluded)}"
                )
            return record
        kept_index += 1
    raise IndexError(
        f"Requested sample index {index} from split='{split_name}' but only "
        f"{kept_index} record(s) remain after excluding trajectory ids {sorted(excluded)}."
    )


def sample_random_records_by_trajectory(
    records: list[WindowRecord],
    *,
    count: int,
    seed: int,
    split_name: str,
    log_every_records: int,
) -> list[WindowRecord]:
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    trajectory_to_record: dict[int, WindowRecord] = {}
    trajectory_counts: dict[int, int] = {}
    log_step(
        f"scanning split='{split_name}' for {count} random example(s) from distinct "
        f"trajectory ids (seed={seed})"
    )
    for seen_count, record in enumerate(records, start=1):
        traj = int(record.trajectory_index)
        seen_for_traj = trajectory_counts.get(traj, 0) + 1
        trajectory_counts[traj] = seen_for_traj
        should_replace = traj not in trajectory_to_record
        if not should_replace:
            should_replace = int(rng.integers(seen_for_traj)) == 0
        if should_replace:
            trajectory_to_record[traj] = record
        if log_every_records > 0 and seen_count % log_every_records == 0:
            log_step(
                f"split='{split_name}' random sampling progress: "
                f"scanned_records={seen_count}, unique_trajectories={len(trajectory_to_record)}"
            )

    sampled = list(trajectory_to_record.values())
    if len(sampled) < count:
        raise ValueError(
            f"Requested {count} random example(s) from distinct trajectories in "
            f"split='{split_name}', but only {len(sampled)} unique trajectories are available."
        )
    rng.shuffle(sampled)
    log_step(
        f"finished scanning split='{split_name}': "
        f"scanned_records={len(records)}, unique_trajectories={len(sampled)}"
    )
    return sampled[:count]


def prepare_pair_records(
    args: argparse.Namespace,
    source_records: list[WindowRecord],
    target_records: list[WindowRecord],
) -> list[dict[str, WindowRecord]]:
    if args.random_pairs:
        if args.num_pairs < 1:
            raise ValueError(f"--num-pairs must be >= 1, got {args.num_pairs}")
        rng = np.random.default_rng(args.seed)
        if args.source_split == args.target_split and args.source_dataset_filter == args.target_dataset_filter:
            sampled = sample_random_records_by_trajectory(
                source_records,
                count=2 * args.num_pairs,
                seed=int(rng.integers(0, 2**31 - 1)),
                split_name=args.source_split,
                log_every_records=args.random_log_every_records,
            )
            rng.shuffle(sampled)
            return [
                {
                    "source": sampled[2 * pair_idx],
                    "target": sampled[2 * pair_idx + 1],
                }
                for pair_idx in range(args.num_pairs)
            ]

        sampled_source = sample_random_records_by_trajectory(
            source_records,
            count=args.num_pairs,
            seed=int(rng.integers(0, 2**31 - 1)),
            split_name=args.source_split,
            log_every_records=args.random_log_every_records,
        )
        sampled_target = sample_random_records_by_trajectory(
            target_records,
            count=args.num_pairs,
            seed=int(rng.integers(0, 2**31 - 1)),
            split_name=args.target_split,
            log_every_records=args.random_log_every_records,
        )
        return [
            {"source": sampled_source[pair_idx], "target": sampled_target[pair_idx]}
            for pair_idx in range(args.num_pairs)
        ]

    if args.num_pairs != 1:
        raise ValueError("--num-pairs > 1 requires --random-pairs.")
    source_record = fetch_record(source_records, args.source_index, split_name=args.source_split)
    excluded = (
        {int(source_record.trajectory_index)}
        if args.source_split == args.target_split and args.source_dataset_filter == args.target_dataset_filter
        else None
    )
    target_record = fetch_record(
        target_records,
        args.target_index,
        exclude_trajectory_indices=excluded,
        split_name=args.target_split,
    )
    return [{"source": source_record, "target": target_record}]


def save_action_transfer_outputs(
    *,
    output_dir: Path,
    source_video: torch.Tensor,
    target_video: torch.Tensor,
    transferred_video: torch.Tensor,
    hist_len: int,
    fps: int,
) -> dict[str, str]:
    from utils.video_utils import write_numpy_to_mp4

    output_dir.mkdir(parents=True, exist_ok=True)

    source_np = video_tensor_to_uint8_numpy(source_video[0])
    target_np = video_tensor_to_uint8_numpy(target_video[0])
    pred_np = video_tensor_to_uint8_numpy(transferred_video[0]).copy()

    if hist_len < pred_np.shape[0]:
        pred_np[hist_len:, :2, :, :] = 255
        pred_np[hist_len:, -2:, :, :] = 255
        pred_np[hist_len:, :, :2, :] = 255
        pred_np[hist_len:, :, -2:, :] = 255

    compare_np = np.concatenate([target_np, pred_np], axis=2)
    triptych_np = np.concatenate([source_np, target_np, pred_np], axis=2)

    source_path = output_dir / "source_video.mp4"
    target_path = output_dir / "target_video.mp4"
    pred_path = output_dir / "transferred_video.mp4"
    compare_path = output_dir / "target_vs_transferred.mp4"
    triptych_path = output_dir / "source_target_transferred_triptych.mp4"

    write_numpy_to_mp4(source_np, str(source_path), fps=fps)
    write_numpy_to_mp4(target_np, str(target_path), fps=fps)
    write_numpy_to_mp4(pred_np, str(pred_path), fps=fps)
    write_numpy_to_mp4(compare_np, str(compare_path), fps=fps)
    write_numpy_to_mp4(triptych_np, str(triptych_path), fps=fps)

    return {
        "source_video": str(source_path),
        "target_video": str(target_path),
        "transferred_video": str(pred_path),
        "target_vs_transferred": str(compare_path),
        "source_target_transferred_triptych": str(triptych_path),
    }


def run_action_transfer_pair(
    *,
    pair_idx: int,
    num_pairs: int,
    pair_output_dir: Path,
    source_record: WindowRecord,
    target_record: WindowRecord,
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
        f"pair {pair_idx + 1}/{num_pairs}: "
        f"source=({source_record.split}, idx={source_record.index}, "
        f"dataset={source_record.dataset_name}, traj={source_record.trajectory_index}), "
        f"target=({target_record.split}, idx={target_record.index}, "
        f"dataset={target_record.dataset_name}, traj={target_record.trajectory_index})"
    )

    source_batch = Batch(**to_device(batchify_window_sample(source_record.sample_np), device))
    target_batch = Batch(**to_device(batchify_window_sample(target_record.sample_np), device))
    source_batch = trim_batch(source_batch, seq_len)
    target_batch = trim_batch(target_batch, seq_len)

    if action_source == "idm":
        with torch.no_grad():
            _, source_video_lat = encode_lvp_video_latents(lvp, source_batch)
            source_conditioning_actions = resolve_conditioning_actions(
                action_source=action_source,
                batch=source_batch,
                video_lat=source_video_lat.detach(),
                idm=idm,
                target_frame_tokens=seq_len,
            )
    else:
        with torch.no_grad():
            source_conditioning_actions = resolve_conditioning_actions(
                action_source=action_source,
                batch=source_batch,
                video_lat=source_batch.observations,
                idm=None,
                target_frame_tokens=seq_len,
            )

    with torch.no_grad():
        sampling_batch = build_sampling_batch(
            lvp,
            target_batch,
            source_conditioning_actions.detach(),
            prompt_text=prompt_text,
            prompt_embed=prompt_embed,
            prompt_embed_len=prompt_embed_len,
            negative_prompt_embed=negative_prompt_embed,
            negative_prompt_embed_len=negative_prompt_embed_len,
        )
        transferred_video = sample_lvp_video(lvp, sampling_batch, seed=seed)

    source_video = to_lvp_range(select_rgb_channels(source_batch.observations))
    target_video = to_lvp_range(select_rgb_channels(target_batch.observations))

    saved_paths = save_action_transfer_outputs(
        output_dir=pair_output_dir,
        source_video=source_video,
        target_video=target_video,
        transferred_video=transferred_video,
        hist_len=int(lvp_cfg.algorithm.hist_len),
        fps=fps,
    )
    for name, path in saved_paths.items():
        log_step(f"pair {pair_idx + 1}/{num_pairs}: {name}={path}")

    return {
        "pair_index": int(pair_idx),
        "source": {
            "split": source_record.split,
            "index": int(source_record.index),
            "dataset_path": source_record.dataset_path,
            "dataset_name": source_record.dataset_name,
            "embodiment_id": int(source_record.embodiment_id),
            "trajectory_index": int(source_record.trajectory_index),
            "episode_index": int(source_record.episode_index),
            "window_start": int(source_record.window_start),
        },
        "target": {
            "split": target_record.split,
            "index": int(target_record.index),
            "dataset_path": target_record.dataset_path,
            "dataset_name": target_record.dataset_name,
            "embodiment_id": int(target_record.embodiment_id),
            "trajectory_index": int(target_record.trajectory_index),
            "episode_index": int(target_record.episode_index),
            "window_start": int(target_record.window_start),
        },
        "conditioning_source": action_source,
        "conditioning_action_abs_mean": float(source_conditioning_actions.detach().abs().mean().cpu()),
        "saved_paths": saved_paths,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir, ckpt_path = resolve_run_dir_and_ckpt(args)
    artifacts = load_run_artifacts(run_dir, ckpt_path)

    if args.output_dir is None:
        output_dir = run_dir / f"action_transfer_{ckpt_path.stem}"
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
    conditioning_source = "idm" if ckpt_probe.get("idm") is not None else "gt"
    conditioning_action_dim = resolve_conditioning_action_dim(
        artifacts.idm_cfg,
        action_source=conditioning_source,
    )
    align_lvp_action_dim(artifacts.lvp_cfg, conditioning_action_dim)
    target_seq_len = get_lvp_target_seq_len(artifacts.lvp_cfg)
    if int(artifacts.idm_cfg.data.seq_len) != target_seq_len:
        log_step(
            f"aligning IDM seq_len from {int(artifacts.idm_cfg.data.seq_len)} to {target_seq_len}"
        )
        align_idm_seq_len(artifacts.idm_cfg, target_seq_len)

    source_records = filter_records(
        build_split_records(artifacts, args.source_split),
        args.source_dataset_filter,
        split_name=args.source_split,
    )
    target_records = filter_records(
        build_split_records(artifacts, args.target_split),
        args.target_dataset_filter,
        split_name=args.target_split,
    )
    pair_records = prepare_pair_records(args, source_records, target_records)

    first_sample_np = pair_records[0]["source"].sample_np
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
        f"conditioning_source={conditioning_source}, action_dim={conditioning_action_dim}, "
        f"num_pairs={len(pair_records)}, lvp_modules="
        + (",".join(sorted(ckpt_lvp_modules)) if ckpt_lvp_modules else "none")
    )

    lvp = build_lvp_prior(
        artifacts.lvp_cfg,
        device=device,
        trainable_modules=ckpt_lvp_modules,
    )
    idm: torch.nn.Module | None = None
    if conditioning_source == "idm":
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

    pair_summaries = []
    for pair_idx, pair_record in enumerate(pair_records):
        pair_output_dir = output_dir / f"pair_{pair_idx:03d}"
        pair_summaries.append(
            run_action_transfer_pair(
                pair_idx=pair_idx,
                num_pairs=len(pair_records),
                pair_output_dir=pair_output_dir,
                source_record=pair_record["source"],
                target_record=pair_record["target"],
                device=device,
                seq_len=seq_len,
                lvp=lvp,
                idm=idm,
                action_source=conditioning_source,
                lvp_cfg=artifacts.lvp_cfg,
                prompt_embed=prompt_embed,
                prompt_embed_len=prompt_embed_len,
                negative_prompt_embed=negative_prompt_embed,
                negative_prompt_embed_len=negative_prompt_embed_len,
                prompt_text=prompt_text,
                fps=args.save_video_fps,
                seed=args.seed + pair_idx,
            )
        )

    summary = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(ckpt_probe.get("step", -1)),
        "source_split": args.source_split,
        "target_split": args.target_split,
        "source_dataset_filter": args.source_dataset_filter,
        "target_dataset_filter": args.target_dataset_filter,
        "random_pairs": bool(args.random_pairs),
        "num_pairs": int(len(pair_summaries)),
        "seq_len": int(seq_len),
        "hist_len": int(artifacts.lvp_cfg.algorithm.hist_len),
        "la_dim": int(artifacts.idm_cfg.model.la_dim),
        "conditioning_source": conditioning_source,
        "lvp_action_dim": int(artifacts.lvp_cfg.algorithm.action_dim),
        "pairs": pair_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_step(f"summary={summary_path}")


if __name__ == "__main__":
    main()
