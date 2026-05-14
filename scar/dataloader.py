"""
PyTorch-native dataloader for the SCAR training pipeline.

Replaces the TF-based ``idm.utils.dataloader`` for this training pipeline only.
Other SCAR trainers are unaffected and continue using the TF pipeline.

Features over the TF pipeline:
- No TF/PyTorch mixed-mode bugs (StopIteration, repeat, etc.)
- DistributedSampler shards *indices* not data → 1x memory instead of Nx
- Standard ``num_workers`` prefetching
- ``embodiment_id`` injected per-sample natively
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Batch dataclass + device transfer (standalone, no legacy training-package dependency)
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """Training batch. All fields are torch.Tensors on the target device."""
    states: Any
    observations: Any
    actions: Any
    rewards: Any = None
    next_observations: Any = None
    dones: Any = None
    tasks: Any = None
    mask: Any = None
    timestep: Any = None
    scene_obs: Any = None
    traj_index: Any = None
    latent_actions: Any = None
    is_first: Any = None
    is_last: Any = None
    is_terminal: Any = None
    discount: Any = None
    images: Any = None
    embeddings: Any = None
    env_state: Any = None
    embodiment_id: Any = None


def to_device(x, device):
    """Convert numpy arrays / dicts of arrays to torch tensors on ``device``."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {
            k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray)
            else v.to(device) if isinstance(v, torch.Tensor)
            else v
            for k, v in x.items()
        }
    return x


# ---------------------------------------------------------------------------
# Episode loading (still uses TF to deserialize the on-disk format)
# ---------------------------------------------------------------------------

def _normalize_episode_indices(
    episode_indices: Sequence[int],
    *,
    num_episodes: int,
    dataset_path: str | Path,
) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in episode_indices:
        index = int(raw_index)
        if index < 0 or index >= num_episodes:
            raise IndexError(
                f"Episode index {index} out of range for dataset {dataset_path} "
                f"with {num_episodes} episodes"
            )
        if index in seen:
            continue
        seen.add(index)
        normalized.append(index)
    return normalized


def load_episodes(
    dataset_path: str | Path,
    *,
    episode_indices: Sequence[int] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Load all episodes from an LMDB database (mmap, multi-process safe)."""
    import lmdb
    import pickle

    ds_path = Path(dataset_path)
    lmdb_path = ds_path / "episodes.lmdb"
    if not lmdb_path.is_dir():
        raise FileNotFoundError(
            f"episodes.lmdb not found in: {ds_path}\n"
            f"Run data_transfer_robotwin.py to convert your data first."
        )

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
    episodes = []
    with env.begin() as txn:
        num_episodes = int(txn.get(b"__len__").decode())
        if episode_indices is None:
            selected_indices = list(range(num_episodes))
        else:
            selected_indices = _normalize_episode_indices(
                episode_indices,
                num_episodes=num_episodes,
                dataset_path=ds_path,
            )
        for i in selected_indices:
            raw = txn.get(f"episode_{i:04d}".encode())
            if raw is None:
                raise KeyError(f"Missing episode_{i:04d} in LMDB dataset: {ds_path}")
            ep = pickle.loads(raw)
            if ep["observations"].shape[0] <= 2:
                continue
            episodes.append(ep)
    env.close()
    return episodes


def load_dataset_metadata(dataset_path: str | Path) -> dict[str, Any]:
    metadata_path = Path(dataset_path) / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_embodiment_key(
    metadata: dict[str, Any],
    dataset_path: str | Path,
) -> str:
    embodiment_name = str(metadata.get("embodiment", "") or "").strip()
    if embodiment_name:
        return embodiment_name
    return str(Path(dataset_path).resolve())


def _build_embodiment_id_mapping(
    dataset_paths: Sequence[str | Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    metadata_by_path: dict[str, dict[str, Any]] = {}
    explicit_ids_by_key: dict[str, int] = {}
    unresolved_keys: set[str] = set()

    for dataset_path in dataset_paths:
        resolved_path = str(Path(dataset_path).resolve())
        metadata = load_dataset_metadata(resolved_path)
        metadata_by_path[resolved_path] = metadata
        embodiment_key = _resolve_embodiment_key(metadata, resolved_path)
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

    return metadata_by_path, embodiment_id_by_key


def _normalize_episode_subset_mapping(
    mapping: dict[str, Sequence[int]] | None,
) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for dataset_path, episode_indices in (mapping or {}).items():
        normalized[str(Path(dataset_path).resolve())] = [int(index) for index in episode_indices]
    return normalized


def _resolve_episode_subset(
    subset_mapping: dict[str, list[int]],
    dataset_path: str | Path,
) -> list[int] | None:
    return subset_mapping.get(str(Path(dataset_path).resolve()))


# ---------------------------------------------------------------------------
# Per-episode preprocessing
# ---------------------------------------------------------------------------

def _preprocess_episode(
    ep: dict[str, np.ndarray],
    *,
    target_action_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """Lightweight preprocessing — images are already float32 CHW from conversion."""
    ep = dict(ep)  # shallow copy

    # --- Action dimension padding ---
    if target_action_dim is not None:
        actions = ep["actions"]
        current_dim = actions.shape[-1]
        if current_dim < target_action_dim:
            pad_width = [(0, 0)] * (actions.ndim - 1) + [(0, target_action_dim - current_dim)]
            ep["actions"] = np.pad(actions, pad_width, mode="constant")

    # --- Mask and timestep ---
    T = ep["actions"].shape[0]
    ep["mask"] = np.ones(T, dtype=np.float32)
    ep["timestep"] = np.arange(T, dtype=np.int64)

    return ep


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

class EpisodeWindowDataset(Dataset):
    """
    Stores preprocessed episodes and yields fixed-length sliding windows.

    Each item is a dict of numpy arrays with shape (seq_len, ...).
    An ``embodiment_id`` scalar is included per-sample.
    """

    _FIELDS = (
        "observations", "actions", "states", "mask", "timestep",
        "is_first", "is_last", "is_terminal", "discount",
        "rewards", "images", "embeddings", "scene_obs",
    )

    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        seq_len: int,
        shift: int = 1,
        embodiment_id: int = 0,
        *,
        cover_tail: bool = False,
    ):
        if shift < 1:
            raise ValueError(f"shift must be >= 1, got {shift}")
        self.seq_len = seq_len
        self.embodiment_id = embodiment_id
        self.cover_tail = bool(cover_tail)

        # Build an index: (episode_idx, start_frame)
        self._episodes = episodes
        self._windows: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            T = ep["observations"].shape[0]
            if T < seq_len:
                continue
            max_start = T - seq_len
            window_starts = list(range(0, max_start + 1, shift))
            if self.cover_tail and window_starts and window_starts[-1] != max_start:
                # Evaluation should always cover the true episode ending even when the
                # final segment is shorter than one full non-overlapping stride.
                window_starts.append(max_start)
            for start in window_starts:
                self._windows.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, start = self._windows[idx]
        ep = self._episodes[ep_idx]
        end = start + self.seq_len

        window: dict[str, Any] = {}
        for key in self._FIELDS:
            if key in ep:
                window[key] = ep[key][start:end]

        window["timestep"] = np.arange(self.seq_len, dtype=np.int64)
        window["mask"] = np.ones(self.seq_len, dtype=np.float32)
        window["embodiment_id"] = np.int32(self.embodiment_id)

        return window


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of window dicts into a batched dict of numpy arrays."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], np.ndarray):
            collated[key] = np.stack(vals)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            collated[key] = np.array(vals)
        else:
            collated[key] = vals
    return collated


# ---------------------------------------------------------------------------
# Infinite iterator (handles epoch for DistributedSampler)
# ---------------------------------------------------------------------------

class InfiniteDataLoaderIter:
    """Wraps a DataLoader to cycle indefinitely."""

    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._epoch = 0
        self._iter = iter(loader)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            sampler = self._loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self._epoch)
            self._iter = iter(self._loader)
            return next(self._iter)


# ---------------------------------------------------------------------------
# High-level builders
# ---------------------------------------------------------------------------

def build_primary_dataloaders(
    dataset_paths: list[str],
    *,
    eval_dataset_paths: list[str] | None = None,
    dataset_episode_subsets: dict[str, Sequence[int]] | None = None,
    eval_dataset_episode_subsets: dict[str, Sequence[int]] | None = None,
    seq_len: int,
    batch_size: int,
    target_action_dim: int | None = None,
    eval_num_trajs: int = 16,
    shift: int = 1,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None, int]:
    """
    Build train and eval DataLoaders for the primary training data.

    Returns (train_loader, eval_loader, num_embodiments).
    """
    explicit_eval = [str(path) for path in (eval_dataset_paths or []) if str(path)]
    train_subset_mapping = _normalize_episode_subset_mapping(dataset_episode_subsets)
    eval_subset_mapping = _normalize_episode_subset_mapping(eval_dataset_episode_subsets)
    all_train: list[Dataset] = []
    all_eval: list[Dataset] = []
    embodiment_ids: set[int] = set()
    metadata_by_path, embodiment_id_by_key = _build_embodiment_id_mapping(
        list(dataset_paths) + explicit_eval
    )

    for eid, ds_path in enumerate(dataset_paths):
        resolved_ds_path = str(Path(ds_path).resolve())
        metadata = metadata_by_path.get(resolved_ds_path, {})
        embodiment_id = int(
            embodiment_id_by_key[_resolve_embodiment_key(metadata, resolved_ds_path)]
        )
        embodiment_ids.add(embodiment_id)
        print(f"[dataloader] loading {ds_path} (embodiment_id={embodiment_id})...")
        selected_train_indices = _resolve_episode_subset(train_subset_mapping, ds_path)
        episodes = load_episodes(ds_path, episode_indices=selected_train_indices)
        if selected_train_indices is None:
            print(f"[dataloader]   {len(episodes)} episodes")
        else:
            print(
                f"[dataloader]   {len(episodes)} selected episodes "
                f"(subset of {len(selected_train_indices)} requested)"
            )

        processed = [
            _preprocess_episode(ep, target_action_dim=target_action_dim)
            for ep in episodes
        ]

        if explicit_eval:
            train_eps = processed
            eval_eps = []
        else:
            # Legacy train/eval split: take the tail episodes as eval.
            n_eval_per_ds = min(eval_num_trajs // max(len(dataset_paths), 1), len(processed))
            if eval_num_trajs <= 0:
                n_eval_per_ds = 0
            train_eps = processed[:-n_eval_per_ds] if n_eval_per_ds > 0 else processed
            eval_eps = processed[-n_eval_per_ds:] if n_eval_per_ds > 0 else []

        if train_eps:
            ds = EpisodeWindowDataset(
                train_eps,
                seq_len=seq_len,
                shift=shift,
                embodiment_id=embodiment_id,
            )
            all_train.append(ds)
            print(f"[dataloader]   train: {len(train_eps)} eps, {len(ds)} windows")

        if eval_eps:
            ds = EpisodeWindowDataset(
                eval_eps,
                seq_len=seq_len,
                shift=seq_len,
                embodiment_id=embodiment_id,
                cover_tail=True,
            )
            all_eval.append(ds)
            print(f"[dataloader]   eval: {len(eval_eps)} eps, {len(ds)} windows")

    for eid, ds_path in enumerate(explicit_eval):
        resolved_ds_path = str(Path(ds_path).resolve())
        metadata = metadata_by_path.get(resolved_ds_path, {})
        embodiment_id = int(
            embodiment_id_by_key[_resolve_embodiment_key(metadata, resolved_ds_path)]
        )
        embodiment_ids.add(embodiment_id)
        print(
            f"[dataloader] loading explicit eval split {ds_path} "
            f"(embodiment_id={embodiment_id})..."
        )
        selected_eval_indices = _resolve_episode_subset(eval_subset_mapping, ds_path)
        episodes = load_episodes(ds_path, episode_indices=selected_eval_indices)
        if selected_eval_indices is None:
            print(f"[dataloader]   {len(episodes)} episodes")
        else:
            print(
                f"[dataloader]   {len(episodes)} selected episodes "
                f"(subset of {len(selected_eval_indices)} requested)"
            )
        processed = [
            _preprocess_episode(ep, target_action_dim=target_action_dim)
            for ep in episodes
        ]
        if processed:
            ds = EpisodeWindowDataset(
                processed,
                seq_len=seq_len,
                shift=seq_len,
                embodiment_id=embodiment_id,
                cover_tail=True,
            )
            all_eval.append(ds)
            print(f"[dataloader]   explicit eval: {len(processed)} eps, {len(ds)} windows")

    train_dataset = ConcatDataset(all_train) if all_train else None
    eval_dataset = ConcatDataset(all_eval) if all_eval else None

    if train_dataset is None or len(train_dataset) == 0:
        raise RuntimeError("No training data after processing!")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
    ) if world_size > 1 else None

    eval_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    eval_loader = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            drop_last=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    num_embodiments = len(embodiment_ids)
    print(
        f"[dataloader] train: {len(train_dataset)} windows, "
        f"eval: {len(eval_dataset) if eval_dataset else 0} windows, "
        f"num_embodiments: {num_embodiments}"
    )
    return train_loader, eval_loader, num_embodiments


def build_explicit_eval_dataloader(
    dataset_paths: list[str],
    *,
    dataset_episode_subsets: dict[str, Sequence[int]] | None = None,
    seq_len: int,
    batch_size: int,
    target_action_dim: int | None = None,
    num_workers: int = 2,
    log_prefix: str = "explicit eval",
) -> DataLoader | None:
    """Build a finite eval-only DataLoader from explicit dataset paths/subsets.

    This is used for auxiliary evaluation suites that should never participate
    in training but should be evaluated with the same fixed-window protocol as
    the main explicit eval split.
    """

    explicit_eval = [str(path) for path in dataset_paths if str(path)]
    if not explicit_eval:
        return None

    eval_subset_mapping = _normalize_episode_subset_mapping(dataset_episode_subsets)
    metadata_by_path, embodiment_id_by_key = _build_embodiment_id_mapping(explicit_eval)
    all_eval: list[Dataset] = []

    for ds_path in explicit_eval:
        resolved_ds_path = str(Path(ds_path).resolve())
        metadata = metadata_by_path.get(resolved_ds_path, {})
        embodiment_id = int(
            embodiment_id_by_key[_resolve_embodiment_key(metadata, resolved_ds_path)]
        )
        print(
            f"[dataloader] loading {log_prefix} {ds_path} "
            f"(embodiment_id={embodiment_id})..."
        )
        selected_eval_indices = _resolve_episode_subset(eval_subset_mapping, ds_path)
        episodes = load_episodes(ds_path, episode_indices=selected_eval_indices)
        if selected_eval_indices is None:
            print(f"[dataloader]   {len(episodes)} episodes")
        else:
            print(
                f"[dataloader]   {len(episodes)} selected episodes "
                f"(subset of {len(selected_eval_indices)} requested)"
            )
        processed = [
            _preprocess_episode(ep, target_action_dim=target_action_dim)
            for ep in episodes
        ]
        if processed:
            ds = EpisodeWindowDataset(
                processed,
                seq_len=seq_len,
                shift=seq_len,
                embodiment_id=embodiment_id,
                cover_tail=True,
            )
            all_eval.append(ds)
            print(f"[dataloader]   {log_prefix}: {len(processed)} eps, {len(ds)} windows")

    eval_dataset = ConcatDataset(all_eval) if all_eval else None
    if eval_dataset is None or len(eval_dataset) == 0:
        return None

    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_secondary_dataloader(
    dataset_paths: list[str],
    *,
    seq_len: int,
    batch_size: int,
    target_action_dim: int | None = None,
    shift: int = 1,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """
    Build a secondary (cross-cycle) DataLoader from empty/random datasets.

    No train/eval split — all data goes to a single loader.
    """
    all_datasets: list[Dataset] = []

    for ds_path in dataset_paths:
        print(f"[secondary-dataloader] loading {ds_path}...")
        episodes = load_episodes(ds_path)
        print(f"[secondary-dataloader]   {len(episodes)} episodes")

        processed = [
            _preprocess_episode(ep, target_action_dim=target_action_dim)
            for ep in episodes
        ]

        ds = EpisodeWindowDataset(processed, seq_len=seq_len, shift=shift, embodiment_id=-1)
        all_datasets.append(ds)
        print(f"[secondary-dataloader]   {len(ds)} windows")

    dataset = ConcatDataset(all_datasets) if all_datasets else None
    if dataset is None or len(dataset) == 0:
        raise RuntimeError(f"Secondary dataset is empty! Paths: {dataset_paths}")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
    ) if world_size > 1 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"[secondary-dataloader] total: {len(dataset)} windows")
    return loader


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "build_primary_dataloaders",
    "build_secondary_dataloader",
    "InfiniteDataLoaderIter",
    "EpisodeWindowDataset",
]
