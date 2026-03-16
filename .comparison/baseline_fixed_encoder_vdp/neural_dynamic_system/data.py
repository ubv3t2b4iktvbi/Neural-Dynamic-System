from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TrajectoryStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "std": np.asarray(self.std, dtype=float).tolist(),
        }


def _coerce_episode_list(
    values: np.ndarray | Sequence[np.ndarray],
    *,
    name: str,
    dtype: np.dtype,
) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.dtype != object:
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"{name} must be a 1D, 2D, or 3D array")
        return [np.asarray(arr[idx], dtype=dtype) for idx in range(arr.shape[0])]

    episodes: list[np.ndarray] = []
    sequence = list(values) if isinstance(values, np.ndarray) else values
    for idx, item in enumerate(sequence):
        episode = np.asarray(item, dtype=dtype)
        if episode.ndim == 1:
            episode = episode.reshape(-1, 1)
        if episode.ndim != 2:
            raise ValueError(f"{name}[{idx}] must be a 1D or 2D array")
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"{name} must contain at least one episode")
    return episodes


def coerce_episode_list(
    values: np.ndarray | Sequence[np.ndarray],
    *,
    name: str = "trajectory",
    dtype: np.dtype | type = np.float32,
) -> list[np.ndarray]:
    return _coerce_episode_list(values, name=name, dtype=np.dtype(dtype))


def _stack_if_uniform(episodes: Sequence[np.ndarray], *, dtype: np.dtype) -> np.ndarray:
    if len(episodes) == 1:
        return np.asarray(episodes[0], dtype=dtype)
    lengths = {episode.shape[0] for episode in episodes}
    dims = {episode.shape[1] for episode in episodes}
    if len(lengths) == 1 and len(dims) == 1:
        return np.stack([np.asarray(episode, dtype=dtype) for episode in episodes], axis=0)
    return np.array([np.asarray(episode, dtype=dtype) for episode in episodes], dtype=object)


class ArrayTrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectory: np.ndarray | Sequence[np.ndarray],
        context_len: int,
        horizons: Sequence[int],
        *,
        labels: np.ndarray | Sequence[np.ndarray] | None = None,
    ):
        episodes = _coerce_episode_list(trajectory, name="trajectory", dtype=np.float32)
        self.context_len = int(context_len)
        self.horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
        self.max_horizon = max(self.horizons)
        self.episodes = episodes
        self.labels = None
        if labels is not None:
            label_episodes = _coerce_episode_list(labels, name="labels", dtype=np.float32)
            if len(label_episodes) != len(episodes):
                raise ValueError("labels must contain the same number of episodes as trajectory")
            for idx, (episode, label_episode) in enumerate(zip(episodes, label_episodes)):
                if len(label_episode) != len(episode):
                    raise ValueError(f"labels[{idx}] must have the same length as trajectory[{idx}]")
            self.labels = label_episodes

        self.sample_index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(episodes):
            count = len(episode) - self.context_len - self.max_horizon + 1
            if count <= 0:
                continue
            self.sample_index.extend((episode_idx, start) for start in range(count))
        if not self.sample_index:
            raise ValueError("trajectory is too short for the requested context_len and horizons")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self.sample_index[int(index)]
        episode = self.episodes[episode_idx]
        stop = start + self.context_len
        window = episode[start:stop]
        current = window[-1]
        future = np.stack([episode[stop - 1 + horizon] for horizon in self.horizons], axis=0)
        future_windows = np.stack(
            [episode[start + horizon : stop + horizon] for horizon in self.horizons],
            axis=0,
        )
        batch = {
            "window": torch.from_numpy(window),
            "current": torch.from_numpy(current),
            "future": torch.from_numpy(future),
            "future_windows": torch.from_numpy(future_windows),
            "flat_window": torch.from_numpy(window.reshape(-1)),
        }
        if self.labels is not None:
            label_episode = self.labels[episode_idx]
            batch["label"] = torch.from_numpy(label_episode[stop - 1])
            batch["future_labels"] = torch.from_numpy(
                np.stack([label_episode[stop - 1 + horizon] for horizon in self.horizons], axis=0)
            )
        return batch

    def current_label_array(self) -> np.ndarray | None:
        if self.labels is None:
            return None
        rows = [self.labels[episode_idx][start + self.context_len - 1] for episode_idx, start in self.sample_index]
        return np.asarray(rows, dtype=np.float32)


def compute_train_val_split(
    length: int,
    *,
    context_len: int,
    horizons: Sequence[int],
    train_fraction: float,
) -> tuple[int, int, int]:
    max_horizon = max(int(h) for h in horizons)
    min_train_len = int(context_len) + max_horizon + 8
    split_index = int(length * float(train_fraction))
    split_index = max(min_train_len, split_index)
    if split_index >= int(length):
        raise ValueError("trajectory is too short for the requested train split")
    val_start = max(split_index - int(context_len) - max_horizon, 0)
    return split_index, val_start, max_horizon


def compute_episode_splits(
    lengths: Sequence[int],
    *,
    context_len: int,
    horizons: Sequence[int],
    train_fraction: float,
) -> list[tuple[int, int, int]]:
    return [
        compute_train_val_split(
            int(length),
            context_len=context_len,
            horizons=horizons,
            train_fraction=train_fraction,
        )
        for length in lengths
    ]


def load_trajectory(path: str | Path, array_key: str | None = None) -> np.ndarray:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(data_path, allow_pickle=True)
    elif suffix == ".npz":
        bundle = np.load(data_path, allow_pickle=True)
        if array_key is not None:
            arr = bundle[array_key]
        elif "trajectory" in bundle.files:
            arr = bundle["trajectory"]
        elif len(bundle.files) == 1:
            arr = bundle[bundle.files[0]]
        else:
            raise ValueError(f"NPZ file has multiple arrays: {bundle.files}; pass --array_key")
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(data_path)
        arr = df.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported trajectory format: {suffix}")
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim not in {2, 3}:
            raise ValueError("Loaded trajectory must be 1D, 2D, or 3D")
        return arr
    episodes = _coerce_episode_list(arr, name="trajectory", dtype=float)
    return _stack_if_uniform(episodes, dtype=float)


def prepare_datasets(
    trajectory: np.ndarray | Sequence[np.ndarray],
    *,
    context_len: int,
    horizons: Sequence[int],
    train_fraction: float,
    labels: np.ndarray | Sequence[np.ndarray] | None = None,
) -> tuple[ArrayTrajectoryDataset, ArrayTrajectoryDataset, TrajectoryStats]:
    episodes = _coerce_episode_list(trajectory, name="trajectory", dtype=float)
    label_episodes = None
    if labels is not None:
        label_episodes = _coerce_episode_list(labels, name="labels", dtype=np.float32)
        if len(label_episodes) != len(episodes):
            raise ValueError("labels must contain the same number of episodes as trajectory")
        for idx, (episode, label_episode) in enumerate(zip(episodes, label_episodes)):
            if len(label_episode) != len(episode):
                raise ValueError(f"labels[{idx}] must have the same length as trajectory[{idx}]")

    splits = compute_episode_splits(
        [len(episode) for episode in episodes],
        context_len=context_len,
        horizons=horizons,
        train_fraction=train_fraction,
    )
    train_raw = np.concatenate(
        [episode[:split_index] for episode, (split_index, _, _) in zip(episodes, splits)],
        axis=0,
    )
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-6
    stats = TrajectoryStats(mean=mean, std=std)

    standardized_episodes = [
        ((episode - mean) / std).astype(np.float32)
        for episode in episodes
    ]
    train_episodes = [
        episode[:split_index]
        for episode, (split_index, _, _) in zip(standardized_episodes, splits)
    ]
    val_episodes = [
        episode[val_start:]
        for episode, (_, val_start, _) in zip(standardized_episodes, splits)
    ]
    train_labels = None
    val_labels = None
    if label_episodes is not None:
        train_labels = [
            label_episode[:split_index]
            for label_episode, (split_index, _, _) in zip(label_episodes, splits)
        ]
        val_labels = [
            label_episode[val_start:]
            for label_episode, (_, val_start, _) in zip(label_episodes, splits)
        ]

    train_dataset = ArrayTrajectoryDataset(
        train_episodes,
        context_len=context_len,
        horizons=horizons,
        labels=train_labels,
    )
    val_dataset = ArrayTrajectoryDataset(
        val_episodes,
        context_len=context_len,
        horizons=horizons,
        labels=val_labels,
    )
    return train_dataset, val_dataset, stats
