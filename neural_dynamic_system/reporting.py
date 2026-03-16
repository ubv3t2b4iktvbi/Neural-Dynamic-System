from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import ArrayTrajectoryDataset, compute_episode_splits, prepare_datasets
from .model import LatentRGManifoldAutoencoder


@dataclass(frozen=True)
class RunVisualPayload:
    run_name: str
    title: str
    color: str
    history: pd.DataFrame
    summary: dict[str, object]
    horizon_metrics: pd.DataFrame
    rollout: dict[str, np.ndarray]
    encoded_episode: dict[str, np.ndarray]
    phase_true: np.ndarray
    phase_pred: np.ndarray
    dt: float
    label_names: list[str]


@torch.no_grad()
def _collect_block(
    model: LatentRGManifoldAutoencoder,
    dataset: ArrayTrajectoryDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    blocks: dict[str, list[np.ndarray]] = {"z": [], "q": [], "h": [], "label": []}
    for raw_batch in loader:
        windows = raw_batch["window"].to(device=device, dtype=torch.float32)
        encoded = model.encode_components(windows, update_whitener=False)
        blocks["z"].append(encoded["z"].detach().cpu().numpy())
        blocks["q"].append(encoded["q"].detach().cpu().numpy())
        blocks["h"].append(encoded["h"].detach().cpu().numpy())
        if "label" in raw_batch:
            blocks["label"].append(raw_batch["label"].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in blocks.items() if values}


def _fit_linear_probe(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1), dtype=features.dtype)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return coeffs


def _apply_linear_probe(features: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1), dtype=features.dtype)], axis=1)
    return design @ coeffs


@torch.no_grad()
def _evaluate_horizons(
    model: LatentRGManifoldAutoencoder,
    dataset: ArrayTrajectoryDataset,
    *,
    device: torch.device,
    batch_size: int,
    dt: float,
    horizons: Sequence[int],
    mean: np.ndarray,
    train_std: np.ndarray,
    norm_scale: np.ndarray,
) -> list[dict[str, float]]:
    horizon_to_index = {int(horizon): idx for idx, horizon in enumerate(dataset.horizons)}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    obs_mean = torch.as_tensor(mean, dtype=torch.float32, device=device).unsqueeze(0)
    obs_scale = torch.as_tensor(train_std, dtype=torch.float32, device=device).unsqueeze(0)
    mse_sums = {int(horizon): 0.0 for horizon in horizons}
    count_sums = {int(horizon): 0 for horizon in horizons}

    for raw_batch in loader:
        windows = raw_batch["window"].to(device=device, dtype=torch.float32)
        future = raw_batch["future"].to(device=device, dtype=torch.float32)
        z = model.encode(windows)
        previous = 0
        current = z
        for horizon in sorted(int(horizon) for horizon in horizons):
            current = model.flow(current, dt=dt, steps=int(horizon) - previous)
            prediction = model.decode(current)
            target = future[:, horizon_to_index[horizon], :]
            pred_raw = prediction * obs_scale + obs_mean
            target_raw = target * obs_scale + obs_mean
            diff = pred_raw - target_raw
            mse_sums[horizon] += float(diff.pow(2).sum().item())
            count_sums[horizon] += int(diff.numel())
            previous = int(horizon)

    norm_value = float(np.mean(norm_scale))
    rows = []
    for horizon in horizons:
        horizon = int(horizon)
        rmse = math.sqrt(mse_sums[horizon] / max(count_sums[horizon], 1))
        rows.append({"horizon": horizon, "rmse": rmse, "nrmse": rmse / max(norm_value, 1e-8)})
    return rows


@torch.no_grad()
def _long_rollout(
    model: LatentRGManifoldAutoencoder,
    episode: np.ndarray,
    *,
    device: torch.device,
    context_len: int,
    dt: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, np.ndarray]:
    obs_mean = torch.as_tensor(mean, dtype=torch.float32, device=device).unsqueeze(0)
    obs_std = torch.as_tensor(std, dtype=torch.float32, device=device).unsqueeze(0)
    standardized_episode = (episode - mean) / std
    window = torch.from_numpy(standardized_episode[:context_len]).unsqueeze(0).to(device=device, dtype=torch.float32)
    z = model.encode(window)

    decoded: list[np.ndarray] = []
    z_series: list[np.ndarray] = []
    q_series: list[np.ndarray] = []
    h_series: list[np.ndarray] = []
    total_steps = int(len(episode) - context_len + 1)
    current = z
    for step_idx in range(total_steps):
        q, h = model.split_latent(current)
        decoded_step = model.decode(current) * obs_std + obs_mean
        decoded.append(decoded_step.squeeze(0).detach().cpu().numpy())
        z_series.append(current.squeeze(0).detach().cpu().numpy())
        q_series.append(q.squeeze(0).detach().cpu().numpy())
        h_series.append(h.squeeze(0).detach().cpu().numpy())
        if step_idx + 1 < total_steps:
            current = model.flow(current, dt=dt, steps=1)

    true = np.asarray(episode[context_len - 1 :], dtype=np.float32)
    pred = np.asarray(decoded, dtype=np.float32)
    drift = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    return {
        "true_obs": true,
        "pred_obs": pred,
        "z": np.asarray(z_series, dtype=np.float32),
        "q": np.asarray(q_series, dtype=np.float32),
        "h": np.asarray(h_series, dtype=np.float32),
        "drift": drift.astype(np.float32),
    }


@torch.no_grad()
def _encode_episode_latents(
    model: LatentRGManifoldAutoencoder,
    episode: np.ndarray,
    *,
    device: torch.device,
    context_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    standardized_episode = (episode - mean) / std
    windows = np.stack(
        [standardized_episode[start : start + context_len] for start in range(len(standardized_episode) - context_len + 1)],
        axis=0,
    ).astype(np.float32)
    loader = DataLoader(torch.from_numpy(windows), batch_size=batch_size, shuffle=False)
    blocks: dict[str, list[np.ndarray]] = {"z": [], "q": [], "h": []}
    for batch in loader:
        encoded = model.encode_components(batch.to(device=device, dtype=torch.float32), update_whitener=False)
        for key in blocks:
            blocks[key].append(encoded[key].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in blocks.items()}


def _first_val_episode(
    trajectory: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    splits = compute_episode_splits(
        [len(episode) for episode in trajectory],
        context_len=model_cfg.context_len,
        horizons=train_cfg.horizons,
        train_fraction=train_cfg.train_fraction,
    )
    val_start = int(splits[0][1])
    return (
        np.asarray(trajectory[0][val_start:], dtype=np.float32),
        np.asarray(labels[0][val_start:], dtype=np.float32),
    )


def build_run_visual_payload(
    *,
    run_name: str,
    title: str,
    color: str,
    trajectory: Sequence[np.ndarray],
    analysis_labels: Sequence[np.ndarray],
    result,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    history: pd.DataFrame,
    summary: dict[str, object],
    eval_batch_size: int,
) -> RunVisualPayload:
    device = next(result.model.parameters()).device
    train_ds, val_ds, stats = prepare_datasets(
        trajectory,
        context_len=model_cfg.context_len,
        horizons=train_cfg.horizons,
        train_fraction=train_cfg.train_fraction,
        labels=analysis_labels,
    )
    block_train = _collect_block(result.model, train_ds, device=device, batch_size=eval_batch_size)
    phase_probe = _fit_linear_probe(block_train["z"], block_train["label"])
    first_val_episode, first_val_labels = _first_val_episode(
        trajectory,
        analysis_labels,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    rollout = _long_rollout(
        result.model,
        first_val_episode,
        device=device,
        context_len=model_cfg.context_len,
        dt=float(train_cfg.dt),
        mean=np.asarray(stats.mean, dtype=np.float32),
        std=np.asarray(stats.std, dtype=np.float32),
    )
    encoded_episode = _encode_episode_latents(
        result.model,
        first_val_episode,
        device=device,
        context_len=model_cfg.context_len,
        mean=np.asarray(stats.mean, dtype=np.float32),
        std=np.asarray(stats.std, dtype=np.float32),
        batch_size=eval_batch_size,
    )
    horizon_metrics = pd.DataFrame(
        _evaluate_horizons(
            result.model,
            val_ds,
            device=device,
            batch_size=eval_batch_size,
            dt=float(train_cfg.dt),
            horizons=train_cfg.horizons,
            mean=np.asarray(stats.mean, dtype=np.float32),
            train_std=np.asarray(stats.std, dtype=np.float32),
            norm_scale=np.std(np.concatenate([episode for episode in trajectory], axis=0), axis=0) + 1e-6,
        )
    )
    phase_pred = _apply_linear_probe(rollout["z"], phase_probe)
    phase_true = first_val_labels[model_cfg.context_len - 1 :]
    label_width = int(phase_true.shape[1])
    default_names = ["x", "v", "phase", "amplitude"][:label_width]
    return RunVisualPayload(
        run_name=run_name,
        title=title,
        color=color,
        history=history,
        summary=summary,
        horizon_metrics=horizon_metrics,
        rollout=rollout,
        encoded_episode=encoded_episode,
        phase_true=phase_true,
        phase_pred=phase_pred[:, :label_width],
        dt=float(train_cfg.dt),
        label_names=default_names,
    )
