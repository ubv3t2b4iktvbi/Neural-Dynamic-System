from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
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


def _paper_rc() -> dict[str, object]:
    return {
        "font.family": "DejaVu Serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def _light_axis_style(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6, axis=grid_axis)
    ax.tick_params(labelsize=9)


def _add_phase_background(ax: plt.Axes, history: pd.DataFrame) -> None:
    val_history = history[history["split"] == "val"].sort_values("epoch", kind="stable")
    if val_history.empty:
        return
    phase_colors = {0: "#f5f7fa", 1: "#fbf6ea", 2: "#eef8ef", 3: "#f3f0fa"}
    for phase, phase_df in val_history.groupby("phase", sort=False):
        start = int(phase_df["epoch"].min())
        end = int(phase_df["epoch"].max())
        ax.axvspan(start - 0.5, end + 0.5, color=phase_colors.get(int(phase), "#f7f7f7"), alpha=0.45)


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


def save_single_run_plots(payload: RunVisualPayload, out_dir: Path, *, trajectory_points: int = 400) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_names = payload.label_names if len(payload.label_names) >= 2 else ["x", "v"]
    with plt.rc_context(_paper_rc()):
        limit = min(int(trajectory_points), payload.phase_true.shape[0], payload.phase_pred.shape[0])
        time_axis = np.arange(limit, dtype=float) * float(payload.dt)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, idx, name in zip(axes, range(2), label_names[:2]):
            ax.plot(time_axis, payload.phase_true[:limit, idx], color="#111111", linewidth=2.0, label="True")
            ax.plot(time_axis, payload.phase_pred[:limit, idx], color=payload.color, linewidth=1.8, label="Predicted")
            ax.set_ylabel(name)
            _light_axis_style(ax)
        axes[1].set_xlabel("Time")
        axes[0].legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_overlay.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.plot(payload.phase_true[:limit, 0], payload.phase_true[:limit, 1], color="#111111", linewidth=2.0, label="True")
        ax.plot(payload.phase_pred[:limit, 0], payload.phase_pred[:limit, 1], color=payload.color, linewidth=1.8, label="Predicted")
        ax.scatter(payload.phase_true[0, 0], payload.phase_true[0, 1], color="#111111", s=18, zorder=3)
        ax.scatter(payload.phase_pred[0, 0], payload.phase_pred[0, 1], color=payload.color, s=18, zorder=3)
        ax.set_xlabel(label_names[0])
        ax.set_ylabel(label_names[1])
        ax.legend(frameon=False, loc="upper right")
        _light_axis_style(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "phase_portrait.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
        horizons = payload.horizon_metrics["horizon"].to_numpy(dtype=int)
        axes[0].bar(np.arange(len(horizons)), payload.horizon_metrics["rmse"], color=payload.color, width=0.68)
        axes[1].bar(np.arange(len(horizons)), payload.horizon_metrics["nrmse"], color=payload.color, width=0.68)
        for ax, column, ylabel in zip(axes, ("rmse", "nrmse"), ("RMSE", "NRMSE")):
            ax.set_xticks(np.arange(len(horizons)), [str(horizon) for horizon in horizons])
            ax.set_xlabel("Prediction Horizon (steps)")
            ax.set_ylabel(ylabel)
            _light_axis_style(ax, grid_axis="y")
        axes[0].set_title("Multi-horizon RMSE", loc="left", fontweight="bold")
        axes[1].set_title("Multi-horizon NRMSE", loc="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "multi_horizon_rmse.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.plot(
            np.arange(len(payload.rollout["drift"])) * float(payload.dt),
            payload.rollout["drift"],
            color=payload.color,
            linewidth=2.0,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Pointwise RMSE")
        ax.set_title("Long-rollout Drift", loc="left", fontweight="bold")
        _light_axis_style(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "long_rollout_drift.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        encoded_limit = min(int(trajectory_points), payload.encoded_episode["q"].shape[0], payload.encoded_episode["h"].shape[0])
        encoded_time = np.arange(encoded_limit, dtype=float) * float(payload.dt)
        axes[0].plot(encoded_time, payload.encoded_episode["q"][:encoded_limit, 0], color=payload.color, linewidth=1.9)
        axes[1].plot(encoded_time, payload.encoded_episode["h"][:encoded_limit, 0], color=payload.color, linewidth=1.9)
        axes[0].set_ylabel("q1")
        axes[1].set_ylabel("h1")
        axes[1].set_xlabel("Time")
        axes[0].set_title("Latent Dynamics", loc="left", fontweight="bold")
        _light_axis_style(axes[0])
        _light_axis_style(axes[1])
        fig.tight_layout()
        fig.savefig(out_dir / "latent_dynamics.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def save_comparison_artifacts(
    payloads: Sequence[RunVisualPayload],
    out_dir: Path,
    *,
    trajectory_points: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    horizon_frames: list[pd.DataFrame] = []
    for payload in payloads:
        horizon_df = payload.horizon_metrics.copy()
        horizon_df.insert(0, "title", payload.title)
        horizon_df.insert(0, "run_name", payload.run_name)
        horizon_frames.append(horizon_df)
        summary_rows.append(
            {
                "run_name": payload.run_name,
                "title": payload.title,
                "best_epoch": int(payload.summary["best_epoch"]),
                "stop_epoch": int(payload.summary["stop_epoch"]),
                "stopped_early": bool(payload.summary.get("stopped_early", False)),
                "best_val_rmse": math.sqrt(max(float(payload.summary["best_val_prediction_loss"]), 0.0)),
                "best_val_one_step_rmse": math.sqrt(max(float(payload.summary["best_val_one_step_prediction_loss"]), 0.0)),
                "best_val_long_horizon_rmse": math.sqrt(
                    max(float(payload.summary["best_val_long_horizon_prediction_loss"]), 0.0)
                ),
                "best_val_semigroup_loss": float(payload.summary["best_val_semigroup_loss"]),
                "best_val_hidden_sym_eig_upper": float(payload.summary["best_val_hidden_sym_eig_upper"]),
                "drift_final": float(payload.rollout["drift"][-1]),
                "drift_mean": float(np.mean(payload.rollout["drift"])),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.concat(horizon_frames, ignore_index=True)
    summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)
    horizon_df.to_csv(out_dir / "horizon_metrics.csv", index=False)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for payload in payloads:
            history = payload.history
            _add_phase_background(ax, history)
            val_history = history[history["split"] == "val"].sort_values("epoch", kind="stable")
            rmse = np.sqrt(np.clip(val_history["prediction_loss"].to_numpy(dtype=float), a_min=0.0, a_max=None))
            ax.plot(val_history["epoch"], rmse, color=payload.color, linewidth=2.0, label=payload.title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation RMSE")
        ax.set_title("Validation RMSE vs Epoch", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        _light_axis_style(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "validation_rmse_vs_epoch.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, len(payloads), figsize=(4.8 * len(payloads), 4.4), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, payload in zip(axes, payloads):
            limit = min(int(trajectory_points), payload.phase_true.shape[0], payload.phase_pred.shape[0])
            ax.plot(payload.phase_true[:limit, 0], payload.phase_true[:limit, 1], color="#111111", linewidth=2.0, label="True")
            ax.plot(payload.phase_pred[:limit, 0], payload.phase_pred[:limit, 1], color=payload.color, linewidth=1.8, label=payload.title)
            ax.set_title(payload.title)
            ax.set_xlabel(payload.label_names[0])
            ax.set_ylabel(payload.label_names[1])
            _light_axis_style(ax)
        axes[0].legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "comparison_phase_portrait.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5.2))
        for payload in payloads:
            time_axis = np.arange(len(payload.rollout["drift"]), dtype=float) * float(payload.dt)
            ax.plot(time_axis, payload.rollout["drift"], color=payload.color, linewidth=2.0, label=payload.title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pointwise RMSE")
        ax.set_title("Comparison Long Rollout", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        _light_axis_style(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "comparison_long_rollout.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    return summary_df, horizon_df
