from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .reporting import RunVisualPayload


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


def save_single_run_plots(payload: "RunVisualPayload", out_dir: Path, *, trajectory_points: int = 400) -> None:
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
        for ax, ylabel in zip(axes, ("RMSE", "NRMSE")):
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
    payloads: Sequence["RunVisualPayload"],
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
                "best_val_rmse": math.sqrt(max(float(payload.summary["best_val_one_step_prediction_loss"]), 0.0)),
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
