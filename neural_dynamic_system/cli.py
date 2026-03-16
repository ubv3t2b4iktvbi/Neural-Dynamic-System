import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_dynamic_system import (  # noqa: E402
    ArrayTrajectoryDataset,
    LossConfig,
    ModelConfig,
    SupervisionConfig,
    SyntheticTrajectoryConfig,
    TrainConfig,
    coerce_episode_list,
    compute_episode_splits,
    fit_model,
    generate_synthetic_trajectory,
    load_trajectory,
)
from neural_dynamic_system.plots import save_single_run_plots  # noqa: E402
from neural_dynamic_system.reporting import build_run_visual_payload  # noqa: E402
from neural_dynamic_system.run_config import (  # noqa: E402
    load_cli_defaults,
    namespace_to_cli_config,
    resolve_repo_path,
    write_cli_config,
)

_HELP_EXPERT_FLAGS = {"--help-expert", "--help-all"}
_COMMON_TRAIN_ARG_DESTS = frozenset(
    {
        "config",
        "out_dir",
        "data_path",
        "array_key",
        "label_path",
        "label_array_key",
        "window",
        "q_dim",
        "h_dim",
        "koopman_dim",
        "curriculum_preset",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "horizons",
        "dt",
        "train_fraction",
        "device",
        "progress_bar",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "seed",
        "steps",
        "num_episodes",
        "obs_dim",
        "burn_in",
        "noise_std",
        "van_der_pol_mu",
        "eval_batch_size",
        "trajectory_points",
        "report_plots",
    }
)
_INTERNAL_TRAIN_DEFAULTS: dict[str, object] = {
    "schedule_mode": None,
    "phase0_min_epochs": None,
    "phase1_min_epochs": None,
    "phase2_min_epochs": None,
    "phase3_min_epochs": None,
    "phase0_reconstruction_improve": None,
    "phase0_koopman_stability_ratio": None,
    "phase0_reconstruction_target": None,
    "phase0_koopman_loss_ceiling": None,
    "phase0_stable_validations": None,
    "phase1_prediction_improve": None,
    "phase1_q_align_improve": None,
    "phase1_prediction_target": None,
    "phase1_q_align_target": None,
    "phase2_prediction_plateau_tolerance": None,
    "phase2_prediction_plateau_checks": None,
    "phase2_separation_target": None,
    "phase2_prediction_plateau_abs_tol": None,
    "phase2_long_horizon_target": None,
    "rollback_prediction_worsen_ratio": None,
    "rollback_long_horizon_worsen_ratio": None,
    "rollback_prediction_worsen_delta": None,
    "rollback_long_horizon_worsen_delta": None,
    "rollback_window": None,
    "rollback_weight_scale": None,
    "phase1_fraction": None,
    "phase2_fraction": None,
    "phase3_lr_scale": None,
    "synthetic_kind": "van_der_pol",
}


def _hide_expert_arguments(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if action.dest == "help" or not action.option_strings:
            continue
        if action.dest not in _COMMON_TRAIN_ARG_DESTS:
            action.help = argparse.SUPPRESS


def build_parser(
    default_overrides: dict[str, object] | None = None,
    *,
    show_expert: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a q,h slow-memory latent dynamics model on synthetic or file-based trajectories.",
        epilog=(
            None
            if show_expert
            else "Use --help-expert to show the full parameter surface."
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML or JSON launch config.")
    parser.add_argument("--out_dir", type=str, default="runs/neural_dynamic_system/demo")
    parser.add_argument("--data_path", type=str, default=None, help="Optional .npy, .npz, or .csv trajectory path.")
    parser.add_argument("--array_key", type=str, default=None, help="Optional NPZ array key.")
    parser.add_argument("--label_path", type=str, default=None, help="Optional label array path aligned with the trajectory.")
    parser.add_argument("--label_array_key", type=str, default=None, help="Optional NPZ label array key.")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--q_dim", type=int, default=2)
    parser.add_argument("--h_dim", "--m_dim", dest="h_dim", type=int, default=2)
    parser.add_argument("--koopman_dim", type=int, default=None)
    parser.add_argument("--latent_scheme", type=str, default="soft_spectrum", choices=["hard_split", "soft_spectrum"])
    parser.add_argument("--koopman_input_mode", type=str, default="slow_only", choices=["joint", "slow_only"])
    parser.add_argument("--hidden_coordinate_mode", type=str, default="normal_residual", choices=["direct", "normal_residual"])
    parser.add_argument("--modal_dim", type=int, default=8)
    parser.add_argument("--modal_temperature", type=float, default=0.35)
    parser.add_argument("--encoder_type", type=str, default="temporal_conv", choices=["temporal_conv", "mlp"])
    parser.add_argument("--encoder_levels", type=int, default=3)
    parser.add_argument("--encoder_kernel_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--vamp_head_depth", type=int, default=1)
    parser.add_argument("--vamp_whitening_momentum", type=float, default=0.05)
    parser.add_argument("--vamp_whitening_eps", type=float, default=1e-5)
    parser.add_argument(
        "--curriculum_preset",
        type=str,
        default="legacy",
        choices=["legacy", "conservative", "alanine_bootstrap", "vdp_paper", "vdp_structured"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--progress_bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early_stopping_monitor", type=str, default="prediction_loss", choices=["loss", "prediction_loss"])
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stopping_min_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_start_phase", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--steps",
        type=int,
        default=8192,
        help="Synthetic steps per episode when --data_path is omitted.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=4,
        help="Number of independent synthetic episodes when --data_path is omitted.",
    )
    parser.add_argument("--obs_dim", type=int, default=8, help="Synthetic observation dimension when --data_path is omitted.")
    parser.add_argument("--burn_in", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--van_der_pol_mu", type=float, default=5.0)

    parser.add_argument("--vamp_weight", type=float, default=0.15)
    parser.add_argument("--vamp_align_weight", type=float, default=0.25)
    parser.add_argument("--koopman_weight", type=float, default=0.2)
    parser.add_argument("--diag_weight", type=float, default=0.02)
    parser.add_argument("--latent_align_weight", type=float, default=0.0)
    parser.add_argument("--semigroup_weight", type=float, default=0.35)
    parser.add_argument("--contract_weight", type=float, default=0.2)
    parser.add_argument("--separation_weight", type=float, default=0.2)
    parser.add_argument("--rg_weight", type=float, default=0.0)
    parser.add_argument("--metric_weight", type=float, default=0.05)
    parser.add_argument("--metric_mode", type=str, default="mahalanobis_dynamics", choices=["euclidean", "mahalanobis_dynamics"])
    parser.add_argument("--hidden_l1_weight", "--memory_l1_weight", dest="hidden_l1_weight", type=float, default=1e-4)
    parser.add_argument("--contract_batch", type=int, default=16)
    parser.add_argument("--rg_horizon", type=int, default=1)
    parser.add_argument("--rg_scale", type=float, default=2.0)
    parser.add_argument("--coarse_strength", type=float, default=0.25)
    parser.add_argument("--hidden_rank", type=int, default=4)
    parser.add_argument("--rg_temperature", type=float, default=0.35)
    parser.add_argument("--q_label_indices", nargs="*", type=int, default=None)
    parser.add_argument("--h_label_indices", "--m_label_indices", dest="h_label_indices", nargs="*", type=int, default=None)
    parser.add_argument("--q_supervised_weight", type=float, default=0.0)
    parser.add_argument("--q_supervision_mode", type=str, default="direct", choices=["direct", "angular"])
    parser.add_argument("--h_supervised_weight", "--m_supervised_weight", dest="h_supervised_weight", type=float, default=0.0)
    parser.add_argument("--label_standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--trajectory_points", type=int, default=400)
    parser.add_argument("--report_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(**_INTERNAL_TRAIN_DEFAULTS)
    if default_overrides:
        parser.set_defaults(**default_overrides)
    if not show_expert:
        _hide_expert_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    show_expert = any(flag in argv for flag in _HELP_EXPERT_FLAGS)
    if show_expert:
        argv = ["--help" if flag in _HELP_EXPERT_FLAGS else flag for flag in argv]
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    known, _ = bootstrap.parse_known_args(argv)
    defaults = None
    if known.config:
        config_path = resolve_repo_path(ROOT, known.config)
        if config_path is None:
            raise ValueError("config path must not be empty")
        defaults = load_cli_defaults(config_path)
    parser = build_parser(default_overrides=defaults, show_expert=show_expert)
    return parser.parse_args(argv)


def _load_array_with_names(path: Path, *, array_key: str | None = None) -> tuple[np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_numpy(dtype=float), [str(col) for col in df.columns]
    arr = load_trajectory(path, array_key=array_key)
    episodes = coerce_episode_list(arr, name=str(path), dtype=float)
    names = [f"value_{idx}" for idx in range(episodes[0].shape[1])]
    return arr, names


def _default_column_names(prefix: str, width: int) -> list[str]:
    return [f"{prefix}_{idx + 1}" for idx in range(int(width))]


def _resolve_curriculum(args: argparse.Namespace) -> dict[str, float | int | str]:
    presets = {
        "legacy": {
            "epochs": 30,
            "schedule_mode": "fractional",
            "phase1_fraction": 0.30,
            "phase2_fraction": 0.80,
            "phase3_lr_scale": 0.10,
        },
        "conservative": {
            "epochs": 48,
            "schedule_mode": "fractional",
            "phase1_fraction": 0.45,
            "phase2_fraction": 0.92,
            "phase3_lr_scale": 0.20,
        },
        "alanine_bootstrap": {
            "epochs": 60,
            "schedule_mode": "fractional",
            "phase1_fraction": 0.55,
            "phase2_fraction": 0.95,
            "phase3_lr_scale": 0.25,
        },
        "vdp_paper": {
            "epochs": 100,
            "schedule_mode": "fractional",
            "phase1_fraction": 0.15,
            "phase2_fraction": 0.65,
            "phase3_lr_scale": 0.20,
        },
        "vdp_structured": {
            "epochs": 72,
            "schedule_mode": "metric_driven",
            "phase1_fraction": 0.30,
            "phase2_fraction": 0.80,
            "phase3_lr_scale": 0.35,
            "phase0_min_epochs": 8,
            "phase1_min_epochs": 12,
            "phase2_min_epochs": 16,
            "phase3_min_epochs": 8,
            "phase0_reconstruction_target": 0.20,
            "phase0_koopman_loss_ceiling": 0.004,
            "phase0_stable_validations": 2,
            "phase1_prediction_target": 0.12,
            "phase1_q_align_target": 0.08,
            "phase2_prediction_plateau_abs_tol": 0.005,
            "phase2_prediction_plateau_checks": 2,
            "phase2_separation_target": 0.08,
            "phase2_long_horizon_target": 0.75,
            "rollback_prediction_worsen_delta": 0.05,
            "rollback_long_horizon_worsen_delta": 0.20,
            "rollback_window": 2,
            "rollback_weight_scale": 0.50,
        },
    }
    resolved = dict(presets[str(args.curriculum_preset)])
    if args.epochs is not None:
        resolved["epochs"] = int(args.epochs)
    if args.schedule_mode is not None:
        resolved["schedule_mode"] = str(args.schedule_mode)
    if args.phase1_fraction is not None:
        resolved["phase1_fraction"] = float(args.phase1_fraction)
    if args.phase2_fraction is not None:
        resolved["phase2_fraction"] = float(args.phase2_fraction)
    if args.phase3_lr_scale is not None:
        resolved["phase3_lr_scale"] = float(args.phase3_lr_scale)
    for key in (
        "phase0_min_epochs",
        "phase1_min_epochs",
        "phase2_min_epochs",
        "phase3_min_epochs",
        "phase0_stable_validations",
        "phase2_prediction_plateau_checks",
        "rollback_window",
    ):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = int(value)
    for key in (
        "phase0_reconstruction_improve",
        "phase0_koopman_stability_ratio",
        "phase0_reconstruction_target",
        "phase0_koopman_loss_ceiling",
        "phase1_prediction_improve",
        "phase1_q_align_improve",
        "phase1_prediction_target",
        "phase1_q_align_target",
        "phase2_prediction_plateau_tolerance",
        "phase2_separation_target",
        "phase2_prediction_plateau_abs_tol",
        "phase2_long_horizon_target",
        "rollback_prediction_worsen_ratio",
        "rollback_long_horizon_worsen_ratio",
        "rollback_prediction_worsen_delta",
        "rollback_long_horizon_worsen_delta",
        "rollback_weight_scale",
    ):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = float(value)
    resolved["curriculum_preset"] = str(args.curriculum_preset)
    return resolved


def _infer_indices(
    requested: list[int] | None,
    *,
    start: int,
    block_dim: int,
    total_dim: int,
) -> tuple[int, ...]:
    if requested is not None:
        return tuple(int(idx) for idx in requested)
    stop = start + int(block_dim)
    if int(block_dim) <= 0 or stop > int(total_dim):
        return ()
    return tuple(range(start, stop))


def _episode_lengths(episodes: Sequence[np.ndarray]) -> list[int]:
    return [int(len(episode)) for episode in episodes]


def _episode_feature_dim(episodes: Sequence[np.ndarray]) -> int:
    return int(episodes[0].shape[1])


def _split_episodes(
    episodes: Sequence[np.ndarray],
    splits: Sequence[tuple[int, int, int]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_episodes = [
        np.asarray(episode[:split_index], dtype=np.float32)
        for episode, (split_index, _, _) in zip(episodes, splits)
    ]
    val_episodes = [
        np.asarray(episode[val_start:], dtype=np.float32)
        for episode, (_, val_start, _) in zip(episodes, splits)
    ]
    return train_episodes, val_episodes


def _flatten_episode_frame(
    episodes: Sequence[np.ndarray],
    *,
    column_names: Sequence[str],
    max_steps_per_episode: int | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for episode_idx, episode in enumerate(episodes):
        limit = len(episode) if max_steps_per_episode is None else min(len(episode), int(max_steps_per_episode))
        sliced = np.asarray(episode[:limit], dtype=float)
        frame = pd.DataFrame(sliced, columns=list(column_names))
        frame.insert(0, "step", np.arange(limit, dtype=int))
        frame.insert(0, "episode", int(episode_idx))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["episode", "step", *column_names])
    return pd.concat(frames, ignore_index=True)


def _standardize_labels(
    labels: Sequence[np.ndarray],
    *,
    splits: Sequence[tuple[int, int, int]],
    enabled: bool,
    skip_indices: tuple[int, ...] = (),
) -> tuple[list[np.ndarray], dict[str, object] | None]:
    label_episodes = [np.asarray(label, dtype=np.float32) for label in labels]
    if not enabled:
        return label_episodes, None
    train_labels = np.concatenate(
        [label_episode[:split_index] for label_episode, (split_index, _, _) in zip(label_episodes, splits)],
        axis=0,
    )
    mean = train_labels.mean(axis=0)
    std = train_labels.std(axis=0) + 1e-6
    if skip_indices:
        mean = mean.copy()
        std = std.copy()
        for idx in skip_indices:
            mean[int(idx)] = 0.0
            std[int(idx)] = 1.0
    standardized = [((label_episode - mean) / std).astype(np.float32) for label_episode in label_episodes]
    return standardized, {
        "mean": np.asarray(mean, dtype=float).tolist(),
        "std": np.asarray(std, dtype=float).tolist(),
        "skip_indices": [int(idx) for idx in skip_indices],
    }


def _r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    residual = np.sum((y_true - y_pred) ** 2, axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    total = np.sum(centered ** 2, axis=0)
    total = np.where(total <= 1e-12, 1e-12, total)
    return 1.0 - residual / total


def _fit_linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, object]:
    if x_train.shape[1] == 0:
        return {"target_r2": {}, "mean_r2": None}
    train_design = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    beta, *_ = np.linalg.lstsq(train_design, y_train, rcond=None)
    val_design = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=x_val.dtype)], axis=1)
    pred = val_design @ beta
    return {"prediction": pred, "r2": _r2_per_target(y_val, pred)}


def _collect_latents(
    model: torch.nn.Module,
    dataset: ArrayTrajectoryDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    outputs = {"koopman": [], "q": [], "h": [], "z": []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            windows = batch["window"].to(device=device, dtype=torch.float32)
            comp = model.encode_components(windows, update_whitener=False)
            for key in outputs:
                outputs[key].append(comp[key].detach().cpu().numpy())
    return {
        key: (np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, 0), dtype=np.float32))
        for key, parts in outputs.items()
    }


def _best_component_corrs(latents: dict[str, np.ndarray], targets: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block_name, arr in latents.items():
        if arr.shape[1] == 0:
            continue
        for idx in range(arr.shape[1]):
            comp = arr[:, idx]
            best_truth = None
            best_corr = None
            for target_idx, target_name in enumerate(target_names):
                corr = float(np.corrcoef(comp, targets[:, target_idx])[0, 1])
                if np.isnan(corr):
                    continue
                if best_corr is None or abs(corr) > abs(best_corr):
                    best_truth = target_name
                    best_corr = corr
            rows.append(
                {
                    "component": f"{block_name}{idx + 1}",
                    "truth": best_truth,
                    "corr": best_corr,
                }
            )
    return pd.DataFrame(rows)


def _label_probe(
    *,
    trajectory: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    label_names: list[str],
    result,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    eval_batch_size: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    trajectory_episodes = [np.asarray(episode, dtype=float) for episode in trajectory]
    label_episodes = [np.asarray(label_episode, dtype=np.float32) for label_episode in labels]
    splits = compute_episode_splits(
        _episode_lengths(trajectory_episodes),
        context_len=model_cfg.context_len,
        horizons=train_cfg.horizons,
        train_fraction=train_cfg.train_fraction,
    )
    standardized_episodes = [
        ((episode - result.stats.mean) / result.stats.std).astype(np.float32)
        for episode in trajectory_episodes
    ]
    train_episodes, val_episodes = _split_episodes(standardized_episodes, splits)
    train_labels, val_labels = _split_episodes(label_episodes, splits)

    train_dataset = ArrayTrajectoryDataset(
        train_episodes,
        context_len=model_cfg.context_len,
        horizons=train_cfg.horizons,
        labels=train_labels,
    )
    val_dataset = ArrayTrajectoryDataset(
        val_episodes,
        context_len=model_cfg.context_len,
        horizons=train_cfg.horizons,
        labels=val_labels,
    )
    device = next(result.model.parameters()).device
    train_latents = _collect_latents(result.model, train_dataset, device=device, batch_size=eval_batch_size)
    val_latents = _collect_latents(result.model, val_dataset, device=device, batch_size=eval_batch_size)
    train_targets = train_dataset.current_label_array()
    val_targets = val_dataset.current_label_array()
    if train_targets is None or val_targets is None:
        raise ValueError("label probe requires datasets with labels")

    probe_summary: dict[str, object] = {
        "train_windows": int(len(train_dataset)),
        "val_windows": int(len(val_dataset)),
        "train_episodes": int(len(train_episodes)),
        "val_episodes": int(len(val_episodes)),
        "target_names": label_names,
        "latent_dims": {
            "koopman": int(model_cfg.koopman_dim),
            "q": int(model_cfg.q_dim),
            "h": int(model_cfg.h_dim),
            "z": int(model_cfg.q_dim + model_cfg.h_dim),
        },
        "block_probe_r2": {},
    }
    for block_name, x_train in train_latents.items():
        x_val = val_latents[block_name]
        fitted = _fit_linear_probe(x_train, train_targets, x_val, val_targets)
        if "r2" not in fitted:
            continue
        r2 = fitted["r2"]
        probe_summary["block_probe_r2"][block_name] = {
            name: float(score)
            for name, score in zip(label_names, r2.tolist())
        }
        probe_summary["block_probe_r2"][block_name]["mean_r2"] = float(np.mean(r2))

    corr_df = _best_component_corrs(val_latents, val_targets, label_names)
    return probe_summary, corr_df


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = resolve_repo_path(ROOT, args.out_dir)
    if out_dir is None:
        raise ValueError("out_dir must not be empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_run_config = namespace_to_cli_config(args)
    write_cli_config(out_dir / "run_config.yaml", effective_run_config)
    synthetic_hidden_state = None
    synthetic_supervision_labels = None
    synthetic_supervision_names: list[str] | None = None
    synthetic_probe_labels = None
    synthetic_probe_names: list[str] | None = None
    label_array = None
    label_names: list[str] | None = None
    label_stats = None
    supervision_cfg = None
    analysis_label_episodes = None
    analysis_label_names: list[str] | None = None
    curriculum_cfg = _resolve_curriculum(args)
    supervision_enabled = any(weight > 0.0 for weight in (args.q_supervised_weight, args.h_supervised_weight))

    if args.data_path:
        data_path = resolve_repo_path(ROOT, args.data_path)
        if data_path is None:
            raise ValueError("data_path must not be empty")
        trajectory = load_trajectory(data_path, array_key=args.array_key)
        source_summary = {"source": "file", "data_path": str(args.data_path)}
    else:
        synth_cfg = SyntheticTrajectoryConfig(
            kind="van_der_pol",
            steps=args.steps,
            dt=args.dt,
            obs_dim=args.obs_dim,
            burn_in=args.burn_in,
            noise_std=args.noise_std,
            seed=args.seed,
            van_der_pol_mu=args.van_der_pol_mu,
            num_episodes=args.num_episodes,
        )
        synth = generate_synthetic_trajectory(synth_cfg)
        trajectory = synth["trajectory"]
        synthetic_hidden_state = synth["hidden_state"]
        synthetic_supervision_labels = np.asarray(synth["labels"], dtype=np.float32) if "labels" in synth else None
        synthetic_supervision_names = list(synth["label_names"]) if "label_names" in synth else None
        synthetic_probe_labels = (
            np.asarray(synth["probe_labels"], dtype=np.float32)
            if "probe_labels" in synth
            else synthetic_supervision_labels
        )
        synthetic_probe_names = (
            list(synth["probe_label_names"])
            if "probe_label_names" in synth
            else synthetic_supervision_names
        )
        source_summary = {
            "source": "synthetic",
            "synthetic_config": synth_cfg.to_dict(),
            "synthetic_metadata": synth.get("metadata", {}),
        }
        if synthetic_probe_labels is not None and synthetic_probe_names is not None:
            analysis_label_episodes = coerce_episode_list(
                synthetic_probe_labels,
                name="synthetic_probe_labels",
                dtype=np.float32,
            )
            analysis_label_names = list(synthetic_probe_names)

    trajectory_episodes = coerce_episode_list(trajectory, name="trajectory", dtype=float)
    input_dim = _episode_feature_dim(trajectory_episodes)
    trajectory_lengths = _episode_lengths(trajectory_episodes)
    obs_names = _default_column_names("obs", input_dim)
    source_summary["episode_count"] = int(len(trajectory_episodes))
    source_summary["episode_lengths"] = trajectory_lengths

    if synthetic_hidden_state is not None:
        hidden_episodes = coerce_episode_list(synthetic_hidden_state, name="synthetic_hidden_state", dtype=np.float32)
        hidden_names = _default_column_names("hidden", _episode_feature_dim(hidden_episodes))
        _flatten_episode_frame(hidden_episodes, column_names=hidden_names).to_csv(
            out_dir / "synthetic_hidden_state.csv",
            index=False,
        )
    if synthetic_supervision_labels is not None and synthetic_supervision_names is not None:
        synthetic_label_episodes = coerce_episode_list(
            synthetic_supervision_labels,
            name="synthetic_labels",
            dtype=np.float32,
        )
        _flatten_episode_frame(synthetic_label_episodes, column_names=synthetic_supervision_names).to_csv(
            out_dir / "synthetic_labels.csv",
            index=False,
        )
    if synthetic_probe_labels is not None and synthetic_probe_names is not None:
        synthetic_probe_episodes = coerce_episode_list(
            synthetic_probe_labels,
            name="synthetic_probe_labels",
            dtype=np.float32,
        )
        _flatten_episode_frame(synthetic_probe_episodes, column_names=synthetic_probe_names).to_csv(
            out_dir / "synthetic_probe_labels.csv",
            index=False,
        )

    if args.label_path:
        label_path = resolve_repo_path(ROOT, args.label_path)
        if label_path is None:
            raise ValueError("label_path must not be empty")
        label_array, label_names = _load_array_with_names(label_path, array_key=args.label_array_key)
        analysis_label_episodes = coerce_episode_list(label_array, name="analysis_labels", dtype=np.float32)
        analysis_label_names = list(label_names)
    elif supervision_enabled and synthetic_supervision_labels is not None and synthetic_supervision_names is not None:
        label_array = np.asarray(synthetic_supervision_labels, dtype=np.float32)
        label_names = list(synthetic_supervision_names)

    label_episodes = None
    if label_array is not None:
        raw_label_episodes = coerce_episode_list(label_array, name="labels", dtype=np.float32)
        if len(raw_label_episodes) != len(trajectory_episodes):
            raise ValueError("label array must contain the same number of episodes as trajectory")
        for idx, (trajectory_episode, label_episode) in enumerate(zip(trajectory_episodes, raw_label_episodes)):
            if len(label_episode) != len(trajectory_episode):
                raise ValueError(f"labels[{idx}] must have the same length as trajectory[{idx}]")
        if analysis_label_episodes is None:
            analysis_label_episodes = raw_label_episodes
            analysis_label_names = list(label_names) if label_names is not None else None

        if supervision_enabled:
            total_label_dim = _episode_feature_dim(raw_label_episodes)
            q_indices = _infer_indices(args.q_label_indices, start=0, block_dim=args.q_dim, total_dim=total_label_dim)
            h_indices = _infer_indices(
                args.h_label_indices,
                start=len(q_indices),
                block_dim=args.h_dim,
                total_dim=total_label_dim,
            )
            splits = compute_episode_splits(
                trajectory_lengths,
                context_len=args.window,
                horizons=tuple(args.horizons),
                train_fraction=args.train_fraction,
            )
            skip_indices = q_indices if (args.q_supervision_mode == "angular" and args.q_supervised_weight > 0.0) else ()
            label_episodes, label_stats = _standardize_labels(
                raw_label_episodes,
                splits=splits,
                enabled=bool(args.label_standardize),
                skip_indices=skip_indices,
            )
            supervision_cfg = SupervisionConfig(
                q_indices=q_indices,
                h_indices=h_indices,
                q_weight=args.q_supervised_weight,
                h_weight=args.h_supervised_weight,
                q_mode=args.q_supervision_mode,
            )

    model_cfg = ModelConfig(
        input_dim=input_dim,
        context_len=args.window,
        q_dim=args.q_dim,
        h_dim=args.h_dim,
        koopman_dim=args.koopman_dim,
        latent_scheme=args.latent_scheme,
        koopman_input_mode=args.koopman_input_mode,
        hidden_coordinate_mode=args.hidden_coordinate_mode,
        modal_dim=args.modal_dim,
        modal_temperature=args.modal_temperature,
        encoder_type=args.encoder_type,
        encoder_levels=args.encoder_levels,
        encoder_kernel_size=args.encoder_kernel_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        vamp_head_depth=args.vamp_head_depth,
        vamp_whitening_momentum=args.vamp_whitening_momentum,
        vamp_whitening_eps=args.vamp_whitening_eps,
        hidden_rank=args.hidden_rank,
        rg_scale=args.rg_scale,
        coarse_strength=args.coarse_strength,
        rg_temperature=args.rg_temperature,
    )
    train_cfg = TrainConfig(
        epochs=int(curriculum_cfg["epochs"]),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        horizons=tuple(args.horizons),
        dt=args.dt,
        device=args.device,
        rg_horizon=args.rg_horizon,
        contract_batch=args.contract_batch,
        schedule_mode=str(curriculum_cfg.get("schedule_mode", "fractional")),
        phase1_fraction=float(curriculum_cfg["phase1_fraction"]),
        phase2_fraction=float(curriculum_cfg["phase2_fraction"]),
        phase3_lr_scale=float(curriculum_cfg["phase3_lr_scale"]),
        phase0_min_epochs=int(curriculum_cfg.get("phase0_min_epochs", 6)),
        phase1_min_epochs=int(curriculum_cfg.get("phase1_min_epochs", 10)),
        phase2_min_epochs=int(curriculum_cfg.get("phase2_min_epochs", 12)),
        phase3_min_epochs=int(curriculum_cfg.get("phase3_min_epochs", 6)),
        phase0_reconstruction_improve=float(curriculum_cfg.get("phase0_reconstruction_improve", 0.20)),
        phase0_koopman_stability_ratio=float(curriculum_cfg.get("phase0_koopman_stability_ratio", 1.25)),
        phase0_reconstruction_target=(
            float(curriculum_cfg["phase0_reconstruction_target"])
            if "phase0_reconstruction_target" in curriculum_cfg
            else None
        ),
        phase0_koopman_loss_ceiling=(
            float(curriculum_cfg["phase0_koopman_loss_ceiling"])
            if "phase0_koopman_loss_ceiling" in curriculum_cfg
            else None
        ),
        phase0_stable_validations=int(curriculum_cfg.get("phase0_stable_validations", 2)),
        phase1_prediction_improve=float(curriculum_cfg.get("phase1_prediction_improve", 0.20)),
        phase1_q_align_improve=float(curriculum_cfg.get("phase1_q_align_improve", 0.20)),
        phase1_prediction_target=(
            float(curriculum_cfg["phase1_prediction_target"])
            if "phase1_prediction_target" in curriculum_cfg
            else None
        ),
        phase1_q_align_target=(
            float(curriculum_cfg["phase1_q_align_target"])
            if "phase1_q_align_target" in curriculum_cfg
            else None
        ),
        phase2_prediction_plateau_tolerance=float(curriculum_cfg.get("phase2_prediction_plateau_tolerance", 0.02)),
        phase2_prediction_plateau_checks=int(curriculum_cfg.get("phase2_prediction_plateau_checks", 2)),
        phase2_separation_target=float(curriculum_cfg.get("phase2_separation_target", 0.08)),
        phase2_prediction_plateau_abs_tol=(
            float(curriculum_cfg["phase2_prediction_plateau_abs_tol"])
            if "phase2_prediction_plateau_abs_tol" in curriculum_cfg
            else None
        ),
        phase2_long_horizon_target=(
            float(curriculum_cfg["phase2_long_horizon_target"])
            if "phase2_long_horizon_target" in curriculum_cfg
            else None
        ),
        rollback_prediction_worsen_ratio=float(curriculum_cfg.get("rollback_prediction_worsen_ratio", 0.25)),
        rollback_long_horizon_worsen_ratio=float(curriculum_cfg.get("rollback_long_horizon_worsen_ratio", 0.35)),
        rollback_prediction_worsen_delta=(
            float(curriculum_cfg["rollback_prediction_worsen_delta"])
            if "rollback_prediction_worsen_delta" in curriculum_cfg
            else None
        ),
        rollback_long_horizon_worsen_delta=(
            float(curriculum_cfg["rollback_long_horizon_worsen_delta"])
            if "rollback_long_horizon_worsen_delta" in curriculum_cfg
            else None
        ),
        rollback_window=int(curriculum_cfg.get("rollback_window", 2)),
        rollback_weight_scale=float(curriculum_cfg.get("rollback_weight_scale", 0.5)),
        log_every=args.log_every,
        validation_interval=args.validation_interval,
        progress_bar=bool(args.progress_bar),
        early_stopping=bool(args.early_stopping),
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_start_phase=args.early_stopping_start_phase,
        seed=args.seed,
    )
    loss_cfg = LossConfig(
        vamp_weight=args.vamp_weight,
        vamp_align_weight=args.vamp_align_weight,
        koopman_weight=args.koopman_weight,
        diag_weight=args.diag_weight,
        latent_align_weight=args.latent_align_weight,
        semigroup_weight=args.semigroup_weight,
        contract_weight=args.contract_weight,
        separation_weight=args.separation_weight,
        rg_weight=args.rg_weight,
        metric_weight=args.metric_weight,
        metric_mode=args.metric_mode,
        hidden_l1_weight=args.hidden_l1_weight,
    )

    result = fit_model(
        trajectory_episodes,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        labels=(label_episodes if supervision_enabled else None),
        supervision_cfg=supervision_cfg,
    )

    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
            "loss_config": loss_cfg.to_dict(),
            "stats": result.stats.to_dict(),
            "supervision_config": (supervision_cfg.to_dict() if supervision_cfg is not None else None),
            "label_stats": label_stats,
        },
        out_dir / "model.pt",
    )
    result.history.to_csv(out_dir / "history.csv", index=False)
    _flatten_episode_frame(
        trajectory_episodes,
        column_names=obs_names,
        max_steps_per_episode=512,
    ).to_csv(out_dir / "trajectory_preview.csv", index=False)

    config_payload = {
        "run_config": effective_run_config,
        "config_source": args.config,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "curriculum_config": curriculum_cfg,
        "loss_config": loss_cfg.to_dict(),
        "stats": result.stats.to_dict(),
        "supervision_config": (supervision_cfg.to_dict() if supervision_cfg is not None else None),
        "label_stats": label_stats,
        "label_names": label_names,
        "analysis_label_names": analysis_label_names,
        "report_plots": bool(args.report_plots),
        "trajectory_points": int(args.trajectory_points),
        **source_summary,
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    if result.summary.get("schedule_events"):
        (out_dir / "phase_events.json").write_text(
            json.dumps(result.summary["schedule_events"], indent=2),
            encoding="utf-8",
        )

    probe_summary = None
    if args.label_path:
        probe_source_labels = analysis_label_episodes
        probe_source_names = analysis_label_names
    elif synthetic_probe_labels is not None and synthetic_probe_names is not None:
        probe_source_labels = coerce_episode_list(
            synthetic_probe_labels,
            name="synthetic_probe_labels",
            dtype=np.float32,
        )
        probe_source_names = list(synthetic_probe_names)
    else:
        probe_source_labels = analysis_label_episodes
        probe_source_names = analysis_label_names
    if probe_source_labels is not None and probe_source_names is not None:
        probe_summary, corr_df = _label_probe(
            trajectory=trajectory_episodes,
            labels=probe_source_labels,
            label_names=probe_source_names,
            result=result,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_batch_size=args.eval_batch_size,
        )
        (out_dir / "analysis_labels_probe.json").write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
        corr_df.to_csv(out_dir / "analysis_component_corrs.csv", index=False)
        if label_episodes is not None:
            (out_dir / "label_probe.json").write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
            corr_df.to_csv(out_dir / "label_component_corrs.csv", index=False)
        elif synthetic_probe_labels is not None:
            (out_dir / "synthetic_hidden_probe.json").write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
            corr_df.to_csv(out_dir / "synthetic_component_corrs.csv", index=False)

    if bool(args.report_plots) and probe_source_labels is not None and probe_source_names is not None and len(probe_source_names) >= 2:
        visual_payload = build_run_visual_payload(
            run_name=out_dir.name,
            title=out_dir.name.replace("_", " "),
            color="#b5542b",
            trajectory=trajectory_episodes,
            analysis_labels=probe_source_labels,
            result=result,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            history=result.history,
            summary=result.summary,
            eval_batch_size=args.eval_batch_size,
        )
        visual_payload.horizon_metrics.to_csv(out_dir / "horizon_metrics.csv", index=False)
        save_single_run_plots(
            visual_payload,
            out_dir,
            trajectory_points=args.trajectory_points,
        )

    print("\n=== neural dynamical system training finished ===")
    print(f"device: {result.summary['device']}")
    print(f"episodes: {len(trajectory_episodes)}")
    print(f"episode lengths: {trajectory_lengths}")
    print(f"best epoch: {result.summary['best_epoch']}")
    print(f"best val loss: {result.summary['best_val_loss']:.6f}")
    print(f"best val phase: {result.summary['best_val_phase']}")
    print(f"best val VAMP score: {result.summary['best_val_vamp_score']:.6f}")
    print(f"best val Koopman loss: {result.summary['best_val_koopman_loss']:.6f}")
    print(f"best val diagonalization loss: {result.summary['best_val_diag_loss']:.6f}")
    print(f"best val prediction loss: {result.summary['best_val_prediction_loss']:.6f}")
    print(f"best val latent-align loss: {result.summary['best_val_latent_align_loss']:.6f}")
    print(f"best val semigroup loss: {result.summary['best_val_semigroup_loss']:.6f}")
    print(f"best val contract loss: {result.summary['best_val_contract_loss']:.6f}")
    print(f"best val separation loss: {result.summary['best_val_separation_loss']:.6f}")
    print(f"best val rg loss: {result.summary['best_val_rg_loss']:.6f}")
    print(f"best val memory rate mean: {result.summary['best_val_memory_rate_mean']:.6f}")
    print(f"best val hidden spectral upper bound: {result.summary['best_val_hidden_sym_eig_upper']:.6f}")
    print(f"best val supervised loss: {result.summary['best_val_supervised_total_loss']:.6f}")
    print(f"last val rg loss: {result.summary['last_val_rg_loss']:.6f}")
    print(f"last val Koopman loss: {result.summary['last_val_koopman_loss']:.6f}")
    print(f"last val diagonalization loss: {result.summary['last_val_diag_loss']:.6f}")
    if result.summary.get("stopped_early"):
        print(
            "early stopping: "
            f"epoch={int(result.summary['stop_epoch'])} "
            f"monitor={result.summary['early_stopping_monitor']} "
            f"reason={result.summary['stop_reason']}"
        )
    if result.summary["best_phase3_epoch"] is not None:
        print(f"best phase-3 epoch: {result.summary['best_phase3_epoch']}")
        print(f"best phase-3 val Koopman loss: {result.summary['best_phase3_val_koopman_loss']:.6f}")
        print(f"best phase-3 val diagonalization loss: {result.summary['best_phase3_val_diag_loss']:.6f}")
        print(f"best phase-3 val contract loss: {result.summary['best_phase3_val_contract_loss']:.6f}")
        print(f"best phase-3 val rg loss: {result.summary['best_phase3_val_rg_loss']:.6f}")
        print(f"best phase-3 val supervised loss: {result.summary['best_phase3_val_supervised_total_loss']:.6f}")
    if result.summary.get("latent_scheme") == "soft_spectrum":
        print(
            "modal rate range: "
            f"{float(result.summary['modal_rate_min']):.6f} .. {float(result.summary['modal_rate_max']):.6f}"
        )
        print(
            "modal entropy (slow/memory): "
            f"{float(result.summary['modal_slow_entropy']):.6f} / {float(result.summary['modal_memory_entropy']):.6f}"
        )
    if probe_summary is not None:
        z_mean_r2 = probe_summary["block_probe_r2"]["z"]["mean_r2"]
        koopman_mean_r2 = probe_summary["block_probe_r2"]["koopman"]["mean_r2"]
        q_mean_r2 = probe_summary["block_probe_r2"]["q"]["mean_r2"]
        print(f"held-out probe mean R2 (koopman): {koopman_mean_r2:.6f}")
        print(f"held-out probe mean R2 (z): {z_mean_r2:.6f}")
        print(f"held-out probe mean R2 (q): {q_mean_r2:.6f}")
    print(f"results saved under: {out_dir}")


if __name__ == "__main__":
    main()
