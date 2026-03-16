from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_dynamic_system.plots import save_comparison_artifacts  # noqa: E402
from neural_dynamic_system.reporting import RunVisualPayload, build_run_visual_payload  # noqa: E402
from neural_dynamic_system.run_artifacts import load_saved_model, regenerate_analysis_data  # noqa: E402
from neural_dynamic_system.run_config import merge_nested_mapping, resolve_repo_path, write_cli_config  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    name: str
    title: str
    koopman_input_mode: str
    hidden_coordinate_mode: str
    metric_mode: str
    semigroup_scale: float
    color: str


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        name="full_geometry_baseline",
        title="Full Geometry Baseline",
        koopman_input_mode="joint",
        hidden_coordinate_mode="normal_residual",
        metric_mode="mahalanobis_dynamics",
        semigroup_scale=1.0,
        color="#1d4e89",
    ),
    RunSpec(
        name="reduced_geometry",
        title="Reduced Geometry",
        koopman_input_mode="slow_only",
        hidden_coordinate_mode="normal_residual",
        metric_mode="mahalanobis_dynamics",
        semigroup_scale=1.0,
        color="#b5542b",
    ),
    RunSpec(
        name="reduced_direct",
        title="Reduced Direct",
        koopman_input_mode="slow_only",
        hidden_coordinate_mode="direct",
        metric_mode="euclidean",
        semigroup_scale=1.0,
        color="#2f7d4a",
    ),
    RunSpec(
        name="reduced_geometry_semigroup_weak",
        title="Reduced Geometry (Weak SG)",
        koopman_input_mode="slow_only",
        hidden_coordinate_mode="normal_residual",
        metric_mode="mahalanobis_dynamics",
        semigroup_scale=0.35,
        color="#7d3f98",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the van der Pol ablation suite.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval"])
    parser.add_argument("--out_root", type=str, default="runs/neural_dynamic_system/vdp_structured_study")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_semigroup_weak", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only", nargs="*", default=None, choices=[spec.name for spec in RUN_SPECS])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--obs_dim", type=int, default=6)
    parser.add_argument("--burn_in", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--van_der_pol_mu", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--q_dim", type=int, default=1)
    parser.add_argument("--h_dim", type=int, default=1)
    parser.add_argument("--koopman_dim", type=int, default=4)
    parser.add_argument("--modal_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--trajectory_points", type=int, default=500)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--vamp_weight", type=float, default=0.15)
    parser.add_argument("--koopman_weight", type=float, default=0.2)
    parser.add_argument("--diag_weight", type=float, default=0.02)
    parser.add_argument("--semigroup_weight", type=float, default=0.35)
    parser.add_argument("--contract_weight", type=float, default=0.2)
    parser.add_argument("--separation_weight", type=float, default=0.2)
    parser.add_argument("--metric_weight", type=float, default=0.05)
    parser.add_argument("--hidden_l1_weight", type=float, default=1e-4)
    return parser.parse_args(argv)


def selected_specs(args: argparse.Namespace) -> list[RunSpec]:
    if args.only:
        allowed = set(args.only)
        return [spec for spec in RUN_SPECS if spec.name in allowed]
    base = [spec for spec in RUN_SPECS if spec.name != "reduced_geometry_semigroup_weak"]
    if bool(args.include_semigroup_weak):
        base.append(next(spec for spec in RUN_SPECS if spec.name == "reduced_geometry_semigroup_weak"))
    return base


def _run_dir(out_root: Path, spec: RunSpec) -> Path:
    return out_root / spec.name


def _base_run_config(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {
            "out_dir": str(run_dir),
        },
        "data": {
            "data_path": None,
            "array_key": None,
            "label_path": None,
            "label_array_key": None,
        },
        "model": {
            "window": int(args.window),
            "q_dim": int(args.q_dim),
            "h_dim": int(args.h_dim),
            "koopman_dim": int(args.koopman_dim),
            "latent_scheme": "soft_spectrum",
            "koopman_input_mode": "slow_only",
            "hidden_coordinate_mode": "normal_residual",
            "modal_dim": int(args.modal_dim),
            "modal_temperature": 0.35,
            "encoder_type": "temporal_conv",
            "encoder_levels": 3,
            "encoder_kernel_size": 5,
            "hidden_dim": int(args.hidden_dim),
            "depth": 2,
            "vamp_head_depth": 1,
            "vamp_whitening_momentum": 0.05,
            "vamp_whitening_eps": 1e-5,
            "hidden_rank": 4,
            "rg_scale": 2.0,
            "coarse_strength": 0.25,
            "rg_temperature": 0.35,
        },
        "curriculum": {
            "curriculum_preset": "vdp_structured",
            "epochs": int(args.epochs),
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
        },
        "train": {
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "horizons": [int(horizon) for horizon in args.horizons],
            "dt": float(args.dt),
            "train_fraction": float(args.train_fraction),
            "device": str(args.device),
            "log_every": int(args.log_every),
            "validation_interval": int(args.validation_interval),
            "schedule_mode": None,
            "progress_bar": False,
            "early_stopping": bool(args.early_stopping),
            "early_stopping_monitor": "prediction_loss",
            "early_stopping_patience": int(args.early_stopping_patience),
            "early_stopping_min_delta": float(args.early_stopping_min_delta),
            "early_stopping_min_epochs": None,
            "early_stopping_start_phase": 3,
            "contract_batch": 16,
            "rg_horizon": 1,
            "eval_batch_size": int(args.eval_batch_size),
            "seed": int(args.seed),
        },
        "synthetic": {
            "steps": int(args.steps),
            "num_episodes": int(args.num_episodes),
            "obs_dim": int(args.obs_dim),
            "burn_in": int(args.burn_in),
            "noise_std": float(args.noise_std),
            "synthetic_kind": "van_der_pol",
            "van_der_pol_mu": float(args.van_der_pol_mu),
        },
        "loss": {
            "vamp_weight": float(args.vamp_weight),
            "vamp_align_weight": 0.25,
            "koopman_weight": float(args.koopman_weight),
            "diag_weight": float(args.diag_weight),
            "latent_align_weight": 0.0,
            "semigroup_weight": float(args.semigroup_weight),
            "contract_weight": float(args.contract_weight),
            "separation_weight": float(args.separation_weight),
            "rg_weight": 0.0,
            "metric_weight": float(args.metric_weight),
            "metric_mode": "mahalanobis_dynamics",
            "hidden_l1_weight": float(args.hidden_l1_weight),
        },
        "supervision": {
            "q_label_indices": None,
            "h_label_indices": None,
            "q_supervised_weight": 0.0,
            "q_supervision_mode": "direct",
            "h_supervised_weight": 0.0,
            "label_standardize": True,
        },
        "reporting": {
            "trajectory_points": int(args.trajectory_points),
            "report_plots": True,
        },
    }


def _run_override_config(spec: RunSpec) -> dict[str, object]:
    return {
        "model": {
            "koopman_input_mode": spec.koopman_input_mode,
            "hidden_coordinate_mode": spec.hidden_coordinate_mode,
        },
        "loss": {
            "metric_mode": spec.metric_mode,
        },
    }


def train_run(spec: RunSpec, args: argparse.Namespace, out_root: Path) -> None:
    run_dir = _run_dir(out_root, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = merge_nested_mapping(_base_run_config(args, run_dir), _run_override_config(spec))
    run_config["loss"]["semigroup_weight"] = float(args.semigroup_weight * spec.semigroup_scale)
    run_config_path = run_dir / "run_config.yaml"
    write_cli_config(run_config_path, run_config)
    if (run_dir / "summary.json").exists() and not args.force:
        print(f"[skip] {spec.title}: {run_dir}")
        return

    cmd = [
        sys.executable,
        "-m",
        "neural_dynamic_system.app",
        "train",
        "--config",
        str(run_config_path),
    ]
    print(f"\n[train] {spec.title}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best_rmse = math.sqrt(max(float(summary["best_val_one_step_prediction_loss"]), 0.0))
    print(
        f"[done] {spec.title}: best_epoch={summary['best_epoch']} "
        f"best_val_rmse={best_rmse:.4f} phase={summary['best_val_phase']}"
    )


def evaluate_runs(args: argparse.Namespace, out_root: Path, specs: list[RunSpec]) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    payloads: list[RunVisualPayload] = []
    for spec in specs:
        run_dir = _run_dir(out_root, spec)
        if not (run_dir / "summary.json").exists():
            raise FileNotFoundError(f"Missing run outputs for {spec.name}: {run_dir}")
        loaded, model_cfg, train_cfg = load_saved_model(run_dir, device=device)
        history = pd.read_csv(run_dir / "history.csv")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        trajectory, analysis_labels = regenerate_analysis_data(run_dir)
        payload = build_run_visual_payload(
            run_name=spec.name,
            title=spec.title,
            color=spec.color,
            trajectory=trajectory,
            analysis_labels=analysis_labels,
            result=loaded,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            history=history,
            summary=summary,
            eval_batch_size=args.eval_batch_size,
        )
        payloads.append(payload)

    plots_dir = out_root / "comparison"
    summary_df, _ = save_comparison_artifacts(payloads, plots_dir, trajectory_points=args.trajectory_points)
    print(f"\n[eval] saved comparison outputs under {plots_dir}")
    print(summary_df.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_root = resolve_repo_path(ROOT, args.out_root)
    if out_root is None:
        raise ValueError("out_root must not be empty")
    out_root.mkdir(parents=True, exist_ok=True)
    specs = selected_specs(args)

    study_config = {
        "mode": args.mode,
        "device": args.device,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "steps": args.steps,
        "dt": args.dt,
        "window": args.window,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_fraction": args.train_fraction,
        "validation_interval": args.validation_interval,
        "early_stopping": bool(args.early_stopping),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "horizons": list(args.horizons),
        "selected_runs": [spec.name for spec in specs],
    }
    (out_root / "study_config.json").write_text(json.dumps(study_config, indent=2), encoding="utf-8")
    write_cli_config(out_root / "study_config.yaml", study_config)

    if args.mode in {"all", "train"}:
        for spec in specs:
            train_run(spec, args, out_root)

    if args.mode in {"all", "eval"}:
        evaluate_runs(args, out_root, specs)


if __name__ == "__main__":
    main()
