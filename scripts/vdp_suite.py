from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_dynamic_system import (  # noqa: E402
    ModelConfig,
    SyntheticTrajectoryConfig,
    TrainConfig,
    TrajectoryStats,
    coerce_episode_list,
    generate_synthetic_trajectory,
)
from neural_dynamic_system.reporting import build_run_visual_payload, save_comparison_artifacts  # noqa: E402


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


@dataclass
class LoadedRun:
    model: torch.nn.Module
    stats: TrajectoryStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the structured van der Pol ablation study.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval"])
    parser.add_argument("--out_root", type=str, default="runs/neural_dynamic_system/vdp_structured_study_20260316")
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
    return parser.parse_args()


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


def _append_bool_flag(args_list: list[str], name: str, enabled: bool) -> None:
    args_list.append(f"--{name}" if enabled else f"--no-{name}")


def _common_cli_args(args: argparse.Namespace) -> list[str]:
    cli_args = [
        "--synthetic_kind",
        "van_der_pol",
        "--van_der_pol_mu",
        str(args.van_der_pol_mu),
        "--num_episodes",
        str(args.num_episodes),
        "--steps",
        str(args.steps),
        "--dt",
        str(args.dt),
        "--obs_dim",
        str(args.obs_dim),
        "--burn_in",
        str(args.burn_in),
        "--noise_std",
        str(args.noise_std),
        "--window",
        str(args.window),
        "--q_dim",
        str(args.q_dim),
        "--h_dim",
        str(args.h_dim),
        "--koopman_dim",
        str(args.koopman_dim),
        "--modal_dim",
        str(args.modal_dim),
        "--hidden_dim",
        str(args.hidden_dim),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--epochs",
        str(args.epochs),
        "--curriculum_preset",
        "vdp_structured",
        "--train_fraction",
        str(args.train_fraction),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--log_every",
        str(args.log_every),
        "--validation_interval",
        str(args.validation_interval),
        "--vamp_weight",
        str(args.vamp_weight),
        "--koopman_weight",
        str(args.koopman_weight),
        "--diag_weight",
        str(args.diag_weight),
        "--latent_align_weight",
        "0.0",
        "--contract_weight",
        str(args.contract_weight),
        "--separation_weight",
        str(args.separation_weight),
        "--rg_weight",
        "0.0",
        "--metric_weight",
        str(args.metric_weight),
        "--hidden_l1_weight",
        str(args.hidden_l1_weight),
        "--q_supervised_weight",
        "0.0",
        "--h_supervised_weight",
        "0.0",
        "--trajectory_points",
        str(args.trajectory_points),
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
    ]
    _append_bool_flag(cli_args, "progress_bar", False)
    _append_bool_flag(cli_args, "report_plots", True)
    _append_bool_flag(cli_args, "early_stopping", bool(args.early_stopping))
    cli_args.extend(
        [
            "--early_stopping_monitor",
            "prediction_loss",
            "--early_stopping_patience",
            str(args.early_stopping_patience),
            "--early_stopping_min_delta",
            str(args.early_stopping_min_delta),
            "--early_stopping_start_phase",
            "3",
        ]
    )
    return cli_args


def train_run(spec: RunSpec, args: argparse.Namespace, out_root: Path) -> None:
    run_dir = _run_dir(out_root, spec)
    if run_dir.exists() and (run_dir / "summary.json").exists() and not args.force:
        print(f"[skip] {spec.title}: {run_dir}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "neural_dynamic_system.cli",
        "--out_dir",
        str(run_dir),
        "--koopman_input_mode",
        spec.koopman_input_mode,
        "--hidden_coordinate_mode",
        spec.hidden_coordinate_mode,
        "--metric_mode",
        spec.metric_mode,
        "--semigroup_weight",
        str(args.semigroup_weight * spec.semigroup_scale),
        *_common_cli_args(args),
    ]
    print(f"\n[train] {spec.title}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best_rmse = math.sqrt(max(float(summary["best_val_prediction_loss"]), 0.0))
    print(
        f"[done] {spec.title}: best_epoch={summary['best_epoch']} "
        f"best_val_rmse={best_rmse:.4f} phase={summary['best_val_phase']}"
    )


def _load_run_config(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, TrajectoryStats, ModelConfig, TrainConfig]:
    payload = torch.load(run_dir / "model.pt", map_location=device)
    model_cfg = ModelConfig(**payload["model_config"])
    train_cfg = TrainConfig(**payload["train_config"])
    model = torch.nn.Module()
    from neural_dynamic_system import LatentRGManifoldAutoencoder  # noqa: E402

    model = LatentRGManifoldAutoencoder(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stats = TrajectoryStats(
        mean=torch.as_tensor(payload["stats"]["mean"]).cpu().numpy(),
        std=torch.as_tensor(payload["stats"]["std"]).cpu().numpy(),
    )
    return model, stats, model_cfg, train_cfg


def _regenerate_analysis_labels(run_dir: Path) -> tuple[list, list]:
    run_config = _load_run_config(run_dir)
    synth_cfg = SyntheticTrajectoryConfig(**run_config["synthetic_config"])
    synth = generate_synthetic_trajectory(synth_cfg)
    trajectory = coerce_episode_list(synth["trajectory"], name="trajectory", dtype=float)
    labels = coerce_episode_list(synth["probe_labels"], name="probe_labels", dtype=float)
    return trajectory, labels


def evaluate_runs(args: argparse.Namespace, out_root: Path, specs: list[RunSpec]) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    payloads: list[RunVisualPayload] = []
    from neural_dynamic_system.reporting import RunVisualPayload  # noqa: E402

    for spec in specs:
        run_dir = _run_dir(out_root, spec)
        if not (run_dir / "summary.json").exists():
            raise FileNotFoundError(f"Missing run outputs for {spec.name}: {run_dir}")
        model, stats, model_cfg, train_cfg = _load_model(run_dir, device=device)
        history = pd.read_csv(run_dir / "history.csv")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        trajectory, analysis_labels = _regenerate_analysis_labels(run_dir)
        loaded = LoadedRun(model=model, stats=stats)
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


def main() -> None:
    args = parse_args()
    out_root = (ROOT / args.out_root) if not Path(args.out_root).is_absolute() else Path(args.out_root)
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

    if args.mode in {"all", "train"}:
        for spec in specs:
            train_run(spec, args, out_root)

    if args.mode in {"all", "eval"}:
        evaluate_runs(args, out_root, specs)


if __name__ == "__main__":
    main()
