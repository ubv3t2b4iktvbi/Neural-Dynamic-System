from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig
from .data import TrajectoryStats, coerce_episode_list
from .model import LatentRGManifoldAutoencoder
from .synthetic import SyntheticTrajectoryConfig, generate_synthetic_trajectory


@dataclass
class LoadedRun:
    model: torch.nn.Module
    stats: TrajectoryStats


def load_run_config(run_dir: Path) -> dict[str, object]:
    return json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))


def load_saved_model(
    run_dir: Path,
    *,
    device: torch.device,
) -> tuple[LoadedRun, ModelConfig, TrainConfig]:
    payload = torch.load(Path(run_dir) / "model.pt", map_location=device)
    model_cfg = ModelConfig(**payload["model_config"])
    train_cfg = TrainConfig(**payload["train_config"])
    model = LatentRGManifoldAutoencoder(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stats = TrajectoryStats(
        mean=torch.as_tensor(payload["stats"]["mean"]).cpu().numpy(),
        std=torch.as_tensor(payload["stats"]["std"]).cpu().numpy(),
    )
    return LoadedRun(model=model, stats=stats), model_cfg, train_cfg


def regenerate_analysis_data(run_dir: Path) -> tuple[list, list]:
    run_config = load_run_config(run_dir)
    synth_cfg = SyntheticTrajectoryConfig(**run_config["synthetic_config"])
    synth = generate_synthetic_trajectory(synth_cfg)
    trajectory = coerce_episode_list(synth["trajectory"], name="trajectory", dtype=float)
    labels = coerce_episode_list(synth["probe_labels"], name="probe_labels", dtype=float)
    return trajectory, labels
