from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import LossConfig, ModelConfig, SupervisionConfig, TrainConfig
from .data import TrajectoryStats, prepare_datasets
from .model import LatentRGManifoldAutoencoder


@dataclass
class FitResult:
    model: LatentRGManifoldAutoencoder
    history: pd.DataFrame
    summary: dict[str, object]
    stats: TrajectoryStats


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


def _normalized_pairwise_distances(values: Tensor) -> Tensor:
    dists = torch.cdist(values, values)
    return dists / (dists.mean().detach() + 1e-6)


def _metric_loss(windows: Tensor, q: Tensor, max_items: int) -> Tensor:
    count = windows.shape[0]
    if count <= 1:
        return windows.new_tensor(0.0)
    if count > max_items:
        idx = torch.randperm(count, device=windows.device)[:max_items]
        windows = windows[idx]
        q = q[idx]
    window_dist = _normalized_pairwise_distances(windows)
    latent_dist = _normalized_pairwise_distances(q)
    return F.mse_loss(latent_dist, window_dist)


def _center_features(x: Tensor) -> Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _matrix_inv_sqrt(mat: Tensor, eps: float = 1e-5) -> Tensor:
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(eps)
    inv_sqrt = evals.rsqrt()
    return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.transpose(-1, -2)


def _vamp2_score(x0: Tensor, x1: Tensor, eps: float = 1e-5) -> Tensor:
    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError("VAMP features must be rank-2 [B, D]")
    if x0.shape != x1.shape:
        raise ValueError(f"VAMP pair shape mismatch: {tuple(x0.shape)} vs {tuple(x1.shape)}")
    batch = int(x0.shape[0])
    if batch < 2:
        return x0.new_tensor(0.0)
    x0c = _center_features(x0)
    x1c = _center_features(x1)
    denom = float(max(batch - 1, 1))
    eye = torch.eye(x0.shape[1], device=x0.device, dtype=x0.dtype)
    c00 = (x0c.transpose(0, 1) @ x0c) / denom + eps * eye
    c11 = (x1c.transpose(0, 1) @ x1c) / denom + eps * eye
    c01 = (x0c.transpose(0, 1) @ x1c) / denom
    k = _matrix_inv_sqrt(c00, eps=eps) @ c01 @ _matrix_inv_sqrt(c11, eps=eps)
    return (k * k).sum()


def _time_lag_covariance(x0: Tensor, x1: Tensor, eps: float = 1e-6) -> Tensor:
    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError("Time-lagged covariance inputs must be rank-2 [B, D]")
    if x0.shape != x1.shape:
        raise ValueError(f"time-lagged covariance shape mismatch: {tuple(x0.shape)} vs {tuple(x1.shape)}")
    batch = int(x0.shape[0])
    if batch < 2:
        dim = int(x0.shape[1])
        return torch.zeros((dim, dim), device=x0.device, dtype=x0.dtype)
    x0c = _center_features(x0)
    x1c = _center_features(x1)
    cov = (x0c.transpose(0, 1) @ x1c) / float(max(batch - 1, 1))
    return cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)


def _offdiag_frobenius_loss(mat: Tensor) -> Tensor:
    offdiag = mat - torch.diag_embed(torch.diagonal(mat, dim1=-2, dim2=-1))
    return offdiag.pow(2).mean()


def _koopman_consistency_loss(
    modal_t: Tensor,
    modal_next: Tensor,
    modal_rates: Tensor,
    *,
    dt: float,
) -> Tensor:
    if modal_t.ndim != 2 or modal_next.ndim != 2:
        raise ValueError("Koopman modal features must be rank-2 [B, K]")
    if modal_t.shape != modal_next.shape:
        raise ValueError(f"Koopman modal shape mismatch: {tuple(modal_t.shape)} vs {tuple(modal_next.shape)}")
    if modal_t.shape[1] == 0:
        return modal_t.new_tensor(0.0)
    if modal_rates.ndim == 2:
        rates = modal_rates
    else:
        rates = modal_rates.unsqueeze(0).expand_as(modal_t)
    decay = torch.exp(-rates * float(dt))
    modal_pred = decay * modal_t
    return F.mse_loss(modal_next, modal_pred)


def _rollout_cache(
    model: LatentRGManifoldAutoencoder,
    z0: Tensor,
    horizons: tuple[int, ...],
    *,
    dt: float,
) -> dict[int, Tensor]:
    cache: dict[int, Tensor] = {}
    prev_horizon = 0
    current = z0
    for horizon in sorted(horizons):
        current = model.flow(current, dt=dt, steps=horizon - prev_horizon)
        cache[horizon] = current
        prev_horizon = horizon
    return cache


def _latent_align_loss(
    rollout: dict[int, Tensor],
    encoded_future: dict[int, Tensor],
) -> Tensor:
    if not rollout:
        ref = next(iter(encoded_future.values()))
        return ref.new_tensor(0.0)
    terms = [F.mse_loss(rollout[h], encoded_future[h]) for h in rollout]
    return torch.stack(terms).mean()


def _q_align_loss(
    model: LatentRGManifoldAutoencoder,
    rollout: dict[int, Tensor],
    future_components: dict[int, dict[str, Tensor]],
) -> Tensor:
    if not rollout:
        ref = next(iter(future_components.values()))["q"]
        return ref.new_tensor(0.0)
    terms = [
        F.mse_loss(model.split_latent(rollout[horizon])[0], future_components[horizon]["q"])
        for horizon in rollout
    ]
    return torch.stack(terms).mean()


def _semigroup_loss(
    model: LatentRGManifoldAutoencoder,
    encoded_future: dict[int, Tensor],
    *,
    dt: float,
) -> Tensor:
    horizons = sorted(encoded_future)
    if len(horizons) < 2:
        ref = next(iter(encoded_future.values()))
        return ref.new_tensor(0.0)
    available = set(horizons)
    terms: list[Tensor] = []
    for h1 in horizons:
        for h2 in horizons:
            total = h1 + h2
            if total not in available:
                continue
            composed = model.flow(encoded_future[h1], dt=dt, steps=h2)
            terms.append(F.mse_loss(composed, encoded_future[total]))
    if not terms:
        ref = next(iter(encoded_future.values()))
        return ref.new_tensor(0.0)
    return torch.stack(terms).mean()


def _memory_contract_loss(hidden_sym_eig_upper: Tensor, *, margin: float) -> Tensor:
    return F.relu(hidden_sym_eig_upper + float(margin)).mean()


def _phase_scales(phase: int) -> dict[str, float]:
    if phase == 1:
        return {
            "koopman": 1.0,
            "diag": 1.0,
            "prediction": 0.0,
            "latent_align": 0.0,
            "semigroup": 0.0,
            "contract": 0.0,
            "separation": 0.0,
            "rg": 0.0,
        }
    if phase == 2:
        return {
            "koopman": 1.0,
            "diag": 1.0,
            "prediction": 1.0,
            "latent_align": 1.0,
            "semigroup": 0.0,
            "contract": 1.0,
            "separation": 1.0,
            "rg": 0.0,
        }
    return {
        "koopman": 1.0,
        "diag": 1.0,
        "prediction": 1.0,
        "latent_align": 1.0,
        "semigroup": 1.0,
        "contract": 1.0,
        "separation": 1.0,
        "rg": 1.0,
    }


def _curriculum_phase(progress: float, train_cfg: TrainConfig) -> int:
    if progress < float(train_cfg.phase1_fraction):
        return 1
    if progress < float(train_cfg.phase2_fraction):
        return 2
    return 3


def _set_trainability(model: LatentRGManifoldAutoencoder, *, dynamics_trainable: bool) -> None:
    for module in model.module_groups().get("dynamics", []):
        for param in module.parameters():
            param.requires_grad_(dynamics_trainable)


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    *,
    scale: float,
) -> None:
    for base_lr, group in zip(base_lrs, optimizer.param_groups):
        group["lr"] = float(base_lr) * float(scale)


def _supervised_component_loss(
    current_values: Tensor,
    future_values: dict[int, Tensor],
    *,
    batch: dict[str, Tensor],
    indices: tuple[int, ...],
    mode: str = "direct",
) -> Tensor:
    if not indices or "label" not in batch:
        return current_values.new_tensor(0.0)
    target_current = batch["label"][:, list(indices)]
    if mode == "angular":

        def _loss_fn(pred: Tensor, target: Tensor) -> Tensor:
            wrapped = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
            return wrapped.pow(2).mean()

    else:

        def _loss_fn(pred: Tensor, target: Tensor) -> Tensor:
            return F.mse_loss(pred, target)

    terms = [_loss_fn(current_values, target_current)]
    if "future_labels" in batch:
        for future_idx, horizon in enumerate(sorted(future_values)):
            target_future = batch["future_labels"][:, future_idx, :][:, list(indices)]
            terms.append(_loss_fn(future_values[horizon], target_future))
    return torch.stack(terms).mean()


def _loss_bundle(
    model: LatentRGManifoldAutoencoder,
    batch: dict[str, Tensor],
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    supervision_cfg: SupervisionConfig | None,
    *,
    phase: int,
) -> dict[str, Tensor]:
    zero = batch["window"].new_tensor(0.0)
    phase_scale = _phase_scales(phase)
    current_comp = model.encode_components(batch["window"], update_whitener=True)
    z0 = current_comp["z"]
    q0 = current_comp["q"]
    koopman0 = current_comp["koopman"]
    stats0 = model.latent_statistics(z0)

    reconstruction = model.decode(z0)
    rec_loss = F.mse_loss(reconstruction, batch["current"])

    rollout = _rollout_cache(model, z0, train_cfg.horizons, dt=train_cfg.dt)
    future_components = {
        horizon: model.encode_components(batch["future_windows"][:, idx, :, :], update_whitener=False)
        for idx, horizon in enumerate(train_cfg.horizons)
    }
    encoded_future = {horizon: comp["z"] for horizon, comp in future_components.items()}
    primary_horizon = train_cfg.horizons[0]
    lag_dt = float(train_cfg.dt) * float(primary_horizon)
    koopman1 = future_components[primary_horizon]["koopman"]

    vamp_score = _vamp2_score(koopman0, koopman1)
    vamp_loss = -vamp_score / float(max(model.koopman_dim, 1))
    diag_loss = _offdiag_frobenius_loss(_time_lag_covariance(koopman0, koopman1))
    koopman_loss = _koopman_consistency_loss(
        current_comp["koopman"],
        future_components[primary_horizon]["koopman"],
        current_comp["koopman_rates"],
        dt=lag_dt,
    )
    q_align_loss = _q_align_loss(model, rollout, future_components)

    pred_terms = [
        F.mse_loss(model.decode(rollout[horizon]), batch["future"][:, idx, :])
        for idx, horizon in enumerate(train_cfg.horizons)
    ]
    pred_loss = torch.stack(pred_terms).mean() if pred_terms else zero
    latent_align_loss = _latent_align_loss(rollout, encoded_future)
    semigroup_loss = _semigroup_loss(model, encoded_future, dt=train_cfg.dt)

    slow_scale = stats0["slow_rates"].mean(dim=-1, keepdim=True)
    memory_scale = stats0["memory_rates"].mean(dim=-1, keepdim=True)
    sep_gap = memory_scale - slow_scale
    separation_loss = F.relu(loss_cfg.separation_margin - sep_gap).mean()
    contract_loss = _memory_contract_loss(
        stats0["hidden_sym_eig_upper"],
        margin=loss_cfg.contraction_margin,
    )

    memory_penalty = stats0["h"].abs().mean()
    memory_drive_mag = stats0["hidden_drive"].abs().mean()
    hidden_sym_eig_upper = stats0["hidden_sym_eig_upper"].mean()
    slow_residual_mag = stats0["slow_residual"].abs().mean()
    koopman_rate_mean = current_comp["koopman_rates"].mean()
    q_rate_mean = current_comp["q_rates"].mean()

    if loss_cfg.rg_weight > 0.0 and phase_scale["rg"] > 0.0:
        coarse_after_fine = model.coarse_grain(model.flow(z0, dt=train_cfg.dt, steps=train_cfg.rg_horizon))
        fine_after_coarse = model.flow(
            model.coarse_grain(z0),
            dt=train_cfg.dt * model.cfg.rg_scale,
            steps=train_cfg.rg_horizon,
        )
        rg_loss = F.mse_loss(coarse_after_fine, fine_after_coarse)
    else:
        rg_loss = zero

    metric_loss = _metric_loss(batch["flat_window"], stats0["q"], train_cfg.metric_subsample)
    q_current = current_comp["q_raw"] if (supervision_cfg is not None and supervision_cfg.q_mode == "angular") else current_comp["q"]
    q_future = {
        horizon: (comp["q_raw"] if (supervision_cfg is not None and supervision_cfg.q_mode == "angular") else comp["q"])
        for horizon, comp in future_components.items()
    }
    q_supervised_loss = _supervised_component_loss(
        q_current,
        q_future,
        batch=batch,
        indices=(supervision_cfg.q_indices if supervision_cfg is not None else ()),
        mode=(supervision_cfg.q_mode if supervision_cfg is not None else "direct"),
    )
    h_supervised_loss = _supervised_component_loss(
        current_comp["h"],
        {horizon: comp["h"] for horizon, comp in future_components.items()},
        batch=batch,
        indices=(supervision_cfg.h_indices if supervision_cfg is not None else ()),
    )
    supervised_total_loss = (
        (float(supervision_cfg.q_weight) * q_supervised_loss if supervision_cfg is not None else zero)
        + (float(supervision_cfg.h_weight) * h_supervised_loss if supervision_cfg is not None else zero)
    )

    total = (
        loss_cfg.reconstruction_weight * rec_loss
        + loss_cfg.vamp_weight * vamp_loss
        + loss_cfg.vamp_align_weight * q_align_loss
        + (loss_cfg.koopman_weight * phase_scale["koopman"]) * koopman_loss
        + (loss_cfg.diag_weight * phase_scale["diag"]) * diag_loss
        + (loss_cfg.prediction_weight * phase_scale["prediction"]) * pred_loss
        + (loss_cfg.latent_align_weight * phase_scale["latent_align"]) * latent_align_loss
        + (loss_cfg.semigroup_weight * phase_scale["semigroup"]) * semigroup_loss
        + (loss_cfg.contract_weight * phase_scale["contract"]) * contract_loss
        + (loss_cfg.separation_weight * phase_scale["separation"]) * separation_loss
        + (loss_cfg.rg_weight * phase_scale["rg"]) * rg_loss
        + loss_cfg.metric_weight * metric_loss
        + loss_cfg.hidden_l1_weight * memory_penalty
        + supervised_total_loss
    )
    return {
        "loss": total,
        "reconstruction_loss": rec_loss,
        "vamp_loss": vamp_loss,
        "vamp_score": vamp_score,
        "koopman_loss": koopman_loss,
        "diag_loss": diag_loss,
        "q_align_loss": q_align_loss,
        "vamp_align_loss": q_align_loss,
        "prediction_loss": pred_loss,
        "latent_align_loss": latent_align_loss,
        "semigroup_loss": semigroup_loss,
        "contract_loss": contract_loss,
        "separation_loss": separation_loss,
        "rg_loss": rg_loss,
        "metric_loss": metric_loss,
        "hidden_l1_loss": memory_penalty,
        "memory_l1_loss": memory_penalty,
        "memory_drive_mag": memory_drive_mag,
        "hidden_sym_eig_upper": hidden_sym_eig_upper,
        "slow_rate_mean": slow_scale.mean(),
        "memory_rate_mean": memory_scale.mean(),
        "koopman_rate_mean": koopman_rate_mean,
        "q_rate_mean": q_rate_mean,
        "slow_residual_mag": slow_residual_mag,
        "q_supervised_loss": q_supervised_loss,
        "h_supervised_loss": h_supervised_loss,
        "m_supervised_loss": h_supervised_loss,
        "supervised_total_loss": supervised_total_loss,
    }


def _run_epoch(
    model: LatentRGManifoldAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    supervision_cfg: SupervisionConfig | None,
    optimizer: torch.optim.Optimizer | None,
    phase: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device=device)
        bundle = _loss_bundle(
            model,
            batch,
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            supervision_cfg=supervision_cfg,
            phase=phase,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            bundle["loss"].backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip))
            optimizer.step()
        batch_size = int(batch["window"].shape[0])
        count += batch_size
        for name, value in bundle.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
    return {name: total / max(count, 1) for name, total in totals.items()}


def fit_model(
    trajectory: np.ndarray | Sequence[np.ndarray],
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    labels: np.ndarray | Sequence[np.ndarray] | None = None,
    supervision_cfg: SupervisionConfig | None = None,
) -> FitResult:
    _seed_everything(int(train_cfg.seed))
    device = _resolve_device(train_cfg.device)
    if supervision_cfg is not None and (
        supervision_cfg.q_weight > 0.0 or supervision_cfg.h_weight > 0.0
    ) and labels is None:
        raise ValueError("supervision_cfg has positive weights but no labels were provided")
    train_ds, val_ds, stats = prepare_datasets(
        trajectory,
        context_len=model_cfg.context_len,
        horizons=train_cfg.horizons,
        train_fraction=train_cfg.train_fraction,
        labels=labels,
    )
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.batch_size), shuffle=False)

    model = LatentRGManifoldAutoencoder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    history_rows: list[dict[str, float | int | str]] = []
    best_state_by_phase: dict[int, dict[str, Tensor]] = {1: copy.deepcopy(model.state_dict())}
    best_val_by_phase: dict[int, float] = {1: float("inf")}

    for epoch in range(1, int(train_cfg.epochs) + 1):
        progress = float(epoch) / float(max(int(train_cfg.epochs), 1))
        phase = _curriculum_phase(progress, train_cfg)
        _set_trainability(model, dynamics_trainable=phase != 1)
        _set_optimizer_lr(
            optimizer,
            base_lrs,
            scale=(float(train_cfg.phase3_lr_scale) if phase == 3 else 1.0),
        )
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            supervision_cfg=supervision_cfg,
            optimizer=optimizer,
            phase=phase,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            supervision_cfg=supervision_cfg,
            optimizer=None,
            phase=phase,
        )

        history_rows.append({"epoch": epoch, "phase": phase, "split": "train", **train_metrics})
        history_rows.append({"epoch": epoch, "phase": phase, "split": "val", **val_metrics})

        if phase not in best_val_by_phase or val_metrics["loss"] < best_val_by_phase[phase]:
            best_val_by_phase[phase] = val_metrics["loss"]
            best_state_by_phase[phase] = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % int(train_cfg.log_every) == 0 or epoch == int(train_cfg.epochs):
            print(
                f"[epoch {epoch:03d}] "
                f"phase={phase} "
                f"train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} "
                f"vamp={val_metrics['vamp_score']:.5f} "
                f"koop={val_metrics['koopman_loss']:.5f} "
                f"diag={val_metrics['diag_loss']:.5f} "
                f"pred={val_metrics['prediction_loss']:.5f} "
                f"align={val_metrics['latent_align_loss']:.5f} "
                f"sg={val_metrics['semigroup_loss']:.5f} "
                f"contract={val_metrics['contract_loss']:.5f} "
                f"gap={val_metrics['separation_loss']:.5f} "
                f"mem={val_metrics['hidden_l1_loss']:.5f}"
            )

    history = pd.DataFrame(history_rows)
    val_history = history[history["split"] == "val"].copy()
    highest_phase = int(val_history["phase"].max())
    selected_phase_history = val_history[val_history["phase"] == highest_phase].copy()
    best_val_row = selected_phase_history.sort_values("loss", kind="stable").iloc[0].to_dict()
    overall_best_val_row = val_history.sort_values("loss", kind="stable").iloc[0].to_dict()
    last_val_row = val_history.sort_values("epoch", kind="stable").iloc[-1].to_dict()
    phase3_history = val_history[val_history["phase"] == 3].copy()
    if not phase3_history.empty:
        best_phase3_row = phase3_history.sort_values("loss", kind="stable").iloc[0].to_dict()
    else:
        best_phase3_row = None
    model.load_state_dict(best_state_by_phase.get(highest_phase, best_state_by_phase[min(best_state_by_phase)]))

    summary: dict[str, object] = {
        "device": str(device),
        "train_episodes": int(len(train_ds.episodes)),
        "val_episodes": int(len(val_ds.episodes)),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "rg_enabled": bool(loss_cfg.rg_weight > 0.0),
        "supervision_enabled": bool(
            supervision_cfg is not None
            and (supervision_cfg.q_weight > 0.0 or supervision_cfg.h_weight > 0.0)
        ),
        "latent_scheme": str(model.cfg.latent_scheme),
        "selection_phase": int(highest_phase),
        "overall_best_val_loss": float(overall_best_val_row["loss"]),
        "overall_best_val_phase": int(overall_best_val_row["phase"]),
        "rg_active_epoch_count": int((val_history["phase"] == 3).sum()),
        "best_val_loss": float(best_val_row["loss"]),
        "best_val_phase": int(best_val_row["phase"]),
        "best_val_vamp_score": float(best_val_row["vamp_score"]),
        "best_val_koopman_loss": float(best_val_row["koopman_loss"]),
        "best_val_diag_loss": float(best_val_row["diag_loss"]),
        "best_val_prediction_loss": float(best_val_row["prediction_loss"]),
        "best_val_q_align_loss": float(best_val_row["q_align_loss"]),
        "best_val_latent_align_loss": float(best_val_row["latent_align_loss"]),
        "best_val_semigroup_loss": float(best_val_row["semigroup_loss"]),
        "best_val_contract_loss": float(best_val_row["contract_loss"]),
        "best_val_separation_loss": float(best_val_row["separation_loss"]),
        "best_val_rg_loss": float(best_val_row["rg_loss"]),
        "best_val_hidden_l1_loss": float(best_val_row["hidden_l1_loss"]),
        "best_val_koopman_rate_mean": float(best_val_row["koopman_rate_mean"]),
        "best_val_q_rate_mean": float(best_val_row["q_rate_mean"]),
        "best_val_memory_rate_mean": float(best_val_row["memory_rate_mean"]),
        "best_val_slow_rate_mean": float(best_val_row["slow_rate_mean"]),
        "best_val_hidden_sym_eig_upper": float(best_val_row["hidden_sym_eig_upper"]),
        "best_val_q_supervised_loss": float(best_val_row["q_supervised_loss"]),
        "best_val_h_supervised_loss": float(best_val_row["h_supervised_loss"]),
        "best_val_supervised_total_loss": float(best_val_row["supervised_total_loss"]),
        "best_epoch": int(best_val_row["epoch"]),
        "last_epoch": int(last_val_row["epoch"]),
        "last_val_loss": float(last_val_row["loss"]),
        "last_val_koopman_loss": float(last_val_row["koopman_loss"]),
        "last_val_diag_loss": float(last_val_row["diag_loss"]),
        "last_val_prediction_loss": float(last_val_row["prediction_loss"]),
        "last_val_q_align_loss": float(last_val_row["q_align_loss"]),
        "last_val_semigroup_loss": float(last_val_row["semigroup_loss"]),
        "last_val_rg_loss": float(last_val_row["rg_loss"]),
        "last_val_contract_loss": float(last_val_row["contract_loss"]),
        "last_val_separation_loss": float(last_val_row["separation_loss"]),
        "last_val_koopman_rate_mean": float(last_val_row["koopman_rate_mean"]),
        "last_val_q_rate_mean": float(last_val_row["q_rate_mean"]),
        "last_val_memory_rate_mean": float(last_val_row["memory_rate_mean"]),
        "last_val_hidden_sym_eig_upper": float(last_val_row["hidden_sym_eig_upper"]),
        "last_val_supervised_total_loss": float(last_val_row["supervised_total_loss"]),
        "best_phase3_epoch": (int(best_phase3_row["epoch"]) if best_phase3_row is not None else None),
        "best_phase3_val_loss": (float(best_phase3_row["loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_koopman_loss": (float(best_phase3_row["koopman_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_diag_loss": (float(best_phase3_row["diag_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_prediction_loss": (float(best_phase3_row["prediction_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_q_align_loss": (float(best_phase3_row["q_align_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_semigroup_loss": (float(best_phase3_row["semigroup_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_rg_loss": (float(best_phase3_row["rg_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_contract_loss": (float(best_phase3_row["contract_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_separation_loss": (float(best_phase3_row["separation_loss"]) if best_phase3_row is not None else None),
        "best_phase3_val_koopman_rate_mean": (float(best_phase3_row["koopman_rate_mean"]) if best_phase3_row is not None else None),
        "best_phase3_val_q_rate_mean": (float(best_phase3_row["q_rate_mean"]) if best_phase3_row is not None else None),
        "best_phase3_val_hidden_sym_eig_upper": (float(best_phase3_row["hidden_sym_eig_upper"]) if best_phase3_row is not None else None),
        "best_phase3_val_supervised_total_loss": (float(best_phase3_row["supervised_total_loss"]) if best_phase3_row is not None else None),
    }
    summary.update(model.spectrum_summary())
    return FitResult(model=model, history=history, summary=summary, stats=stats)
