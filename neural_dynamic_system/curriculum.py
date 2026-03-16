from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .config import TrainConfig

LOSS_NAMES = (
    "reconstruction",
    "vamp",
    "koopman",
    "diag",
    "prediction",
    "q_align",
    "latent_align",
    "semigroup",
    "contract",
    "separation",
    "rg",
    "metric",
    "hidden_l1",
)


@dataclass(frozen=True)
class PhaseSpec:
    phase: int
    name: str
    min_epochs: int
    lr_scale: float
    prediction_horizons: str
    q_align_horizons: str
    dynamics_trainable: bool
    loss_scales: Mapping[str, float]
    rollback_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseEvent:
    epoch: int
    action: str
    from_phase: int
    to_phase: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": int(self.epoch),
            "action": str(self.action),
            "from_phase": int(self.from_phase),
            "to_phase": int(self.to_phase),
            "reason": str(self.reason),
        }


@dataclass
class _MetricTracker:
    entry: dict[str, float] | None = None
    entry_epoch: int = 0
    best_prediction: float | None = None
    best_long_horizon: float | None = None
    stable_validations: int = 0
    plateau_checks: int = 0
    validations: int = 0


def _relative_improvement(start: float, current: float) -> float:
    denom = max(abs(float(start)), 1e-8)
    return (float(start) - float(current)) / denom


def _finite_metrics(metrics: Mapping[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def _base_scales(overrides: Mapping[str, float]) -> dict[str, float]:
    scales = {name: 0.0 for name in LOSS_NAMES}
    scales.update({str(name): float(value) for name, value in overrides.items()})
    return scales


def _legacy_fractional_specs(train_cfg: TrainConfig) -> tuple[PhaseSpec, ...]:
    return (
        PhaseSpec(
            phase=0,
            name="representation_bootstrap",
            min_epochs=1,
            lr_scale=1.0,
            prediction_horizons="none",
            q_align_horizons="none",
            dynamics_trainable=False,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 1.0,
                    "koopman": 1.0,
                    "diag": 1.0,
                    "metric": 1.0,
                }
            ),
        ),
        PhaseSpec(
            phase=1,
            name="local_dynamics",
            min_epochs=1,
            lr_scale=1.0,
            prediction_horizons="all",
            q_align_horizons="all",
            dynamics_trainable=True,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 1.0,
                    "koopman": 1.0,
                    "diag": 1.0,
                    "prediction": 1.0,
                    "q_align": 1.0,
                    "latent_align": 1.0,
                    "contract": 1.0,
                    "separation": 1.0,
                    "metric": 1.0,
                    "hidden_l1": 1.0,
                }
            ),
        ),
        PhaseSpec(
            phase=2,
            name="global_consistency",
            min_epochs=1,
            lr_scale=float(train_cfg.phase3_lr_scale),
            prediction_horizons="all",
            q_align_horizons="all",
            dynamics_trainable=True,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 1.0,
                    "koopman": 1.0,
                    "diag": 1.0,
                    "prediction": 1.0,
                    "q_align": 1.0,
                    "latent_align": 1.0,
                    "semigroup": 1.0,
                    "contract": 1.0,
                    "separation": 1.0,
                    "rg": 1.0,
                    "metric": 1.0,
                    "hidden_l1": 1.0,
                }
            ),
        ),
    )


def _metric_driven_specs(train_cfg: TrainConfig) -> tuple[PhaseSpec, ...]:
    return (
        PhaseSpec(
            phase=0,
            name="representation_bootstrap",
            min_epochs=int(train_cfg.phase0_min_epochs),
            lr_scale=1.0,
            prediction_horizons="none",
            q_align_horizons="none",
            dynamics_trainable=False,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 1.0,
                    "koopman": 1.0,
                    "diag": 0.35,
                    "metric": 0.25,
                }
            ),
        ),
        PhaseSpec(
            phase=1,
            name="local_one_step_dynamics",
            min_epochs=int(train_cfg.phase1_min_epochs),
            lr_scale=1.0,
            prediction_horizons="first",
            q_align_horizons="first",
            dynamics_trainable=True,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 1.0,
                    "koopman": 1.0,
                    "diag": 0.25,
                    "prediction": 1.0,
                    "q_align": 1.0,
                    "contract": 1.0,
                    "metric": 0.35,
                    "hidden_l1": 1.0,
                }
            ),
            rollback_keys=("prediction", "q_align", "contract"),
        ),
        PhaseSpec(
            phase=2,
            name="short_mid_rollout",
            min_epochs=int(train_cfg.phase2_min_epochs),
            lr_scale=1.0,
            prediction_horizons="all",
            q_align_horizons="all",
            dynamics_trainable=True,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 0.5,
                    "koopman": 0.5,
                    "diag": 0.20,
                    "prediction": 1.0,
                    "q_align": 1.0,
                    "semigroup": 0.35,
                    "contract": 1.0,
                    "separation": 1.0,
                    "metric": 0.35,
                    "hidden_l1": 1.0,
                }
            ),
            rollback_keys=("semigroup", "separation"),
        ),
        PhaseSpec(
            phase=3,
            name="consistency_refinement",
            min_epochs=int(train_cfg.phase3_min_epochs),
            lr_scale=float(train_cfg.phase3_lr_scale),
            prediction_horizons="all",
            q_align_horizons="all",
            dynamics_trainable=True,
            loss_scales=_base_scales(
                {
                    "reconstruction": 1.0,
                    "vamp": 0.4,
                    "koopman": 0.4,
                    "diag": 0.15,
                    "prediction": 1.0,
                    "q_align": 1.0,
                    "semigroup": 1.0,
                    "contract": 1.0,
                    "separation": 1.0,
                    "metric": 0.30,
                    "hidden_l1": 1.0,
                }
            ),
            rollback_keys=("semigroup",),
        ),
    )


def _select_horizons(mode: str, horizons: tuple[int, ...]) -> tuple[int, ...]:
    if not horizons or mode == "none":
        return ()
    if mode == "first":
        return (int(horizons[0]),)
    if mode == "all":
        return tuple(int(horizon) for horizon in horizons)
    raise ValueError(f"Unsupported horizon mode: {mode}")


class FractionalPhaseController:
    def __init__(self, train_cfg: TrainConfig):
        self.train_cfg = train_cfg
        self.specs = _legacy_fractional_specs(train_cfg)
        self.current_phase = 0
        self.events: list[PhaseEvent] = []

    def current_spec(self, epoch: int) -> PhaseSpec:
        progress = float(epoch) / float(max(int(self.train_cfg.epochs), 1))
        if progress < float(self.train_cfg.phase1_fraction):
            self.current_phase = 0
        elif progress < float(self.train_cfg.phase2_fraction):
            self.current_phase = 1
        else:
            self.current_phase = 2
        return self.specs[self.current_phase]

    def loss_scales(self, epoch: int) -> dict[str, float]:
        return dict(self.current_spec(epoch).loss_scales)

    def prediction_horizons(self, epoch: int, horizons: tuple[int, ...]) -> tuple[int, ...]:
        return _select_horizons(self.current_spec(epoch).prediction_horizons, horizons)

    def q_align_horizons(self, epoch: int, horizons: tuple[int, ...]) -> tuple[int, ...]:
        return _select_horizons(self.current_spec(epoch).q_align_horizons, horizons)

    def observe_validation(self, epoch: int, val_metrics: Mapping[str, float]) -> PhaseEvent | None:
        del epoch, val_metrics
        return None


class MetricDrivenPhaseController:
    def __init__(self, train_cfg: TrainConfig):
        self.train_cfg = train_cfg
        self.specs = _metric_driven_specs(train_cfg)
        self.current_phase = 0
        self.events: list[PhaseEvent] = []
        self._tracker = _MetricTracker()
        self._rollback_reference: dict[str, float] | None = None
        self._rollback_reference_phase: int | None = None
        self._phase_penalties: dict[int, float] = {spec.phase: 1.0 for spec in self.specs}

    @property
    def spec(self) -> PhaseSpec:
        return self.specs[self.current_phase]

    def current_spec(self, epoch: int) -> PhaseSpec:
        del epoch
        return self.spec

    def loss_scales(self, epoch: int) -> dict[str, float]:
        del epoch
        scales = dict(self.spec.loss_scales)
        for phase_idx in range(self.current_phase + 1):
            penalty = float(self._phase_penalties.get(phase_idx, 1.0))
            if abs(penalty - 1.0) < 1e-8:
                continue
            for key in self.specs[phase_idx].rollback_keys:
                if key in scales:
                    scales[key] *= penalty
        return scales

    def prediction_horizons(self, epoch: int, horizons: tuple[int, ...]) -> tuple[int, ...]:
        del epoch
        return _select_horizons(self.spec.prediction_horizons, horizons)

    def q_align_horizons(self, epoch: int, horizons: tuple[int, ...]) -> tuple[int, ...]:
        del epoch
        return _select_horizons(self.spec.q_align_horizons, horizons)

    def _reset_tracker(self, metrics: Mapping[str, float], *, epoch: int) -> None:
        self._tracker = _MetricTracker(
            entry={key: float(value) for key, value in metrics.items()},
            entry_epoch=int(epoch),
            best_prediction=float(metrics["prediction_loss"]),
            best_long_horizon=float(metrics["long_horizon_prediction_loss"]),
            stable_validations=0,
            plateau_checks=0,
            validations=0,
        )

    def _advance(self, epoch: int, reason: str, reference: Mapping[str, float]) -> PhaseEvent:
        from_phase = int(self.current_phase)
        self._rollback_reference = {key: float(value) for key, value in reference.items()}
        self._rollback_reference_phase = from_phase
        self.current_phase += 1
        self._tracker = _MetricTracker()
        event = PhaseEvent(
            epoch=epoch,
            action="advance",
            from_phase=from_phase,
            to_phase=int(self.current_phase),
            reason=reason,
        )
        self.events.append(event)
        return event

    def _rollback(self, epoch: int, reason: str) -> PhaseEvent | None:
        if self.current_phase <= 0 or self._rollback_reference is None or self._rollback_reference_phase is None:
            return None
        from_phase = int(self.current_phase)
        to_phase = int(self._rollback_reference_phase)
        penalty = float(self._phase_penalties.get(from_phase, 1.0))
        self._phase_penalties[from_phase] = penalty * float(self.train_cfg.rollback_weight_scale)
        self.current_phase = to_phase
        self._reset_tracker(self._rollback_reference, epoch=epoch)
        event = PhaseEvent(
            epoch=epoch,
            action="rollback",
            from_phase=from_phase,
            to_phase=to_phase,
            reason=reason,
        )
        self.events.append(event)
        return event

    def _phase0_ready(self, epoch: int, metrics: Mapping[str, float]) -> tuple[bool, str]:
        assert self._tracker.entry is not None
        if not _finite_metrics(metrics):
            self._tracker.stable_validations = 0
            return False, "validation metrics not finite"
        phase_epochs = int(epoch - self._tracker.entry_epoch + 1)
        koopman_entry = float(self._tracker.entry["koopman_loss"])
        koopman_limit = max(
            koopman_entry * float(self.train_cfg.phase0_koopman_stability_ratio),
            koopman_entry + 1e-6,
        )
        if float(metrics["koopman_loss"]) <= koopman_limit:
            self._tracker.stable_validations += 1
        else:
            self._tracker.stable_validations = 0
        rec_gain = _relative_improvement(
            float(self._tracker.entry["reconstruction_loss"]),
            float(metrics["reconstruction_loss"]),
        )
        ready = (
            phase_epochs >= int(self.spec.min_epochs)
            and rec_gain >= float(self.train_cfg.phase0_reconstruction_improve)
            and self._tracker.stable_validations >= int(self.train_cfg.phase0_stable_validations)
        )
        reason = (
            f"reconstruction improved by {rec_gain:.1%} after {phase_epochs} epoch(s); "
            f"{self._tracker.stable_validations} stable validation(s)"
        )
        return ready, reason

    def _phase1_ready(self, epoch: int, metrics: Mapping[str, float]) -> tuple[bool, str]:
        assert self._tracker.entry is not None
        phase_epochs = int(epoch - self._tracker.entry_epoch + 1)
        pred_gain = _relative_improvement(
            float(self._tracker.entry["one_step_prediction_loss"]),
            float(metrics["one_step_prediction_loss"]),
        )
        align_gain = _relative_improvement(
            float(self._tracker.entry["q_align_loss"]),
            float(metrics["q_align_loss"]),
        )
        stable_hidden = float(metrics["hidden_sym_eig_upper"]) < 0.0
        ready = (
            phase_epochs >= int(self.spec.min_epochs)
            and pred_gain >= float(self.train_cfg.phase1_prediction_improve)
            and align_gain >= float(self.train_cfg.phase1_q_align_improve)
            and stable_hidden
        )
        reason = (
            f"1-step prediction improved by {pred_gain:.1%} after {phase_epochs} epoch(s); "
            f"q-align improved by {align_gain:.1%}; "
            f"hidden_sym_eig_upper={float(metrics['hidden_sym_eig_upper']):.4f}"
        )
        return ready, reason

    def _phase2_ready(self, epoch: int, metrics: Mapping[str, float]) -> tuple[bool, str]:
        phase_epochs = int(epoch - self._tracker.entry_epoch + 1)
        best_prediction = float(self._tracker.best_prediction or metrics["prediction_loss"])
        current_prediction = float(metrics["prediction_loss"])
        tolerance = float(self.train_cfg.phase2_prediction_plateau_tolerance)
        if current_prediction < best_prediction * (1.0 - tolerance):
            self._tracker.best_prediction = current_prediction
            self._tracker.plateau_checks = 0
        else:
            self._tracker.plateau_checks += 1
        best_long = float(self._tracker.best_long_horizon or metrics["long_horizon_prediction_loss"])
        long_ok = float(metrics["long_horizon_prediction_loss"]) <= best_long * (1.0 + tolerance)
        separation_ok = float(metrics["separation_loss"]) <= float(self.train_cfg.phase2_separation_target)
        ready = (
            phase_epochs >= int(self.spec.min_epochs)
            and separation_ok
            and long_ok
            and self._tracker.plateau_checks >= int(self.train_cfg.phase2_prediction_plateau_checks)
        )
        reason = (
            f"plateau checks={self._tracker.plateau_checks} after {phase_epochs} epoch(s); "
            f"separation={float(metrics['separation_loss']):.4f}; "
            f"long_horizon={float(metrics['long_horizon_prediction_loss']):.4f}"
        )
        return ready, reason

    def _rollback_needed(self, metrics: Mapping[str, float]) -> tuple[bool, str]:
        if (
            self.current_phase <= 0
            or self._rollback_reference is None
            or self._tracker.validations > int(self.train_cfg.rollback_window)
        ):
            return False, ""
        pred_ref = float(self._rollback_reference["one_step_prediction_loss"])
        long_ref = float(self._rollback_reference["long_horizon_prediction_loss"])
        pred_worse = float(metrics["one_step_prediction_loss"]) > pred_ref * (
            1.0 + float(self.train_cfg.rollback_prediction_worsen_ratio)
        )
        long_worse = float(metrics["long_horizon_prediction_loss"]) > long_ref * (
            1.0 + float(self.train_cfg.rollback_long_horizon_worsen_ratio)
        )
        hidden_unstable = float(metrics["hidden_sym_eig_upper"]) > 0.0
        if pred_worse or long_worse or hidden_unstable:
            reason_bits = []
            if pred_worse:
                reason_bits.append("prediction worsened")
            if long_worse:
                reason_bits.append("long-horizon loss worsened")
            if hidden_unstable:
                reason_bits.append("hidden contraction violated")
            return True, "; ".join(reason_bits)
        return False, ""

    def observe_validation(self, epoch: int, val_metrics: Mapping[str, float]) -> PhaseEvent | None:
        metrics = {key: float(value) for key, value in val_metrics.items()}
        if self._tracker.entry is None:
            self._reset_tracker(metrics, epoch=epoch)
        self._tracker.validations += 1
        self._tracker.best_prediction = min(
            float(self._tracker.best_prediction or metrics["prediction_loss"]),
            float(metrics["prediction_loss"]),
        )
        self._tracker.best_long_horizon = min(
            float(self._tracker.best_long_horizon or metrics["long_horizon_prediction_loss"]),
            float(metrics["long_horizon_prediction_loss"]),
        )
        rollback_needed, rollback_reason = self._rollback_needed(metrics)
        if rollback_needed:
            return self._rollback(epoch, rollback_reason)
        if self.current_phase >= len(self.specs) - 1:
            return None
        if self.current_phase == 0:
            ready, reason = self._phase0_ready(epoch, metrics)
        elif self.current_phase == 1:
            ready, reason = self._phase1_ready(epoch, metrics)
        else:
            ready, reason = self._phase2_ready(epoch, metrics)
        if ready:
            return self._advance(epoch, reason, metrics)
        return None


def build_phase_controller(train_cfg: TrainConfig) -> FractionalPhaseController | MetricDrivenPhaseController:
    if train_cfg.schedule_mode == "metric_driven":
        return MetricDrivenPhaseController(train_cfg)
    return FractionalPhaseController(train_cfg)
