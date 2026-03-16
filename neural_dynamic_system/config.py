from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    input_dim: int
    context_len: int = 32
    q_dim: int = 2
    h_dim: int = 2
    m_dim: int | None = None
    koopman_dim: int | None = None
    latent_scheme: str = "soft_spectrum"
    koopman_input_mode: str = "slow_only"
    hidden_coordinate_mode: str = "normal_residual"
    modal_dim: int = 8
    modal_temperature: float = 0.35
    encoder_type: str = "temporal_conv"
    encoder_levels: int = 3
    encoder_kernel_size: int = 5
    hidden_dim: int = 128
    depth: int = 2
    vamp_head_depth: int = 1
    vamp_whitening_momentum: float = 0.05
    vamp_whitening_eps: float = 1e-5
    min_slow_rate: float = 0.02
    min_fast_rate: float = 0.15
    min_memory_rate: float | None = None
    hidden_rank: int = 4
    hidden_operator_scale: float = 0.10
    hidden_drive_scale: float = 0.50
    slow_residual_scale: float = 0.50
    rg_scale: float = 2.0
    coarse_strength: float = 0.25
    rg_temperature: float = 0.35
    rg_eps: float = 1e-4

    def __post_init__(self) -> None:
        self.latent_scheme = str(self.latent_scheme).lower()
        self.koopman_input_mode = str(self.koopman_input_mode).lower()
        self.hidden_coordinate_mode = str(self.hidden_coordinate_mode).lower()
        if self.m_dim is not None:
            if int(self.h_dim) != int(self.m_dim):
                if int(self.h_dim) != 2:
                    raise ValueError("Pass either h_dim or m_dim, or keep them equal")
                self.h_dim = int(self.m_dim)
        if self.min_memory_rate is not None:
            self.min_fast_rate = float(self.min_memory_rate)
        if self.koopman_dim is None:
            default_koopman_dim = max(int(self.q_dim), int(self.modal_dim))
            self.koopman_dim = default_koopman_dim
        self.koopman_dim = int(self.koopman_dim)
        self.hidden_rank = min(int(self.hidden_rank), int(self.h_dim))

        if self.latent_scheme not in {"hard_split", "soft_spectrum"}:
            raise ValueError("latent_scheme must be 'hard_split' or 'soft_spectrum'")
        if self.koopman_input_mode not in {"joint", "slow_only"}:
            raise ValueError("koopman_input_mode must be 'joint' or 'slow_only'")
        if self.hidden_coordinate_mode not in {"direct", "normal_residual"}:
            raise ValueError("hidden_coordinate_mode must be 'direct' or 'normal_residual'")
        if int(self.modal_dim) < 2:
            raise ValueError("modal_dim must be >= 2")
        if float(self.modal_temperature) <= 0.0:
            raise ValueError("modal_temperature must be > 0")
        if int(self.q_dim) < 1:
            raise ValueError("q_dim must be >= 1")
        if int(self.h_dim) < 1:
            raise ValueError("h_dim must be >= 1")
        if int(self.koopman_dim) < int(self.q_dim):
            raise ValueError("koopman_dim must be >= q_dim")
        if float(self.min_fast_rate) <= 0.0:
            raise ValueError("min_fast_rate must be > 0")
        if float(self.min_slow_rate) <= 0.0:
            raise ValueError("min_slow_rate must be > 0")
        if int(self.hidden_rank) < 1:
            raise ValueError("hidden_rank must be >= 1")
        if float(self.hidden_operator_scale) < 0.0:
            raise ValueError("hidden_operator_scale must be >= 0")
        if float(self.hidden_drive_scale) < 0.0:
            raise ValueError("hidden_drive_scale must be >= 0")
        if float(self.slow_residual_scale) < 0.0:
            raise ValueError("slow_residual_scale must be >= 0")
        if float(self.rg_temperature) <= 0.0:
            raise ValueError("rg_temperature must be > 0")
        if float(self.rg_eps) <= 0.0:
            raise ValueError("rg_eps must be > 0")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["h_dim"] = int(self.h_dim)
        payload["koopman_dim"] = int(self.koopman_dim)
        payload.pop("m_dim", None)
        payload["min_fast_rate"] = float(self.min_fast_rate)
        payload.pop("min_memory_rate", None)
        return payload


@dataclass
class LossConfig:
    reconstruction_weight: float = 1.0
    vamp_weight: float = 0.2
    vamp_align_weight: float = 0.25
    koopman_weight: float = 0.25
    diag_weight: float = 0.05
    prediction_weight: float = 1.0
    latent_align_weight: float = 0.75
    semigroup_weight: float = 0.5
    separation_weight: float = 0.2
    contract_weight: float = 0.2
    rg_weight: float = 0.05
    metric_weight: float = 0.1
    metric_mode: str = "mahalanobis_dynamics"
    hidden_l1_weight: float = 1e-4
    memory_l1_weight: float | None = None
    separation_margin: float = 0.5
    contraction_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.memory_l1_weight is not None:
            self.hidden_l1_weight = float(self.memory_l1_weight)
        self.metric_mode = str(self.metric_mode).lower()
        if self.metric_mode not in {"euclidean", "mahalanobis_dynamics"}:
            raise ValueError("metric_mode must be 'euclidean' or 'mahalanobis_dynamics'")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["hidden_l1_weight"] = float(self.hidden_l1_weight)
        payload.pop("memory_l1_weight", None)
        return payload


@dataclass
class SupervisionConfig:
    q_indices: tuple[int, ...] = ()
    h_indices: tuple[int, ...] = ()
    m_indices: tuple[int, ...] | None = None
    q_weight: float = 0.0
    h_weight: float = 0.0
    m_weight: float | None = None
    q_mode: str = "direct"

    def __post_init__(self) -> None:
        self.q_indices = tuple(int(idx) for idx in self.q_indices)
        if self.m_indices is not None and not self.h_indices:
            self.h_indices = tuple(int(idx) for idx in self.m_indices)
        else:
            self.h_indices = tuple(int(idx) for idx in self.h_indices)
        if self.m_weight is not None and float(self.h_weight) == 0.0:
            self.h_weight = float(self.m_weight)
        self.q_mode = str(self.q_mode).lower()
        if self.q_mode not in {"direct", "angular"}:
            raise ValueError("q_mode must be 'direct' or 'angular'")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["q_indices"] = list(self.q_indices)
        payload["h_indices"] = list(self.h_indices)
        payload.pop("m_indices", None)
        payload["q_weight"] = float(self.q_weight)
        payload["h_weight"] = float(self.h_weight)
        payload.pop("m_weight", None)
        return payload


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    train_fraction: float = 0.8
    horizons: tuple[int, ...] = (10, 20, 40, 80)
    dt: float = 0.05
    grad_clip: float = 1.0
    device: str = "auto"
    metric_subsample: int = 32
    rg_horizon: int = 1
    contract_batch: int = 16
    schedule_mode: str = "fractional"
    phase1_fraction: float = 0.3
    phase2_fraction: float = 0.8
    phase3_lr_scale: float = 0.1
    phase0_min_epochs: int = 6
    phase1_min_epochs: int = 10
    phase2_min_epochs: int = 12
    phase3_min_epochs: int = 6
    phase0_reconstruction_improve: float = 0.20
    phase0_koopman_stability_ratio: float = 1.25
    phase0_stable_validations: int = 2
    phase1_prediction_improve: float = 0.20
    phase1_q_align_improve: float = 0.20
    phase2_prediction_plateau_tolerance: float = 0.02
    phase2_prediction_plateau_checks: int = 2
    phase2_separation_target: float = 0.08
    rollback_prediction_worsen_ratio: float = 0.25
    rollback_long_horizon_worsen_ratio: float = 0.35
    rollback_window: int = 2
    rollback_weight_scale: float = 0.5
    log_every: int = 5
    validation_interval: int = 1
    progress_bar: bool = True
    early_stopping: bool = False
    early_stopping_monitor: str = "prediction_loss"
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    early_stopping_min_epochs: int | None = None
    early_stopping_start_phase: int = 3
    seed: int = 123

    def __post_init__(self) -> None:
        horizons = tuple(sorted({int(h) for h in self.horizons if int(h) > 0}))
        if not horizons:
            raise ValueError("horizons must contain at least one positive step")
        self.horizons = horizons
        self.schedule_mode = str(self.schedule_mode).lower()
        if not 0.0 < float(self.train_fraction) < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.rg_horizon < 1:
            raise ValueError("rg_horizon must be >= 1")
        if self.schedule_mode not in {"fractional", "metric_driven"}:
            raise ValueError("schedule_mode must be 'fractional' or 'metric_driven'")
        if self.schedule_mode == "fractional":
            if not 0.0 < float(self.phase1_fraction) < 1.0:
                raise ValueError("phase1_fraction must be in (0, 1)")
            if not float(self.phase1_fraction) < float(self.phase2_fraction) <= 1.0:
                raise ValueError("phase2_fraction must be in (phase1_fraction, 1]")
        for name in ("phase0_min_epochs", "phase1_min_epochs", "phase2_min_epochs", "phase3_min_epochs"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in (
            "phase0_reconstruction_improve",
            "phase0_koopman_stability_ratio",
            "phase1_prediction_improve",
            "phase1_q_align_improve",
            "phase2_prediction_plateau_tolerance",
            "phase2_separation_target",
            "rollback_prediction_worsen_ratio",
            "rollback_long_horizon_worsen_ratio",
            "rollback_weight_scale",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.phase0_stable_validations) < 1:
            raise ValueError("phase0_stable_validations must be >= 1")
        if int(self.phase2_prediction_plateau_checks) < 1:
            raise ValueError("phase2_prediction_plateau_checks must be >= 1")
        if int(self.rollback_window) < 1:
            raise ValueError("rollback_window must be >= 1")
        if int(self.validation_interval) < 1:
            raise ValueError("validation_interval must be >= 1")
        self.early_stopping_monitor = str(self.early_stopping_monitor).lower()
        if self.early_stopping_monitor not in {"loss", "prediction_loss"}:
            raise ValueError("early_stopping_monitor must be 'loss' or 'prediction_loss'")
        if int(self.early_stopping_patience) < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if float(self.early_stopping_min_delta) < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if self.early_stopping_min_epochs is not None and int(self.early_stopping_min_epochs) < 1:
            raise ValueError("early_stopping_min_epochs must be >= 1 when provided")
        if int(self.early_stopping_start_phase) not in {0, 1, 2, 3}:
            raise ValueError("early_stopping_start_phase must be one of {0, 1, 2, 3}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
