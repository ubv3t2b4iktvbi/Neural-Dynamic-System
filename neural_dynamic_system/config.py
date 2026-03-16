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
    hidden_l1_weight: float = 1e-4
    memory_l1_weight: float | None = None
    separation_margin: float = 0.5
    contraction_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.memory_l1_weight is not None:
            self.hidden_l1_weight = float(self.memory_l1_weight)

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
    phase1_fraction: float = 0.3
    phase2_fraction: float = 0.8
    phase3_lr_scale: float = 0.1
    log_every: int = 5
    seed: int = 123

    def __post_init__(self) -> None:
        horizons = tuple(sorted({int(h) for h in self.horizons if int(h) > 0}))
        if not horizons:
            raise ValueError("horizons must contain at least one positive step")
        self.horizons = horizons
        if not 0.0 < float(self.train_fraction) < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.rg_horizon < 1:
            raise ValueError("rg_horizon must be >= 1")
        if not 0.0 < float(self.phase1_fraction) < 1.0:
            raise ValueError("phase1_fraction must be in (0, 1)")
        if not float(self.phase1_fraction) < float(self.phase2_fraction) <= 1.0:
            raise ValueError("phase2_fraction must be in (phase1_fraction, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
