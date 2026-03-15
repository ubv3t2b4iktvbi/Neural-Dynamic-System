from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    input_dim: int
    context_len: int = 32
    q_dim: int = 2
    m_dim: int = 2
    latent_scheme: str = "hard_split"
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
    min_memory_rate: float = 0.15
    rg_scale: float = 2.0
    coarse_strength: float = 0.25

    def __post_init__(self) -> None:
        self.latent_scheme = str(self.latent_scheme).lower()
        if self.latent_scheme not in {"hard_split", "soft_spectrum"}:
            raise ValueError("latent_scheme must be 'hard_split' or 'soft_spectrum'")
        if int(self.modal_dim) < 2:
            raise ValueError("modal_dim must be >= 2")
        if float(self.modal_temperature) <= 0.0:
            raise ValueError("modal_temperature must be > 0")
        if int(self.q_dim) < 1:
            raise ValueError("q_dim must be >= 1")
        if int(self.m_dim) < 1:
            raise ValueError("m_dim must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
    memory_l1_weight: float = 1e-4
    separation_margin: float = 0.5
    contraction_margin: float = 0.10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SupervisionConfig:
    q_indices: tuple[int, ...] = ()
    m_indices: tuple[int, ...] = ()
    q_weight: float = 0.0
    m_weight: float = 0.0
    q_mode: str = "direct"

    def __post_init__(self) -> None:
        self.q_indices = tuple(int(idx) for idx in self.q_indices)
        self.m_indices = tuple(int(idx) for idx in self.m_indices)
        self.q_mode = str(self.q_mode).lower()
        if self.q_mode not in {"direct", "angular"}:
            raise ValueError("q_mode must be 'direct' or 'angular'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    train_fraction: float = 0.8
    horizons: tuple[int, ...] = (10, 20, 40, 80)
    dt: float = 1.0
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
