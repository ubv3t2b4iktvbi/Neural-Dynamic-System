from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig

try:
    from torchdiffeq import odeint  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    odeint = None


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = in_dim
    hidden_layers = max(int(depth), 1)
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.SiLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


def _spectral_last(module: nn.Sequential) -> nn.Sequential:
    if module and isinstance(module[-1], nn.Linear):
        module[-1] = nn.utils.spectral_norm(module[-1])
    return module


def _group_count(channels: int, max_groups: int = 8) -> int:
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


def _matrix_inv_sqrt(mat: Tensor, eps: float = 1e-5) -> Tensor:
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(eps)
    inv_sqrt = evals.rsqrt()
    return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.transpose(-1, -2)


class _TemporalResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups=_group_count(in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=_group_count(out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class RunningWhitenedVAMP(nn.Module):
    def __init__(self, dim: int, *, momentum: float, eps: float):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(self.dim))
        self.register_buffer("running_cov", torch.eye(self.dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _batch_stats(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        mean = raw.mean(dim=0)
        centered = raw - mean
        if raw.shape[0] <= 1:
            cov = torch.eye(self.dim, device=raw.device, dtype=raw.dtype)
        else:
            cov = (centered.transpose(0, 1) @ centered) / float(raw.shape[0] - 1)
        eye = torch.eye(self.dim, device=raw.device, dtype=raw.dtype)
        cov = cov + self.eps * eye
        return mean, cov

    def _update_running(self, mean: Tensor, cov: Tensor) -> None:
        if not bool(self.initialized.item()):
            self.running_mean.copy_(mean)
            self.running_cov.copy_(cov)
            self.initialized.fill_(True)
            return
        self.running_mean.mul_(1.0 - self.momentum).add_(mean, alpha=self.momentum)
        self.running_cov.mul_(1.0 - self.momentum).add_(cov, alpha=self.momentum)

    def forward(self, raw: Tensor, *, update: bool = True) -> tuple[Tensor, Tensor]:
        if self.training and update and raw.shape[0] > 1:
            batch_mean, batch_cov = self._batch_stats(raw.detach())
            self._update_running(batch_mean, batch_cov)
        if bool(self.initialized.item()):
            mean = self.running_mean.to(device=raw.device, dtype=raw.dtype)
            cov = self.running_cov.to(device=raw.device, dtype=raw.dtype)
        else:
            mean, cov = self._batch_stats(raw.detach())
        whitening = _matrix_inv_sqrt(cov, eps=self.eps).detach()
        q = (raw - mean.detach()) @ whitening
        return q, raw


class TemporalMultiscaleEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        levels = max(1, int(cfg.encoder_levels))
        kernel_size = max(3, int(cfg.encoder_kernel_size) | 1)
        base_ch = max(16, cfg.hidden_dim // 4)
        channels = [min(cfg.hidden_dim, base_ch * (2**idx)) for idx in range(levels)]
        channels[-1] = max(channels[-1], min(cfg.hidden_dim, base_ch * 2))
        self.channels = channels

        self.stem = nn.Conv1d(int(cfg.input_dim), channels[0], kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, out_ch in enumerate(channels):
            in_ch = channels[idx]
            self.blocks.append(_TemporalResBlock(in_ch, out_ch, kernel_size))
            if idx < len(channels) - 1:
                next_ch = channels[idx + 1]
                self.downsamples.append(
                    nn.Conv1d(out_ch, next_ch, kernel_size=4, stride=2, padding=1)
                )

        self.bottleneck = _TemporalResBlock(channels[-1], channels[-1], kernel_size)
        self.q_proj = nn.Sequential(
            nn.Linear(channels[-1], cfg.hidden_dim),
            nn.SiLU(),
        )
        self.memory_proj = nn.Sequential(
            nn.Linear(sum(channels) + channels[-1], cfg.hidden_dim),
            nn.SiLU(),
        )

    def forward(self, window: Tensor) -> tuple[Tensor, Tensor]:
        x = window.transpose(1, 2)
        h = self.stem(x)
        summaries: list[Tensor] = []
        for idx, block in enumerate(self.blocks):
            h = block(h)
            summaries.append(h.mean(dim=-1))
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)
        h = self.bottleneck(h)
        bottleneck = h.mean(dim=-1)
        q_hidden = self.q_proj(bottleneck)
        memory_hidden = self.memory_proj(torch.cat([bottleneck, *summaries], dim=-1))
        return q_hidden, memory_hidden


class LatentRGManifoldAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_scheme = str(cfg.latent_scheme).lower()
        flat_dim = int(cfg.input_dim) * int(cfg.context_len)
        encoder_type = str(cfg.encoder_type).lower()

        if encoder_type == "temporal_conv":
            self.encoder_backbone = TemporalMultiscaleEncoder(cfg)
            q_hidden_dim = int(cfg.hidden_dim)
            memory_hidden_dim = int(cfg.hidden_dim)
        elif encoder_type == "mlp":
            self.encoder_backbone = _mlp(flat_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.depth)
            q_hidden_dim = int(cfg.hidden_dim)
            memory_hidden_dim = int(cfg.hidden_dim)
        else:
            raise ValueError(f"Unknown encoder_type={cfg.encoder_type!r}")

        if self.latent_scheme == "soft_spectrum":
            modal_input_dim = int(q_hidden_dim + memory_hidden_dim)
            self.modal_backbone = nn.Sequential(
                nn.LayerNorm(modal_input_dim),
                _mlp(modal_input_dim, cfg.modal_dim, cfg.hidden_dim, 1),
            )
            self.modal_rate_logits = nn.Parameter(torch.linspace(-1.5, 1.5, steps=int(cfg.modal_dim)))
            q_input_dim = int(cfg.modal_dim)
            memory_input_dim = int(cfg.modal_dim)
        else:
            self.modal_backbone = None
            self.modal_rate_logits = None
            q_input_dim = int(q_hidden_dim)
            memory_input_dim = int(memory_hidden_dim)

        self.vamp_head = nn.Sequential(
            nn.LayerNorm(q_input_dim),
            _mlp(q_input_dim, cfg.q_dim, cfg.hidden_dim, cfg.vamp_head_depth),
        )
        self.vamp_whitener = RunningWhitenedVAMP(
            cfg.q_dim,
            momentum=cfg.vamp_whitening_momentum,
            eps=cfg.vamp_whitening_eps,
        )
        self.m_head = nn.Linear(memory_input_dim, cfg.m_dim)

        dyn_input_dim = cfg.q_dim + cfg.m_dim
        self.q_drift_net = _mlp(dyn_input_dim, cfg.q_dim, cfg.hidden_dim, cfg.depth)
        self.q_rate_net = _mlp(dyn_input_dim, cfg.q_dim, cfg.hidden_dim, 1)
        self.memory_rate_net = _mlp(dyn_input_dim, cfg.m_dim, cfg.hidden_dim, 1)
        self.memory_kernel_net = _spectral_last(_mlp(dyn_input_dim, cfg.m_dim, cfg.hidden_dim, 1))

        self.manifold_decoder = _mlp(cfg.q_dim, cfg.input_dim, cfg.hidden_dim, cfg.depth)
        self.memory_readout_net = _mlp(cfg.q_dim, cfg.input_dim * cfg.m_dim, cfg.hidden_dim, cfg.depth)
        self.coarse_q_net = _mlp(cfg.q_dim, cfg.q_dim, cfg.hidden_dim, 1)

    @property
    def latent_dim(self) -> int:
        return int(self.cfg.q_dim + self.cfg.m_dim)

    def _zeros(self, batch_size: int, dim: int, ref: Tensor) -> Tensor:
        return ref.new_zeros((batch_size, dim))

    def _encode_backbone(self, window: Tensor) -> tuple[Tensor, Tensor]:
        if str(self.cfg.encoder_type).lower() == "temporal_conv":
            return self.encoder_backbone(window)
        hidden = self.encoder_backbone(window.reshape(window.shape[0], -1))
        return hidden, hidden

    def modal_rates(self) -> Tensor:
        if self.latent_scheme != "soft_spectrum" or self.modal_rate_logits is None:
            return torch.zeros(0, device=self.coarse_q_net[0].weight.device, dtype=self.coarse_q_net[0].weight.dtype)
        return self.cfg.min_slow_rate + F.softplus(self.modal_rate_logits)

    def modal_weight_vectors(self, rates: Tensor) -> tuple[Tensor, Tensor]:
        if rates.numel() == 0:
            empty = rates.new_zeros((0,))
            return empty, empty
        log_rates = torch.log(rates.clamp_min(1e-6))
        centered = (log_rates - log_rates.mean()) / float(self.cfg.modal_temperature)
        slow_weights = torch.softmax(-centered, dim=-1)
        memory_weights = torch.softmax(centered, dim=-1)
        return slow_weights, memory_weights

    def spectrum_summary(self) -> dict[str, float | list[float] | str]:
        if self.latent_scheme != "soft_spectrum":
            return {"latent_scheme": self.latent_scheme}
        with torch.no_grad():
            rates = self.modal_rates().detach().cpu()
            slow_weights, memory_weights = self.modal_weight_vectors(rates)
            norm = math.log(max(int(rates.numel()), 2))
            slow_entropy = float((-(slow_weights * (slow_weights + 1e-8).log()).sum() / norm).item())
            memory_entropy = float((-(memory_weights * (memory_weights + 1e-8).log()).sum() / norm).item())
            return {
                "latent_scheme": self.latent_scheme,
                "modal_dim": int(rates.numel()),
                "modal_rates": rates.tolist(),
                "modal_rate_min": float(rates.min().item()),
                "modal_rate_max": float(rates.max().item()),
                "modal_rate_mean": float(rates.mean().item()),
                "modal_slow_weights": slow_weights.cpu().tolist(),
                "modal_memory_weights": memory_weights.cpu().tolist(),
                "modal_slow_entropy": slow_entropy,
                "modal_memory_entropy": memory_entropy,
            }

    def encode_components(self, window: Tensor, *, update_whitener: bool = True) -> dict[str, Tensor]:
        batch = window.shape[0]
        q_hidden, memory_hidden = self._encode_backbone(window)
        if self.latent_scheme == "soft_spectrum":
            modal_input = torch.cat([q_hidden, memory_hidden], dim=-1)
            modal = torch.tanh(self.modal_backbone(modal_input))
            modal_rates = self.modal_rates().to(device=modal.device, dtype=modal.dtype)
            slow_weights, memory_weights = self.modal_weight_vectors(modal_rates)
            slow_weights_batch = slow_weights.unsqueeze(0).expand(batch, -1)
            memory_weights_batch = memory_weights.unsqueeze(0).expand(batch, -1)
            q_source = modal * slow_weights_batch
            memory_source = modal * memory_weights_batch
        else:
            modal = self._zeros(batch, 0, q_hidden)
            modal_rates = self._zeros(0, 0, q_hidden).reshape(0)
            slow_weights_batch = self._zeros(batch, 0, q_hidden)
            memory_weights_batch = self._zeros(batch, 0, q_hidden)
            q_source = q_hidden
            memory_source = memory_hidden

        q_raw = self.vamp_head(q_source)
        q, q_vamp_raw = self.vamp_whitener(q_raw, update=update_whitener)
        m = self.m_head(memory_source)
        z = self.join_latent(q, m)
        return {
            "z": z,
            "q": q,
            "m": m,
            "q_hidden": q_hidden,
            "memory_hidden": memory_hidden,
            "q_raw": q_vamp_raw,
            "q_vamp": q,
            "modal": modal,
            "modal_rates": modal_rates.unsqueeze(0).expand(batch, -1) if modal_rates.numel() > 0 else self._zeros(batch, 0, q_hidden),
            "modal_slow_weights": slow_weights_batch,
            "modal_memory_weights": memory_weights_batch,
        }

    def split_latent(self, z: Tensor) -> tuple[Tensor, Tensor]:
        q_end = self.cfg.q_dim
        q = z[:, :q_end]
        m = z[:, q_end:]
        return q, m

    def join_latent(self, q: Tensor, m: Tensor) -> Tensor:
        return torch.cat([q, m], dim=-1)

    def encode(self, window: Tensor) -> Tensor:
        return self.encode_components(window)["z"]

    def decode_parts(self, q: Tensor, m: Tensor) -> Tensor:
        base = self.manifold_decoder(q)
        memory_basis = self.memory_readout_net(q).view(q.shape[0], self.cfg.input_dim, self.cfg.m_dim)
        correction = torch.bmm(memory_basis, m.unsqueeze(-1)).squeeze(-1)
        return base + correction

    def decode(self, z: Tensor) -> Tensor:
        q, m = self.split_latent(z)
        return self.decode_parts(q, m)

    def memory_kernel(self, q: Tensor, m: Tensor) -> Tensor:
        kernel_input = torch.cat([q, m], dim=-1)
        return 0.1 * torch.tanh(self.memory_kernel_net(kernel_input))

    def latent_statistics(self, z: Tensor) -> dict[str, Tensor]:
        q, m = self.split_latent(z)
        qm = self.join_latent(q, m)
        slow_rates = self.cfg.min_slow_rate + F.softplus(self.q_rate_net(qm))
        memory_rates = self.cfg.min_memory_rate + F.softplus(self.memory_rate_net(qm))
        return {
            "q": q,
            "m": m,
            "slow_rates": slow_rates,
            "memory_rates": memory_rates,
        }

    def derivative(self, z: Tensor) -> dict[str, Tensor]:
        stats = self.latent_statistics(z)
        q = stats["q"]
        m = stats["m"]
        qm = self.join_latent(q, m)
        q_raw = torch.tanh(self.q_drift_net(qm))
        dq = stats["slow_rates"] * q_raw
        memory_kernel = self.memory_kernel(q, m)
        dm = -(stats["memory_rates"] * m) + memory_kernel
        dz = self.join_latent(dq, dm)
        stats.update(
            {
                "dq": dq,
                "dm": dm,
                "dz": dz,
                "memory_kernel": memory_kernel,
            }
        )
        return stats

    @staticmethod
    def _phi1(x: Tensor) -> Tensor:
        return torch.where(x.abs() < 1e-6, torch.ones_like(x), (1.0 - torch.exp(-x)) / x)

    def step(self, z: Tensor, dt: float) -> Tensor:
        dt_value = float(dt)
        q0, m0 = self.split_latent(z)
        stats0 = self.derivative(z)

        q_mid = q0 + 0.5 * dt_value * stats0["dq"]
        half_mem = 0.5 * dt_value * stats0["memory_rates"]
        m_mid = torch.exp(-half_mem) * m0 + 0.5 * dt_value * self._phi1(half_mem) * stats0["memory_kernel"]

        z_mid = self.join_latent(q_mid, m_mid)
        stats_mid = self.derivative(z_mid)
        q_next = q0 + dt_value * stats_mid["dq"]

        full_mem = dt_value * stats_mid["memory_rates"]
        m_next = torch.exp(-full_mem) * m0 + dt_value * self._phi1(full_mem) * stats_mid["memory_kernel"]
        return self.join_latent(q_next, m_next)

    def slow_vector_field(self, _t: Tensor, state: Tensor) -> Tensor:
        return self.derivative(state)["dz"]

    def flow(self, z: Tensor, *, dt: float, steps: int) -> Tensor:
        num_steps = int(steps)
        if num_steps <= 0:
            return z
        if odeint is None:
            out = z
            for _ in range(num_steps):
                out = self.step(out, dt=dt)
            return out
        t_eval = torch.linspace(
            0.0,
            float(dt) * float(num_steps),
            steps=num_steps + 1,
            device=z.device,
            dtype=z.dtype,
        )
        return odeint(self.slow_vector_field, z, t_eval, method="dopri5")[-1]

    def project_to_manifold(self, z: Tensor) -> Tensor:
        q, m = self.split_latent(z)
        return self.join_latent(q, torch.zeros_like(m))

    def coarse_grain(self, z: Tensor) -> Tensor:
        q, m = self.split_latent(z)
        scale = max(self.cfg.rg_scale, 1e-6)
        delta_q = torch.tanh(self.coarse_q_net(q)) * (self.cfg.coarse_strength / scale)
        coarse_q = q + delta_q
        coarse_m = m / scale
        return self.join_latent(coarse_q, coarse_m)

    def module_groups(self) -> dict[str, list[nn.Module]]:
        encoder_modules: list[nn.Module] = [self.encoder_backbone, self.vamp_head, self.m_head]
        if self.modal_backbone is not None:
            encoder_modules.append(self.modal_backbone)

        decoder_modules: list[nn.Module] = [self.manifold_decoder, self.memory_readout_net]
        dynamics_modules: list[nn.Module] = [
            self.q_drift_net,
            self.q_rate_net,
            self.memory_rate_net,
            self.memory_kernel_net,
        ]

        return {
            "encoder": encoder_modules,
            "decoder": decoder_modules,
            "dynamics": dynamics_modules,
            "rg": [self.coarse_q_net],
        }
