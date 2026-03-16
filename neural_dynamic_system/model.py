from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


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
        self.fast_proj = nn.Sequential(
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
        fast_hidden = self.fast_proj(torch.cat([bottleneck, *summaries], dim=-1))
        return q_hidden, fast_hidden


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
            fast_hidden_dim = int(cfg.hidden_dim)
        elif encoder_type == "mlp":
            self.encoder_backbone = _mlp(flat_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.depth)
            q_hidden_dim = int(cfg.hidden_dim)
            fast_hidden_dim = int(cfg.hidden_dim)
        else:
            raise ValueError(f"Unknown encoder_type={cfg.encoder_type!r}")

        if self.latent_scheme == "soft_spectrum":
            modal_input_dim = int(q_hidden_dim + fast_hidden_dim)
            self.modal_backbone = nn.Sequential(
                nn.LayerNorm(modal_input_dim),
                _mlp(modal_input_dim, cfg.modal_dim, cfg.hidden_dim, 1),
            )
            self.modal_rate_logits = nn.Parameter(torch.linspace(-1.5, 1.5, steps=int(cfg.modal_dim)))
            q_input_dim = int(cfg.modal_dim)
            fast_input_dim = int(cfg.modal_dim)
        else:
            self.modal_backbone = None
            self.modal_rate_logits = None
            q_input_dim = int(q_hidden_dim)
            fast_input_dim = int(fast_hidden_dim)

        self.vamp_head = nn.Sequential(
            nn.LayerNorm(q_input_dim),
            _mlp(q_input_dim, cfg.q_dim, cfg.hidden_dim, cfg.vamp_head_depth),
        )
        self.vamp_whitener = RunningWhitenedVAMP(
            cfg.q_dim,
            momentum=cfg.vamp_whitening_momentum,
            eps=cfg.vamp_whitening_eps,
        )
        self.h_head = nn.Linear(fast_input_dim, cfg.h_dim)

        self.q_residual_net = _mlp(cfg.q_dim, cfg.q_dim, cfg.hidden_dim, cfg.depth)
        self.q_rate_net = _mlp(cfg.q_dim, cfg.q_dim, cfg.hidden_dim, 1)
        self.q_coupling_net = _mlp(cfg.q_dim, cfg.q_dim * cfg.h_dim, cfg.hidden_dim, 1)

        self.h_rate_net = _mlp(cfg.q_dim, cfg.h_dim, cfg.hidden_dim, 1)
        self.h_skew_net = _mlp(cfg.q_dim, cfg.h_dim * cfg.h_dim, cfg.hidden_dim, 1)
        self.h_dissipation_net = _mlp(cfg.q_dim, cfg.h_dim * cfg.h_dim, cfg.hidden_dim, 1)
        self.h_drive_net = _mlp(cfg.q_dim, cfg.h_dim, cfg.hidden_dim, cfg.depth)

        self.manifold_decoder = _mlp(cfg.q_dim, cfg.input_dim, cfg.hidden_dim, cfg.depth)
        self.hidden_readout_net = _mlp(cfg.q_dim, cfg.input_dim * cfg.h_dim, cfg.hidden_dim, cfg.depth)
        self.coarse_q_net = _mlp(cfg.q_dim, cfg.q_dim, cfg.hidden_dim, 1)

    @property
    def latent_dim(self) -> int:
        return int(self.cfg.q_dim + self.cfg.h_dim)

    def _zeros(self, batch_size: int, dim: int, ref: Tensor) -> Tensor:
        return ref.new_zeros((batch_size, dim))

    def _encode_backbone(self, window: Tensor) -> tuple[Tensor, Tensor]:
        if str(self.cfg.encoder_type).lower() == "temporal_conv":
            return self.encoder_backbone(window)
        hidden = self.encoder_backbone(window.reshape(window.shape[0], -1))
        return hidden, hidden

    def modal_rates(self) -> Tensor:
        if self.latent_scheme != "soft_spectrum" or self.modal_rate_logits is None:
            ref = self.coarse_q_net[0].weight
            return torch.zeros(0, device=ref.device, dtype=ref.dtype)
        return self.cfg.min_slow_rate + F.softplus(self.modal_rate_logits)

    def modal_weight_vectors(self, rates: Tensor) -> tuple[Tensor, Tensor]:
        if rates.numel() == 0:
            empty = rates.new_zeros((0,))
            return empty, empty
        log_rates = torch.log(rates.clamp_min(1e-6))
        centered = (log_rates - log_rates.mean()) / float(self.cfg.modal_temperature)
        slow_weights = torch.softmax(-centered, dim=-1)
        fast_weights = torch.softmax(centered, dim=-1)
        return slow_weights, fast_weights

    def spectrum_summary(self) -> dict[str, float | list[float] | str]:
        base_summary: dict[str, float | list[float] | str] = {
            "latent_structure": "slow_fast_state_space",
            "latent_scheme": self.latent_scheme,
            "integrator": "midpoint_q_plus_exact_affine_h",
        }
        if self.latent_scheme != "soft_spectrum":
            return base_summary
        with torch.no_grad():
            rates = self.modal_rates().detach().cpu()
            slow_weights, fast_weights = self.modal_weight_vectors(rates)
            norm = math.log(max(int(rates.numel()), 2))
            slow_entropy = float((-(slow_weights * (slow_weights + 1e-8).log()).sum() / norm).item())
            fast_entropy = float((-(fast_weights * (fast_weights + 1e-8).log()).sum() / norm).item())
            base_summary.update(
                {
                    "modal_dim": int(rates.numel()),
                    "modal_rates": rates.tolist(),
                    "modal_rate_min": float(rates.min().item()),
                    "modal_rate_max": float(rates.max().item()),
                    "modal_rate_mean": float(rates.mean().item()),
                    "modal_slow_weights": slow_weights.cpu().tolist(),
                    "modal_fast_weights": fast_weights.cpu().tolist(),
                    "modal_memory_weights": fast_weights.cpu().tolist(),
                    "modal_slow_entropy": slow_entropy,
                    "modal_fast_entropy": fast_entropy,
                    "modal_memory_entropy": fast_entropy,
                }
            )
            return base_summary

    def encode_components(self, window: Tensor, *, update_whitener: bool = True) -> dict[str, Tensor]:
        batch = window.shape[0]
        q_hidden, fast_hidden = self._encode_backbone(window)
        if self.latent_scheme == "soft_spectrum":
            modal_input = torch.cat([q_hidden, fast_hidden], dim=-1)
            modal = torch.tanh(self.modal_backbone(modal_input))
            modal_rates = self.modal_rates().to(device=modal.device, dtype=modal.dtype)
            slow_weights, fast_weights = self.modal_weight_vectors(modal_rates)
            slow_weights_batch = slow_weights.unsqueeze(0).expand(batch, -1)
            fast_weights_batch = fast_weights.unsqueeze(0).expand(batch, -1)
            q_source = modal * slow_weights_batch
            fast_source = modal * fast_weights_batch
        else:
            modal = self._zeros(batch, 0, q_hidden)
            modal_rates = self._zeros(0, 0, q_hidden).reshape(0)
            slow_weights_batch = self._zeros(batch, 0, q_hidden)
            fast_weights_batch = self._zeros(batch, 0, q_hidden)
            q_source = q_hidden
            fast_source = fast_hidden

        q_raw = self.vamp_head(q_source)
        q, q_vamp_raw = self.vamp_whitener(q_raw, update=update_whitener)
        h = self.h_head(fast_source)
        z = self.join_latent(q, h)
        modal_rate_batch = (
            modal_rates.unsqueeze(0).expand(batch, -1)
            if modal_rates.numel() > 0
            else self._zeros(batch, 0, q_hidden)
        )
        return {
            "z": z,
            "q": q,
            "h": h,
            "m": h,
            "q_hidden": q_hidden,
            "h_hidden": fast_hidden,
            "memory_hidden": fast_hidden,
            "q_raw": q_vamp_raw,
            "q_vamp": q,
            "modal": modal,
            "modal_rates": modal_rate_batch,
            "modal_slow_weights": slow_weights_batch,
            "modal_fast_weights": fast_weights_batch,
            "modal_memory_weights": fast_weights_batch,
        }

    def split_latent(self, z: Tensor) -> tuple[Tensor, Tensor]:
        q_end = self.cfg.q_dim
        q = z[:, :q_end]
        h = z[:, q_end:]
        return q, h

    def join_latent(self, q: Tensor, h: Tensor) -> Tensor:
        return torch.cat([q, h], dim=-1)

    def encode(self, window: Tensor) -> Tensor:
        return self.encode_components(window)["z"]

    def decode_parts(self, q: Tensor, h: Tensor) -> Tensor:
        base = self.manifold_decoder(q)
        hidden_basis = self.hidden_readout_net(q).view(q.shape[0], self.cfg.input_dim, self.cfg.h_dim)
        correction = torch.bmm(hidden_basis, h.unsqueeze(-1)).squeeze(-1)
        return base + correction

    def decode(self, z: Tensor) -> Tensor:
        q, h = self.split_latent(z)
        return self.decode_parts(q, h)

    def _slow_coupling(self, q: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        coupling = self.q_coupling_net(q).view(q.shape[0], self.cfg.q_dim, self.cfg.h_dim)
        coupling = coupling / math.sqrt(max(int(self.cfg.h_dim), 1))
        value = torch.bmm(coupling, h.unsqueeze(-1)).squeeze(-1)
        return coupling, value

    def _hidden_operator(self, q: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        fast_rates = self.cfg.min_fast_rate + F.softplus(self.h_rate_net(q))
        skew_raw = self.h_skew_net(q).view(q.shape[0], self.cfg.h_dim, self.cfg.h_dim)
        skew = 0.5 * (skew_raw - skew_raw.transpose(-1, -2))
        dissipation_raw = self.h_dissipation_net(q).view(q.shape[0], self.cfg.h_dim, self.cfg.h_dim)
        dissipation = torch.bmm(dissipation_raw.transpose(-1, -2), dissipation_raw)
        dissipation = dissipation / float(max(int(self.cfg.h_dim), 1))
        operator = (self.cfg.hidden_operator_scale * skew) - dissipation - torch.diag_embed(fast_rates)
        sym = 0.5 * (operator + operator.transpose(-1, -2))
        sym_eig_upper = torch.linalg.eigvalsh(sym).amax(dim=-1, keepdim=True)
        return operator, fast_rates, dissipation, sym_eig_upper

    def latent_statistics(self, z: Tensor) -> dict[str, Tensor]:
        q, h = self.split_latent(z)
        slow_rates = self.cfg.min_slow_rate + F.softplus(self.q_rate_net(q))
        slow_residual = self.cfg.slow_residual_scale * torch.tanh(self.q_residual_net(q))
        coupling_matrix, q_coupling = self._slow_coupling(q, h)
        dq = -(slow_rates * q) + slow_residual + q_coupling

        hidden_operator, fast_rates, dissipation, hidden_sym_eig_upper = self._hidden_operator(q)
        hidden_drive = self.cfg.hidden_drive_scale * torch.tanh(self.h_drive_net(q))
        dh = torch.bmm(hidden_operator, h.unsqueeze(-1)).squeeze(-1) + hidden_drive

        return {
            "q": q,
            "h": h,
            "m": h,
            "slow_rates": slow_rates,
            "fast_rates": fast_rates,
            "memory_rates": fast_rates,
            "slow_residual": slow_residual,
            "q_coupling_matrix": coupling_matrix,
            "q_coupling": q_coupling,
            "hidden_operator": hidden_operator,
            "hidden_dissipation": dissipation,
            "hidden_drive": hidden_drive,
            "hidden_sym_eig_upper": hidden_sym_eig_upper,
            "dq": dq,
            "dh": dh,
            "dm": dh,
            "dz": self.join_latent(dq, dh),
        }

    def derivative(self, z: Tensor) -> dict[str, Tensor]:
        return self.latent_statistics(z)

    @staticmethod
    def _affine_hidden_transition(operator: Tensor, drive: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        batch_size, dim, _ = operator.shape
        aug = operator.new_zeros((batch_size, dim + 1, dim + 1))
        aug[:, :dim, :dim] = operator * float(dt)
        aug[:, :dim, dim] = drive * float(dt)
        exp_aug = torch.matrix_exp(aug)
        transition = exp_aug[:, :dim, :dim]
        bias = exp_aug[:, :dim, dim]
        return transition, bias

    def hidden_ssm_matrices(self, q: Tensor, *, dt: float) -> dict[str, Tensor]:
        operator, fast_rates, dissipation, hidden_sym_eig_upper = self._hidden_operator(q)
        drive = self.cfg.hidden_drive_scale * torch.tanh(self.h_drive_net(q))
        transition, bias = self._affine_hidden_transition(operator, drive, dt=float(dt))
        return {
            "generator": operator,
            "transition": transition,
            "bias": bias,
            "memory_rates": fast_rates,
            "fast_rates": fast_rates,
            "hidden_dissipation": dissipation,
            "hidden_sym_eig_upper": hidden_sym_eig_upper,
            "hidden_drive": drive,
        }

    def _affine_hidden_step(self, q: Tensor, state: Tensor, dt: float) -> Tensor:
        ssm = self.hidden_ssm_matrices(q, dt=dt)
        next_state = torch.bmm(ssm["transition"], state.unsqueeze(-1)).squeeze(-1)
        return next_state + ssm["bias"]

    def step(self, z: Tensor, dt: float) -> Tensor:
        dt_value = float(dt)
        q0, h0 = self.split_latent(z)
        stats0 = self.derivative(z)

        q_mid = q0 + 0.5 * dt_value * stats0["dq"]
        h_mid = self._affine_hidden_step(q0, h0, 0.5 * dt_value)

        z_mid = self.join_latent(q_mid, h_mid)
        stats_mid = self.derivative(z_mid)
        q_next = q0 + dt_value * stats_mid["dq"]
        h_next = self._affine_hidden_step(q_mid, h0, dt_value)
        return self.join_latent(q_next, h_next)

    def flow(self, z: Tensor, *, dt: float, steps: int) -> Tensor:
        num_steps = int(steps)
        if num_steps <= 0:
            return z
        out = z
        for _ in range(num_steps):
            out = self.step(out, dt=dt)
        return out

    def project_to_manifold(self, z: Tensor) -> Tensor:
        q, h = self.split_latent(z)
        return self.join_latent(q, torch.zeros_like(h))

    def coarse_grain(self, z: Tensor) -> Tensor:
        q, h = self.split_latent(z)
        scale = max(self.cfg.rg_scale, 1e-6)
        delta_q = torch.tanh(self.coarse_q_net(q)) * (self.cfg.coarse_strength / scale)
        coarse_q = q + delta_q
        coarse_h = h / scale
        return self.join_latent(coarse_q, coarse_h)

    def module_groups(self) -> dict[str, list[nn.Module]]:
        encoder_modules: list[nn.Module] = [self.encoder_backbone, self.vamp_head, self.h_head]
        if self.modal_backbone is not None:
            encoder_modules.append(self.modal_backbone)

        decoder_modules: list[nn.Module] = [self.manifold_decoder, self.hidden_readout_net]
        dynamics_modules: list[nn.Module] = [
            self.q_residual_net,
            self.q_rate_net,
            self.q_coupling_net,
            self.h_rate_net,
            self.h_skew_net,
            self.h_dissipation_net,
            self.h_drive_net,
        ]

        return {
            "encoder": encoder_modules,
            "decoder": decoder_modules,
            "dynamics": dynamics_modules,
            "rg": [self.coarse_q_net],
        }
