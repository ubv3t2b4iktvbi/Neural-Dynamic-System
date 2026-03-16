from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

_TWO_PI = 2.0 * np.pi
_ALANINE_BASINS = np.array(
    [
        [-2.35, 2.45],   # C7eq
        [1.15, -1.10],   # C7ax
        [-1.25, -0.75],  # alpha_R
        [0.95, 1.05],    # alpha_L
    ],
    dtype=float,
)
_ALANINE_BASIN_NAMES = ["C7eq", "C7ax", "alpha_R", "alpha_L"]
_ALANINE_WIDTHS = np.array([0.52, 0.60], dtype=float)


@dataclass
class SyntheticTrajectoryConfig:
    kind: str = "toy"
    steps: int = 4096
    dt: float = 0.05
    obs_dim: int = 8
    burn_in: int = 256
    noise_std: float = 0.01
    seed: int = 123
    alanine_fast_dim: int = 4
    van_der_pol_mu: float = 5.0
    num_episodes: int = 1

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind not in {"toy", "no_gap_toy", "alanine_like", "van_der_pol"}:
            raise ValueError("kind must be one of: toy, no_gap_toy, alanine_like, van_der_pol")
        if int(self.alanine_fast_dim) < 2:
            raise ValueError("alanine_fast_dim must be >= 2")
        if float(self.van_der_pol_mu) <= 0.0:
            raise ValueError("van_der_pol_mu must be > 0")
        if int(self.num_episodes) < 1:
            raise ValueError("num_episodes must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return ((value + np.pi) % _TWO_PI) - np.pi


def _mix_observations(
    feature_stack: np.ndarray,
    *,
    obs_dim: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = rng.normal(scale=0.7, size=(int(obs_dim), feature_stack.shape[1]))
    trajectory = np.tanh(feature_stack @ weights.T)
    linear_term = np.zeros_like(trajectory)
    cols = min(trajectory.shape[1], feature_stack.shape[1])
    linear_term[:, :cols] = feature_stack[:, :cols]
    trajectory += 0.15 * linear_term
    trajectory += rng.normal(scale=float(noise_std), size=trajectory.shape)
    return trajectory.astype(np.float32)


def _toy_rhs(state: np.ndarray, *, no_gap: bool) -> np.ndarray:
    q1, q2, a, m = state
    dq1 = 0.22 * q2
    if no_gap:
        dq2 = 0.19 * (0.85 * (1.0 - q1 * q1) * q2 - q1 + 0.20 * np.tanh(m)) + 0.10 * a
        da = -0.55 * a - 0.18 * q2 + 0.25 * np.sin(q1) + 0.08 * m
        dm = -0.42 * m + 0.30 * a
    else:
        dq2 = 0.22 * (0.9 * (1.0 - q1 * q1) * q2 - q1 + 0.25 * np.tanh(m)) + 0.05 * a
        da = -3.2 * a - 0.25 * q2 + 0.35 * np.sin(q1)
        dm = -0.65 * m + 0.45 * a
    return np.array([dq1, dq2, da, dm], dtype=float)


def _rk4_step(state: np.ndarray, dt: float, rhs) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _build_toy_features(hidden: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            hidden[:, 0],
            hidden[:, 1],
            hidden[:, 2],
            hidden[:, 3],
            hidden[:, 0] * hidden[:, 1],
            hidden[:, 0] ** 2 - hidden[:, 1] ** 2,
            np.sin(hidden[:, 0]),
            np.cos(hidden[:, 1]),
            hidden[:, 2] * hidden[:, 0],
            hidden[:, 3] * hidden[:, 1],
        ]
    )


def _van_der_pol_rhs(state: np.ndarray, *, mu: float) -> np.ndarray:
    x, v = state
    dx = v
    dv = float(mu) * (1.0 - x * x) * v - x
    return np.array([dx, dv], dtype=float)


def _generate_toy_episode(
    cfg: SyntheticTrajectoryConfig,
    *,
    no_gap: bool,
) -> dict[str, np.ndarray | list[str] | dict[str, object]]:
    rng = np.random.default_rng(int(cfg.seed))
    total_steps = int(cfg.steps) + int(cfg.burn_in)
    hidden = np.zeros((total_steps, 4), dtype=float)
    hidden[0] = np.array([0.8, -0.2, 0.6, 0.0], dtype=float) + rng.normal(
        scale=np.array([0.20, 0.20, 0.12, 0.08], dtype=float),
        size=4,
    )
    rhs = lambda state: _toy_rhs(state, no_gap=no_gap)
    system_noise = 0.015 if no_gap else 0.01
    for idx in range(1, total_steps):
        next_state = _rk4_step(hidden[idx - 1], dt=float(cfg.dt), rhs=rhs)
        next_state += rng.normal(scale=system_noise, size=4)
        hidden[idx] = next_state

    hidden = hidden[int(cfg.burn_in) :]
    feature_stack = _build_toy_features(hidden)
    trajectory = _mix_observations(
        feature_stack,
        obs_dim=int(cfg.obs_dim),
        noise_std=float(cfg.noise_std),
        rng=rng,
    )
    label_names = ["q1", "q2", "a", "m"]
    return {
        "trajectory": trajectory,
        "hidden_state": hidden.astype(np.float32),
        "labels": hidden.astype(np.float32),
        "label_names": label_names,
        "probe_labels": hidden.astype(np.float32),
        "probe_label_names": label_names,
        "metadata": {
            "kind": ("no_gap_toy" if no_gap else "toy"),
            "label_names": label_names,
            "seed": int(cfg.seed),
        },
    }


def _generate_van_der_pol_episode(
    cfg: SyntheticTrajectoryConfig,
) -> dict[str, np.ndarray | list[str] | dict[str, object]]:
    rng = np.random.default_rng(int(cfg.seed))
    total_steps = int(cfg.steps) + int(cfg.burn_in)
    hidden = np.zeros((total_steps, 2), dtype=float)
    hidden[0] = np.array([1.8, 0.0], dtype=float) + rng.normal(scale=np.array([0.35, 0.25], dtype=float), size=2)
    rhs = lambda state: _van_der_pol_rhs(state, mu=float(cfg.van_der_pol_mu))
    system_noise = np.array([0.004, 0.006], dtype=float)
    for idx in range(1, total_steps):
        next_state = _rk4_step(hidden[idx - 1], dt=float(cfg.dt), rhs=rhs)
        next_state += rng.normal(scale=system_noise, size=2)
        hidden[idx] = next_state

    hidden = hidden[int(cfg.burn_in) :]
    x = hidden[:, 0]
    v = hidden[:, 1]
    feature_stack = np.column_stack(
        [
            x,
            v,
            x * v,
            x**2,
            v**2,
            np.sin(x),
            np.cos(x),
            np.tanh(v),
            x**3,
            v * np.sin(x),
        ]
    )
    trajectory = _mix_observations(
        feature_stack,
        obs_dim=int(cfg.obs_dim),
        noise_std=float(cfg.noise_std),
        rng=rng,
    )
    amplitude = np.sqrt(np.maximum(x * x + v * v, 1e-8))
    phase = np.arctan2(v, x)
    labels = np.column_stack([x, v]).astype(np.float32)
    label_names = ["x", "v"]
    probe_labels = np.column_stack([x, v, phase, amplitude]).astype(np.float32)
    probe_label_names = ["x", "v", "phase", "amplitude"]
    return {
        "trajectory": trajectory,
        "hidden_state": hidden.astype(np.float32),
        "labels": labels,
        "label_names": label_names,
        "probe_labels": probe_labels,
        "probe_label_names": probe_label_names,
        "metadata": {
            "kind": "van_der_pol",
            "label_names": label_names,
            "probe_label_names": probe_label_names,
            "mu": float(cfg.van_der_pol_mu),
            "seed": int(cfg.seed),
        },
    }


def _alanine_basin_weights(phi: float, psi: float) -> np.ndarray:
    point = np.array([phi, psi], dtype=float)
    deltas = _wrap_angle(point[None, :] - _ALANINE_BASINS)
    dist2 = np.sum((deltas / _ALANINE_WIDTHS[None, :]) ** 2, axis=1)
    logits = -0.5 * dist2
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / (weights.sum() + 1e-12)


def _generate_alanine_like_episode(
    cfg: SyntheticTrajectoryConfig,
) -> dict[str, np.ndarray | list[str] | dict[str, object]]:
    rng = np.random.default_rng(int(cfg.seed))
    fast_dim = int(cfg.alanine_fast_dim)
    total_steps = int(cfg.steps) + int(cfg.burn_in)
    hidden = np.zeros((total_steps, 3 + fast_dim), dtype=float)
    start_basin = int(rng.integers(0, len(_ALANINE_BASINS)))
    hidden[0, 0:2] = _wrap_angle(
        _ALANINE_BASINS[start_basin] + rng.normal(scale=np.array([0.28, 0.28], dtype=float), size=2)
    )
    hidden[0, 2 : 2 + fast_dim] = rng.normal(scale=0.18, size=fast_dim)
    hidden[0, -1] = float(rng.normal(scale=0.08))
    fast_rates = np.linspace(6.0, 14.0, fast_dim, dtype=float)
    basin_hist = np.zeros((total_steps, len(_ALANINE_BASINS)), dtype=float)

    for idx in range(1, total_steps):
        phi, psi = hidden[idx - 1, 0], hidden[idx - 1, 1]
        basin_weights = _alanine_basin_weights(phi, psi)
        basin_hist[idx - 1] = basin_weights

        deltas = _wrap_angle(np.array([phi, psi], dtype=float)[None, :] - _ALANINE_BASINS)
        pull = -np.sum(basin_weights[:, None] * deltas / (_ALANINE_WIDTHS[None, :] ** 2), axis=0)
        coupling = np.array(
            [
                0.18 * np.sin(psi - phi) + 0.06 * basin_weights[3] - 0.05 * basin_weights[0],
                0.16 * np.sin(phi + 0.5 * psi) + 0.04 * basin_weights[0] - 0.06 * basin_weights[1],
            ],
            dtype=float,
        )
        slow_noise = rng.normal(scale=0.11, size=2)
        slow_next = (
            np.array([phi, psi], dtype=float)
            + float(cfg.dt) * (0.26 * pull + coupling)
            + np.sqrt(float(cfg.dt)) * slow_noise
        )
        slow_next = _wrap_angle(slow_next)

        fast_prev = hidden[idx - 1, 2 : 2 + fast_dim]
        forcing = np.zeros(fast_dim, dtype=float)
        forcing[0] = 0.35 * np.sin(phi) + 0.22 * np.cos(psi)
        forcing[1] = 0.30 * np.sin(phi - psi) + 0.18 * basin_weights[2]
        for fast_idx in range(2, fast_dim):
            forcing[fast_idx] = (
                0.18 * np.cos((fast_idx + 1) * phi)
                + 0.14 * np.sin((fast_idx + 2) * psi)
                + 0.10 * basin_weights[fast_idx % len(_ALANINE_BASINS)]
            )
        fast_noise = rng.normal(scale=0.16, size=fast_dim)
        fast_next = (
            fast_prev
            + float(cfg.dt) * (-fast_rates * fast_prev + forcing)
            + np.sqrt(float(cfg.dt)) * fast_noise
        )

        closure_prev = hidden[idx - 1, -1]
        closure_drive = 0.22 * (basin_weights[0] - basin_weights[1]) + 0.16 * fast_next[0] - 0.10 * fast_next[1]
        closure_next = (
            closure_prev
            + float(cfg.dt) * (-1.25 * closure_prev + closure_drive)
            + np.sqrt(float(cfg.dt)) * rng.normal(scale=0.05)
        )

        hidden[idx, 0:2] = slow_next
        hidden[idx, 2 : 2 + fast_dim] = fast_next
        hidden[idx, -1] = closure_next

    basin_hist[-1] = _alanine_basin_weights(hidden[-1, 0], hidden[-1, 1])
    hidden = hidden[int(cfg.burn_in) :]
    basin_hist = basin_hist[int(cfg.burn_in) :]
    phi = hidden[:, 0]
    psi = hidden[:, 1]
    fast = hidden[:, 2 : 2 + fast_dim]
    closure = hidden[:, -1]
    feature_stack = np.column_stack(
        [
            np.sin(phi),
            np.cos(phi),
            np.sin(psi),
            np.cos(psi),
            np.sin(phi + psi),
            np.cos(phi - psi),
            basin_hist[:, 0],
            basin_hist[:, 1],
            basin_hist[:, 2],
            basin_hist[:, 3],
            fast,
            fast**2,
            fast[:, 0] * np.sin(phi),
            fast[:, 1] * np.cos(psi),
            closure,
            closure * np.sin(phi - psi),
        ]
    )
    trajectory = _mix_observations(
        feature_stack,
        obs_dim=int(cfg.obs_dim),
        noise_std=float(cfg.noise_std),
        rng=rng,
    )
    labels = np.column_stack([phi, psi, fast]).astype(np.float32)
    label_names = ["phi", "psi"] + [f"fast_{idx + 1}" for idx in range(fast_dim)]
    probe_labels = np.column_stack(
        [
            np.sin(phi),
            np.cos(phi),
            np.sin(psi),
            np.cos(psi),
            fast,
        ]
    ).astype(np.float32)
    probe_label_names = ["sin_phi", "cos_phi", "sin_psi", "cos_psi"] + [f"fast_{idx + 1}" for idx in range(fast_dim)]
    basin_assign = basin_hist.argmax(axis=1)
    basin_counts = {
        name: int(np.sum(basin_assign == basin_idx))
        for basin_idx, name in enumerate(_ALANINE_BASIN_NAMES)
    }
    return {
        "trajectory": trajectory,
        "hidden_state": hidden.astype(np.float32),
        "labels": labels,
        "label_names": label_names,
        "probe_labels": probe_labels,
        "probe_label_names": probe_label_names,
        "metadata": {
            "kind": "alanine_like",
            "label_names": label_names,
            "probe_label_names": probe_label_names,
            "basin_names": list(_ALANINE_BASIN_NAMES),
            "basin_counts": basin_counts,
            "seed": int(cfg.seed),
        },
    }


def _generate_single_synthetic_trajectory(
    cfg: SyntheticTrajectoryConfig,
) -> dict[str, np.ndarray | list[str] | dict[str, object]]:
    if cfg.kind == "toy":
        return _generate_toy_episode(cfg, no_gap=False)
    if cfg.kind == "no_gap_toy":
        return _generate_toy_episode(cfg, no_gap=True)
    if cfg.kind == "van_der_pol":
        return _generate_van_der_pol_episode(cfg)
    return _generate_alanine_like_episode(cfg)


def _stack_episode_payload(
    episodes: list[dict[str, np.ndarray | list[str] | dict[str, object]]],
    key: str,
) -> np.ndarray:
    values = [np.asarray(bundle[key], dtype=np.float32) for bundle in episodes]
    return np.stack(values, axis=0)


def generate_synthetic_trajectory(cfg: SyntheticTrajectoryConfig) -> dict[str, np.ndarray | list[str] | dict[str, object]]:
    episode_count = int(cfg.num_episodes)
    if episode_count == 1:
        episode_seeds = [int(cfg.seed)]
    else:
        seed_rng = np.random.default_rng(int(cfg.seed))
        episode_seeds = [
            int(seed)
            for seed in seed_rng.integers(
                low=0,
                high=np.iinfo(np.int32).max,
                size=episode_count,
                dtype=np.int64,
            )
        ]

    episode_bundles: list[dict[str, np.ndarray | list[str] | dict[str, object]]] = []
    for episode_idx, episode_seed in enumerate(episode_seeds):
        episode_cfg = replace(cfg, seed=episode_seed, num_episodes=1)
        bundle = _generate_single_synthetic_trajectory(episode_cfg)
        metadata = dict(bundle.get("metadata", {}))
        metadata["episode_index"] = int(episode_idx)
        metadata["seed"] = int(episode_seed)
        bundle["metadata"] = metadata
        episode_bundles.append(bundle)

    first = episode_bundles[0]
    metadata = {
        "kind": str(cfg.kind),
        "num_episodes": episode_count,
        "steps_per_episode": int(cfg.steps),
        "episode_seeds": [int(seed) for seed in episode_seeds],
        "label_names": list(first["label_names"]),
        "probe_label_names": list(first["probe_label_names"]),
    }
    if "basin_names" in first["metadata"]:
        metadata["basin_names"] = list(first["metadata"]["basin_names"])
        metadata["basin_counts"] = {
            name: int(
                sum(
                    int(bundle["metadata"]["basin_counts"][name])
                    for bundle in episode_bundles
                )
            )
            for name in metadata["basin_names"]
        }

    return {
        "trajectory": _stack_episode_payload(episode_bundles, "trajectory"),
        "hidden_state": _stack_episode_payload(episode_bundles, "hidden_state"),
        "labels": _stack_episode_payload(episode_bundles, "labels"),
        "label_names": list(first["label_names"]),
        "probe_labels": _stack_episode_payload(episode_bundles, "probe_labels"),
        "probe_label_names": list(first["probe_label_names"]),
        "metadata": metadata,
    }
