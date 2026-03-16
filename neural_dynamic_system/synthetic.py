from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

SYNTHETIC_KINDS = ("van_der_pol",)


@dataclass
class SyntheticTrajectoryConfig:
    kind: str = "van_der_pol"
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
        if self.kind not in SYNTHETIC_KINDS:
            choices = ", ".join(SYNTHETIC_KINDS)
            raise ValueError(f"kind must be one of: {choices}")
        if float(self.van_der_pol_mu) <= 0.0:
            raise ValueError("van_der_pol_mu must be > 0")
        if int(self.num_episodes) < 1:
            raise ValueError("num_episodes must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def available_synthetic_kinds() -> tuple[str, ...]:
    return SYNTHETIC_KINDS


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


def _rk4_step(state: np.ndarray, dt: float, rhs) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _van_der_pol_rhs(state: np.ndarray, *, mu: float) -> np.ndarray:
    x, v = state
    dx = v
    dv = float(mu) * (1.0 - x * x) * v - x
    return np.array([dx, dv], dtype=float)


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
        episode_cfg = replace(cfg, seed=episode_seed, num_episodes=1, kind="van_der_pol")
        bundle = _generate_van_der_pol_episode(episode_cfg)
        metadata = dict(bundle.get("metadata", {}))
        metadata["episode_index"] = int(episode_idx)
        metadata["seed"] = int(episode_seed)
        bundle["metadata"] = metadata
        episode_bundles.append(bundle)

    first = episode_bundles[0]
    return {
        "trajectory": _stack_episode_payload(episode_bundles, "trajectory"),
        "hidden_state": _stack_episode_payload(episode_bundles, "hidden_state"),
        "labels": _stack_episode_payload(episode_bundles, "labels"),
        "label_names": list(first["label_names"]),
        "probe_labels": _stack_episode_payload(episode_bundles, "probe_labels"),
        "probe_label_names": list(first["probe_label_names"]),
        "metadata": {
            "kind": "van_der_pol",
            "num_episodes": episode_count,
            "steps_per_episode": int(cfg.steps),
            "episode_seeds": [int(seed) for seed in episode_seeds],
            "label_names": list(first["label_names"]),
            "probe_label_names": list(first["probe_label_names"]),
            "mu": float(cfg.van_der_pol_mu),
        },
    }
