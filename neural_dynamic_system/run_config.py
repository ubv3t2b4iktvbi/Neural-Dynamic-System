from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


CLI_CONFIG_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("run", ("out_dir",)),
    ("data", ("data_path", "array_key", "label_path", "label_array_key")),
    (
        "model",
        (
            "window",
            "q_dim",
            "h_dim",
            "koopman_dim",
            "latent_scheme",
            "koopman_input_mode",
            "hidden_coordinate_mode",
            "modal_dim",
            "modal_temperature",
            "encoder_type",
            "encoder_levels",
            "encoder_kernel_size",
            "hidden_dim",
            "depth",
            "vamp_head_depth",
            "vamp_whitening_momentum",
            "vamp_whitening_eps",
            "hidden_rank",
            "rg_scale",
            "coarse_strength",
            "rg_temperature",
        ),
    ),
    (
        "curriculum",
        (
            "curriculum_preset",
            "epochs",
            "phase0_min_epochs",
            "phase1_min_epochs",
            "phase2_min_epochs",
            "phase3_min_epochs",
            "phase0_reconstruction_improve",
            "phase0_koopman_stability_ratio",
            "phase0_reconstruction_target",
            "phase0_koopman_loss_ceiling",
            "phase0_stable_validations",
            "phase1_prediction_improve",
            "phase1_q_align_improve",
            "phase1_prediction_target",
            "phase1_q_align_target",
            "phase2_prediction_plateau_tolerance",
            "phase2_prediction_plateau_checks",
            "phase2_separation_target",
            "phase2_prediction_plateau_abs_tol",
            "phase2_long_horizon_target",
            "rollback_prediction_worsen_ratio",
            "rollback_long_horizon_worsen_ratio",
            "rollback_prediction_worsen_delta",
            "rollback_long_horizon_worsen_delta",
            "rollback_window",
            "rollback_weight_scale",
            "phase1_fraction",
            "phase2_fraction",
            "phase3_lr_scale",
        ),
    ),
    (
        "train",
        (
            "batch_size",
            "lr",
            "weight_decay",
            "horizons",
            "dt",
            "train_fraction",
            "device",
            "log_every",
            "validation_interval",
            "schedule_mode",
            "progress_bar",
            "early_stopping",
            "early_stopping_monitor",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "early_stopping_min_epochs",
            "early_stopping_start_phase",
            "contract_batch",
            "rg_horizon",
            "eval_batch_size",
            "seed",
        ),
    ),
    (
        "synthetic",
        (
            "steps",
            "num_episodes",
            "obs_dim",
            "burn_in",
            "noise_std",
            "synthetic_kind",
            "van_der_pol_mu",
        ),
    ),
    (
        "loss",
        (
            "vamp_weight",
            "vamp_align_weight",
            "koopman_weight",
            "diag_weight",
            "latent_align_weight",
            "semigroup_weight",
            "contract_weight",
            "separation_weight",
            "rg_weight",
            "metric_weight",
            "metric_mode",
            "hidden_l1_weight",
        ),
    ),
    (
        "supervision",
        (
            "q_label_indices",
            "h_label_indices",
            "q_supervised_weight",
            "q_supervision_mode",
            "h_supervised_weight",
            "label_standardize",
        ),
    ),
    ("reporting", ("trajectory_points", "report_plots")),
)

_SECTION_KEYS = {section: set(keys) for section, keys in CLI_CONFIG_SECTIONS}
_KNOWN_DESTS = {key for _, keys in CLI_CONFIG_SECTIONS for key in keys}


def resolve_repo_path(root: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(root) / path


def merge_nested_mapping(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_nested_mapping(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _coerce_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping in the config file")
    return dict(value)


def _normalize_cli_defaults(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    unsupported: list[str] = []
    for key, value in data.items():
        if key == "schema_version":
            continue
        if key in _SECTION_KEYS:
            section_values = _coerce_mapping(value, context=key)
            for field_name, field_value in section_values.items():
                if field_name not in _SECTION_KEYS[key]:
                    unsupported.append(f"{key}.{field_name}")
                    continue
                normalized[field_name] = deepcopy(field_value)
            continue
        if key in _KNOWN_DESTS:
            normalized[key] = deepcopy(value)
            continue
        unsupported.append(str(key))
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported config keys: {joined}")
    return normalized


def load_cli_defaults(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(raw, Mapping):
        raise ValueError("Launch config must deserialize to a mapping")
    return _normalize_cli_defaults(raw)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    return deepcopy(value)


def namespace_to_cli_config(args: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": 1}
    for section, keys in CLI_CONFIG_SECTIONS:
        payload[section] = {
            key: _serialize_value(getattr(args, key))
            for key in keys
        }
    return payload


def write_cli_config(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _serialize_value(dict(payload))
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Unsupported config format: {path}")
    path.write_text(
        yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
