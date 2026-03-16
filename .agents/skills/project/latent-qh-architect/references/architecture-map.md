# Architecture Map

## Config Layer
- `neural_dynamic_system/config.py`
  - `ModelConfig`: latent sizes, encoder, rate constraints, RG settings
  - `LossConfig`: loss weights and metric mode
  - `SupervisionConfig`: q/h label indices and weights
  - `TrainConfig`: horizons, curriculum, device, seeds

## CLI Layer
- `neural_dynamic_system/cli.py`
  - owns user-facing flags
  - builds config dataclasses
  - writes `config.json` and `summary.json`
  - controls synthetic vs file-backed data flow

## Model Layer
- `neural_dynamic_system/model.py`
  - encoder backbone
  - q/h latent construction
  - Koopman head and modal rates
  - hidden SSM matrices and `flow`
  - decoder geometry and `rg_transform`

## Training Layer
- `neural_dynamic_system/training.py`
  - curriculum phases
  - loss bundle and summary keys
  - model selection across phases

## Coupling Rules
- New CLI flags should usually map to one config dataclass field.
- New model behavior should be reflected in saved config payloads and, when useful, `summary.json`.
- Changes to latent meanings often require checking label-probe logic in `cli.py` and loss computation in `training.py`.
