# Repo Map

## Main Entry Surface
- `scripts/run_neural_dynamic_system.py`: thin wrapper that calls `neural_dynamic_system.cli.main`.
- `neural_dynamic_system/cli.py`: user-facing CLI, file loading, synthetic-data branch, output writing.
- `neural_dynamic_system/config.py`: dataclass-backed config contracts for model, loss, supervision, and training.

## Core Modules
- `neural_dynamic_system/data.py`: trajectory loading, episode coercion, dataset windows, train/val splits, standardization.
- `neural_dynamic_system/synthetic.py`: synthetic generators for `toy`, `no_gap_toy`, `alanine_like`, and `van_der_pol`.
- `neural_dynamic_system/model.py`: encoder, q/h latent split, Koopman rates, hidden SSM, decoder, RG transform.
- `neural_dynamic_system/training.py`: loss bundle, curriculum phases, fit loop, summary creation.

## Output Surface
- Default runs live under `runs/neural_dynamic_system/`.
- Common artifacts: `config.json`, `summary.json`, `history.csv`, `model.pt`, `trajectory_preview.csv`.
- Synthetic runs may also emit `synthetic_hidden_state.csv`, `synthetic_labels.csv`, and `synthetic_probe_labels.csv`.
- Supervised or synthetic-probe runs may emit `label_probe.json` or `synthetic_hidden_probe.json`, plus a correlations CSV.

## Fast Validation
- CLI sanity: `python scripts/run_neural_dynamic_system.py --help`
- Syntax sanity: `python -m compileall neural_dynamic_system scripts`
- Tiny run:

```bash
python scripts/run_neural_dynamic_system.py \
  --out_dir runs/neural_dynamic_system/smoke \
  --synthetic_kind toy \
  --steps 512 \
  --num_episodes 2 \
  --epochs 2 \
  --batch_size 64 \
  --horizons 1 2 4 \
  --device cpu
```
