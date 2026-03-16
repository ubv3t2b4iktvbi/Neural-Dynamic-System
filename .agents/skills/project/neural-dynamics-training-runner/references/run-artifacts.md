# Run Artifacts

## Common Outputs
- `config.json`: full run configuration, including source summary and standardization stats.
- `summary.json`: best and last validation metrics, selected phase, device, modal summaries.
- `history.csv`: per-epoch train and val metrics with `phase`.
- `model.pt`: saved model state plus config payload.
- `trajectory_preview.csv`: flattened preview of the loaded or generated trajectory episodes.

## Synthetic-Only Extras
- `synthetic_hidden_state.csv`
- `synthetic_labels.csv`
- `synthetic_probe_labels.csv`

## Probe Outputs
- If explicit labels were used: `label_probe.json` and `label_component_corrs.csv`
- If synthetic hidden probes were used: `synthetic_hidden_probe.json` and `synthetic_component_corrs.csv`

## Good Smoke Commands

Synthetic smoke:

```bash
python scripts/run_neural_dynamic_system.py \
  --out_dir runs/neural_dynamic_system/smoke-toy \
  --synthetic_kind toy \
  --steps 512 \
  --num_episodes 2 \
  --epochs 2 \
  --batch_size 64 \
  --horizons 1 2 4 \
  --device cpu
```

Supervised synthetic smoke:

```bash
python scripts/run_neural_dynamic_system.py \
  --out_dir runs/neural_dynamic_system/smoke-supervised \
  --synthetic_kind van_der_pol \
  --steps 768 \
  --num_episodes 2 \
  --epochs 2 \
  --q_supervised_weight 0.25 \
  --q_label_indices 0 \
  --h_supervised_weight 0.25 \
  --h_label_indices 1 \
  --device cpu
```

## When Comparing Runs
- Keep the data source, `seed`, `steps`, and `curriculum_preset` fixed unless the comparison is specifically about those settings.
- Compare both `best_val_*` metrics and probe outputs, not just one scalar loss.
