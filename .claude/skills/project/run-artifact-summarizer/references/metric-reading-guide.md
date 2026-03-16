# Metric Reading Guide

## Read Order
1. `config.json`
2. `summary.json`
3. `history.csv`
4. probe JSON and correlations CSV
5. optional trajectory or synthetic preview CSVs

## Metric Direction
- Higher is usually better:
  - `best_val_vamp_score`
  - probe `mean_r2`
- Lower is usually better:
  - `best_val_loss`
  - `best_val_koopman_loss`
  - `best_val_diag_loss`
  - `best_val_prediction_loss`
  - `best_val_latent_align_loss`
  - `best_val_semigroup_loss`
  - `best_val_contract_loss`
  - `best_val_separation_loss`
  - `best_val_rg_loss`
- For `best_val_hidden_sym_eig_upper`, more negative is usually more contractive.

## Phase Notes
- `best_val_phase` tells you which curriculum phase produced the selected checkpoint.
- `best_phase3_*` is useful when RG or later-stage constraints matter more than the overall minimum loss.
- `last_val_*` helps catch late-epoch instability or regression after the best checkpoint.

## Comparison Discipline
- Compare runs with the same data source and roughly the same training budget unless the budget itself is the experiment.
- When `latent_scheme` is `soft_spectrum`, check the modal summaries as part of the interpretation.
