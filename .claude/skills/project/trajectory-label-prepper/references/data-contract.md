# Data Contract

## Supported Trajectory Inputs
- `.npy`: loaded with `numpy.load`
- `.npz`: uses `trajectory` by default, the only array if exactly one exists, or `--array_key`
- `.csv` and `.txt`: read with `pandas.read_csv`

## Accepted Shapes
- `[T]` -> coerced to `[T, 1]`
- `[T, D]` -> one episode
- `[E, T, D]` -> multiple episodes
- object arrays or sequences -> each item becomes one `[T, D]` episode

## Label Rules
- Labels must have the same episode count as the trajectory.
- Each label episode must have the same number of time steps as the matching trajectory episode.
- `q_label_indices` and `h_label_indices` refer to columns inside the label array, not the observation array.
- `--label_standardize` is enabled by default.

## Split Constraint
- Every episode must be long enough for:
  - `context_len`
  - the largest prediction horizon
  - the train/val split margin used by `compute_train_val_split`

## Synthetic Alternatives
- `toy`: slow-fast toy system with a clearer timescale gap
- `no_gap_toy`: toy system with weaker separation
- `alanine_like`: alanine-style angular slow coordinates plus fast modes
- `van_der_pol`: noisy Van der Pol oscillator
