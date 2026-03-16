# Koopman Factor View

Use this note when interpreting factor mining or coordinate ablations through Koopman theory.

## Core statement

A good factor can be treated as an approximate Koopman eigenfunction when it evolves close to:

`f_{t+1} ~= lambda f_t`

or, in the multivariate case,

`z_{t+1} ~= A z_t`

for a small linear operator `A`.

## Translation table

- `predictive factor` -> approximate Koopman observable
- `stable factor dynamics` -> low linear residual in factor coordinates
- `factor set` -> candidate Koopman-invariant subspace
- `factor drift / breakdown` -> coordinate not closed or not invariant enough

## What to check

1. Markov closure:
   If lagged coordinates help a lot, the observable is not close to a closed Koopman coordinate.
2. Linear invariance:
   If `z_{t+1}` is well approximated by `A z_t`, the coordinate is closer to a Koopman-invariant subspace.
3. Spectral preservation:
   If local spectral structure is badly distorted, short-term fit is not enough.
4. Factor-level eigenfunction score:
   If a scalar factor has strong `f_{t+1} ~= lambda f_t` behavior, it is a stronger Koopman-style candidate.

## Interpretation patterns

- High Koopman score + good spectral preservation:
  Strong candidate coordinate for structured modeling.
- High one-step gain + low Koopman score:
  Likely a predictive shortcut, not a stable invariant coordinate.
- Factor family wins only after heavy lagging:
  The factor is useful, but the closure still lives in an augmented state, not the scalar factor alone.
