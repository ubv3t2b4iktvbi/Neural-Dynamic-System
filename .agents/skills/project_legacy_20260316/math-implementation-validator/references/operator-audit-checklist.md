# Operator Audit Checklist

Use this checklist when validating gradients, Jacobians, Hessians, PDE operators, or other math-heavy array code.

## Translation Anchors
- State the domain and codomain of the operator.
- Record tensor shapes, axis order, batch semantics, and dtype.
- Note units, normalization constants, and coordinate spacing such as `dt`, `dx`, or `sigma`.
- Write the intended formula first, then mark where each symbol appears in code.

## Gradient-Family Checks
- Confirm whether the implementation returns `d output / d input`, a Jacobian-vector product, a vector-Jacobian product, or an aggregated gradient.
- Check Jacobian layout explicitly: output-by-input versus input-by-output.
- Verify transpose rules in matrix calculus and tensor contractions.
- Check `detach`, `stop_gradient`, `no_grad`, cached states, and in-place mutation.
- For Hessians, confirm symmetry assumptions and whether mixed partials are actually computed.

## Discrete Operator Checks
- Identify the stencil: forward, backward, central, spectral, or learned surrogate.
- Check spacing factors such as division by `dx`, `dt`, `dx**2`, or sample count.
- Confirm boundary handling: periodic, mirrored, clamped, ghost cells, or truncated edges.
- Distinguish convolution from correlation and note kernel flips.
- Verify whether normalization belongs inside or outside square roots, norms, logs, or softmax-like reductions.

## Array Semantics Checks
- Check broadcasting against the intended formula, especially for per-channel or per-sample factors.
- Verify `sum` versus `mean`, `keepdims`, masked reductions, and `N` versus `N - 1` normalization.
- Confirm index ordering for channels, time, space, and coordinate dimensions.
- Check whether a scalar factor should be applied before or after reduction.

## High-Signal Probe Cases
- Constant field should often yield zero gradient, zero divergence, or zero residual.
- Linear field should often yield constant first derivative and zero second derivative.
- Quadratic field should often expose Hessian symmetry and scaling factors.
- Axis permutation or coordinate swap can reveal index-order bugs.
- A one-cell or one-step boundary fixture can reveal padding and normalization mistakes.
