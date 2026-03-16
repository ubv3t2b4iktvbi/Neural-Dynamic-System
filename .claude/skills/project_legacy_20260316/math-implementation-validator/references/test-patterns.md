# Test Patterns

Choose the smallest independent oracle that can falsify the mathematical claim.

## Closed-Form Oracles
- Use analytic functions with known derivatives, integrals, or conserved quantities.
- Prefer low-dimensional fixtures that keep the expected value readable in the test itself.

## Numerical Derivative Checks
- Use finite differences for gradients, Jacobians, or update rules when the implementation is smooth enough.
- Use complex-step checks when the code path supports complex numbers and subtraction error would dominate.
- Scale perturbation size to the dtype and conditioning; do not hardcode one epsilon everywhere.

## Structural Property Checks
- Check linearity, symmetry, antisymmetry, positive-definiteness, or skew-symmetry when the operator should have them.
- Check invariance under translation, rotation, rescaling, or permutation when the math claims it.
- Check conservation, monotonicity, or energy decay for dynamical updates when the model should preserve them.

## Cross-Implementation Checks
- Compare the optimized implementation with a slow reference implementation on tiny fixtures.
- Compare autograd output with an independent numerical oracle, not only with another autograd path.
- Use manufactured solutions for PDE-style operators when no closed-form full pipeline oracle exists.

## Fixture Design Rules
- Keep one deterministic fixture with exact expected behavior.
- Add one randomized test only if it covers a new failure mode such as broadcasting, batching, or conditioning.
- Make tolerance scaling explicit and document whether it is absolute, relative, or mixed.
- Prefer tiny fixtures that isolate one operator over full training loops or long rollouts.
