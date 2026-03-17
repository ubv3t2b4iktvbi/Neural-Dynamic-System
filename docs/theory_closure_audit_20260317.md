# Theory Closure Audit (2026-03-17)

This note audits the revised first-principles derivation against strict mathematical closure.

## Verdict

The revised draft is much stronger than the earlier version, but it is **not yet fully rigorous**.
The main remaining issues are:

1. The Koopman operator is defined on `L^2(X, mu)` with `mu` supported on the attractor, but the stated stable eigenvalues and Floquet exponents live on a neighborhood or basin, not on `L^2(mu)` itself.
2. The fixed-point and limit-cycle spectrum theorems need stronger smooth-linearization hypotheses than Hartman-Grobman alone.
3. The tubular coordinates `(q, h_tilde)` are not explicitly closed until the implicit `qdot` dependence inside `Ddot(q)` is resolved.
4. The normal block `A(q) = D^T Df D` is only a projected normal-normal block, not automatically the full transverse Jacobian.
5. The `rho`-interpolated spectral ansatz is still an engineering parameterization, not a theorem.
6. The exponential `q` integrator still has a sign error in the draft.
7. The type-loss proposition is false as written.
8. The curvature penalty should use the second fundamental form, not a raw summed Hessian.

The sections below record the exact fixes.

## 1. Base Dynamical Setup

### What is correct

- `X` as a finite-dimensional `C^r` manifold with `r >= 3`.
- A smooth autonomous flow `F_t`.
- A compact attracting set restricted to either a hyperbolic equilibrium or a hyperbolic limit cycle.

### Required fix

For a stable limit cycle, one Floquet exponent is always zero in the tangential direction.
So assumption A2 should read:

- equilibrium: all eigenvalues of `Df(x*)` have negative real part;
- limit cycle: all **nontrivial transverse** Floquet exponents have negative real part.

## 2. Koopman Operator: Function Space Mismatch

### Current issue

The draft defines

`K_tau : L^2(X, mu) -> L^2(X, mu)`

with `mu` supported on the attractor `A`.

That is fine for the measure-preserving attractor dynamics, but then:

- for a stable equilibrium with `mu = delta_{x*}`, `L^2(mu)` only sees the value at `x*`, so the generator spectrum is only `{0}`;
- for a stable limit cycle with `mu` the invariant cycle measure, `L^2(mu)` only sees functions restricted to the cycle, so the point spectrum contains the harmonic phase modes `ik omega`, but not the stable transverse Floquet modes.

### Correct repair

Use two different Koopman settings:

1. **Attractor-restricted Koopman operator**

   `K_tau^A : L^2(A, mu_A) -> L^2(A, mu_A)`

   This is the right object for phase harmonics on the attractor.

2. **Neighborhood / basin Koopman operator**

   `K_tau^U phi = phi o F_tau`

   acting on `C^1(U)` or another smooth observable space on a forward-invariant neighborhood `U` of `A`.

The stable eigenvalues from linearization and Floquet theory belong to the second setting, not the first.

## 3. Equilibrium Spectrum Theorem

### What fails in the current proof

Hartman-Grobman gives only a topological conjugacy.
That is not enough to pull back polynomial Koopman eigenfunctions as differentiable eigenfunctions satisfying

`f . grad psi = lambda psi`.

### Correct repair

Replace Theorem 2.1 by:

If the equilibrium is hyperbolic and the vector field satisfies the additional smooth-linearization assumptions needed for a `C^1` or analytic conjugacy near `x*`, then the local point spectrum contains the nonnegative integer combinations of the linearization eigenvalues.

So the theorem is valid only after adding a smooth or analytic conjugacy hypothesis.

## 4. Limit-Cycle Spectrum Theorem

### What is correct

- the phase harmonics `e^{ik theta}`;
- the spectral family `ik omega + sum_j m_j nu_j`.

### Required repair

State explicitly that this is a **local neighborhood theorem** near a hyperbolic periodic orbit, using:

- a smooth asymptotic phase / isochron map;
- Floquet coordinates;
- smooth dependence of the transverse linearization.

Also distinguish:

- spectrum on the attractor itself: `ik omega`;
- transverse stable modes: only visible on a neighborhood function space.

## 5. Delay Embedding

### What is correct

The conjugacy argument on the embedded attractor is valid once an actual embedding `h : A -> h(A)` is available.

### Required repair

Takens/Sauer guarantees an embedding of the attractor dynamics, not of the full basin.
Therefore:

- point spectrum on the attractor is preserved under delay reconstruction;
- recovery of stable off-attractor modes is not a direct consequence of Takens alone.

So the sentence "all later Koopman eigenfunctions on the delay space have the same eigenvalues as the original system" must be narrowed to:

"the attractor-restricted Koopman point spectrum is preserved under the embedding."

## 6. Tubular Coordinates and Exact Closure

### Good part

The derivation

`x = g(q) + D(q) h_tilde`

`xdot = Dg qdot + Ddot h_tilde + D hdot_tilde`

and the projected normal equation

`hdot_tilde = D^T f(g(q) + D h_tilde) - Omega(q, qdot) h_tilde`

with `Omega = D^T Ddot`

is correct.

### Main closure issue

The displayed `qdot` formula is still implicit, because `Ddot` depends on `qdot`.

From

`qdot = G^{-1} Dg^T f(g(q) + D h_tilde) - G^{-1} Dg^T Ddot h_tilde`

and

`Ddot = sum_i (partial_i D) qdot_i`,

one gets

`(I + C(q, h_tilde)) qdot = G^{-1} Dg^T f(g(q) + D h_tilde)`,

where the `i`-th column of `C(q, h_tilde)` is

`C_i(q, h_tilde) = G^{-1} Dg^T (partial_i D) h_tilde`.

Therefore the explicit closed equation is

`qdot = (I + C(q, h_tilde))^{-1} G^{-1} Dg^T f(g(q) + D h_tilde)`,

provided `I + C(q, h_tilde)` is invertible.

That invertibility is guaranteed locally for sufficiently small `h_tilde`.

## 7. Exact Coordinate Validity Needs a Tubular-Neighborhood Assumption

The map `(q, h_tilde) -> g(q) + D(q) h_tilde` is an exact local chart only if:

- `D(q)` spans the full normal bundle, so `d_h = d_x - d_q`;
- the chart is restricted to a small tubular neighborhood.

If `d_h` is chosen smaller than the codimension, the representation is only a reduced normal approximation, not an exact coordinate system.

## 8. Normal Dynamics and the Meaning of `A(q)`

### Correct linearization

On an invariant manifold, the normal linearization is

`hdot_tilde = [D^T Df(g(q)) D - Omega(q, qdot)] h_tilde + O(||h_tilde||^2)`.

### Required repair

`A(q) = D^T Df(g(q)) D` is only the normal-normal projected Jacobian in the chosen frame.
Its eigenvalues are not automatically eigenvalues of the full Jacobian unless the normal bundle is invariant under `Df`.

So the rigorous statement is:

- `A(q)` is the projected transverse block;
- transverse contraction is controlled by the symmetric part of
  `A_full(q, qdot) = A(q) - Omega(q, qdot)`.

## 9. Koopman Structure on the Manifold

### Core statement

If `M = g(Q)` is invariant and `qdot = v(q)` is the induced flow, then Koopman eigenfunctions of `v` pull back to Koopman eigenfunctions on `M`.

### Proof repair

Avoid the incorrect identity involving `Dg^{-L} grad_q psi`.
The clean derivation is:

- along a trajectory `x(t) = g(q(t))`,
- `d/dt [psi(q(t))] = grad_q psi(q) . qdot`,
- and `qdot = G^{-1} Dg^T f(g(q))`.

So

`d/dt [psi o g^{-1}](x(t)) = grad_q psi(q) . G^{-1} Dg^T f(g(q))`.

If `grad_q psi . v = lambda psi`, then `(psi o g^{-1})` is a Koopman eigenfunction on `M`.

## 10. The Spectral Ansatz with `rho`

This is still **not** a theorem.

What the earlier layers justify is:

- equilibrium: local stable eigenvalues from the slow tangent block selected by the invariant manifold;
- limit cycle: neutral phase harmonics on the cycle and stable transverse Floquet exponents in a neighborhood.

What they do **not** justify is a single continuous interpolation

`lambda_j = -mu_j + i rho k_j omega`

between the two cases.

So `rho` must be presented as a learnable architectural gate, not as something "emerging from the previous theorems."

## 11. Whitening

This part is correct:

- multiplicative rescaling preserves Koopman eigenfunctions;
- additive centering breaks them unless `lambda = 0`.

Implementation note:

- the repo's q/Koopman whitening has been updated so that it scales channels without subtracting the running mean.

## 12. Integrator

### Sign fix

If the linear part is

`qdot = Lambda q`,

then the exact flow is

`q(t + dt) = exp(Lambda dt) q(t)`,

not `exp(-Lambda dt)`.

If `lambda_j = -mu_j + i omega_j`, then

`exp(lambda_j dt) = exp(-mu_j dt) exp(i omega_j dt)`.

The current draft has the sign flipped.

### Method-name fix

The displayed formula is not a literal Strang splitting unless the nonlinear subflow is separately integrated.
It is closer to an exponential midpoint / ETD-style approximation.

So the rigorous statement should be:

- exact on the linear block;
- second-order local truncation error from midpoint quadrature on the variation-of-constants integral.

## 13. Geometry and Curvature Losses

### `L_geo`

This is mathematically closed only if an ambient vector field estimate `f_theta` on `X` is part of the model.
Without such an `f_theta`, `P_perp(q) f_theta(g(q))` is not an internal quantity of the current q/h architecture.

### `L_curv`

The current formula

`|| sum_{i,j} partial_{ij} g ||_F^2`

is not the right geometric object.

The correct coordinate-free curvature regularizer is based on the second fundamental form:

`L_curv = E_q [ || II(q) ||_F^2 ]`

with

`|| II(q) ||_F^2 = sum_{i,j} || P_perp(q) partial_{ij} g(q) ||_2^2`.

That is the object directly linked to extrinsic bending.

## 14. Separation Loss

### Current issue

The draft compares real parts of eigenvalues of `-A(q)` via a Weyl bound written for a generic DPLR matrix.
That is not rigorous for nonnormal matrices if the goal is contraction.

### Correct repair

Use the symmetric-part contraction rate

`alpha_h(q) = -lambda_max( (A_full + A_full^T) / 2 )`.

Then require

`alpha_h(q) >= kappa alpha_q + delta`

samplewise or via a worst-batch lower bound.

This is the correct object because it controls

`d/dt ||h_tilde||^2`.

## 15. Type Loss

The proposition in the draft is false.

For

`L_type(rho) = beta rho (1 - rho) - gamma C_long rho`,

the stationary point

`rho* = 1/2 - gamma C_long / (2 beta)`

has second derivative `-2 beta < 0`, so it is a maximum, not a stable minimum.

Under the box constraint `rho in [0, 1]`, the minima are the endpoints:

- `L_type(0) = 0`,
- `L_type(1) = -gamma C_long`.

So:

- if `C_long > 0`, minimization prefers `rho = 1`;
- if `C_long = 0`, `rho = 0` and `rho = 1` tie.

Therefore this loss does **not** implement the claimed threshold rule.

A thresholded type classifier needs an additional linear bias or a different statistic, ideally one that detects a nonzero-frequency spectral peak rather than just long correlation.

## 16. Dependency Graph Repair

The logic order should be:

1. `L_embed` ensures `Dg` has full rank locally.
2. Then `P_perp` and `G^{-1}` are well defined.
3. Then geometry and normal-coordinate losses make sense.
4. Then separation and curvature control the validity of the reduced normal dynamics.
5. Then Koopman constraints become semantically meaningful.
6. Then semigroup / global-consistency penalties are added.

So `L_embed` is a prerequisite for `L_geo`, not the other way around.

## 17. Literature Fit

- Sauer-style reconstruction supports the claim that shared low-dimensional dynamics can be recovered from multivariate observations, but it does not by itself prove slow/fast semantic disentanglement.
- Gilpin-style generative nonlinear-dynamics work supports latent-space attractor reconstruction and rollout learning, but it is a representation argument, not a proof that the learned coordinates are Koopman eigenfunctions or invariant fiber coordinates.
- mHC-style manifold / parallel-flow mixers are useful as optimization and routing priors, but they do not replace explicit invariance, transverse contraction, or closure diagnostics.

## Bottom Line

The revised draft becomes mathematically defensible after the following edits:

1. Separate attractor-restricted Koopman theory from basin/neighborhood Koopman theory.
2. Strengthen the spectral theorems with smooth-linearization hypotheses.
3. Replace the implicit `qdot` equation by the solved `(I + C)^{-1}` form.
4. Downgrade `A(q)` from "the transverse spectrum" to "the projected normal block."
5. Mark `rho` as an engineering gate, not a theorem.
6. Fix the sign in the exponential `q` integrator.
7. Replace the type-loss proposition.
8. Replace the curvature loss by a second-fundamental-form penalty.
9. Use symmetric-part contraction rates in the separation theorem.

Without these fixes, the draft is still better than the earlier version, but it overclaims rigor at several key steps.
