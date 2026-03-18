# Closure-First Koopman Semantic Refinement Note

Date: 2026-03-19

Status: extension note for the March 2026 closure-first theory drafts. This note refines the semantic contract of the model family in `closure_first_info_koopman_full_model_20260318.md` without replacing its construction order. When wording differs, the March 18 canonical note still controls the base model family, while this note states the stronger semantic version that should guide later chart-local `q / h / m` upgrades.

## 1. Purpose

The canonical note already fixed one major problem:

- `q / h / m` are not primitive encoder coordinates;
- they emerge only after closure-first encoding, charting, flag decomposition, and promotion.

The next problem is more subtle:

- "tangent / normal" is a geometric split;
- "persistent / fast" is a temporal or spectral split;
- "memory" is a closure-history split.

These three splits are related, but they are not identical.
So the semantic refinement in this note is:

1. keep the construction order of the canonical model;
2. separate geometric closure coordinates from true fast-fiber coordinates;
3. make `q`, `h`, and `m` pass different role tests instead of one shared heuristic.

## 2. Main Correction: `h` Should Not Mean Two Different Things

In the March 18 canonical model, the promoted hidden block `h` has the mixed meaning

- normal or closure correction in the decoder;
- contractive hidden state in the dynamics;
- possibly memory-driven correction through terms such as `D_k(q) m_t` and `b_k(q)`.

This is mathematically useful, but semantically overloaded.
It mixes two different objects:

1. a general chart-local closure coordinate;
2. a recentered fast transverse deviation around a tracked section.

Those should not be named by the same symbol.

This note therefore proposes a two-level hidden semantics:

- `u_t^{(k)}`: promoted closure-normal block before recentering;
- `\eta_t^{(k)}`: recentered fast transverse deviation after section removal.

Only `\eta` is allowed to claim approximate fast-fiber semantics.
`u` is the weaker and safer object.

## 3. Refined Construction Order

The canonical order remains

```text
observation window
-> closure-first state s_t
-> chart-local summaries
-> chart-local flag blocks
-> promotion into persistent / closure / memory roles
-> geometry and dynamics
```

The semantic refinement inserts one extra layer after promotion:

```text
promoted closure block u
-> tracked section h_*(q)
-> recentered deviation eta = u - h_*(q)
```

So the stronger reading is

```text
s_t
-> chart-local blocks
-> promoted q / u / m
-> tracked section and recentering
-> q / eta / m
```

This preserves the canonical construction while avoiding premature "fiber" claims.

## 4. Geometry Layer with Tracked Section

Keep the chart-local base manifold

```math
g_k : \mathbb R^{d_q^{(k)}} \to \mathbb R^{d_x},
\qquad
T_k(q) = \partial_q g_k(q),
```

and the tangent-removed normal basis

```math
N_k(q)
=
B_k(q)
-
T_k(q)
\bigl(T_k(q)^\top T_k(q) + \varepsilon I\bigr)^{-1}
T_k(q)^\top B_k(q).
```

The canonical geometric hidden coordinate is still the least-squares normal coefficient.
But instead of interpreting that coefficient directly as a fast variable, introduce a chart-local tracked section

```math
h_{*,k}(q) \in \mathbb R^{d_u^{(k)}}.
```

Then write

```math
u_t^{(k)} = h_{*,k}(q_t^{(k)}) + \eta_t^{(k)}.
```

The decoder should be read in section-centered form:

```math
\Sigma_k(q) = g_k(q) + N_k(q) h_{*,k}(q),
```

```math
\hat x_t^{(k)} = \Sigma_k(q_t^{(k)}) + N_k(q_t^{(k)}) \eta_t^{(k)}.
```

This keeps the observable geometry attached to a tracked section, while `\eta` measures off-section normal deviation.

Important consequence:

- `q` explains persistent tangent or section-following variation;
- `\eta` explains off-section transverse deviation;
- `m` still does not enter the current-time decoder directly.

## 5. Section Invariance as the Missing Middle Object

The canonical note already distinguishes hard guarantees from empirical targets.
This refinement adds one missing intermediate object:
an approximately invariant tracked section.

For the pre-recentered hidden update

```math
u_{t+1}^{(k)}
=
\Psi_k^\perp(q_t^{(k)}) u_t^{(k)}
+ b_k(q_t^{(k)})
+ D_k(q_t^{(k)}) m_t^{(k)}
+ \varepsilon_{u,k},
```

define the section defect

```math
\delta_{*,k}(q_t)
=
h_{*,k}(\phi_k(q_t))
-
\Psi_k^\perp(q_t) h_{*,k}(q_t)
-
b_k(q_t).
```

If `\delta_{*,k}` is small, the tracked section follows the hidden drift induced by `q`.
After recentering, one obtains

```math
\eta_{t+1}^{(k)}
=
\Psi_k^\perp(q_t^{(k)}) \eta_t^{(k)}
+ \widetilde D_k(q_t^{(k)}) m_t^{(k)}
+ \widetilde \varepsilon_{\eta,k},
```

where all residual section mismatch is absorbed into the recentered residual terms.

This yields a strict semantic rule:

- if the section defect is small and memory leakage into `\eta` is also small, then `\eta` may be interpreted as an approximate fast fiber coordinate;
- otherwise the model should only claim a closure coordinate `u`, not a true fast fiber.

## 6. Refined Local Dynamics

After recentering, the preferred chart-local model is

```math
q_{t+1}^{(k)}
=
\phi_k(q_t^{(k)})
+ B_k(q_t^{(k)}) \eta_t^{(k)}
+ C_k(q_t^{(k)}) m_t^{(k)}
+ \varepsilon_{q,k},
```

```math
\eta_{t+1}^{(k)}
=
\Psi_k^\perp(q_t^{(k)}) \eta_t^{(k)}
+ \widetilde D_k(q_t^{(k)}) m_t^{(k)}
+ \varepsilon_{\eta,k},
```

```math
m_{t+1}^{(k)}
=
A_k^m(q_t^{(k)}) m_t^{(k)}
+ \beta_k(q_t^{(k)}) \psi_k(q_t^{(k)}, \eta_t^{(k)})
+ \varepsilon_{m,k}.
```

Two semantic regimes are then possible.

### 6.1 Conservative closure regime

Allow `\widetilde D_k(q)` to be nonzero.
Then `\eta` is still a valid recentered closure coordinate, but not yet a clean fiber coordinate.
This is the safe default for partially observed or strongly nonnormal systems.

### 6.2 Strong fiber regime

Require both

```math
\delta_{*,k}(q) \approx 0,
\qquad
\widetilde D_k(q) \approx 0.
```

Then `\eta = 0` becomes an approximately invariant chart-local section and `\eta` deserves the name fast transverse or fiber deviation.

This is the regime in which "normal" and "fast" are locally aligned.

## 7. `q / h / m` Should Be Decided by Three Axes, Not One

The canonical note already defines information-based promotion scores.
That is necessary, but not sufficient.

A block should be judged on three distinct axes:

1. geometric axis:
   - tangent affinity,
   - normal affinity,
   - section-following vs off-section behavior;
2. temporal or spectral axis:
   - long-horizon persistence,
   - phase-like neutrality,
   - strong contraction;
3. closure-history axis:
   - instantaneous correction value,
   - delayed predictive value,
   - residual memory value after conditioning on persistent and closure blocks.

So the refined semantics are:

- promote to `q` when a block is persistent, chart-tangent or section-following, and spectrally slow or phase-like;
- promote to `u` when a block is geometrically normal or closure-like and mainly improves short-horizon closure;
- refine `u` into `\eta` only after section recentering and invariance checks;
- promote to `m` when delayed predictive utility remains after conditioning on `q` and `u` or `\eta`.

This is the key conceptual correction:
do not let information utility alone determine geometry semantics.

## 8. Role Tests and Evidence Objects

To support the refined semantics, the theory should introduce explicit evidence objects.

### 8.1 Persistent-state evidence

For `q`, the relevant evidence is:

- long-horizon predictive gain;
- chart-local tangent or section alignment;
- phase-harmonic or slow-mode compatibility;
- low reliance on fast corrective couplings.

### 8.2 Closure-vs-fiber evidence

For the hidden normal block, the relevant evidence is:

- section defect:

```math
I_{\mathrm{sec}} = \mathbb E \|\delta_{*,k}(q_t)\|^2;
```

- recentered invariance defect:

```math
I_{\eta} = \mathbb E \|\eta_{t+1}(q_t, 0, 0)\|^2;
```

- memory leakage into recentered normal dynamics:

```math
J_{m \to \eta}
=
\mathbb E
\left\|
\frac{\partial \eta_{t+1}}{\partial m_t}
\right\|_F^2;
```

- transverse contraction certificate for fixed `q`:

```math
\|\Psi_k^\perp(q)\|_2 \le \bar\rho_h < 1
```

or its continuous-time symmetric-part analogue.

If contraction is strong but section and leakage defects are not small, call the state contractive closure, not fiber.

### 8.3 Memory evidence

For `m`, the relevant evidence is:

- delayed gain after conditioning on `q` and `u` or `\eta`;
- no direct current-time decoder shortcut;
- stable memory poles or contraction of `A_k^m(q)`;
- persistence across short coarse-graining intervals when the kernel time constants justify it.

This yields a sharper role split:

- `q`: persistent geometry-bearing state;
- `\eta`: recentered fast normal deviation;
- `m`: retained memory lift.

## 9. Chart Semantics Should Be Typed

The canonical note distinguishes equilibrium, limit-cycle, and more general charts.
The refinement here is to make that distinction part of the semantic contract.

Each chart should carry a semantic type

```math
\tau_k \in \{\mathrm{eq}, \mathrm{lc}, \mathrm{gen}\}.
```

### 9.1 Equilibrium charts

For `\tau_k = eq`:

- `q` contains persistent real modes only;
- there is no phase block;
- "slow" means weak decay, not oscillation.

### 9.2 Limit-cycle charts

For `\tau_k = lc`:

- `q` must contain an explicit `2D` phase block or equivalent rotational coordinate;
- nearby amplitude or auxiliary persistent directions may be appended as slow real blocks;
- `\eta` describes transverse Floquet-like contraction only after recentering.

This avoids the common error of forcing phase into a nearly neutral real scalar channel.

### 9.3 General local charts

For `\tau_k = gen`:

- `q` is only a chart-local persistent summary;
- no global slow manifold or pure-point Koopman claim should be made;
- the safe targets are short-horizon prediction, local linearization consistency, local contraction when present, and invariant-statistics approximation.

This chart typing prevents the theory from making one global spectral promise for every regime.

## 10. Spectral Semantics Should Be Split by Role

The refined spectral rule is:

- Koopman or Floquet regularity belongs to `q`;
- transverse contraction belongs to `\eta`;
- memory-pole stability belongs to `m`.

These are different spectral objects.
They should not be trained or interpreted as if they were interchangeable.

So the semantic reading is:

- `q`: near-neutral phase, slow decay, or persistent local modes;
- `\eta`: fast contractive transverse modes;
- `m`: stable memory poles approximating an effective kernel.

This is the temporal counterpart of the geometry split.

## 11. RG and Coarse-Graining Semantics

The canonical family already includes RG and coarse-graining ideas.
The semantic refinement should state explicitly how the three roles behave under scale change.

Under coarse-graining:

- `\eta` should decay or be eliminated first;
- `q` should be the primary retained state;
- `m` may survive only when the retained memory poles are not yet negligible at the coarser scale.

So the correct coarse-graining intuition is not

```text
all latent channels are rescaled equally
```

but

```text
q is retained,
eta is rapidly damped,
m is retained only if kernel time scales justify it.
```

This is the RG form of the same semantic split.

## 12. Guarantee Ladder

The canonical note already has a guarantee hierarchy.
This refinement sharpens it into four levels.

### 12.1 Construction-level guarantees

These may hold by parameterization:

- chart weights lie on a simplex;
- block structure respects chart-local ordered increments;
- `m` does not enter the current-time decoder directly;
- `\eta` and `m` can be parameterized to be contractive for fixed `q`;
- limit-cycle charts can contain explicit rotation blocks.

### 12.2 Local geometric guarantees under assumptions

These require explicit local hypotheses:

- tangent/normal coordinates are valid in a tubular neighborhood;
- section-centered coordinates are valid only where the tracked section exists smoothly;
- exact normal coordinates require full codimension;
- local Koopman or Floquet semantics require the relevant smooth-linearization assumptions.

### 12.3 Semantic certification targets

These are stronger than construction but weaker than theorem-level recovery:

- `q` passes persistent plus tangent or section alignment tests;
- `u` passes closure-normal tests;
- `\eta` passes recentered invariance and low memory leakage tests;
- `m` passes delayed-gain and memory-pole tests.

### 12.4 Empirical-only claims

These should remain non-claims:

- exact recovery of the true physical slow variables;
- exact recovery of the true chart partition;
- exact recovery of the true Mori-Zwanzig truncation;
- exact equality between predictive-loss proxies and conditional mutual information;
- exact identification of fast fiber coordinates from observations alone in the generic nonlinear case.

## 13. Canonical Summary Statement for the Stronger Semantic Version

The refined theory can therefore be read as follows.

1. Start from a closure-first latent state `s_t`.
2. Build chart-local summaries, chart-local flag blocks, and promoted persistent / closure / memory roles.
3. Call the promoted hidden geometric block `u`, not yet fast fiber.
4. Introduce a tracked section `h_*(q)` and recenter `u = h_*(q) + \eta`.
5. Interpret only the recentered `\eta` as a candidate fast transverse coordinate.
6. Keep Koopman or Floquet semantics on `q`, contraction semantics on `\eta`, and kernel-lift semantics on `m`.
7. Treat alignment between tangent/normal and persistent/fast as a local certified compatibility condition, not as an initialization assumption.

In one sentence:

```text
the model should first discover closure structure,
then promote q / u / m,
then recenter u into eta,
and only after that claim any local alignment between persistent-tangent, fast-normal, and retained memory roles.
```

## 14. Practical Consequence for Later Implementation

If later repository work wants the stronger semantics, the upgrade order should be:

1. preserve the canonical closure-first `s -> chart -> flag -> promotion` stack;
2. add tracked-section diagnostics before claiming fiber semantics;
3. separate `u` and `\eta` in summaries and losses;
4. allow chart typing to decide whether `q` should use equilibrium, limit-cycle, or generic local spectral structure;
5. apply RG, Koopman, and memory constraints to their own roles rather than to one shared hidden block.

This is the minimal theory change that keeps the closure-first architecture while removing the remaining semantic ambiguity around `q / h / m`.
