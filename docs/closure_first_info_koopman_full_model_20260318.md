# Closure-First Canonical Theory Model

Date: 2026-03-18

Status: canonical theory note for the March 2026 closure-first drafts. When this file conflicts with earlier blueprint or landing-path notes, this file wins on mathematical wording. It is intentionally theory-only and does not try to describe the current repository implementation. The technology stack and dependency order are recorded separately in `docs/closure_first_technical_outline_20260318.md`, and the literature-backed re-audit is recorded in `docs/closure_first_literature_audit_20260318.md`.

## 1. Purpose

This note gives one mathematically careful statement of the intended model family.
Its job is to keep three things separate:

1. objects that are well defined by construction;
2. objects that are valid only under explicit local assumptions;
3. objects that are only loss-driven or empirical targets.

The main correction relative to earlier drafts is:

- `q / h / m` are not primitive coordinates at initialization;
- they are roles that emerge from a closure-first state after charting, flag decomposition, block decomposition, and promotion;
- but once those roles are defined, their dimensions are still part of the mathematical model.

## 2. Scope and Non-Claims

This model family is meant to cover, in one formalism:

- stable equilibria;
- stable limit cycles;
- chart-local modeling of more complicated or chaotic regimes;
- local tangent/normal geometry;
- finite-dimensional memory closure.

It does not claim, without extra assumptions:

- exact recovery of the physically true slow variables;
- exact recovery of a globally correct basin partition;
- exact recovery of the true Mori-Zwanzig kernel;
- exact global low-dimensional coordinates for chaotic dynamics;
- unique or identifiable `q/h/m` semantics from observations alone.

## 3. Core Objects and Order of Construction

Let the observations be

```math
x_t \in \mathbb R^{d_x},
\qquad
W_t = (x_{t-L+1}, \dots, x_t).
```

The primary latent object is a closure-first state

```math
s_t = E_\theta(W_t) \in \mathbb R^{d_s}.
```

This is the only primitive latent state.
At this stage no coordinate is yet declared to be slow, transverse, or memory-like.

For each chart `k \in \{1,\dots,K\}`, choose either a fixed chart frame or a frame parameterized by a provisional chart-local coordinate

```math
\bar q_t^{(k)} = R_k s_t,
```

where `R_k` is any admissible chart-local summary map.
This provisional coordinate is used only to avoid circularity in the chart frame.
The final promoted `q_t` is defined later.

Now introduce a chart-local partial flag with fixed signature

```math
\mathcal F_k(\bar q_t^{(k)})
:
V_{k,1} \subset V_{k,2} \subset \cdots \subset V_{k,M_k} \subset \mathbb R^{d_s},
```

with block increments

```math
\dim(V_{k,j} / V_{k,j-1}) \in \{1,2\}.
```

For fixed increment dimensions, these objects lie on a partial flag manifold; when they vary with the chart coordinate, the mathematically precise object is a smooth section of the corresponding partial-flag bundle over the chart domain.

An adapted orthonormal frame `U_k(\bar q_t^{(k)})` is then any frame whose first coordinates follow this ordered nested decomposition.
This is the precise place where the flag-manifold structure enters:

- the chart chooses a local state region;
- the flag chooses an ordered latent decomposition inside that chart;
- only afterwards do we decide which blocks become `q`, `h`, or `m`.

Then define chart-local block coordinates

```math
y_t^{(k)}
=
U_k(\bar q_t^{(k)})^\top s_t
=
\bigl(
b_t^{(k,1)}, \dots, b_t^{(k,M_k)}, r_t^{(k)}
\bigr),
```

with block dimensions constrained to `1` or `2`:

- `1D` blocks for real decay or recovery directions;
- `2D` blocks for oscillatory or phase-like planes;
- `r_t^{(k)}` as the residual closure block.

The modeling order is therefore

```text
closure state s
-> chart-local summaries
-> chart-local flag
-> chart-local block coordinates
-> block scoring and promotion
-> emergent q / h / m
```

and not

```text
raw observations
-> primitive q / h / m
```

## 4. Why `q`, `h`, and `m` Still Need Dimensions

The correct criticism is not that the theory should remove `q/h/m` dimensions.
The correct criticism is that the theory should stop treating those dimensions as primitive encoder outputs.

Once the model introduces

- a persistent coordinate `q`,
- a transverse or closure coordinate `h`,
- and a memory coordinate `m`,

their dimensions are mathematically unavoidable because the following maps must be typed:

```math
g_k : \mathbb R^{d_q^{(k)}} \to \mathbb R^{d_x},
\qquad
N_k(q) \in \mathbb R^{d_x \times d_h^{(k)}},
\qquad
m_t^{(k)} \in \mathbb R^{d_m^{(k)}}.
```

So there are two distinct notions of dimension:

1. closure capacity:

```math
d_s = \dim s_t;
```

2. realized role dimensions after promotion:

```math
d_q^{(k)} = \dim q_t^{(k)},
\qquad
d_h^{(k)} = \dim h_t^{(k)},
\qquad
d_m^{(k)} = \dim m_t^{(k)}.
```

If one wants global capacity budgets, write instead

```math
d_q^{\max}, \qquad d_h^{\max}, \qquad d_m^{\max},
```

and let promotion decide the active dimensions inside those budgets.

So the mathematically correct statement is:

- `q/h/m` should not be primitive at initialization;
- `q/h/m` do still have dimensions after they are defined as promoted subspaces or local coordinates.

## 5. Geometry Layer

For chart `k`, let

```math
g_k : \mathbb R^{d_q^{(k)}} \to \mathbb R^{d_x}
```

be the chart-local base manifold and let

```math
T_k(q) = \partial_q g_k(q) \in \mathbb R^{d_x \times d_q^{(k)}}
```

be its tangent Jacobian.

Given a raw local basis `B_k(q)`, define its normal component by tangent removal:

```math
N_k(q)
=
B_k(q)
-
T_k(q)
\bigl(T_k(q)^\top T_k(q) + \varepsilon I\bigr)^{-1}
T_k(q)^\top B_k(q).
```

Then

```math
P_{T,k}(q)
=
T_k(q)
\bigl(T_k(q)^\top T_k(q) + \varepsilon I\bigr)^{-1}
T_k(q)^\top,
\qquad
P_{N,k}(q) = I - P_{T,k}(q).
```

For residual

```math
r_t^{(k)} = x_t - g_k(q_t^{(k)}),
```

define the chart-local geometric hidden coordinate by least squares:

```math
h_{t,k}^{\mathrm{geo}}
=
\arg\min_h
\|
P_{N,k}(q_t^{(k)}) r_t^{(k)}
-
N_k(q_t^{(k)}) h
\|^2,
```

with closed form

```math
h_{t,k}^{\mathrm{geo}}
=
\bigl(N_k^\top N_k + \varepsilon I\bigr)^{-1}
N_k^\top P_{N,k} r_t^{(k)}.
```

This gives the intended semantics:

- `q` explains tangent or persistent variation;
- `h` explains normal or closure correction.

Important limitation:

- this is an exact local coordinate description only when the chart is restricted to a tubular neighborhood and the hidden normal span has full codimension;
- if `d_h^{(k)} < d_x - d_q^{(k)}`, then `h` is only a reduced normal approximation, not an exact normal coordinate system.

## 6. Information-Theoretic Promotion

For block `b_j`, define long-horizon incremental predictive information

```math
U_j^L
=
I(b_j; F_t^L \mid b_{<j}, c_t),
```

short-horizon closure value

```math
U_j^S
=
I(b_j; F_t^S \mid q_t, c_t),
```

and conditional redundancy

```math
R_j
=
I(b_j; b_{<j} \mid F_t^L, c_t).
```

Here `q_t` means the persistent state formed from blocks that have already been promoted before the current residual-role decision is made.
This prevents a circular definition between promotion scores and the promoted state itself.

Then a mathematically honest promotion logic is:

```math
S_j^{(q)}
=
U_j^L
- \lambda_R R_j
- \lambda_C \kappa_j
+ \lambda_P \mathrm{persist}_j
- \lambda_X \mathrm{cross}_j,
```

```math
S_j^{(h)}
=
U_j^S
- \lambda_{Lh} I(b_j; F_t^L \mid q_t, c_t),
```

```math
S_j^{(m)}
=
I(b_j; F_t^{S,+} \mid q_t, h_t, c_t)
- \lambda_{Lm} I(b_j; F_t^L \mid q_t, c_t)
- \lambda_{Cm} \kappa_j.
```

Decision semantics:

- promote to `q` when long-horizon incremental value is high;
- keep in `h` when short-horizon closure value is high but long-horizon value is low;
- keep in `m` when memory-sensitive value remains after conditioning on `q` and `h`;
- prune otherwise.

With chart weights `c_{t,k}`, the promoted subspaces are

```math
q_t = \sum_k c_{t,k} U_k \Pi_k^{(q)} U_k^\top s_t,
```

```math
h_t = \sum_k c_{t,k} U_k \Pi_k^{(h)} U_k^\top s_t,
```

```math
m_t = \sum_k c_{t,k} U_k \Pi_k^{(m)} U_k^\top s_t,
```

with

```math
\Pi_k^{(q)} + \Pi_k^{(h)} + \Pi_k^{(m)} + \Pi_k^{(\mathrm{prune})} = I
```

only when the chart-local promoted subspaces are actually constrained to be complementary.

So the partition identity is exact linear algebra only under enforced orthogonality or complementary-subspace constraints.
Equivalently: the promotion masks must respect the chart-local flag blocks rather than arbitrary coordinates.

## 7. Merge-vs-Coexist Diagnostic: What `\Delta_{ij}` Really Measures

Earlier drafts called

```math
\Delta_{ij}
=
I((b_i,b_j); F_t^L \mid q_t, c_t)
- I(b_i; F_t^L \mid q_t, c_t)
- I(b_j; F_t^L \mid q_t, c_t)
```

a synergy measure.
That wording is too strong.

By the chain rule,

```math
\Delta_{ij}
=
I(b_j; F_t^L \mid b_i, q_t, c_t)
- I(b_j; F_t^L \mid q_t, c_t).
```

So `\Delta_{ij}` is an interaction-information-style proxy:

- positive when `b_j` becomes more useful after conditioning on `b_i`;
- negative when conditioning on `b_i` removes part of `b_j`'s utility;
- zero when the incremental future utility of `b_j` is unchanged by `b_i`.

It is not a full PID synergy decomposition.
Therefore the correct wording is:

- use `\Delta_{ij}` as a merge-vs-coexist heuristic;
- do not present it as an exact PID synergy theorem.

## 8. Koopman Spectral Separation

Two function-space settings must be kept separate.

### 8.1 Attractor-restricted spectrum

For an attractor `A` with invariant measure `\mu_A`,

```math
K_\tau^A : L^2(A,\mu_A) \to L^2(A,\mu_A),
\qquad
K_\tau^A \varphi = \varphi \circ F_\tau.
```

Then:

- for a stable equilibrium, the attractor-restricted point spectrum is trivial;
- for a stable limit cycle, the attractor-restricted spectrum contains phase harmonics

```math
\varphi_k(\theta) = e^{ik\theta},
\qquad
K_\tau^A \varphi_k = e^{ik\omega\tau}\varphi_k.
```

### 8.2 Neighborhood spectrum

To see stable off-attractor modes, move to a forward-invariant neighborhood `U`:

```math
K_\tau^U \varphi = \varphi \circ F_\tau,
\qquad
\varphi \in C^1(U)
```

or another smooth observable space.

Under smooth-linearization hypotheses:

- near a hyperbolic equilibrium, the local point spectrum contains combinations of stable linearization eigenvalues;
- near a hyperbolic limit cycle, the local neighborhood spectrum contains

```math
ik\omega + \sum_j m_j \nu_j,
```

where `ik\omega` are phase harmonics and `\nu_j` are stable transverse Floquet exponents.

So the mathematically correct spectral split is:

- phase harmonics on the attractor;
- stable transverse modes in a neighborhood.

This is why limit-cycle charts need an explicit phase or rotation block rather than only nearly neutral real-decay channels.

### 8.3 Chaotic regimes

For chaotic dynamics one should not promise a single global low-dimensional slow chart with pure point Koopman spectrum.
The safe target is instead:

- chart-local short-horizon prediction;
- local linearization consistency;
- local stable-fiber or transverse contraction when present;
- invariant-statistics or occupancy approximation, not exact global reduction.

## 9. Unified Chart-Local Dynamics

The clean local discrete-time model is

```math
q_{t+1}^{(k)}
=
\phi_k(q_t^{(k)})
+ B_k(q_t^{(k)}) h_t^{(k)}
+ C_k(q_t^{(k)}) m_t^{(k)}
+ \varepsilon_{q,k},
```

```math
h_{t+1}^{(k)}
=
\Psi_k^\perp(q_t^{(k)}) h_t^{(k)}
+ D_k(q_t^{(k)}) m_t^{(k)}
+ b_k(q_t^{(k)})
+ \varepsilon_{h,k},
```

```math
m_{t+1}^{(k)}
=
A_k^m(q_t^{(k)}) m_t^{(k)}
+ \beta_k(q_t^{(k)}) \psi_k(q_t^{(k)}, h_t^{(k)})
+ \varepsilon_{m,k}.
```

The decoder is

```math
\hat x_t^{(k)} = g_k(q_t^{(k)}) + N_k(q_t^{(k)}) h_t^{(k)},
\qquad
\hat x_t = \sum_k c_{t,k} \hat x_t^{(k)}.
```

The role of `m` is dynamic only:
it should influence future evolution, not become a shortcut in the current-time decoder.

### 9.1 Equilibrium charts

For equilibrium-dominant charts,

```math
\phi_k(q) = A_k^{\mathrm{eq}} q + \delta_k^{(q)}(q),
```

with stable real blocks.
Here "persistent" means slow decay, not phase.

### 9.2 Limit-cycle charts

For stable limit-cycle charts, `q` must contain a phase block:

```math
\phi_k(q)
=
\operatorname{blkdiag}\!\bigl(
R(\omega_k(q)),
A_k^{\mathrm{sl}}(q)
\bigr) q
+ \Gamma_k(q)
+ \delta_k^{(q)}(q).
```

This is the correct local expression of phase plus slower amplitude or auxiliary directions.

### 9.3 Memory meaning

The role of `m` is justified by a finite-dimensional approximation to a non-Markov memory term.
Starting from the formal projected equation

```math
q_{t+1}
=
\bar\phi(q_t)
+ \sum_{\ell \ge 0} K_\ell(q_t)\psi(q_{t-\ell})
+ \xi_t,
```

approximate the kernel by exponentials:

```math
K_\ell(q_t)\psi(q_{t-\ell})
\approx
\sum_{j=1}^{d_m}
C_j(q_t)\alpha_j^\ell \beta_j(q_{t-\ell})\psi(q_{t-\ell}),
\qquad
|\alpha_j| < 1,
```

which yields the memory lift

```math
m_{j,t+1}
=
\alpha_j(q_t) m_{j,t}
+ \beta_j(q_t)\psi(q_t,h_t).
```

This gives `m` a precise semantics:

- `h` is instantaneous transverse or closure correction;
- `m` is finite-dimensional retained memory.

### 9.4 Energy-based chart routing

Chart routing is part of the model, not an external classifier.
Each chart proposes a local explanation and receives an energy

```math
E_{t,k}
=
\|x_t - \hat x_t^{(k)}\|^2
+ \lambda_{\mathrm{dyn}} \|x_{t+1} - \hat x_{t+1}^{(k)}\|^2
+ \lambda_{\mathrm{ctr}} [\sigma_{\max}(\Psi_k^\perp(q_t^{(k)})) - \bar\rho_h]_+^2
+ \lambda_{\mathrm{mz}} [\sigma_{\max}(A_k^m(q_t^{(k)})) - \bar\rho_m]_+^2
+ \lambda_{\mathrm{sw}} S(k,c_{t-1}),
```

where a convenient soft switching penalty is

```math
S(k,c_{t-1}) = 1 - c_{t-1,k}.
```

If one uses hard chart assignments instead of soft routing, this reduces to the discrete penalty `\mathbf 1[k \ne k_{t-1}]`.

Then the routing weights are

```math
c_{t,k} = \operatorname{softmax}(-E_{t,k}/\tau_t).
```

So routing depends on:

- chart-local geometry;
- chart-local dynamics;
- contraction legality;
- memory legality;
- switching stickiness.

This keeps routing downstream of the chart-local model rather than making routing a semantically unconstrained classifier.

## 10. Contraction and Spectral-Gap Claims

The strongest hard guarantee in this family is transverse or memory contraction.

In discrete time, if the structured core parameterization enforces

```math
\|\Psi_k^\perp(q)\|_2 \le \bar\rho_h < 1,
\qquad
\|A_k^m(q)\|_2 \le \bar\rho_m < 1,
```

then the structured core updates for `h` and `m` are contractive by construction for fixed `q`.

If residual terms such as `\varepsilon_{h,k}` or `\varepsilon_{m,k}` are retained, then contraction of the full update requires additional small-gain or Lipschitz assumptions on those residual terms.

In continuous time, a useful contraction certificate is the symmetric-part rate

```math
\alpha_h(q)
=
-\lambda_{\max}\!\left(
\frac{G_h(q) + G_h(q)^\top}{2}
\right).
```

This quantity correctly controls instantaneous energy decay.
But for nonnormal operators it is not identical to the full asymptotic spectral description.
So the correct wording is:

- symmetric-part bounds certify contraction or dissipation;
- they do not by themselves fully characterize every transient effect of a nonnormal operator.

## 11. Proxies, Losses, and What They Mean

The information quantities above are theoretical objects, not directly observed statistics.
So one needs proxies.

### 11.1 Exact Gaussian benchmark

For linear-Gaussian smoke systems,

```math
\widehat I(X;Y \mid Z)
=
\frac{1}{2}
\log
\frac{\det \Sigma_{X \mid Z}}
{\det \Sigma_{X \mid Y,Z}}
```

is an exact conditional mutual information formula.

### 11.2 Predictive-loss proxies

Loss differences such as

```math
\widehat U_j^L
=
\mathcal L_{\mathrm{pred}}(b_{<j}, c_t)
- \mathcal L_{\mathrm{pred}}(b_{\le j}, c_t)
```

or

```math
\widehat U_j^S
=
\mathcal L_{\mathrm{short}}(q_t, c_t)
- \mathcal L_{\mathrm{short}}(q_t, b_j, c_t)
```

should be described as predictive-utility proxies.

They match conditional mutual information only under additional conditions such as:

- the loss is a calibrated negative log-likelihood;
- the distribution family is correctly specified;
- the estimator error is controlled well enough.

So the safe wording is:

- mutual-information definitions are theory objects;
- predictive-loss differences are training surrogates, not exact equalities in general.

### 11.3 Objective library

The safe global objective form is

```math
L = \sum_\bullet \lambda_\bullet L_\bullet,
```

where the loss library may include reconstruction, prediction, geometry, contraction, spectral-gap, routing, promotion, redundancy, semigroup, and memory terms.

But this is only an objective library.
It is not a theorem that one parameter setting simultaneously optimizes all desiderata.

So the correct statement is:

- the losses encode competing structural preferences;
- joint success is an empirical multi-objective optimization question, not a proved fact.

## 12. Guarantee Hierarchy

### 12.1 Hard guarantees by construction

These can be true purely by parameterization:

- the structured core of `h` and `m` can be made contractive for fixed `q`;
- the persistent core can contain genuine `2D` rotation blocks;
- chart weights lie on the probability simplex;
- `m` does not enter the current-time decoder directly;
- block structure can be constrained to `1D/2D` increments plus residual closure blocks.

### 12.2 Local theorem-level statements under assumptions

These require explicit assumptions:

- stable equilibrium charts can be locally modeled by stable real modes;
- stable limit cycles admit phase harmonics on the attractor and transverse stable modes in a neighborhood;
- tangent/normal coordinates are locally valid in a tubular neighborhood;
- exact normal coordinates require full codimension;
- neighborhood Koopman statements require smooth-linearization hypotheses, not Hartman-Grobman alone.

### 12.3 Loss-driven or empirical targets only

These are not theorem-level guarantees:

- exact recovery of the true physical slow variables;
- exact recovery of the correct chart count or basin structure;
- exact PID synergy from `\Delta_{ij}`;
- exact equality between loss-difference proxies and mutual information in the generic nonlinear case;
- exact recovery of the true finite-dimensional MZ truncation;
- a globally correct low-dimensional representation for chaotic dynamics.

For memory in particular, the literature supports two weaker claims:

- Mori-Zwanzig yields an exact non-Markov equation with memory kernel and noise after projection;
- finite-dimensional auxiliary-state or rational-kernel embeddings are useful approximations, not exact consequences of the formalism in general.

## 13. Canonical Summary Statement

The mathematically careful summary of the model family is:

1. start from a closure-first latent state `s_t`;
2. build chart-local summaries and a chart-local flag decomposition;
3. turn the flag blocks into persistent `q`, transverse `h`, and memory `m` using information and geometry criteria;
4. evolve `q`, `h`, and `m` with chart-local structured dynamics;
5. route between charts by energy competition;
6. claim only those guarantees that are justified by parameterization or by explicit local assumptions.

In equations, the canonical model is

```math
s_t = E_\theta(W_t),
```

```math
\bar q_t^{(k)} = R_k s_t,
\qquad
\mathcal F_k(\bar q_t^{(k)})
:
V_{k,1} \subset \cdots \subset V_{k,M_k} \subset \mathbb R^{d_s},
```

```math
y_t^{(k)}
=
U_k(\bar q_t^{(k)})^\top s_t
=
\bigl(
b_t^{(k,1)}, \dots, b_t^{(k,M_k)}, r_t^{(k)}
\bigr),
```

```math
q_t = \sum_k c_{t,k} U_k \Pi_k^{(q)} U_k^\top s_t,
\qquad
h_t = \sum_k c_{t,k} U_k \Pi_k^{(h)} U_k^\top s_t,
\qquad
m_t = \sum_k c_{t,k} U_k \Pi_k^{(m)} U_k^\top s_t,
```

```math
\hat x_t^{(k)} = g_k(q_t^{(k)}) + N_k(q_t^{(k)}) h_t^{(k)},
\qquad
\hat x_t = \sum_k c_{t,k} \hat x_t^{(k)},
```

```math
E_{t,k}
=
\|x_t - \hat x_t^{(k)}\|^2
+ \lambda_{\mathrm{dyn}} \|x_{t+1} - \hat x_{t+1}^{(k)}\|^2
+ \lambda_{\mathrm{ctr}} [\sigma_{\max}(\Psi_k^\perp(q_t^{(k)})) - \bar\rho_h]_+^2
+ \lambda_{\mathrm{mz}} [\sigma_{\max}(A_k^m(q_t^{(k)})) - \bar\rho_m]_+^2
+ \lambda_{\mathrm{sw}} S(k,c_{t-1}),
```

```math
c_{t,k} = \operatorname{softmax}(-E_{t,k}/\tau_t),
```

```math
q_{t+1}^{(k)}
=
\phi_k(q_t^{(k)})
+ B_k(q_t^{(k)}) h_t^{(k)}
+ C_k(q_t^{(k)}) m_t^{(k)}
+ \varepsilon_{q,k},
```

```math
h_{t+1}^{(k)}
=
\Psi_k^\perp(q_t^{(k)}) h_t^{(k)}
+ D_k(q_t^{(k)}) m_t^{(k)}
+ b_k(q_t^{(k)})
+ \varepsilon_{h,k},
```

```math
m_{t+1}^{(k)}
=
A_k^m(q_t^{(k)}) m_t^{(k)}
+ \beta_k(q_t^{(k)}) \psi_k(q_t^{(k)}, h_t^{(k)})
+ \varepsilon_{m,k}.
```

This is the version that is complete, implementable in principle, and mathematically honest about what is guaranteed and what is not.
