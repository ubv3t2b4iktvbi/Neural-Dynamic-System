# Closure-First Literature Audit

Date: 2026-03-18

Status: literature-backed audit note for `docs/closure_first_info_koopman_full_model_20260318.md`.

## 1. Purpose

This note records the external literature used to re-check the mathematical wording of the closure-first canonical theory model.

It is not a proof document.
Its purpose is narrower:

- identify where the current theory note is mathematically sound;
- identify where the wording must stay weaker than a theorem;
- record which repairs were made after checking the literature.

## 2. Primary Sources Checked

### 2.1 Flag-manifold geometry

- Absil, Hueper, and Trumpf, *Optimization on flag manifolds*, arXiv:1907.00949  
  Link: https://arxiv.org/abs/1907.00949

Use:

- confirms that the collection of ordered nested subspaces with fixed signature is a smooth manifold;
- supports the wording that our chart-local ordered block structure should be treated as a partial flag object, not just an arbitrary orthogonal basis.

### 2.2 Koopman eigenfunctions near fixed points and periodic orbits

- Kvalheim and Revzen, *Existence and uniqueness of global Koopman eigenfunctions for stable fixed points and periodic orbits*, arXiv:1911.11996  
  Link: https://arxiv.org/abs/1911.11996

Use:

- supports the statement that one needs stronger hypotheses than Hartman-Grobman alone for smooth Koopman eigenfunction claims;
- supports separate treatment of stable fixed points and periodic orbits;
- supports careful wording around semiconjugacies, Floquet normal forms, isostables, and smoothness assumptions.

### 2.3 Partial information decomposition versus interaction information

- Williams and Beer, *Nonnegative Decomposition of Multivariate Information*, arXiv:1004.2515  
  Link: https://arxiv.org/abs/1004.2515

Use:

- supports the correction that the pairwise quantity `\Delta_{ij}` in the theory note is not a full PID synergy quantity;
- justifies downgrading that quantity to an interaction-information-style merge/coexist proxy.

### 2.4 Mori-Zwanzig and auxiliary memory embeddings

- Lei, Baker, and Li, *Data-driven parameterization of the generalized Langevin equation*, PNAS 2016  
  Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC5167214/

Use:

- supports the distinction between exact projected memory equations and approximate finite-dimensional auxiliary-state embeddings;
- supports the claim that rational or auxiliary-variable approximations can turn memory terms into extended Markovian systems;
- supports the caution that this is an approximation hierarchy, not an exact theorem of the Mori-Zwanzig formalism in general.

## 3. Main Audit Conclusions

### 3.1 What survived the audit cleanly

- closure-first state `s_t` as the primitive latent object;
- chart-local decomposition before semantic promotion;
- explicit separation of attractor-restricted and neighborhood Koopman statements;
- local tangent/normal geometry with tubular-neighborhood caveats;
- treating finite-dimensional memory lifting as approximate rather than exact;
- explicit separation between hard guarantees, local theorem-level claims, and empirical targets.

### 3.2 What needed correction or tightening

- `flag manifold` needed to be promoted from an implicit basis trick to an explicit partial-flag object with fixed signature;
- `\Delta_{ij}` needed to stop being described as exact synergy;
- routing energy needed a soft switching penalty, because `\mathbf 1[k \ne k_{t-1}]` is only well defined for hard chart assignments;
- contraction guarantees needed to be restricted to the structured core map unless residual terms satisfy additional small-gain assumptions;
- rotational or phase blocks needed to be described as structured core components, not as exact full-map guarantees when arbitrary residual corrections are still present.

## 4. Repairs Applied to the Canonical Theory Note

The following repairs were made in `docs/closure_first_info_koopman_full_model_20260318.md`:

1. the chart-local ordered decomposition is now described as a partial flag with fixed signature;
2. the dependence of varying flags is now described as a smooth section of a partial-flag bundle over the chart domain;
3. the residual-role information scores now clarify that `q_t` means the already-promoted persistent state, avoiding a circular definition;
4. the routing switching term is now written in a soft form `S(k,c_{t-1}) = 1 - c_{t-1,k}`, with the hard indicator listed only as a special case;
5. contraction claims are now stated for the structured core operators, with extra assumptions required when residual terms are retained;
6. the memory section now explicitly distinguishes exact Mori-Zwanzig projection from approximate finite-dimensional auxiliary embeddings.

## 5. Remaining Non-Theorem Areas

Even after the literature check, the following parts remain intentionally weaker than theorem statements:

- promotion from information scores to true physical slow variables;
- exact chart recovery from data;
- exact identification of memory order;
- exact PID-style multivariate information decomposition;
- global low-dimensional closure for chaotic systems;
- joint optimality of the full multi-loss objective.

These should continue to be described as empirical targets, diagnostics, or model-selection heuristics.

## 6. Bottom-Line Verdict

After the literature audit, the current canonical theory note is materially stronger than the earlier drafts.

The remaining risk is not a hidden contradiction between the main technical layers.
The remaining risk is instead semantic overreach:

- saying that a proxy is a theorem;
- saying that a structured approximation is an exact recovered object;
- or saying that a local chart-level statement is a global identification result.

As long as those boundaries are kept explicit, the current mathematical model is internally coherent.
