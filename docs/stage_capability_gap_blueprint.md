# Stage Capability Gap Closure Blueprint

## Goal

Turn stage-by-stage theory discussions into a monotone upgrade path:

1. Reuse the current model whenever possible.
2. Prefer non-training operations over retraining.
3. If training is needed, warm-start from the previous stage and fine-tune the smallest affected surface.
4. Require an explicit certificate before moving to the next stage.

This blueprint is written for the current `neural_dynamic_system/` stack, where the existing implementation is strongest as:

- a closure-oriented latent autoencoder,
- a `q + driven hidden block` dynamical model,
- a warm-start base for future `q/h/m` structure discovery.

It is not yet a full automatic structure-discovery system.

## Current Diagnosis

The current model family is effectively:

- `q`: slow Koopman-aligned block
- `h`: driven stable hidden block
- `m`: alias of `h`, not an independent memory block

That means the current checkpoint already supports:

- closure-oriented prediction,
- stable rollout,
- rate-ordered modal summaries,
- a hidden block that can act as generic closure state.

It does not yet support, in a structurally faithful way:

- independent `q+h` versus `q+m` versus `q+h+m` model-family choice,
- explicit fiber invariance evidence,
- effective-dimension reporting,
- checkpoint-preserving mode commitment with clear certificates.

So the shortest path is not "throw away the model and retrain a new family".
The shortest path is "wrap the current model in progressively stronger views, diagnostics, and only then narrow warm-start upgrades".

## Design Principle

Treat every stage as a proof obligation:

- closure stage asks: "Is the latent sufficiently Markov and predictive?"
- structure stage asks: "Do the added states behave like fiber, memory, both, or neither?"
- commitment stage asks: "Which model family is justified by evidence?"
- refinement stage asks: "Can geometry or memory consistency be strengthened without breaking closure?"
- consolidation stage asks: "Can inactive structure be frozen or pruned safely?"

For each stage, separate:

- expressivity gap: the family cannot represent the needed object,
- identifiability gap: the current losses and summaries cannot distinguish candidates,
- curriculum gap: the stage gate advances on the wrong evidence.

Add two more checks before proposing a path:

- transition-reuse gap: what the next stage needs that the current checkpoint cannot yet supply directly,
- literature gap: whether the proposed remedy is already covered by mature theory or algorithms.

## Research First, Then Path Search

Before proposing a model change, build a small evidence stack:

1. Theory paper or theorem source
2. Mature algorithm source
3. Empirical or benchmarking source
4. Review or synthesis source
5. Existing repository path that is closest to the literature idea

Then run a closed loop:

1. derive a hypothesis from the literature,
2. map it to the current checkpoint,
3. test the smallest falsifiable prediction,
4. update the hypothesis,
5. only then escalate to a model change.

This prevents the process from becoming "invent a clever architecture and hope it works".
It forces each proposed path to be literature-backed, repo-grounded, or clearly marked as speculative.

## Monotone Upgrade Ladder

Always try upgrades in this order:

1. Pure analysis
2. Analytic coordinate transform
3. Checkpoint masking or freezing
4. Checkpoint surgery with zero or minimal new parameters
5. Narrow warm-start fine-tune
6. Broader warm-start fine-tune
7. Full-family retraining only if all earlier steps fail

The default assumption is that stage `k+1` should inherit the checkpoint from stage `k`.

## Reuse Matrix

The old blueprint treated reuse as a preference.
From here on, treat reuse as a first-class design constraint.

Each stage should export a reusable bundle:

- checkpoint
- encoded latent cache when practical
- local operator cache such as `A(q)`, `b(q)`, rates, and adapter parameters
- evidence table
- path log

How later stages reuse them:

- `A -> B`: reuse the checkpoint, rollout cache, and latent cache to test whether added state is needed at all
- `B -> C`: reuse the same checkpoint and evidence table to compare candidate families without retraining separate models
- `C -> D`: reuse the committed-family checkpoint and family adapter; refine semantics rather than relearn representation
- `D -> E`: reuse all evidence plus masks to prune without restarting

This also means failed paths are not wasted work.
A failed path should still leave behind diagnostics that eliminate part of the search space.

## Hard Checklist Rule

The conservative main workflow should be written as a stack of strict checklists.
Each step must specify:

- input artifacts
- theoretical object
- executable operations
- output artifacts
- allowed conclusions
- forbidden conclusions

This is necessary because the main risk in this project is not only underfitting.
It is also semantic overclaiming: saying "fiber", "memory", or "slow manifold" before the current model and diagnostics actually justify those words.

## Mature Algorithm Reuse Menu

Use mature algorithms to shrink the search space before adding trainable structure.

### For closure or Markov surrogates

- delay embedding and reconstruction maps
- DMD or EDMD for low-cost linear or lifted dynamics checks
- VAMP for Koopman-singular feature scoring and model selection
- ERA or N4SID style subspace identification when a linear state-space proxy is informative

These are useful for the `Stage A -> Stage B` gap, where the main question is whether the current latent is already predictive enough and whether additional state is even needed.

### For stage-to-stage dimensional or block selection

- SVD and low-rank residual analysis
- balanced truncation or balanced POD
- Schur reordering and modal sorting
- controllability or observability diagnostics

These are useful for the `Stage B -> Stage C` gap, where the question is no longer "can we predict?" but "which part of the state is structurally worth keeping?"

Important limitation:

These methods usually solve rank, energy, controllability, observability, or prediction usefulness.
They do not by themselves solve the semantic distinction between fiber and memory.

### For fiber hypotheses

- fixed-point recentering of driven hidden dynamics
- local linearization around the candidate slow manifold
- invariant-subspace and tangent-normal projection diagnostics
- spectral submanifold or nonlinear normal mode reasoning

These are useful for the `Stage C -> Stage D1` gap, where the issue is whether a candidate fast block can really be interpreted as a fiber rather than generic hidden closure.

### For memory hypotheses

- delay-coordinate closure models
- autoregressive or state-space residual closure
- Mori-Zwanzig-inspired memory closure
- equation-free coarse dynamics

These are useful for the `Stage C -> Stage D2` gap, where the main question is whether the unresolved effect should stay as driven memory rather than be forced into an invariant-fiber interpretation.

### For pruning and consolidation

- thresholding
- masking
- checkpoint surgery
- frozen reduced-family evaluation

These are useful for the `Stage D -> Stage E` gap, where the objective is to compress only after the evidence is stable.

## Stage Flow

### Stage A: Closure Bootstrap

Target:

- obtain a latent state that is predictive enough to serve as a dynamical proxy,
- avoid early semantic commitments about fiber versus memory.

Certificate:

- one-step prediction improves and stays stable,
- short rollout improves and stays stable,
- no numerical instability,
- latent rollout remains better than reconstruction-only behavior.

Allowed operations before training:

- add reporting for one-step, short-rollout, and latent-vs-observation errors,
- evaluate selective horizons on the current checkpoint,
- compare `z`, `q`, and `h` predictive usefulness without modifying weights.

Minimal training if needed:

- warm-start from the current checkpoint,
- train only prediction-facing losses first,
- do not strengthen geometry, separation, or semigroup aggressively.

Why the current model is already close:

- the existing implementation already has reconstruction, one-step/multi-step prediction, Koopman loss, and rollback-aware curriculum.

Gap to the next stage:

- Stage A proves predictive usefulness,
- Stage B additionally requires structural evidence,
- so the real gap is not "better prediction" but "prediction decomposed into interpretable causes".

### Stage B: Structural Discovery

Target:

- decide whether the extra state behaves like fiber, memory, both, or neither,
- expose evidence without forcing a new family too early.

Required evidence:

- invariance residual `I_h`
- fast-fiber gain `G_h`
- memory gain `G_m`
- effective active dimensions or equivalent block activity summaries

Recommended definitions:

- `I_h = E || f_h(q, 0) ||^2`
- `G_h = L_pred(q-only) - L_pred(q+h)`
- `G_m = L_pred(q-only) - L_pred(q+m)`

Non-training-first upgrades:

1. Add selective ablation evaluators on top of the current checkpoint:
   - decode and roll out with `h=0`
   - decode and roll out with memory-only view
   - compare against the full state
2. Add hidden fixed-point recentering:
   - if current hidden dynamics is `dh/dt = A(q) h + b(q)`,
   - define the driven equilibrium `h_star(q) = -A(q)^(-1) b(q)` when stable,
   - define fiber-view coordinate `eta = h - h_star(q)`.

This recentering is the key non-training upgrade. It converts the current driven hidden block into a candidate fiber coordinate system without discarding the checkpoint.

Interpretation after recentering:

- if `I_eta` becomes small and predictive gain survives, the current hidden block is compatible with a fiber view,
- if predictive gain exists but invariance remains poor, the block should stay memory-like.

Minimal training if needed:

- fine-tune only the new diagnostic heads or split heads,
- freeze encoder and validated `q` dynamics first.

Gap to the next stage:

- Stage B proves that different views are plausible,
- Stage C additionally requires choosing one family,
- so the real gap is not representational capacity alone but decision-quality under competing explanations.

Do existing algorithms already solve `B -> C`?

Answer: partially, but not fully.

Already solved or mostly solved:

- whether additional state improves closure:
  delay embedding, VAMP scores, subspace identification baselines, and residual autoregressive closure can answer this well
- how many additional effective directions may matter:
  SVD, Hankel singular values, balanced truncation, and order-selection criteria help here
- whether a memory-style closure view is plausible:
  delay-coordinate and state-space closure baselines help here

Not fully solved in the general setting:

- whether the additional state should be interpreted as an invariant fast fiber rather than generic hidden closure
- whether the family decision should be `q+h`, `q+m`, or `q+h+m` from observations alone without structural assumptions

Therefore `B -> C` should be treated as:

- first, an algorithmically narrowed decision,
- then, a semantics-aware commitment step.

### Stage C: Mode Commitment

Target:

- choose among `q-only`, `q+h`, `q+m`, or `q+h+m`.

Commitment rule:

- keep `h` if `G_h` is positive and invariance is small,
- keep `m` if `G_m` is positive and invariance is not small,
- keep both if both gains are positive and behavior is distinct,
- collapse to `q-only` if neither gain is significant.

Non-training-first path:

- represent family choice as masks over a shared checkpoint:
  - `q-only`: zero hidden contribution,
  - `q+h`: use recentered fiber coordinate only,
  - `q+m`: use driven memory coordinate only,
  - `q+h+m`: split the existing hidden block into both views.

At this stage, "model-family selection" should first be a view over the checkpoint, not a new training run.

Minimal training if needed:

- only after one family is favored,
- initialize new block heads from the existing hidden block,
- train the chosen branch while freezing the rest.

Gap to the next stage:

- Stage C proves a family choice,
- Stage D requires that the chosen family behave according to its claimed semantics,
- so the next gap is semantic faithfulness, not family selection.

Concrete `C -> D` model design:

Use a shared-base, adapter-first architecture before any heavy retraining.

Shared inherited base:

- encoder `E`
- slow block `q`
- legacy hidden block `r`

Adapter layer:

- memory adapter `m = P_m r`
- fiber adapter `h = P_h (r - h_star(q))`
- family mask selecting `q-only`, `q+h`, `q+m`, or `q+h+m`

Recommended initialization:

- `P_m` starts as identity or a balanced-truncation style retained projection
- `h_star(q)` comes from the driven equilibrium of the inherited hidden dynamics
- `P_h` starts as identity on the recentered coordinates or a tangent-normal projection

Only after this shared adapter stage is stable, move to the explicit split model:

- `qdot = f_q(q) + B_qh(q) h + B_qm(q) m`
- `hdot = A_h(q) h + r_h(q,h)` with `r_h = O(||h||^2)`
- `mdot = A_m(q) m + b_m(q) + r_m(q,m)`
- `xhat = g(q) + D_h(q) h + D_m(q) m`

Loss design by branch:

- always keep closure and rollout losses
- for `h`: add invariance, transverse contraction, tangent-normal consistency
- for `m`: add long-horizon gain, closure consistency, and stability
- do not apply fiber losses to `m`
- do not force `m` to satisfy `m=0` invariance

Checkpoint-preserving training rule:

- first freeze encoder and slow block
- train only adapters or newly split readouts
- then, if needed, unfreeze the matching branch
- only last unfreeze the full model

### Stage D1: Fiber Refinement

Enter only if `q+h` is justified.

Target:

- make `h=0` a credible approximate invariant manifold in the chosen coordinate system,
- preserve predictive quality.

Needed capability:

- explicit fiber coordinate `h_fiber`,
- explicit invariance metric,
- contraction in the transverse block,
- optional tangent-normal consistency terms.

Non-training-first upgrades:

- continue to use the recentered coordinate view,
- compute `I_h`, transverse contraction, and projection diagnostics on the checkpoint.

Minimal training if needed:

- warm-start from the committed checkpoint,
- fine-tune only hidden geometry and coupling terms,
- keep encoder and slow block mostly frozen initially.

### Stage D2: Memory Refinement

Enter only if `q+m` is justified.

Target:

- make memory contribution predictive and stable over longer horizons,
- preserve closure without pretending the memory block is a fiber.

Needed capability:

- explicit memory state `m`,
- stable driven transition,
- long-horizon gain reporting,
- closure consistency summaries.

Non-training-first upgrades:

- preserve the original driven hidden dynamics as the memory view,
- evaluate longer-horizon gain and stability on the checkpoint.

Minimal training if needed:

- warm-start memory block only,
- strengthen rollout and consistency losses before touching the encoder.

### Stage E: Prune And Consolidate

Target:

- freeze or prune unsupported structure,
- preserve the chosen family,
- keep the checkpoint usable as the next baseline.

Safe prune rule:

- prune only after activity, gains, and invariance or memory certificates stay stable for multiple validations.

Non-training-first upgrades:

- hard masks,
- block freezing,
- checkpoint surgery that removes inactive slices without changing surviving weights.

Minimal training if needed:

- short consolidation fine-tune after pruning,
- no full restart.

Gap from Stage D to Stage E:

- Stage D proves semantic validity,
- Stage E requires compactness without losing that validity,
- so the real question becomes "which structure is provably redundant?" rather than "can the model still fit?".

## Target Model Family

The next model family should be expressed as:

`u -> (q, h, m, gates, diagnostics)`

with the following interpretation:

- `q`: slow structural coordinate
- `h`: fiber coordinate, expected to be approximately invariant at zero after recentering or refinement
- `m`: driven memory or closure coordinate
- `gates`: activity or family masks
- `diagnostics`: certificate metrics used for stage transition

The important point is that this family should be backward-compatible with the current checkpoint.

## Checkpoint-Preserving Upgrade Path

### Step 1: Legacy View

Interpret the current checkpoint as:

- `q = current slow block`
- `m = current driven hidden block`
- `h = empty`

This gives an immediate `q+m` baseline with zero retraining.

### Step 2: Recentered Fiber View

Construct an analytic view from the same checkpoint:

- compute `h_star(q)` from the hidden equilibrium,
- define `h = current_hidden - h_star(q)`.

This produces a candidate `q+h` view without retraining.

### Step 3: Dual View

Expose both from the same checkpoint:

- `h_fiber = current_hidden - h_star(q)`
- `m_memory = h_star(q)` or the driven component summary

This gives a provisional `q+h+m` interpretation using shared inherited weights.

### Step 4: Explicit Split Module

Only after the evidence favors a split, add trainable explicit `h` and `m` blocks:

- initialize shared readout and dynamics from the legacy hidden block,
- freeze the encoder and slow block,
- fine-tune the new split minimally.

This is the first point where new training should usually happen.

## Repo-Level Design

### Config Surface

Add fields that support monotone upgrades instead of forcing a fresh branch:

- `structure_mode = legacy_qh | split_view | split_trainable`
- `structure_family = auto | q_only | q_h | q_m | q_h_m`
- `fiber_view_mode = none | recentered_equilibrium | learned_offset`
- `use_structure_diagnostics = true/false`
- `h_dim`, `m_dim` as independent dimensions
- `gate_mode = none | fixed_mask | learned_mask`
- `warm_start_upgrade_mode = inherit | surgery | branch_init`

### Model Surface

Refactor the latent interface so it can expose both legacy and upgraded views:

- `encode_components()` should return independent `q`, `h`, `m`, plus legacy aliases where needed,
- latent split and join should support all three blocks,
- add helper methods for:
  - `fiber_equilibrium(q)`
  - `recenter_hidden_to_fiber(q, h_legacy)`
  - selective block masking
  - `predict_with_family(family_name)`

### Training Surface

Training should log stage evidence directly:

- `I_h`
- `G_h`
- `G_m`
- family-specific prediction losses
- effective active dimensions or block activity

The training loop should allow:

- zero-training evaluation of family views,
- warm-start fine-tuning of selected modules,
- preservation of old summaries for comparison.

### Curriculum Surface

Stage transitions should be evidence-driven, not epoch-driven alone.

Recommended stage gates:

- A -> B: closure metrics plateau in a stable region
- B -> C: at least one of `G_h` or `G_m` is materially positive and diagnostics are stable
- C -> D: family choice stops oscillating
- D -> E: invariance or memory certificates remain stable through repeated validation

## Minimal Implementation Roadmap

1. Add diagnostics first, with no weight changes.
2. Add analytic hidden recentering and family-view evaluation.
3. Expose independent `m` in config and summaries while preserving checkpoint aliases.
4. Allow `q-only`, `q+h`, `q+m`, and `q+h+m` evaluation from one inherited checkpoint.
5. Only then add explicit trainable split blocks and warm-start them from the legacy hidden block.
6. Add pruning or hard masks only after the committed family is stable.

## Practical Rule

If a stage can be certified by:

- reporting,
- coordinate recentering,
- masking,
- freezing,
- or checkpoint surgery,

do that and move on.

Only pay for optimization when those tools cannot close the gap.

That keeps each stage directly usable by the next one and minimizes training volume.
