---
name: stage-capability-gap-closer
description: Iteratively analyze stage objectives versus model capability in Neural Dynamic System, derive expressivity, identifiability, and curriculum gaps, and upgrade config, model, training, or curriculum until each stage has enough capacity and diagnostics to satisfy its goal. Use when Codex needs to turn theory-driven gap analysis into warm-start architecture changes, stage certificates, checkpoint-preserving upgrades, or reduced-retraining research loops for closure, fiber-vs-memory separation, mode commitment, or pruning.
---

# Stage Capability Gap Closer

Use this skill to turn repeated "stage goal vs model ability" discussions into incremental repository changes and explicit stage certificates.

## Core Principle
1. Prefer carrying the current checkpoint forward.
2. Treat full retraining as the last resort.
3. Start with research, existing paths, and evidence before inventing a new upgrade path.
4. Use non-training or low-training operations first when they can satisfy the next stage:
   - add diagnostics or summaries without changing weights
   - reorder or relabel modes with analytic transforms
   - zero, gate, freeze, or mask blocks with existing weights
   - split outputs by deterministic post-processing or checkpoint surgery
   - warm-start new heads or blocks from existing parameters
5. If training is unavoidable, run the narrowest warm-start fine-tune that preserves the previous stage.
6. Do not reinitialize the encoder, q block, or already-validated dynamics unless the current model family is proven insufficient.

## Research First
1. Start every stage with a literature-and-path review before proposing a model change.
2. Gather four buckets:
   - theory: what object should exist if the stage is valid
   - algorithm: which mature methods already approximate or test that object
   - experiment: what empirical signatures the literature uses to validate it
   - summary or review: what conclusions or caveats are already established
3. Reuse existing repository paths before proposing new architecture:
   - current checkpoint behavior
   - current summaries, probes, and ablations
   - existing scripts, configs, and curriculum branches
4. Mark every new architectural idea as either:
   - literature-backed
   - repo-backed
   - hypothesis needing validation
5. Do not skip directly to a custom neural solution when a mature algorithm can first narrow the hypothesis space.
6. Before numerical validation, write the strongest non-overclaiming theory statement the current model family can support:
   - what object the model claims to recover
   - which weaker proxy it actually learns if the full object is not identifiable
   - which claims are only valid for multi-response shared-driver data versus single-sequence data
7. When the system structure is unknown, treat a single global latent family as a hypothesis rather than a default:
   - compare routed or piecewise family explanations before locking in one chart
   - consider mixture-of-charts, operator-family routing, or regime dispatch ahead of stronger theorem claims
   - do not let an intra-state channel mixer stand in for structural routing

## Gap -> Path -> Feasibility Workflow
1. Identify the exact gap between the current stage and the next stage:
   - stage target
   - evidence currently satisfied
   - missing capability
   - whether the missing part is expressivity, identifiability, curriculum, or transition-reuse
2. Decompose the remediation into candidate paths, ordered from cheapest to most invasive:
   - path A: pure diagnostics or reporting
   - path B: literature-backed analytic transform or mature algorithm
   - path C: checkpoint masking, freezing, or surgery
   - path D: narrow warm-start fine-tune
   - path E: broader family upgrade
3. Break each candidate path into the smallest executable steps.
4. Rate each step for feasibility:
   - no training needed
   - mature algorithm available
   - narrow warm-start needed
   - blocked by missing representation or tooling
5. Choose the lowest-cost feasible path that preserves the most checkpoint reuse and is best supported by theory and prior experiments.
6. Re-evaluate after each small step before escalating to the next path.

## Reuse Contract
Treat every stage output as a reusable package for all later stages and all candidate paths.

Required reusable artifacts:
- checkpoint or frozen model state
- latent caches or encoded summaries when cheap to store
- operator summaries such as local linearizations, rate tables, and family-view adapters
- stage evidence table such as `I_h`, `G_h`, `G_m`, prediction deltas, and ablation results
- decision log recording which paths were ruled out and why

Before proposing retraining, ask:
- can the next path reuse the same checkpoint?
- can the same encoded latents be reused for multiple path tests?
- can an analytic adapter expose the next family without changing weights?
- can a failed path still leave diagnostics that prune the search space?

## Hard Checklist Format
For every proposed step, write a strict checklist with exactly these fields:

- input artifacts
- theoretical object
- executable operations
- output artifacts
- allowed conclusions
- forbidden conclusions

Use this checklist to prevent semantic overreach.

Checklist rules:
- `input artifacts` must list the specific checkpoint, caches, metrics, adapters, or literature assumptions used by the step.
- `theoretical object` must name the mathematical object being tested or approximated, not just an engineering goal.
- `executable operations` must be limited to actions actually supported by the current repo, current checkpoint, or mature algorithms cited in the research pass.
- `output artifacts` must be reusable by later steps.
- `allowed conclusions` must be strictly weaker than the theoretical object unless the step truly proves the full claim.
- `forbidden conclusions` must explicitly block the common over-interpretations for that step.

If a step cannot be written in this format, it is not mature enough to become part of the main workflow.

## Iteration Loop
1. Write the current stage target as a proof obligation: closure, structural discovery, mode commitment, geometric refinement, or prune and consolidate.
2. Write the stage-to-stage transition gap explicitly:
   - what the current stage proves
   - what the next stage additionally requires
   - what can be inherited unchanged
3. Decompose the gap into:
   - expressivity: the model family cannot represent the needed state or branch
   - identifiability: the losses or diagnostics cannot distinguish the desired structure
   - curriculum: the phase gates or rollback logic do not match the evidence
4. Check whether the gap can be closed without retraining:
   - add a summary metric
   - run a selective rollout or ablation on the current checkpoint
   - recenter coordinates or remap a block analytically
   - threshold gates, prune inactive blocks, or freeze modules
5. If not, patch the narrowest layer:
   - `neural_dynamic_system/config.py` for dimensions, flags, and serialization
   - `neural_dynamic_system/model.py` for latent blocks, dynamics, and diagnostics
   - `neural_dynamic_system/training.py` for losses and logged evidence
   - `neural_dynamic_system/curriculum.py` for stage logic
   - `neural_dynamic_system/cli.py` when the user-facing contract changes
6. Keep old checkpoints forward-compatible whenever possible; add aliases instead of silent breaking changes.
7. Define an exit certificate for the stage before moving on.

## Mature Algorithm Reuse
- For closure or Markov tests, consider delay embedding, DMD or EDMD, Koopman spectral estimation, VAMP, ERA, N4SID, and related subspace-identification baselines before adding new learned structure.
- For stage-to-stage dimensional or block selection, consider SVD, balanced truncation, balanced POD, Schur reordering, controllability or observability analysis, and mode-energy ranking before learning new gates.
- For fiber hypotheses, consider equilibrium recentering, local linearization, invariant-subspace or spectral-submanifold analysis, and tangent-normal projections before adding strong geometric losses.
- For memory hypotheses, consider delay coordinates, autoregressive or state-space closure models, Mori-Zwanzig-inspired coarse closure, and equation-free coarse dynamics before inventing a custom memory block.
- For pruning or consolidation, consider thresholding, masking, frozen reduced views, and checkpoint surgery before retraining a compact model.
- Record which mature algorithm was tried, what assumption it relies on, and whether it narrowed the hypothesis space.
- Distinguish clearly between:
  - algorithms that solve "extra state is needed"
  - algorithms that solve "how many extra states are needed"
  - algorithms that solve "the extra state is memory-like"
  - algorithms that solve "the extra state is invariant-fiber-like"
- Do not claim that a mature algorithm solves the full `B -> C` or `C -> D` transition unless it covers the semantic decision, not just rank or prediction.

## Required Certificates
- Closure stage: improve one-step and short-rollout behavior without numerical instability.
- Structural discovery stage: expose explicit evidence such as `I_h`, `G_h`, `G_m`, selective ablations, or effective dimensions.
- Mode commitment stage: justify a choice among `q-only`, `q+h`, `q+m`, or `q+h+m`.
- Fiber refinement stage: make invariance residuals small and transverse contraction stable.
- Memory refinement stage: show positive memory gain with stable long rollout.
- Consolidation stage: prune or freeze only after the evidence remains stable.

## Warm-Start Upgrade Ladder
1. Make pure analysis or reporting changes without retraining.
2. Make diagnostic-only model or summary changes and reuse the checkpoint for zero-shot evaluation.
3. Apply block masking, thresholding, freezing, or checkpoint surgery before considering optimization.
4. Add a narrow head or split an existing block only after initializing it from current weights and fine-tuning the affected modules first.
5. Escalate to a full-family change only after writing down why non-training and warm-start variants cannot satisfy the stage goal.

## Guardrails
- Do not start from private intuition alone; start from theory papers, algorithm papers, empirical validations, and review conclusions.
- Do not present a hypothesis as a conclusion until it survives an explicit theory -> experiment -> conclusion loop.
- Do not let experiments choose the claim after the fact; first name the mathematical object being tested, then design the validation.
- Do not turn on strong geometry or semigroup losses before closure evidence exists.
- Do not call a driven hidden state a fiber unless `h=0` is approximately invariant or explicitly recentered to an invariant offset.
- Do not treat a rate gap alone as proof of fast-slow separation.
- Do not promote a single-sequence latent state into a Sauer-style shared driver or shared semiconjugate factor unless the data actually contain multiple responses with a common driver and the objective includes an explicit shared-response coupling.
- Do not call `q` an exact Koopman eigenfunction, or `h` an exact slow manifold or fiber coordinate, unless the model family and diagnostics identify those stronger objects rather than only an approximate latent SSM coordinate.
- When the structure is unknown, do not rely on channel mixing alone; routing must happen at the chart, operator-family, or branch-selection level if the goal is automatic regime separation.
- Do not reset training from scratch when checkpoint surgery, gating, freezing, or a narrow fine-tune can preserve prior work.
- Preserve CLI flags, config serialization, summary keys, and checkpoint compatibility when surfacing new structure.

## Verification
1. Run `python -m compileall neural_dynamic_system scripts` first.
2. Run `python scripts/run_neural_dynamic_system.py --help` if the CLI or config surface changed.
3. Prefer a tiny synthetic warm-start run over a fresh large run when behavior changed.
4. Report what changed, what was reused from the previous stage, what retraining was avoided, and what gap remains.

## Pair With
- Use `$latent-qh-architect` for narrow q, h, or m architecture edits.
- Use `$neural-dynamics-vibe-coding` for general repository changes around the architecture work.
- Use `$run-artifact-summarizer` to judge whether a stage certificate is actually met.
