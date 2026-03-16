# Validation Protocol

Use this protocol after translating external motifs into local dynamical factors.

## Minimal loop
1. Translate 1-3 candidate motifs and document the translation table.
2. Run factor-mining smoke tests on one narrow task.
3. Inspect candidate and selected-factor artifacts.
4. Run coordinate analysis if the claim involves closure, spectrum, separability, or memory.
5. Escalate to the closed-loop workflow only after the smoke pass is interpretable.
6. Write one follow-up experiment matrix before promoting any factor.

## Command patterns

### Smoke factor mining

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --out_dir runs/factor_mining/external_repo/<repo_slug>_smoke
```

Use `--mode accumulate` for direct library extension and `--mode identify` when importing motif families rather than fixed formulas.

### Coordinate analysis

```bash
python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/external_repo/<repo_slug>_smoke
```

### Closed-loop validation

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/external_repo/<repo_slug>_smoke --mining_mode identify
```

## Claim-to-experiment mapping

| Claim | What to compare | Evidence to seek | Promotion gate |
|---|---|---|---|
| translated factor improves closure | `raw` vs `factor` coordinates, with `delay` as sanity check | factor coordinates reduce memory deficit without obvious spectrum collapse | closure improves and spectral/separability metrics do not degrade materially |
| translated factor is a slow-fast observable | smoke task plus at least one slow-fast-friendly task | same family is selected or remains near top across tasks | survives beyond one task and keeps mechanistic meaning |
| translated factor captures regime surprise | identifier-enabled runs with and without surprise-style factor | factor ranks well and remains interpretable in `manual_review.md` | improvement is not only a one-step artifact and review notes stay coherent |
| translated factor captures compression-release or multiscale structure | smoke task plus one harder task or research-loop pass | factor helps while preserving or improving separability / local geometry | not confined to a single task and no obvious rollout instability warning |

## Artifacts to read before promotion
- `candidate_scores.csv`
- `selected_factor_library.json`
- `manual_review.md`
- `metrics.json`
- `run_summary.md`
- `coordinate_summary.csv` and `coordinate_summary.md`
- `loop_summary.md` when the closed loop is used

## Follow-up experiment matrix

Write a table with:

| field | meaning |
|---|---|
| hypothesis | the mechanistic claim |
| candidate factors | translated factor names |
| comparison | baseline vs translated condition |
| script | one of the local entrypoints |
| task set | smoke first, then one follow-up task |
| success criterion | metric and qualitative review gate |
| next action | promote, revise, or reject |

## Promotion rules
- Do not promote on a single metric bump alone.
- Require a mechanistic explanation that matches `dynamics_meaning`.
- Prefer cross-task consistency or a second-seed rerun for shortlisted factors.
- Leave unresolved cases in the review queue instead of forcing a library update.
