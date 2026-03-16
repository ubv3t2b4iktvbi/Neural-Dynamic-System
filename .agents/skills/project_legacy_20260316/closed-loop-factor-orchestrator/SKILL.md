---
name: closed-loop-factor-orchestrator
description: Run the end-to-end research loop across ablation benchmarks, coordinate diagnostics, factor mining, and next-step planning in one command. Use when Codex should operate as a project-level control skill for analyzing ablations, proposing experiments, and mining new factors without manually stitching together separate scripts.
---

# Closed Loop Factor Orchestrator

Use this skill when the user wants a single control surface for the research loop.

## Workflow
1. Choose the loop mode:
   - `accumulate`: expand or stress-test the factor library across known tasks.
   - `identify`: treat the task as an unknown system, run property analysis first, and bias factor search with those properties.
2. Run `scripts/run_research_loop.py` for the requested suite or tasks.
3. Read `loop_summary.md` before opening detailed subreports.
4. Use benchmark ablations to detect memory, coordinate, or stability failures.
5. Use coordinate diagnostics to test Markov closure, spectral preservation, and separability.
6. Use factor-mining outputs to shortlist or discover interpretable factors.
7. Read `confidence_report.json` and `expert_review_template.md` before proposing actions.
8. Propose the next experiment or factor edit only after citing evidence from the generated artifacts.
9. Mark all LLM-produced recommendations as pending dynamics-expert review.

## Default command
```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo
```

## Guardrails
- Do not claim the loop is fully autonomous in code alone; the LLM still supplies theory-aware interpretation and experimental redesign.
- Do not modify `factor_bank.py` unless the proposed factor has both finance origin and dynamics meaning.
- In `identify` mode, do not skip the property-guided coordinate analysis stage.
- Preserve `loop_manifest.json` and `loop_summary.md` so each loop remains reviewable.
- Never bypass the expert-review gate. `expert_review_template.md` must be completed by a human dynamics expert before conclusions are treated as accepted.
