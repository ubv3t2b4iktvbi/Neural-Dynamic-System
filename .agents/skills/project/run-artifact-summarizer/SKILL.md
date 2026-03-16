---
name: run-artifact-summarizer
description: Summarize and compare Neural Dynamic System run artifacts such as `summary.json`, `history.csv`, `config.json`, label probes, and synthetic previews into actionable conclusions. Use when Codex needs experiment readouts, run comparisons, regression triage, or recommendations about the next training change.
---

# Run Artifact Summarizer

Use this skill when the main task is to interpret completed runs rather than edit code.

## Workflow
1. Read `config.json` first so the data source, latent sizes, and curriculum are explicit.
2. Read `summary.json` next for the selected best metrics.
3. Read `history.csv` to understand phase transitions and whether the best checkpoint is representative.
4. If present, read probe outputs:
   - `label_probe.json`
   - `synthetic_hidden_probe.json`
   - the corresponding correlations CSV
5. Summarize in this order:
   - data source and run shape
   - best metrics
   - phase behavior
   - probe signal quality
   - likely next change

## Comparison Rules
- Compare runs only when the differences in config are actually understood.
- Keep seed, synthetic kind, steps, and curriculum fixed when making causal claims about architecture or loss changes.
- Separate "best validation row" claims from "last epoch" behavior.

## Guardrails
- Do not claim improvement from one scalar alone when other diagnostics regress.
- Do not mix phase-1-only wins with phase-3 stabilized results without saying so.
- Do not infer q/h interpretability unless the probe outputs support it.
- Keep metric names exact when citing them from `summary.json`.

## References
- Metric-reading order and interpretation notes: [references/metric-reading-guide.md](references/metric-reading-guide.md)
