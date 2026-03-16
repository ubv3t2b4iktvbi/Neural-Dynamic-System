# External Repo Intake

Use this note to turn an outside quant repo into a bounded translation task instead of reading the whole project.

## First-pass triage
1. Clone or open the repo locally.
2. Run:

```bash
python .agents/skills/project/quant-factor-dynamics-updater/scripts/repo_factor_manifest.py <repo_path> --format md --max-files 20
```

3. Read, in order:
   - the main `README`
   - the highest-ranked factor-definition files
   - the highest-ranked experiment or evolution files
   - the factor-library output format or loader
   - one config file that shows what the system is actually optimizing

## Repo archetypes

### Static factor library
- Typical signs: many factor-expression files, loaders, factor catalogs, little evolution logic.
- Translation target: factor formulas and naming conventions.
- Local action: map directly into `FactorSpec` or new base features.

### Evolutionary mining system
- Typical signs: `planning`, `trajectory`, `mutation`, `crossover`, `hypothesis`, `library`.
- Translation target: mechanism families and search priors, not only final formulas.
- Local action: translate trajectory outcomes into dynamical families, then screen them locally with RC and coordinate diagnostics.

### Backtest-first framework
- Typical signs: strong emphasis on `qlib`, `rankic`, `returns`, `portfolio`, but factor code is secondary.
- Translation target: factor semantics and data assumptions.
- Local action: treat backtest metrics as weak evidence until the factor survives local dynamics diagnostics.

## QuantaAlpha worked example

### High-value files
- `README.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/user_guide.md`
- `docs/experiment_guide.md`
- `configs/experiment.yaml`
- `experiment/original_direction.json`

### What matters
- The main loop is `research direction -> diversified planning -> trajectory evolution -> validated factors`.
- `all_factors_library*.json` is the output library, but the richer signal often lives earlier in the trajectory and hypothesis artifacts.
- The useful import for this project is usually not a verbatim Qlib formula. It is the mechanism family behind that formula: momentum breakout, reversal recovery, compression-release, regime surprise, or multi-timescale interaction.

### Translation stance
- Treat QuantaAlpha directions and trajectories as proposal generators for dynamical motifs.
- Treat the local project as the validation environment: `run_factor_mining.py`, `run_coordinate_analysis.py`, and `run_research_loop.py`.
- Port only motifs that can be expressed causally with local benchmark observations or with a justified new base feature.

## Intake output template

Produce a short repo note with these fields before editing code:

| Field | What to record |
|---|---|
| source repo | URL or local path |
| repo archetype | static library / evolutionary miner / backtest-first |
| top files | 3-8 files and why they matter |
| motif families | candidate alpha families worth translating |
| blocked semantics | assumptions that do not map cleanly to local state variables |
| first local target | `feature_engine.py`, `factor_bank.py`, `finance_to_dynamics.py`, or experiment-only hypothesis |
