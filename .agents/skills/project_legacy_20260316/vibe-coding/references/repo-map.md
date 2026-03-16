# Repo Map

Use this file when the correct edit surface or the smallest useful validation command is unclear.

## Read Order
1. `scripts/README.md`
2. `src/README.md`
3. `src/fsrc_sindy/README.md`

## File Placement
- CLI flags or entrypoint behavior: the matching `scripts/run_*.py` file, then the underlying module in `src/fsrc_sindy/`.
- Benchmark suite changes or model-group wiring: `src/fsrc_sindy/selection.py`, `src/fsrc_sindy/experiment.py`, `scripts/run_benchmarks.py`.
- New model families: `src/fsrc_sindy/models/*.py`, `src/fsrc_sindy/models/__init__.py`, `src/fsrc_sindy/selection.py`.
- Factor formulas or features: `src/fsrc_sindy/factors/factor_bank.py`, `src/fsrc_sindy/factors/feature_engine.py`, `src/fsrc_sindy/factors/miner.py`.
- Identifier backends: `src/fsrc_sindy/factors/identifiers.py`, `src/fsrc_sindy/factors/miner.py`, `src/fsrc_sindy/factors/property_analyzer.py`.
- Factor-mining orchestration or config plumbing: `src/fsrc_sindy/pipeline/factor_mining.py`, `scripts/run_factor_mining.py`, `configs/factor_mining.yaml`.
- Coordinate diagnostics: `src/fsrc_sindy/research/coordinate_analysis.py`, `scripts/run_coordinate_analysis.py`.
- Research-loop logic, gating, confidence, or report generation: `src/fsrc_sindy/research/loop.py`, `scripts/run_research_loop.py`.
- User-facing docs: `README.md`, `scripts/README.md`, `src/README.md`, `src/fsrc_sindy/README.md`.

## Output Contracts To Preserve
- Factor mining usually emits `candidate_scores.csv`, `selected_factor_library.json`, `manual_review.md`, `run_summary.md`, and `manifest.txt`.
- Coordinate analysis usually emits `coordinate_summary.csv`, `coordinate_summary.md`, and `coordinate_details.json`.
- Research loop usually emits `loop_summary.md`, `loop_manifest.json`, `confidence_report.json`, and `expert_review_template.md`.

## Minimal Validation
- Broad import, syntax, or plumbing changes:
  `python -m compileall src scripts`
- Benchmark CLI or registry wiring:
  `python scripts/run_benchmarks.py --help`
  `python scripts/run_benchmarks.py --suite smoke --tasks vanderpol_smoke --out_dir runs/benchmarks/_skill_smoke`
- Coordinate-analysis changes:
  `python scripts/run_coordinate_analysis.py --help`
  `python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/_skill_smoke`
- Factor-mining changes:
  `python scripts/run_factor_mining.py --help`
  `python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --out_dir runs/factor_mining/_skill_smoke`
- Research-loop changes:
  `python scripts/run_research_loop.py --help`
  If runtime is acceptable, run:
  `python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/_skill_smoke`
