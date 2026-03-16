from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_dynamic_system.plots import save_single_run_plots  # noqa: E402
from neural_dynamic_system.reporting import build_run_visual_payload  # noqa: E402
from neural_dynamic_system.run_artifacts import load_saved_model, regenerate_analysis_data  # noqa: E402
from neural_dynamic_system.run_config import resolve_repo_path  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate plots for a finished van der Pol run.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing model.pt/config.json/summary.json.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional output directory for regenerated plots.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--trajectory_points", type=int, default=400)
    parser.add_argument("--color", type=str, default="#b5542b")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = resolve_repo_path(ROOT, args.run_dir)
    if run_dir is None:
        raise ValueError("run_dir must not be empty")
    out_dir = run_dir if args.out_dir is None else resolve_repo_path(ROOT, args.out_dir)
    if out_dir is None:
        raise ValueError("out_dir must not be empty")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    loaded, model_cfg, train_cfg = load_saved_model(run_dir, device=device)
    history = pd.read_csv(run_dir / "history.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trajectory, analysis_labels = regenerate_analysis_data(run_dir)
    payload = build_run_visual_payload(
        run_name=run_dir.name,
        title=run_dir.name.replace("_", " "),
        color=str(args.color),
        trajectory=trajectory,
        analysis_labels=analysis_labels,
        result=loaded,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        history=history,
        summary=summary,
        eval_batch_size=int(args.eval_batch_size),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload.horizon_metrics.to_csv(out_dir / "horizon_metrics.csv", index=False)
    save_single_run_plots(payload, out_dir, trajectory_points=int(args.trajectory_points))
    print(f"plots regenerated under: {out_dir}")


if __name__ == "__main__":
    main()
