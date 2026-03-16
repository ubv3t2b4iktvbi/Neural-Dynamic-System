from __future__ import annotations

import argparse
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural Dynamic System command line.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", help="Train a model or run a single experiment config.")
    subparsers.add_parser("plot", help="Regenerate plots for a finished run.")
    subparsers.add_parser("suite", help="Run the van der Pol ablation suite.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command, rest = argv[0], argv[1:]
    if command == "train":
        from . import cli as train_cli

        train_cli.main(rest)
        return
    if command == "plot":
        from . import plot_cli

        plot_cli.main(rest)
        return
    if command == "suite":
        from . import suite_cli

        suite_cli.main(rest)
        return
    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    main()
