from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "node_modules",
    "venv",
}

CATEGORY_TERMS: dict[str, tuple[str, ...]] = {
    "factor_definition": ("factor", "alpha", "expression", "formula", "signal"),
    "evolution_or_hypothesis": ("trajectory", "evolution", "mutation", "crossover", "planning", "direction", "hypothesis"),
    "experiment_or_config": ("experiment", "config", "benchmark", "eval", "validation"),
    "backtest_or_metrics": ("backtest", "rankic", "ic", "return", "drawdown", "qlib", "metric"),
    "library_or_storage": ("library", "catalog", "repository", "archive", "json"),
}

IMPORTANT_NAMES = {
    "readme.md",
    "readme_cn.md",
    "project_structure.md",
    "user_guide.md",
    "experiment_guide.md",
    "backtest.yaml",
    "experiment.yaml",
}


@dataclass
class FileHit:
    path: str
    score: int
    categories: list[str]
    matched_terms: list[str]
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank the most factor-relevant files inside a local quant repo clone.")
    parser.add_argument("repo_path", type=str, help="Path to a local repo checkout.")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Optional output file. Defaults to stdout.")
    return parser.parse_args()


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def can_read_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.lower() in IMPORTANT_NAMES


def read_text_preview(path: Path, limit: int = 16000) -> str:
    if not can_read_text(path):
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit].lower()
    except OSError:
        return ""


def score_file(root: Path, path: Path) -> FileHit | None:
    rel = path.relative_to(root).as_posix()
    rel_lower = rel.lower()
    preview = read_text_preview(path)
    score = 0
    categories: set[str] = set()
    matched_terms: list[str] = []

    if path.name.lower() in IMPORTANT_NAMES:
        score += 6
        matched_terms.append(f"name:{path.name}")

    for category, terms in CATEGORY_TERMS.items():
        category_hit = False
        for term in terms:
            path_hit = term in rel_lower
            text_hit = bool(preview) and term in preview
            if path_hit:
                score += 3
                matched_terms.append(f"path:{term}")
                category_hit = True
            if text_hit:
                score += 1
                matched_terms.append(f"text:{term}")
                category_hit = True
        if category_hit:
            categories.add(category)

    if "readme" in rel_lower or "/docs/" in f"/{rel_lower}/":
        score += 1
    if path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}:
        score += 1

    if score <= 0:
        return None

    unique_terms = list(dict.fromkeys(matched_terms))[:8]
    reason = ", ".join(unique_terms[:4]) if unique_terms else "keyword hit"
    return FileHit(
        path=rel,
        score=score,
        categories=sorted(categories),
        matched_terms=unique_terms,
        reason=reason,
    )


def build_manifest(root: Path, max_files: int) -> dict[str, object]:
    hits = [hit for path in iter_files(root) if (hit := score_file(root, path)) is not None]
    hits.sort(key=lambda item: (-item.score, item.path))
    top_hits = hits[:max_files]

    category_counts: dict[str, int] = {}
    for hit in top_hits:
        for category in hit.categories:
            category_counts[category] = category_counts.get(category, 0) + 1

    suggested_reads = [hit.path for hit in top_hits[: min(8, len(top_hits))]]
    return {
        "repo_root": str(root.resolve()),
        "scanned_files": len(hits),
        "top_categories": category_counts,
        "suggested_first_reads": suggested_reads,
        "files": [asdict(hit) for hit in top_hits],
    }


def render_markdown(manifest: dict[str, object]) -> str:
    lines = [
        f"# Repo factor manifest: {manifest['repo_root']}",
        "",
        f"- scanned relevant files: {manifest['scanned_files']}",
        "",
        "## Suggested first reads",
        "",
    ]
    suggested = manifest.get("suggested_first_reads", [])
    if suggested:
        for path in suggested:
            lines.append(f"- `{path}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Ranked files",
            "",
            "| rank | path | score | categories | reason |",
            "|---|---|---:|---|---|",
        ]
    )
    for idx, row in enumerate(manifest.get("files", []), start=1):
        categories = ", ".join(row["categories"]) if row["categories"] else "-"
        reason = str(row["reason"]).replace("|", "/")
        lines.append(f"| {idx} | `{row['path']}` | {row['score']} | {categories} | {reason} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repo path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repo path is not a directory: {root}")

    manifest = build_manifest(root, max_files=max(1, args.max_files))
    if args.format == "json":
        content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    else:
        content = render_markdown(manifest)

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
