---
name: code-change-summarizer
description: Summarize tracked or working-tree changes in DynamicAlpha Lab into subsystem-level notes, validation status, and README-ready bullets. Use when Codex needs PR summaries, changelog entries, commit-scope explanations, release notes, or a concise explanation of how current diffs affect models, factor mining, research-loop behavior, scripts, skills, or documentation.
---

# Code Change Summarizer

Turn raw diffs into a short maintenance narrative that another engineer or agent can act on quickly.

## Workflow
1. Inspect scope with `git status --short`, `git diff --stat`, and `git diff --name-only`.
2. Split the diff into buckets:
   - behavior changes in `src/fsrc_sindy/`
   - entrypoint or tooling changes in `scripts/`
   - skill or workflow changes under `.agents/` and `.claude/`
   - docs/config changes
3. Map each changed code file to the nearest subsystem:
   - models and selection
   - factors and readouts
   - coordinate analysis
   - research loop orchestration
   - skill infrastructure
4. Capture validation that already ran, plus validation that is still missing.
5. Emit summaries in this order:
   - high-level behavior change
   - impacted files or subsystems
   - generated artifacts or new outputs
   - verification
   - open risks or excluded files

## Output Shape
- Lead with 2-5 grouped bullets, not a file-by-file dump.
- Separate user-facing behavior from internal refactors.
- Call out new outputs such as reports, manifests, or JSON artifacts explicitly.
- Mention untracked draft files when they exist, and say whether they are part of the intended push scope.

## Guardrails
- Do not just restate `git diff --stat`; synthesize what changed and why it matters.
- Do not mix committed scope with unrelated local drafts without saying so.
- Do not claim validation passed unless the command actually ran.
- Keep README-ready bullets concrete enough to paste into docs or a commit summary.
