---
name: readme-change-syncer
description: Update DynamicAlpha Lab folder READMEs and the root README after code, script, output-contract, or skill changes. Use when new modules, CLI flags, generated artifacts, maintenance workflows, or project-local skills change what future contributors need to read first, and the docs should stay aligned with the latest implementation.
---

# README Change Syncer

Sync documentation from the leaf directories upward so the nearest README stays the most concrete and the root README stays the best project entrypoint.

## Workflow
1. Inspect the current diff and determine which directories actually changed.
2. Update the nearest README first:
   - `scripts/README.md` for CLI and maintenance scripts
   - `src/README.md` for implementation-layer shifts
   - `src/fsrc_sindy/README.md` for package/module changes
3. Promote only the high-level implications into `README.md`.
4. Refresh command examples, artifact names, and skill lists when they changed.
5. Keep doc edits additive when legacy multilingual content or encoding is fragile.

## What To Document
- New modules, classes, or registries that future maintainers should discover quickly.
- New output artifacts, reports, manifests, or gate files.
- New maintenance scripts or project-local skills.
- Changes in the recommended read order or extension points.

## Guardrails
- Do not rewrite large README sections when a short update block is enough.
- Do not document speculative behavior that is not in the diff.
- Keep examples runnable from the repository root.
- Prefer the narrowest README set that matches the changed code surface, then update the root README with only the distilled summary.
