---
name: github-publisher
description: Prepare DynamicAlpha Lab changes for GitHub by checking branch and remote state, running minimal validation, creating a clean non-interactive commit, and pushing safely. Use when the user asks to upload, publish, sync, or push the current branch after local code, documentation, or skill changes.
---

# GitHub Publisher

Publish the current branch cleanly without force-pushes, accidental scope creep, or undocumented leftovers.

## Workflow
1. Inspect `git status --short`, `git branch --show-current`, and `git remote -v`.
2. Decide the intended push scope before staging anything.
3. Run the smallest validation that matches the changed areas.
4. Stage only the files that belong in the publish set.
5. Commit with one imperative summary line.
6. Push with a non-interactive command such as `git push origin <branch>`.
7. Report the branch, remote, commit hash, validation, and any files left unpushed.

## Validation Policy
- Prefer `python -m compileall src scripts` for broad Python changes.
- Add focused CLI or smoke runs when scripts, research loop logic, or output contracts changed.
- If validation is skipped or partial, say so before pushing.

## Guardrails
- Do not use `--force` or `--force-with-lease` unless the user explicitly asks.
- Do not amend an existing commit unless the user explicitly asks.
- Do not silently include unrelated draft files or unfinished skill stubs.
- Prefer one coherent commit over many tiny commits for a single upload request.
