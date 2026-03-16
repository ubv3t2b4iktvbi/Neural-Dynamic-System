---
name: parallel-workflow-planner
description: Plan and govern human-in-the-loop parallel execution before work begins. Use when multi-thread, multi-agent, or multi-branch work may conflict on shared files, registries, interfaces, or unclear requirements, and Codex must first clarify scope, map a dependency tree, isolate conflict zones, define serial versus parallel batches, and wait for human approval before execution.
---

# Parallel Workflow Planner

Treat parallelism as a staged design problem, not a default speed optimization. Use this skill to front-load alignment, dependency ordering, and merge safety before any worker starts changing files.

## Workflow
1. Restate the requested outcome, non-goals, and any ambiguity that could invalidate parallel work.
2. Inventory the likely touch surface:
   - shared files or directories
   - shared registries, configs, schemas, or CLI entrypoints
   - generated artifacts or tests that depend on fixed ordering
3. Build a dependency tree before proposing workers:
   - `Layer 0`: requirements, constraints, and approval bar
   - `Layer 1`: shared prerequisites or interfaces that must land first
   - `Layer 2`: parallel-safe leaves once the prerequisites are stable
   - `Layer 3`: integration, regression checks, and documentation sync
4. Mark each node as `serial`, `parallel-safe`, or `needs-human-gate`.
5. Surface the conflict map:
   - file overlap
   - interface coupling
   - merge-order risk
   - validation coupling
6. Present the plan to the human before execution. Use `references/plan-template.md` when the structure would help.
7. Execute only the batches that became approved and whose prerequisites are complete.
8. Re-check the dependency tree after each completed layer before opening the next batch.
9. Finish with an integration pass that consolidates outputs, validation, and remaining risk.

## Human Checkpoints
- Pause before execution if requirements are still ambiguous, ownership overlaps, or two threads would edit the same file or registry.
- Ask for approval when changing the dependency ordering would alter scope, merge cost, or rollback difficulty.
- Prefer one short targeted question over a broad brainstorming prompt.

## Guardrails
- Do not parallelize tasks only because they look independent at the feature level; check shared files, generated outputs, registries, and docs first.
- Do not let two workers edit the same file, manifest, config, or public interface unless the user explicitly accepts that merge risk.
- Do not execute downstream tasks while an upstream interface or dependency contract is still unsettled.
- Do not skip the final integration and validation step, even if each worker completed local checks.
- Prefer serial execution when the coordination overhead or conflict risk outweighs the expected speedup.

## Pair With Other Skills
- Use this skill first when the user wants discussion, dependency planning, or staged approval before work starts.
- Pair it with `vibe-coding` or a research skill only after the plan and dependency layers are approved.
