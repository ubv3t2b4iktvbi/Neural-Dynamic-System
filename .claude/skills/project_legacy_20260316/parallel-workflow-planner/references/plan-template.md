# Parallel Workflow Plan Template

Use this template when presenting a human-gated parallel plan before execution.

## Goal
- Requested outcome:
- Out of scope:
- Assumptions to confirm:

## Dependency Tree
- Layer 0:
- Layer 1:
- Layer 2:
- Layer 3:

For each node, mark one of:
- `serial`
- `parallel-safe`
- `needs-human-gate`

## Conflict Map
- Shared files or directories:
- Shared interfaces, registries, or configs:
- Merge hotspots:
- Validation hotspots:

## Proposed Batches
- Batch A (serial first):
- Batch B (parallel after A):
- Batch C (integration and verification):

## Human Gates
- Gate 1:
- Gate 2:
- Gate 3:

## Exit Criteria
- Required checks:
- Docs to sync:
- Rollback point:
