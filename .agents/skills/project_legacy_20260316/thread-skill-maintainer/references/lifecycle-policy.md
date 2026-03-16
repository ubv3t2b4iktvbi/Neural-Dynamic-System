# Skill Lifecycle Policy

## Decision Order
1. Reuse an existing skill when the user feedback mainly changes an established workflow, guardrail, or output contract.
2. Merge into an existing skill when the candidate overlaps on trigger, domain, and expected artifacts with another active skill.
3. Draft a temporary skill when the pattern is promising but only appears once, depends on unsettled code, or still lacks a stable trigger phrase.
4. Promote a new permanent skill only when the behavior is reusable across sessions and can be validated locally.

## Active And Draft Locations
- Active project skills: `.agents/skills/project/<skill-name>/`
- Compatibility mirror: `.claude/skills/project/<skill-name>/`
- Draft-only skills: `archive/skill_drafts/YYYY-MM-DD/<skill-name>/`
- Daily governance notes: `archive/skill_evolution/YYYY-MM-DD/`

## Merge Checklist
- Keep the older stable skill name unless the new name materially improves triggering.
- Merge trigger language in frontmatter before expanding the body.
- Regenerate or update `agents/openai.yaml` when the skill scope changes.
- Move shared helper scripts to top-level `scripts/` when both mirrors need them.
- Preserve any existing reference files that still explain non-obvious workflows.

## Promotion Checklist
- Confirm the behavior appeared in repeated user feedback or clearly generalizes beyond one thread.
- Confirm every referenced command and path exists in the workspace.
- Validate `.agents/skills/project` and `.claude/skills/project` with the local validator.
- Run `python scripts/skill_inventory_report.py` and review mirror drift before promotion.

## Draft Checklist
- Keep the draft outside active skill trees.
- Capture what evidence is still missing before promotion.
- Prefer concise SKILL bodies and only add scripts or references that are needed for the next validation pass.
