---
name: thread-skill-maintainer
description: Summarize current or recently persisted conversation threads, distill reusable user feedback, and decide whether to update an existing skill, create a draft skill, or merge overlapping skills. Use when Codex needs to turn recurring thread guidance into maintainable project-local skills, keep temporary skills out of the active skill trees, or check compatibility across `.agents` and `.claude` skill directories before promotion.
---

# Thread Skill Maintainer

Use this skill when conversation history should become a durable skill change rather than a one-off reply.

## Workflow
1. Gather inputs in this order:
   - the current thread, if the runtime exposes it;
   - the newest persisted memo or feedback note under `archive/skill_evolution/`;
   - the latest direct user feedback in the active request.
2. Reduce the inputs to three buckets:
   - repeated requests or corrections;
   - durable constraints that should survive across sessions;
   - one-off context that should stay out of permanent skills.
3. Run `python scripts/skill_inventory_report.py` before proposing any new or updated skill.
4. Use the system `skill-creator` skill when creating a new permanent skill or restructuring an existing one.
5. Read [the lifecycle policy](references/lifecycle-policy.md) before choosing between update, merge, draft, or no-op.
6. Prefer updating an existing skill when the requested behavior extends an established workflow more than it creates a new domain.
7. Write unstable or single-session ideas as draft skills under `archive/skill_drafts/YYYY-MM-DD/<skill-name>/` so they do not trigger implicitly.
8. Promote a skill into the active trees only when the behavior is reusable, stable, and specific enough to test.
9. When promoting, mirror the skill under both `.agents/skills/project/<skill-name>/` and `.claude/skills/project/<skill-name>/`.
10. Keep `SKILL.md` and `agents/openai.yaml` aligned, and place shared helper code under `scripts/` instead of duplicating it across mirrors.
11. Validate both trees with:
    - `python .agents/skills/upstream/dynamics-research-skills/validate_skills.py .agents/skills/project`
    - `python .agents/skills/upstream/dynamics-research-skills/validate_skills.py .claude/skills/project`
12. Write the session outcome with [the daily report template](references/daily-report-template.md) under `archive/skill_evolution/YYYY-MM-DD/`.

## Guardrails
- Do not assume a scheduled automation can read thread history unless the runtime actually exposes it; fall back to persisted notes and say so explicitly.
- Do not create a new permanent skill when a patch or merge into an existing skill is sufficient.
- Do not place temporary or speculative skills in active skill directories.
- Do not reference scripts, files, or commands that do not exist in the current workspace.
- Preserve compatibility with the current research code; prefer additive skill-layer changes over edits to `src/` or research CLI entry points unless the user explicitly asks for code changes.
