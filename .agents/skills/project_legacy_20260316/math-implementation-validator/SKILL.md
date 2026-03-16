---
name: math-implementation-validator
description: Validate math-heavy implementations by reading source code, translating code paths into explicit formulas, auditing operators and tensor semantics, and designing focused numerical tests. Use when Codex needs to verify gradients, Jacobians, Hessians, PDE or discrete differential operators, loss derivations, update rules, tensor contractions, sign or axis conventions, or code-vs-formula consistency before expert review.
---

# Math Implementation Validator

Turn mathematical code review into a short evidence loop: map code to formulas, isolate the operator semantics, build a minimal oracle, then implement narrow tests that catch the most plausible failure modes.

## Workflow
1. Define the mathematical surface:
   - identify the entrypoints, helper functions, and configuration flags that change the math;
   - collect adjacent formulas from docstrings, markdown, comments, papers, or issue context;
   - write down assumptions explicitly when the intended formula is only implicit in code.
2. Translate code into formulas:
   - restate each intermediate in mathematical notation while preserving indices, axes, reductions, and broadcasting;
   - name the state, parameter, dtype, and shape of each tensor or array;
   - collapse branching logic into piecewise formulas when branches change the operator.
3. Audit the operator:
   - read [the operator checklist](references/operator-audit-checklist.md) for gradients, Jacobians, Hessians, divergences, curls, Laplacians, convolutions, normalizations, or time-steppers;
   - check sign conventions, transpose or index order, boundary handling, epsilon placement, detach or stop-gradient behavior, and units or scale factors;
   - compare the implemented discrete operator against the intended continuous or discrete form, not just against variable names.
4. Choose the smallest convincing oracle:
   - prefer a closed-form case when one exists;
   - otherwise use finite differences, complex-step checks, symmetry or invariance checks, conservation laws, manufactured solutions, or autograd parity;
   - read [the test patterns reference](references/test-patterns.md) when the best oracle is unclear.
5. Implement focused tests:
   - start with one deterministic fixture that exposes the operator clearly;
   - add one perturbation or randomized property test only if it checks a different failure mode;
   - keep tolerances explicit and justify them from conditioning, discretization error, or dtype limits.
6. Report for expert review:
   - separate confirmed matches, suspected mismatches, and assumptions that remain unproven;
   - include the code-to-formula translation, operator-audit findings, chosen oracle, and residual risk;
   - if tests were not run, say exactly what remains unverified and what fixture should be added next.

## Preferred Outputs
- A short code-to-math translation table or markdown section.
- A mismatch list with file or function references and the specific mathematical claim at risk.
- A test plan or implemented tests that distinguish algebraic bugs from tolerance or conditioning issues.
- An expert-facing note on what is proved, what is only numerically supported, and what still depends on domain judgment.

## Guardrails
- Do not claim a mathematical proof from passing numerical tests alone.
- Do not stop at "the shapes match"; verify axes, reductions, broadcasting, and semantics.
- Do not compare autograd against itself when an independent oracle is available.
- Do not broaden into full refactors before pinning down the target operator and failure mode.
- Prefer the narrowest reproducible fixture over large end-to-end pipelines.
- Preserve unrelated user changes; add focused tests before rewriting unrelated implementation layers.

## Pair With Other Skills
- Pair with `vibe-coding` when the review should end with code changes or new tests in this repository.
- Pair with `readme-change-syncer` when the validation workflow changes documented entrypoints or contributor guidance.
- Pair with theory or experiment skills when the question is about research interpretation rather than implementation correctness.
