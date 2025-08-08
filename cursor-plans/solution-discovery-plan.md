# Solution Discovery Mode - TDD Plan

## Goal

Enable guided exploration of solution directions informed by user insights. The mode should:

- Synthesize candidate solution concepts from insights
- Explore tradeoffs, risks, constraints
- Propose MVP scope, metrics, and validation plans
- Maintain conversational, LLM-first analyses and outputs

## Modes

- Concept generation: divergent brainstorming
- Concept refinement: converge toward MVP
- Critique round: identify risks and mitigations

## Artifacts

- `solution_briefs/` per run:
  - `concepts.md` (top N concepts with rationales)
  - `mvp_proposal.md` (scope, assumptions, risks, metrics)
  - `validation_plan.md` (experiments, success criteria)

## Tests (incremental)

1. Unit

- Generate N concepts from insights (mock LLM)
- Prioritize by value/feasibility
- Produce MVP with clear scope and metrics

2. Integration

- End-to-end solution discovery session from sample insights
- Snapshot MVP proposal header/sections

3. Edge Cases

- Sparse insights → graceful defaults
- Conflicting insights → address with branching options

## Markers

- pytest -m solution_discovery

## CI

- PRs: run only -m solution_discovery
- main merges: run full suite

## Implementation Steps

1. Tests first (fail):

- test_solution_discovery_mode.py

2. Minimal `AsyncSolutionDiscoveryEngine` with LLM prompts
3. Iterate: prioritization logic, MVP extraction, validation plan
4. Snapshot tests for MVP doc structure
5. Docs: add docstrings, usage in README
