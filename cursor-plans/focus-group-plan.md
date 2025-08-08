# Focus Group Mode - TDD Plan

## Goal

Enable multi-agent focus group interviews with a single facilitator in two formats:

- Round-the-table: turn-based contributions from each participant
- Open-table: natural, free-flowing conversation with mediated turns

## Outcomes

- Rich group transcript with clear speaker labels
- LLM-first synthesis of group insights, agreements, disagreements, and themes
- Actionable recommendations based on consensus and conflict

## Scope

- New `AsyncFocusGroupEngine` orchestrates sessions
- Uses existing persona generator to produce diverse participants
- Works with current config and hypotheses
- Produces transcript.md, insights.json, and summary.md per session

## Tests (incremental)

1. Unit

- Generate N unique personas (reuse AsyncPersonaGenerator)
- Orchestrate round-table: ensure exactly one turn per participant per round
- Orchestrate open-table: ensure facilitation logic and turn-taking constraints
- Aggregate group insights via LLM

2. Integration

- Run a full focus group session (mock LLM) and assert artifacts
- Snapshot transcript header and structure

3. Edge Cases

- Participant drops out (skip turn)
- Over-talker (facilitator limits turns)
- Conflicting opinions (ensure conflict summary)

## Markers

- pytest -m focus_group

## CI

- PRs: run only -m focus_group
- main merges: run full suite

## Implementation Steps

1. Tests first (fail):

- test_focus_group_mode.py

2. Minimal orchestrator + LLM prompt scaffolds
3. Iterate: turn-taking, transcript shaping, insight aggregation
4. Snapshot tests for transcript structure
5. Docs: add docstrings, usage in README
