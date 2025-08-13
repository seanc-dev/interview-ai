# Auto Hypothesis Iteration Plan

## Goal

- Automatically promote or retire hypotheses across runs based on alignment and frequency.

## Behavior

- Track per-hypothesis stats per run: total mentions, aligned count, misaligned count.
- Persist `hypotheses_state.json` in each run directory; optionally aggregate at project root later.
- Decide promotions when alignment ratio >= threshold and frequency >= threshold.
- Decide retirements when misaligned ratio >= threshold and frequency <= max.

## Tests

- Unit: tracker aggregates counts and computes promote/retire lists given sample insights.
- Integration-light: tracker writes JSON structure with expected keys.

## CI

- Marker: `hypothesis_iteration` for targeted runs.
