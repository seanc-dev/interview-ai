# Cross-Functional Insights Synthesis Plan

## Goal

- Generate cross-functional insights (marketing, sales, strategy, GTM) each run and integrate into master report.

## Behavior

- Synthesize a `general_insights.md` and a structured `general_insights.json` with sections for Marketing, Sales, Strategy, GTM.
- Append a "Cross-Functional Insights" section to the per-run master report and integrated master report.

## Tests

- Unit: synthesizer formats output with expected keys.
- Integration-light: per-run master report contains section header and bullets for each function.

## CI

- Marker: `cross_insights` for targeted runs.
