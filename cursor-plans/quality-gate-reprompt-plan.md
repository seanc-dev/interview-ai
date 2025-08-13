# Quality Gate AB-Reprompt Plan

## Goal

- For low-scoring dimensions, auto re-prompt with targeted guidance and optionally A/B variants; include retry results if better.

## Behavior

- Use `QualityEvaluator` scores per-insight. If any dimension < threshold, create a targeted reprompt with guidance to improve missing items.
- Support `prompt_variant` A/B: `A` (original) vs `B` (explicit bullets + examples). Keep best result by score.
- Persist `reprompt_log.json` in run dir and annotate insights with `re_prompted: true` when used.

## Tests

- Unit: reprompt policy selects candidates and builds prompts.
- Integration-light: improved score replaces original in aggregated insights.

## CI

- Marker: `quality_gate` for targeted runs.
