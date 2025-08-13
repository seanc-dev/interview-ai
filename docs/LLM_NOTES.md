### LLM Notes (for agent use)

Prompts/contracts:

- Interview output format requires sections with Aligned?, Pain Points, Desired Outcomes, Micro-feature Suggestions.
- Report generator consumes normalized insight dicts: `{aligned, pain_points, desired_outcomes, ...}` or raw markdown via `_prepare_insights_summary`.

Judging rubric (heuristic):

- Alignment: derives from presence of "Aligned? Yes/No".
- Pain points/outcomes: counts list bullets near respective headers.
- Micro-features: counts bullets after "Micro-feature Suggestions:".

Quality gate:

- If any dimension < thresholds, build targeted reprompt including explicit requirements and original output.

Cross-functional synthesis:

- Bucket bullet lines into Marketing/Sales/Strategy/GTM in order; LLM-backed clustering can replace heuristic later.
