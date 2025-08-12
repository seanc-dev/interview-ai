# Quality Loop Plan (LLM-as-Judge)

## Goals

- Automated, objective evaluation of interview outputs and reports
- Scores with rationales for: alignment, pain points relevance, outcomes specificity, micro-feature actionability, roadmap prioritization, formatting
- JSON/Markdown artifacts; CI gating via thresholds

## Scope (MVP)

- LLMEvaluator with rubric (heuristic fallback + optional LLM judge)
- Evaluate per-insight and per-run
- Write `quality_report.md` + `quality.json` to run dir
- Append Quality Assessment to run master report

## Tests

- Unit: heuristic scoring from sample insight
- Unit: aggregate run scores + pass/fail gate
- Integration-light: ensure Quality Assessment text is appended (string contains)

## Future

- CI workflow job by tag `quality`
- Dashboard HTML, trend charts
