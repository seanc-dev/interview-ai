### Architecture Overview

This document summarizes the high-level structure and flow.

Core flows:

- Async Iterative Research: `AsyncIterativeResearchEngine.run_iterative_research_async()`
  - Persona generation → interviews → aggregation → report/roadmap → (optional) evolution.
- Report/roadmap: `LLMReportGenerator`, `LLMRoadmapGenerator` (LLM-first with fallbacks).
- Quality: `QualityEvaluator` for heuristic scoring; appended to master report.
- Extras: `HypothesisStateTracker`, `CrossFunctionalSynthesizer`, `QualityGateReprompter`.

Artifacts per run (under `outputs/<Project>/runs/run_<ts>/`):

- `master_report.md`, `roadmap.md`, `metrics.json`, `insights.json`
- `general_insights.{json,md}`, `hypotheses_state.json`

Refactor plan:

- Keep `llm_interview_engine.py` as compatibility entry while migrating modules into `interview_ai/` package.
