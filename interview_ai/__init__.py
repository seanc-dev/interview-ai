"""Interview AI package

This package provides modular building blocks for the interview engine.
For backward compatibility, the legacy entry remains available as `llm_interview_engine.py`.
"""

# During refactor, re-export from legacy module for engines and LLM utilities
from ..llm_interview_engine import (
    AsyncIterativeResearchEngine,
    AsyncPersonaGenerator,
    AsyncInterviewProcessor,
    AsyncInsightAggregator,
    AsyncFocusGroupEngine,
    AsyncSolutionDiscoveryEngine,
    LLMInterviewEngine,
    LLMInsightAnalyzer,
    LLMReportGenerator,
    LLMRoadmapGenerator,
    LLMProductEvolution,
    QualityGateReprompter,
)

# New modular exports
from .datamodel import ProjectConfig, InterviewMode, ProblemHypothesis
from .quality import QualityEvaluator
from .synthesis import CrossFunctionalSynthesizer, HypothesisStateTracker

__all__ = [
    "ProjectConfig",
    "InterviewMode",
    "ProblemHypothesis",
    "AsyncIterativeResearchEngine",
    "AsyncPersonaGenerator",
    "AsyncInterviewProcessor",
    "AsyncInsightAggregator",
    "AsyncFocusGroupEngine",
    "AsyncSolutionDiscoveryEngine",
    "LLMInterviewEngine",
    "LLMInsightAnalyzer",
    "LLMReportGenerator",
    "LLMRoadmapGenerator",
    "LLMProductEvolution",
    "QualityEvaluator",
    "HypothesisStateTracker",
    "CrossFunctionalSynthesizer",
    "QualityGateReprompter",
]
