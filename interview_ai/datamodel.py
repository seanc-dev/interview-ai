from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProblemHypothesis:
    """
    Represents a problem hypothesis to be tested in LLM interviews.

    Attributes:
        label (str): A short identifier for the hypothesis
        description (str): Detailed description of the problem hypothesis
    """

    label: str
    description: str


@dataclass
class InterviewMode:
    """
    Represents an interview mode with its parameters and hypotheses.

    Attributes:
        mode (str): The interview mode identifier (e.g., "trauma_informed", "cognitive_behavioral")
        persona_count (int): Number of contrasting personas to generate (default: 3)
        problem_hypotheses (List[ProblemHypothesis]): List of hypotheses to test in this mode
    """

    mode: str
    persona_count: int = 3
    problem_hypotheses: List[ProblemHypothesis] = None

    def __post_init__(self):
        if self.problem_hypotheses is None:
            self.problem_hypotheses = []


@dataclass
class ProjectConfig:
    """
    Configuration for an LLM interview project.

    Attributes:
        project_name (str): Unique name for the project
        llm_model (str): OpenAI model to use for interviews (default: "gpt-5-mini")
        product_sketch (str): Internal product description (not shared with personas)
        interview_modes (List[InterviewMode]): List of interview modes and their configurations
        output_format (str): Output format for interviews ("markdown" or "json")
        version (str): Version identifier for this configuration
    """

    project_name: str
    llm_model: str = "gpt-5-mini"
    product_sketch: str = ""
    interview_modes: List[InterviewMode] = None
    output_format: str = "markdown"
    version: str = "v1"
    max_tokens: int = 2000
    temperature: float = 0.7
    enable_jsonl_logging: bool = False
    prompt_variant: Optional[str] = None

    def __post_init__(self):
        if self.interview_modes is None:
            self.interview_modes = []


__all__ = [
    "ProblemHypothesis",
    "InterviewMode",
    "ProjectConfig",
]
