#!/usr/bin/env python3
"""
Test Async Engine Structure without API calls.
Verifies the async engine components work correctly.
"""

import pytest
import json
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

# Import the classes we'll be testing
from llm_interview_engine import (
    AsyncIterativeResearchEngine,
    ProjectConfig,
    InterviewMode,
    ProblemHypothesis,
    AsyncPersonaGenerator,
    AsyncInterviewProcessor,
    AsyncInsightAggregator,
)


class TestAsyncEngineStructure:
    """Test suite for async engine structure without API calls."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return {
            "project_name": "TestProject",
            "llm_model": "gpt-4o-mini",
            "product_sketch": "A wellness app that helps users manage stress and build healthy habits.",
            "interview_modes": [
                {
                    "mode": "Recovery",
                    "persona_count": 2,
                    "problem_hypotheses": [
                        {
                            "label": "Stress Management",
                            "description": "Users struggle to manage daily stress and need tools to cope effectively.",
                        }
                    ],
                }
            ],
            "output_format": "markdown",
            "version": "v1",
        }

    @pytest.mark.asyncio
    async def test_async_persona_generator_structure(self, test_config):
        """Test that AsyncPersonaGenerator generates personas correctly."""
        generator = AsyncPersonaGenerator()

        # Generate personas
        personas = await generator.generate_personas_async(count=2, cycle_number=1)

        assert len(personas) == 2
        assert all(isinstance(p, dict) for p in personas)

        # Check that personas have expected structure
        for persona in personas:
            assert "name" in persona
            assert "emotional_baseline" in persona
            assert "background" in persona
            assert isinstance(persona["name"], str)
            assert len(persona["name"]) > 0

    @pytest.mark.asyncio
    async def test_async_interview_processor_prompt_generation(self, test_config):
        """Test that AsyncInterviewProcessor generates prompts correctly."""
        processor = AsyncInterviewProcessor(
            api_key="fake_key",
            max_concurrent=2,
            rate_limit_per_minute=60,
        )

        config = ProjectConfig(
            project_name=test_config["project_name"],
            llm_model=test_config["llm_model"],
            product_sketch=test_config["product_sketch"],
            interview_modes=[
                InterviewMode(
                    mode=mode["mode"],
                    persona_count=mode["persona_count"],
                    problem_hypotheses=[
                        ProblemHypothesis(
                            label=h["label"], description=h["description"]
                        )
                        for h in mode["problem_hypotheses"]
                    ],
                )
                for mode in test_config["interview_modes"]
            ],
            output_format=test_config["output_format"],
            version=test_config["version"],
        )

        mode = config.interview_modes[0]
        hypothesis = mode.problem_hypotheses[0]
        persona = {
            "name": "Test Persona",
            "emotional_baseline": "stressed",
            "background": "Test background",
            "coping_style": "support-seeking",
            "readiness_level": "ready",
        }

        prompt = processor._generate_interview_prompt_async(
            config, mode, hypothesis, persona, 1, "20240101_120000"
        )

        # Check prompt quality
        assert len(prompt) > 100
        assert config.product_sketch in prompt
        assert hypothesis.label in prompt
        assert hypothesis.description in prompt
        assert persona["name"] in prompt
        assert "INTERNAL CONTEXT" in prompt
        assert "ASSIGNMENT" in prompt
        assert "INTERVIEW INSTRUCTIONS" in prompt

    @pytest.mark.asyncio
    async def test_async_insight_aggregator_structure(self, test_config):
        """Test that AsyncInsightAggregator processes insights correctly."""
        aggregator = AsyncInsightAggregator()

        # Mock insights
        insights = [
            {
                "mode": "Recovery",
                "hypothesis": "Stress Management",
                "insights": "Aligned? Yes\nPain Points: - Too many notifications\n- App feels overwhelming\nDesired Outcomes: - Simpler interface\n- More actionable advice",
                "alignment": "aligned",
            },
            {
                "mode": "Recovery",
                "hypothesis": "Stress Management",
                "insights": "Aligned? No\nPain Points: - Not enough guidance\n- Too generic\nDesired Outcomes: - More personalization\n- Better onboarding",
                "alignment": "misaligned",
            },
        ]

        aggregated = await aggregator.aggregate_insights_async(insights)

        assert "alignment_rate" in aggregated
        assert "total_insights" in aggregated
        assert aggregated["total_insights"] == 2
        assert 0.0 <= aggregated["alignment_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_complete_async_engine_structure(self, test_config):
        """Test complete async engine structure without API calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            engine = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=2,
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "test_async_structure"
            engine.config_dir_path = engine.project_dir / "config"
            engine.runs_dir = engine.project_dir / "runs"
            engine.project_dir.mkdir(parents=True, exist_ok=True)
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

            # Mock the API calls
            with patch.object(
                AsyncInterviewProcessor, "_call_openai_async"
            ) as mock_api:
                mock_api.return_value = """
# Interview Results

## Alignment
Aligned? Yes

## Key Insights
- User finds stress management tools helpful
- Daily check-ins are valuable
- Would like more personalized recommendations

## Pain Points
- Too many notifications
- App feels overwhelming at times
- Needs better progress tracking

## Desired Outcomes
- Simpler interface
- More actionable advice
- Better integration with daily routine
"""

                results = await engine.run_iterative_research_async()

                assert len(results) == 1
                result = results[0]
                assert result["success"] is True
                assert result["insights_count"] > 0
                assert result["personas_generated"] > 0
                assert result["alignment_rate"] >= 0.0

    @pytest.mark.asyncio
    async def test_async_engine_with_mocked_llm(self, test_config):
        """Test async engine with mocked LLM responses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            engine = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=1,
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "test_mocked_llm"
            engine.config_dir_path = engine.project_dir / "config"
            engine.runs_dir = engine.project_dir / "runs"
            engine.project_dir.mkdir(parents=True, exist_ok=True)
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

            # Mock the API calls with realistic responses
            with patch.object(
                AsyncInterviewProcessor, "_call_openai_async"
            ) as mock_api:
                mock_responses = [
                    # Persona generation response
                    '{"name": "Sarah, 28", "emotional_baseline": "overwhelmed", "background": "Early career professional struggling with work-life balance", "coping_style": "support-seeking", "readiness_level": "ready"}',
                    # Interview response
                    """
# Interview Results

## Alignment
Aligned? Yes

## Key Insights
- User finds stress management tools helpful
- Daily check-ins are valuable
- Would like more personalized recommendations

## Pain Points
- Too many notifications
- App feels overwhelming at times
- Needs better progress tracking

## Desired Outcomes
- Simpler interface
- More actionable advice
- Better integration with daily routine

## Interview Notes
Sarah expressed feeling overwhelmed by her current workload and found the concept of daily wellness check-ins appealing. She mentioned struggling with work-life balance and would appreciate tools that help her set boundaries and manage stress more effectively.
""",
                ]

                mock_api.side_effect = mock_responses

                results = await engine.run_iterative_research_async()

                assert len(results) == 1
                result = results[0]
                assert result["success"] is True
                assert result["insights_count"] > 0
                assert result["personas_generated"] > 0

                # Check that reports were generated
                run_dirs = list(engine.runs_dir.glob("run_*"))
                assert len(run_dirs) > 0

                latest_run = run_dirs[-1]
                master_report = latest_run / "master_report.md"
                roadmap = latest_run / "roadmap.md"

                assert master_report.exists()
                assert roadmap.exists()

    def test_prompt_generation_quality_detailed(self, test_config):
        """Test detailed prompt generation quality."""
        processor = AsyncInterviewProcessor(
            api_key="fake_key",
            max_concurrent=1,
            rate_limit_per_minute=60,
        )

        config = ProjectConfig(
            project_name=test_config["project_name"],
            llm_model=test_config["llm_model"],
            product_sketch=test_config["product_sketch"],
            interview_modes=[
                InterviewMode(
                    mode=mode["mode"],
                    persona_count=mode["persona_count"],
                    problem_hypotheses=[
                        ProblemHypothesis(
                            label=h["label"], description=h["description"]
                        )
                        for h in mode["problem_hypotheses"]
                    ],
                )
                for mode in test_config["interview_modes"]
            ],
            output_format=test_config["output_format"],
            version=test_config["version"],
        )

        mode = config.interview_modes[0]
        hypothesis = mode.problem_hypotheses[0]
        persona = {
            "name": "Test Persona",
            "emotional_baseline": "stressed",
            "background": "Test background",
            "coping_style": "support-seeking",
            "readiness_level": "ready",
        }

        prompt = processor._generate_interview_prompt_async(
            config, mode, hypothesis, persona, 1, "20240101_120000"
        )

        # Detailed prompt quality checks
        assert "INTERNAL CONTEXT" in prompt
        assert "Product sketch" in prompt
        assert config.product_sketch in prompt
        assert "ASSIGNMENT" in prompt
        assert config.project_name in prompt
        assert mode.mode in prompt
        assert hypothesis.label in prompt
        assert hypothesis.description in prompt
        assert "PERSONA CONTEXT" in prompt
        assert persona["name"] in prompt
        assert "INTERVIEW INSTRUCTIONS" in prompt
        assert "Current State" in prompt
        assert "Challenges" in prompt
        assert "Coping Mechanisms" in prompt
        assert "Desired Outcomes" in prompt
        assert "Barriers" in prompt
        assert "Support Needs" in prompt
        assert "OUTPUT FORMAT" in prompt
        assert "Interview Results" in prompt
        assert "Alignment" in prompt
        assert "Key Insights" in prompt
        assert "Pain Points" in prompt
        assert "Desired Outcomes" in prompt


def cleanup_test_outputs():
    """Clean up test outputs directory after tests."""
    import shutil

    test_outputs = Path("test_outputs")
    if test_outputs.exists():
        shutil.rmtree(test_outputs)


if __name__ == "__main__":
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    finally:
        cleanup_test_outputs()
