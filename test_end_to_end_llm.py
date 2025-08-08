#!/usr/bin/env python3
"""
End-to-End LLM Tests with Real API Calls.
Tests the async engine with actual LLM integration.
"""

import pytest
import json
import tempfile
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from typing import Dict, List, Any

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


class TestRealLLMIntegration:
    """Test suite for real LLM integration."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        # Load from .env file
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set in .env file")
        return api_key

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return {
            "project_name": "TestProject",
            "llm_model": "gpt-4o-mini",  # Use cheaper model for testing
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
    async def test_async_persona_generator_with_real_llm(self, api_key, test_config):
        """Test that AsyncPersonaGenerator calls LLM to generate personas."""
        generator = AsyncPersonaGenerator()

        # Generate personas with real LLM
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
    async def test_async_interview_processor_with_real_llm(self, api_key, test_config):
        """Test that AsyncInterviewProcessor calls LLM for interviews."""
        processor = AsyncInterviewProcessor(
            api_key=api_key,
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

        # Create test personas
        personas = [
            {
                "name": "Sarah, 28",
                "emotional_baseline": "overwhelmed",
                "background": "Early career professional struggling with work-life balance",
            },
            {
                "name": "Mike, 35",
                "emotional_baseline": "stressed",
                "background": "Mid-career manager dealing with high pressure and burnout",
            },
        ]

        async with processor:
            results = await processor.process_interviews_concurrently(
                config, mode, hypothesis, personas, "20240101_120000"
            )

        assert len(results) == 2
        for result in results:
            assert result["success"] is True
            assert "insights" in result
            assert len(result["insights"]) > 0
            assert "mode" in result
            assert "hypothesis" in result

    @pytest.mark.asyncio
    async def test_complete_cycle_with_real_llm(self, api_key, test_config):
        """Test complete cycle with real LLM calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            engine = AsyncIterativeResearchEngine(
                api_key=api_key,
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=2,
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "test_real_llm"
            engine.config_dir_path = engine.project_dir / "config"
            engine.runs_dir = engine.project_dir / "runs"
            engine.project_dir.mkdir(parents=True, exist_ok=True)
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

            results = await engine.run_iterative_research_async()

            assert len(results) == 1
            result = results[0]
            assert result["success"] is True
            assert result["insights_count"] > 0
            assert result["personas_generated"] > 0
            assert result["alignment_rate"] >= 0.0

    @pytest.mark.asyncio
    async def test_prompt_generation_quality(self, api_key, test_config):
        """Test that prompts are generated with proper quality and context."""
        processor = AsyncInterviewProcessor(
            api_key=api_key,
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

    @pytest.mark.asyncio
    async def test_response_parsing_quality(self, api_key, test_config):
        """Test parsing of real LLM responses."""
        processor = AsyncInterviewProcessor(
            api_key=api_key,
            max_concurrent=1,
            rate_limit_per_minute=60,
        )

        # Test markdown response parsing
        test_response = """
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

        parsed = processor._parse_markdown_response(test_response)

        assert "insights" in parsed
        assert "Aligned? Yes" in parsed["insights"]
        assert "Pain Points" in parsed["insights"]
        assert "Desired Outcomes" in parsed["insights"]

    @pytest.mark.asyncio
    async def test_insight_aggregation_with_real_data(self, api_key, test_config):
        """Test insight aggregation with real LLM responses."""
        aggregator = AsyncInsightAggregator()

        # Mock insights from real LLM responses
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
    async def test_rate_limiting_and_concurrency(self, api_key, test_config):
        """Test rate limiting and concurrency with real API calls."""
        processor = AsyncInterviewProcessor(
            api_key=api_key,
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
        personas = [
            {
                "name": f"Persona {i}",
                "emotional_baseline": "stressed",
                "background": f"Background {i}",
            }
            for i in range(3)
        ]

        start_time = datetime.now()

        async with processor:
            results = await processor.process_interviews_concurrently(
                config, mode, hypothesis, personas, "20240101_120000"
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should process 3 interviews concurrently (max 2 at a time)
        # Should take less than 3 * individual_time due to concurrency
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert duration < 30  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_error_handling_and_retries(self, api_key, test_config):
        """Test error handling and retries with real API calls."""
        processor = AsyncInterviewProcessor(
            api_key=api_key,
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
            "name": "Test",
            "emotional_baseline": "stressed",
            "background": "Test",
        }

        # Test with invalid model (should handle error gracefully)
        config.llm_model = "invalid-model"

        async with processor:
            result = await processor.process_interview_async(
                config, mode, hypothesis, persona, 1, "20240101_120000"
            )

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_production_ygt_config_with_real_llm(self, api_key):
        """Test with actual YGT config and real LLM calls."""
        # Use the actual YGT config
        config_data = {
            "project_name": "YGT",
            "llm_model": "gpt-4o-mini",  # Use cheaper model for testing
            "product_sketch": "YGT is an emotionally intelligent AI companion designed to support people across all stages of personal wellness — from burnout recovery to high-performance thriving. It adapts to the user's emotional state over time, offering a grounded sense of continuity across four evolving modes: Recovery, Stability, Growth, and Thriving. Rather than offering advice or productivity pressure, YGT provides a psychologically attuned daily rhythm: check-ins, reflections, soft nudges, and deep emotional validation. It helps users track their internal patterns, reconnect to their values, and build resilience at their own pace. Its tone is adaptive — gentle when needed, energized when invited. YGT is not a therapist or coach, but a compassionate guide that helps users stabilize, rebuild, and flourish — without shame, overwhelm, or toxic positivity.",
            "interview_modes": [
                {
                    "mode": "Recovery",
                    "persona_count": 1,  # Reduced for testing
                    "problem_hypotheses": [
                        {
                            "label": "Emotional Overwhelm + Self-Shaming in Burnout",
                            "description": "Users experiencing burnout or trauma often internalize failure and become emotionally overloaded. A psychologically attuned assistant might help them externalize guilt, track emotional patterns, and reframe setbacks — without triggering shame or collapse.",
                        }
                    ],
                }
            ],
            "output_format": "markdown",
            "version": "v1",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "ygt_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            engine = AsyncIterativeResearchEngine(
                api_key=api_key,
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=1,
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "test_ygt_real_llm"
            engine.config_dir_path = engine.project_dir / "config"
            engine.runs_dir = engine.project_dir / "runs"
            engine.project_dir.mkdir(parents=True, exist_ok=True)
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

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

            # Check report content
            with open(master_report, "r") as f:
                report_content = f.read()
                assert "Master Report" in report_content
                assert "Total Insights" in report_content
                assert "Total Personas" in report_content


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
