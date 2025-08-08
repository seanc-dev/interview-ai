import asyncio
import time
import tempfile
from datetime import datetime
from pathlib import Path
import json
import pytest
from unittest.mock import patch, AsyncMock

from llm_interview_engine import (
    AsyncIterativeResearchEngine,
    AsyncPersonaGenerator,
    AsyncInterviewProcessor,
    ProjectConfig,
    InterviewMode,
    ProblemHypothesis,
)


class TestTimingAnalysis:
    """Test timing analysis for async engine bottlenecks."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return {
            "project_name": "TimingTest",
            "llm_model": "gpt-4o-mini",
            "product_sketch": "A wellness app for stress management",
            "interview_modes": [
                {
                    "mode": "trauma_informed",
                    "persona_count": 3,
                    "problem_hypotheses": [
                        {
                            "label": "Stress Management",
                            "description": "Users struggle to manage daily stress and need tools to cope effectively.",
                        },
                        {
                            "label": "Emotional Regulation",
                            "description": "Users need help regulating emotions during high-stress periods.",
                        },
                    ],
                },
                {
                    "mode": "cognitive_behavioral",
                    "persona_count": 2,
                    "problem_hypotheses": [
                        {
                            "label": "Thought Patterns",
                            "description": "Users have negative thought patterns that affect their wellness.",
                        },
                    ],
                },
            ],
            "output_format": "markdown",
            "version": "v1",
        }

    async def test_persona_generation_timing(self, test_config):
        """Test timing of persona generation."""
        print("\n=== PERSONA GENERATION TIMING ===")

        generator = AsyncPersonaGenerator(api_key="fake_key")

        # Test with different counts
        for count in [3, 5, 10]:
            start_time = time.time()
            personas = await generator.generate_personas_async(count, 1)
            end_time = time.time()

            duration = end_time - start_time
            print(
                f"Generated {count} personas in {duration:.2f}s ({duration/count:.2f}s per persona)"
            )

            assert len(personas) == count

    async def test_interview_processing_timing(self, test_config):
        """Test timing of interview processing."""
        print("\n=== INTERVIEW PROCESSING TIMING ===")

        # Create test personas
        personas = [
            {
                "name": f"Persona {i}",
                "emotional_baseline": "stressed",
                "background": f"Background {i}",
            }
            for i in range(5)
        ]

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

        # Mock the API call to measure timing without real API calls
        with patch.object(AsyncInterviewProcessor, "_call_openai_async") as mock_api:
            mock_api.return_value = """
# Interview Results

## Alignment
Aligned? Yes

## Key Insights
- Users need better stress management tools
- Emotional regulation is a key concern
- Thought patterns affect wellness outcomes

## Pain Points
- Difficulty managing daily stress
- Lack of effective coping mechanisms
- Negative thought patterns

## Desired Outcomes
- Better stress management tools
- Improved emotional regulation
- Positive thought pattern changes

## Interview Notes
The persona expressed feeling overwhelmed by daily stressors and would benefit from tools that help manage stress and regulate emotions.
"""

            processor = AsyncInterviewProcessor(api_key="fake_key", max_concurrent=5)

            # Test different concurrency levels
            for max_concurrent in [1, 3, 5]:
                processor.max_concurrent = max_concurrent

                start_time = time.time()
                async with processor:
                    results = await processor.process_interviews_concurrently(
                        config, mode, hypothesis, personas[:3], "20240101_120000"
                    )
                end_time = time.time()

                duration = end_time - start_time
                print(
                    f"Processed 3 interviews with max_concurrent={max_concurrent} in {duration:.2f}s ({duration/3:.2f}s per interview)"
                )

                assert len(results) == 3

    async def test_cycle_timing_breakdown(self, test_config):
        """Test timing breakdown of a complete cycle."""
        print("\n=== CYCLE TIMING BREAKDOWN ===")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            engine = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=5,
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "timing_test"
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

            # Mock API calls
            with patch.object(
                AsyncInterviewProcessor, "_call_openai_async"
            ) as mock_api:
                mock_api.return_value = """
# Interview Results

## Alignment
Aligned? Yes

## Key Insights
- Users need better stress management tools
- Emotional regulation is a key concern

## Pain Points
- Difficulty managing daily stress
- Lack of effective coping mechanisms

## Desired Outcomes
- Better stress management tools
- Improved emotional regulation

## Interview Notes
The persona expressed feeling overwhelmed by daily stressors.
"""

                start_time = time.time()
                results = await engine.run_iterative_research_async()
                end_time = time.time()

                total_duration = end_time - start_time
                print(f"Complete cycle took {total_duration:.2f}s")

                assert len(results) == 1

    async def test_concurrent_vs_sequential_timing(self, test_config):
        """Compare concurrent vs sequential processing timing."""
        print("\n=== CONCURRENT VS SEQUENTIAL TIMING ===")

        # Test data
        personas = [
            {
                "name": f"Persona {i}",
                "emotional_baseline": "stressed",
                "background": f"Background {i}",
            }
            for i in range(10)
        ]

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

        # Mock API call with realistic timing
        async def mock_api_call(prompt, model):
            await asyncio.sleep(2)  # Simulate 2-second API call
            return """
# Interview Results

## Alignment
Aligned? Yes

## Key Insights
- Users need better stress management tools

## Pain Points
- Difficulty managing daily stress

## Desired Outcomes
- Better stress management tools

## Interview Notes
The persona expressed feeling overwhelmed by daily stressors.
"""

        with patch.object(
            AsyncInterviewProcessor, "_call_openai_async", side_effect=mock_api_call
        ):
            processor = AsyncInterviewProcessor(api_key="fake_key", max_concurrent=5)

            # Test concurrent processing
            start_time = time.time()
            async with processor:
                results = await processor.process_interviews_concurrently(
                    config,
                    config.interview_modes[0],
                    config.interview_modes[0].problem_hypotheses[0],
                    personas[:5],
                    "20240101_120000",
                )
            concurrent_duration = time.time() - start_time

            print(f"Concurrent processing of 5 interviews: {concurrent_duration:.2f}s")
            print(f"Expected sequential time: {5 * 2:.2f}s")
            print(f"Speedup: {(5 * 2) / concurrent_duration:.2f}x")

            assert len(results) == 5
            assert concurrent_duration < 5 * 2  # Should be faster than sequential


if __name__ == "__main__":

    async def run_timing_tests():
        """Run all timing tests."""
        test = TestTimingAnalysis()
        test_config = {
            "project_name": "TimingTest",
            "llm_model": "gpt-4o-mini",
            "product_sketch": "A wellness app for stress management",
            "interview_modes": [
                {
                    "mode": "trauma_informed",
                    "persona_count": 3,
                    "problem_hypotheses": [
                        {
                            "label": "Stress Management",
                            "description": "Users struggle to manage daily stress and need tools to cope effectively.",
                        },
                        {
                            "label": "Emotional Regulation",
                            "description": "Users need help regulating emotions during high-stress periods.",
                        },
                    ],
                },
                {
                    "mode": "cognitive_behavioral",
                    "persona_count": 2,
                    "problem_hypotheses": [
                        {
                            "label": "Thought Patterns",
                            "description": "Users have negative thought patterns that affect their wellness.",
                        },
                    ],
                },
            ],
            "output_format": "markdown",
            "version": "v1",
        }

        # Run tests
        await test.test_persona_generation_timing(test_config)
        await test.test_interview_processing_timing(test_config)
        await test.test_cycle_timing_breakdown(test_config)
        await test.test_concurrent_vs_sequential_timing(test_config)

        print("\n=== TIMING ANALYSIS COMPLETE ===")

    asyncio.run(run_timing_tests())
