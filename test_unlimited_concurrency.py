import asyncio
import time
import pytest
from datetime import datetime
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

from llm_interview_engine import (
    AsyncIterativeResearchEngine,
    AsyncInterviewProcessor,
    ProjectConfig,
    InterviewMode,
    ProblemHypothesis,
)


class TestUnlimitedConcurrency:
    """Test unlimited concurrency performance."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        return {
            "project_name": "ConcurrencyTest",
            "llm_model": "gpt-4o-mini",
            "product_sketch": "A wellness app for stress management",
            "interview_modes": [
                {
                    "mode": "trauma_informed",
                    "persona_count": 5,
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
                    "persona_count": 5,
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

    async def test_limited_vs_unlimited_concurrency(self, test_config):
        """Compare limited vs unlimited concurrency performance."""
        print("\n=== LIMITED VS UNLIMITED CONCURRENCY ===")

        # Mock API call with realistic timing
        async def mock_api_call(prompt, model):
            await asyncio.sleep(1.5)  # Simulate 1.5-second API call
            return """
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

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            # Test with limited concurrency (5)
            print("Testing with limited concurrency (max_concurrent=5)...")
            engine_limited = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=5,
            )

            # Override to use test_outputs
            engine_limited.project_dir = (
                Path("test_outputs") / "concurrency_test_limited"
            )
            engine_limited.config_dir_path.mkdir(exist_ok=True)
            engine_limited.runs_dir.mkdir(exist_ok=True)

            with patch.object(
                AsyncInterviewProcessor, "_call_openai_async", side_effect=mock_api_call
            ):
                start_time = time.time()
                results_limited = await engine_limited.run_iterative_research_async()
                limited_duration = time.time() - start_time

            print(f"Limited concurrency (5): {limited_duration:.2f}s")

            # Test with unlimited concurrency (None)
            print("Testing with unlimited concurrency (max_concurrent=None)...")
            engine_unlimited = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=None,
            )

            # Override to use test_outputs
            engine_unlimited.project_dir = (
                Path("test_outputs") / "concurrency_test_unlimited"
            )
            engine_unlimited.config_dir_path.mkdir(exist_ok=True)
            engine_unlimited.runs_dir.mkdir(exist_ok=True)

            with patch.object(
                AsyncInterviewProcessor, "_call_openai_async", side_effect=mock_api_call
            ):
                start_time = time.time()
                results_unlimited = (
                    await engine_unlimited.run_iterative_research_async()
                )
                unlimited_duration = time.time() - start_time

            print(f"Unlimited concurrency: {unlimited_duration:.2f}s")

            # Calculate speedup
            speedup = limited_duration / unlimited_duration
            print(f"Speedup: {speedup:.2f}x")

            # Both should complete successfully
            assert len(results_limited) == 1
            assert len(results_unlimited) == 1

            # Unlimited should be faster (or at least not slower)
            assert unlimited_duration <= limited_duration

    async def test_openai_rate_limit_handling(self, test_config):
        """Test that we handle OpenAI rate limits gracefully."""
        print("\n=== OPENAI RATE LIMIT HANDLING ===")

        # Mock API call that simulates rate limiting
        call_count = 0

        async def mock_api_call_with_rate_limit(prompt, model):
            nonlocal call_count
            call_count += 1

            # Simulate rate limit after 10 calls
            if call_count > 10:
                await asyncio.sleep(2)  # Simulate rate limit delay

            await asyncio.sleep(0.5)  # Simulate API call
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

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            engine = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=None,  # Unlimited
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "rate_limit_test"
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

            with patch.object(
                AsyncInterviewProcessor,
                "_call_openai_async",
                side_effect=mock_api_call_with_rate_limit,
            ):
                start_time = time.time()
                results = await engine.run_iterative_research_async()
                duration = time.time() - start_time

            print(f"Rate limit test completed in {duration:.2f}s")
            print(f"Total API calls made: {call_count}")

            assert len(results) == 1
            assert call_count > 0

    async def test_massive_concurrency_test(self, test_config):
        """Test with a very large number of concurrent interviews."""
        print("\n=== MASSIVE CONCURRENCY TEST ===")

        # Create a config with many interviews
        massive_config = {
            "project_name": "MassiveConcurrencyTest",
            "llm_model": "gpt-4o-mini",
            "product_sketch": "A wellness app for stress management",
            "interview_modes": [
                {
                    "mode": "mode_1",
                    "persona_count": 10,
                    "problem_hypotheses": [
                        {"label": "Hypothesis 1", "description": "Test hypothesis 1"},
                        {"label": "Hypothesis 2", "description": "Test hypothesis 2"},
                    ],
                },
                {
                    "mode": "mode_2",
                    "persona_count": 10,
                    "problem_hypotheses": [
                        {"label": "Hypothesis 3", "description": "Test hypothesis 3"},
                    ],
                },
            ],
            "output_format": "markdown",
            "version": "v1",
        }

        # Mock API call
        async def mock_api_call(prompt, model):
            await asyncio.sleep(0.1)  # Fast mock call
            return """
# Interview Results

## Alignment
Aligned? Yes

## Key Insights
- Test insight

## Pain Points
- Test pain point

## Desired Outcomes
- Test outcome

## Interview Notes
Test interview.
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "massive_test_config.json"
            with open(config_path, "w") as f:
                json.dump(massive_config, f)

            engine = AsyncIterativeResearchEngine(
                api_key="fake_key",
                config_dir=temp_dir,
                cycles=1,
                evolution_enabled=False,
                max_concurrent_interviews=None,  # Unlimited
            )

            # Override to use test_outputs
            engine.project_dir = Path("test_outputs") / "massive_concurrency_test"
            engine.config_dir_path.mkdir(exist_ok=True)
            engine.runs_dir.mkdir(exist_ok=True)

            with patch.object(
                AsyncInterviewProcessor, "_call_openai_async", side_effect=mock_api_call
            ):
                start_time = time.time()
                results = await engine.run_iterative_research_async()
                duration = time.time() - start_time

            total_interviews = 30  # 2 modes * (2+1 hypotheses) * 10 personas
            print(
                f"Massive concurrency test: {total_interviews} interviews in {duration:.2f}s"
            )
            print(f"Average time per interview: {duration/total_interviews:.2f}s")

            assert len(results) == 1
            assert (
                duration < total_interviews * 0.2
            )  # Should be much faster than sequential


if __name__ == "__main__":
    import pytest

    async def run_concurrency_tests():
        """Run all concurrency tests."""
        test = TestUnlimitedConcurrency()
        test_config = {
            "project_name": "ConcurrencyTest",
            "llm_model": "gpt-4o-mini",
            "product_sketch": "A wellness app for stress management",
            "interview_modes": [
                {
                    "mode": "trauma_informed",
                    "persona_count": 5,
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
                    "persona_count": 5,
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
        await test.test_limited_vs_unlimited_concurrency(test_config)
        await test.test_openai_rate_limit_handling(test_config)
        await test.test_massive_concurrency_test(test_config)

        print("\n=== CONCURRENCY TESTS COMPLETE ===")

    asyncio.run(run_concurrency_tests())
