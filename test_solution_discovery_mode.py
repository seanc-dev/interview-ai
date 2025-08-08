#!/usr/bin/env python3
import pytest
import asyncio
from typing import List, Dict
from unittest.mock import AsyncMock, patch
from pathlib import Path

from llm_interview_engine import (
    AsyncSolutionDiscoveryEngine,
)

pytestmark = pytest.mark.solution_discovery


class TestSolutionDiscoveryMode:
    @pytest.mark.asyncio
    async def test_concept_generation(self):
        engine = AsyncSolutionDiscoveryEngine(
            api_key="test_key", project_name="sd_concepts"
        )

        sample_insights = [
            {
                "pain_points": ["overwhelm", "time debt"],
                "desired_outcomes": ["better boundaries"],
            }
        ]

        with patch.object(engine, "_llm_generate_concepts_async") as mock_gen:
            mock_gen.return_value = {
                "concepts": ["Boundary Coach", "Time Budgeter"],
                "reasoning": "addresses overwhelm and time debt",
            }
            out = await engine.generate_concepts_async(sample_insights, top_n=2)
            assert len(out["concepts"]) == 2
            assert "Boundary" in out["concepts"][0]

    @pytest.mark.asyncio
    async def test_mvp_proposal(self, tmp_path: Path):
        engine = AsyncSolutionDiscoveryEngine(
            api_key="test_key", project_name=str(tmp_path / "sd_mvp")
        )

        concepts = ["Boundary Coach", "Time Budgeter"]
        with patch.object(engine, "_llm_propose_mvp_async") as mock_mvp:
            mock_mvp.return_value = {
                "title": "Boundary Coach MVP",
                "scope": ["smart notifications", "boundary prompts"],
                "metrics": ["overwhelm reduction", "adherence"],
                "risks": ["false positives"],
            }
            mvp = await engine.propose_mvp_async(concepts)
            assert "scope" in mvp and len(mvp["scope"]) >= 2

            # save artifacts
            await engine.save_artifacts_async(
                concepts={"concepts": concepts}, mvp=mvp, validation_plan={}
            )
            assert (tmp_path / "sd_mvp" / "concepts.md").exists()
            assert (tmp_path / "sd_mvp" / "mvp_proposal.md").exists()

    @pytest.mark.asyncio
    async def test_validation_plan(self, tmp_path: Path):
        engine = AsyncSolutionDiscoveryEngine(
            api_key="test_key", project_name=str(tmp_path / "sd_validate")
        )

        with patch.object(engine, "_llm_validation_plan_async") as mock_val:
            mock_val.return_value = {
                "experiments": ["A/B test notifications", "diary study"],
                "success_criteria": [">25% overwhelm reduction"],
            }
            plan = await engine.create_validation_plan_async(
                {"title": "Boundary Coach MVP"}
            )
            assert "experiments" in plan and len(plan["experiments"]) > 0

            await engine.save_artifacts_async(
                concepts={"concepts": ["Boundary Coach"]}, mvp={"title": "Boundary Coach MVP"}, validation_plan=plan
            )
            assert (tmp_path / "sd_validate" / "validation_plan.md").exists()
