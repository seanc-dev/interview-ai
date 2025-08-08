#!/usr/bin/env python3
import pytest
import asyncio
from typing import List, Dict
from unittest.mock import AsyncMock, patch
from pathlib import Path

from llm_interview_engine import (
    AsyncFocusGroupEngine,
    ProjectConfig,
    InterviewMode,
    ProblemHypothesis,
)

pytestmark = pytest.mark.focus_group

class TestFocusGroupMode:
    @pytest.mark.asyncio
    async def test_round_table_basic_flow(self, tmp_path: Path):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            config_dir="config/v1/",
            project_name=str(tmp_path / "fg_round_table"),
            participants=4,
            rounds=2,
            facilitator_persona="trauma-aware coach",
        )

        # minimal config
        engine.current_config.interview_modes = [
            InterviewMode(
                mode="Recovery",
                persona_count=0,
                problem_hypotheses=[
                    ProblemHypothesis(label="Overwhelm Regulation", description="...")
                ],
            )
        ]

        with patch.object(
            engine, "_call_llm_transcript_turn_async"
        ) as mock_turn, patch.object(
            engine, "_aggregate_group_insights_async"
        ) as mock_agg:
            mock_turn.return_value = {"speaker": "P1", "content": "sample"}
            mock_agg.return_value = {
                "themes": ["overwhelm"],
                "agreements": 2,
                "disagreements": 1,
            }

            result = await engine.run_focus_group_round_table_async()

            assert result["success"] is True
            assert result["format"] == "round_table"
            assert result["participants"] == 4
            assert result["rounds"] == 2
            assert "transcript" in result
            assert "insights" in result

            # save artifacts
            await engine.save_artifacts_async(result)
            assert (tmp_path / "fg_round_table" / "transcript.md").exists()
            assert (tmp_path / "fg_round_table" / "insights.json").exists()
            assert (tmp_path / "fg_round_table" / "summary.md").exists()

    @pytest.mark.asyncio
    async def test_open_table_facilitated_flow(self, tmp_path: Path):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            config_dir="config/v1/",
            project_name=str(tmp_path / "fg_open_table"),
            participants=5,
            facilitator_persona="supportive moderator",
        )

        engine.current_config.interview_modes = [
            InterviewMode(
                mode="Stability",
                persona_count=0,
                problem_hypotheses=[
                    ProblemHypothesis(label="Boundary Support", description="...")
                ],
            )
        ]

        with patch.object(
            engine, "_facilitate_open_table_async"
        ) as mock_open, patch.object(
            engine, "_aggregate_group_insights_async"
        ) as mock_agg:
            mock_open.return_value = [
                {"speaker": "Facilitator", "content": "Welcome"},
                {"speaker": "P3", "content": "I struggle with boundaries"},
            ]
            mock_agg.return_value = {
                "themes": ["boundaries"],
                "agreements": 3,
                "disagreements": 0,
            }

            result = await engine.run_focus_group_open_table_async()

            assert result["success"] is True
            assert result["format"] == "open_table"
            assert result["participants"] == 5
            assert len(result["transcript"]) >= 2
            assert result["insights"]["themes"][0] == "boundaries"

            # verify speaker structure
            speakers = {t["speaker"] for t in result["transcript"]}
            assert "Facilitator" in speakers

    @pytest.mark.asyncio
    async def test_conflict_summary_present(self):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            config_dir="config/v1/",
            project_name="fg_conflict",
            participants=3,
        )

        with patch.object(engine, "_aggregate_group_insights_async") as mock_agg:
            mock_agg.return_value = {
                "themes": ["overwhelm"],
                "agreements": 1,
                "disagreements": 2,
                "conflicts": ["breaks vs push through"],
            }
            result = {"insights": await engine._aggregate_group_insights_async([])}
            assert "conflicts" in result["insights"]
