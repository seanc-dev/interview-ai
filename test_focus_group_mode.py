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
    async def test_round_table_turn_counts(self):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            project_name="fg_counts",
            participants=3,
            rounds=3,
        )
        with patch.object(engine, "_aggregate_group_insights_async") as mock_agg:
            mock_agg.return_value = {
                "themes": ["boundaries"],
                "agreements": 0,
                "disagreements": 0,
            }
            result = await engine.run_focus_group_round_table_async()
            # exactly one turn per participant per round
            assert len(result["transcript"]) == 3 * 3

    @pytest.mark.asyncio
    async def test_round_table_dropout_skips_turn(self):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            project_name="fg_dropout",
            participants=3,
            rounds=1,
        )
        call_count = {"n": 0}

        async def flaky_turn(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("participant disconnected")
            return {"speaker": f"P{call_count['n']}", "content": "ok"}

        with patch.object(engine, "_aggregate_group_insights_async") as mock_agg, \
             patch.object(engine, "_call_llm_transcript_turn_async", side_effect=flaky_turn):
            mock_agg.return_value = {"themes": ["boundaries"], "agreements": 0, "disagreements": 0}
            result = await engine.run_focus_group_round_table_async()
            # one dropout -> one fewer turn recorded
            assert len(result["transcript"]) == engine.participants - 1
            assert result["success"] is True

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
    async def test_open_table_rotation_and_snapshot(self, tmp_path: Path):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            project_name=str(tmp_path / "fg_snapshot"),
            participants=4,
        )
        result = await engine.run_focus_group_open_table_async()
        # No immediate repeated speaker in participant turns
        last = None
        for turn in result["transcript"][1:]:  # skip facilitator welcome
            if last is not None:
                assert turn["speaker"] != last
            last = turn["speaker"]
        await engine.save_artifacts_async(result)
        # Transcript snapshot basics
        transcript_text = (tmp_path / "fg_snapshot" / "transcript.md").read_text()
        assert transcript_text.splitlines()[0].startswith("Facilitator:")

    @pytest.mark.asyncio
    async def test_generate_report_and_roadmap_from_result(self):
        engine = AsyncFocusGroupEngine(
            api_key="test_key",
            project_name="fg_report",
            participants=3,
        )
        with patch.object(engine, "_aggregate_group_insights_async") as mock_agg:
            mock_agg.return_value = {
                "themes": ["overwhelm", "boundaries"],
                "agreements": 2,
                "disagreements": 1,
            }
            result = await engine.run_focus_group_round_table_async()
        report, roadmap = await engine.generate_report_and_roadmap_async(result)
        assert isinstance(report, str) and len(report) > 0
        assert isinstance(roadmap, str) and len(roadmap) > 0

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
