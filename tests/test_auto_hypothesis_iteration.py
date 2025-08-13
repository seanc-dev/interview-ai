import json
from pathlib import Path

import pytest


@pytest.mark.hypothesis_iteration
def test_hypothesis_tracker_promote_and_retire(tmp_path: Path):
    from llm_interview_engine import HypothesisStateTracker

    insights = [
        {
            "mode": "Recovery",
            "hypothesis": "Overwhelm Regulation",
            "insights": "Aligned? Yes\nPain Points:\n- Stress\n- Anxiety",
        },
        {
            "mode": "Recovery",
            "hypothesis": "Overwhelm Regulation",
            "insights": "Aligned? Yes\nPain Points:\n- Stress",
        },
        {
            "mode": "Stability",
            "hypothesis": "Energy Awareness and Boundary Support",
            "insights": "Aligned? No\nPain Points:\n- Fatigue",
        },
    ]

    tracker = HypothesisStateTracker(
        alignment_threshold=0.6, min_frequency=2, retire_threshold=0.8
    )
    summary = tracker.summarize(insights)

    assert summary["hypotheses"]["Overwhelm Regulation"]["count"] == 2
    assert summary["hypotheses"]["Overwhelm Regulation"]["aligned"] == 2
    assert "promote" in summary
    assert "Overwhelm Regulation" in summary["promote"]

    # The Stability hypothesis has 1 mention and is misaligned; consider for retire
    assert "retire" in summary
    assert "Energy Awareness and Boundary Support" in summary["retire"]

    # Write to disk
    out = tmp_path / "hypotheses_state.json"
    tracker.write_state(summary, out)
    data = json.loads(out.read_text())
    assert "hypotheses" in data and "promote" in data and "retire" in data
