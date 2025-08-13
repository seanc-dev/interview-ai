import pytest


@pytest.mark.quality_gate
def test_quality_gate_selects_and_builds_reprompts(monkeypatch):
    from llm_interview_engine import QualityGateReprompter, QualityEvaluator

    low = {
        "mode": "Recovery",
        "hypothesis": "Overwhelm Regulation",
        "insights": "Aligned? No\nPain Points:\n- (none)",
    }
    ok = {
        "mode": "Recovery",
        "hypothesis": "Trauma-Aware Productivity",
        "insights": "Aligned? Yes\nPain Points:\n- Stress\nDesired Outcomes:\n- Focus",
    }

    qe = QualityEvaluator()
    gate = QualityGateReprompter(qe, thresholds={"alignment": 0.5, "pain_points": 0.2})

    candidates = gate.select_candidates([low, ok])
    assert low in candidates and ok not in candidates

    prompt = gate.build_reprompt(low)
    assert "Please improve" in prompt and "Overwhelm Regulation" in prompt
