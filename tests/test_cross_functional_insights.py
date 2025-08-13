import pytest


@pytest.mark.cross_insights
def test_cross_functional_synthesis_and_report_integration(tmp_path):
    from llm_interview_engine import CrossFunctionalSynthesizer

    insights = [
        {
            "insights": "Aligned? Yes\nPain Points:\n- Awareness\nDesired Outcomes:\n- Clear next steps\nMicro-feature Suggestions:\n- Gentle nudge",
        }
    ]

    synth = CrossFunctionalSynthesizer()
    structured = synth.synthesize(insights)

    # Expect top-level keys
    for key in ["marketing", "sales", "strategy", "gtm"]:
        assert key in structured
        assert isinstance(structured[key], list)

    # Integration helper returns a section string with a header
    section = synth.to_markdown_section(structured)
    assert section.startswith("\n## Cross-Functional Insights\n")
    assert "- Marketing:" in section
