from typing import Dict, Any, List, Optional


class QualityEvaluator:
    """Evaluates quality of insights and runs using heuristics and optional LLM judge."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def score_insight_heuristic(self, insight_text: str) -> Dict[str, float]:
        text = (insight_text or "").strip()
        if not text:
            return {
                "alignment": 0.0,
                "pain_points": 0.0,
                "desired_outcomes": 0.0,
                "micro_features": 0.0,
            }

        if ("Aligned? Yes" in text) or ("Aligned: Yes" in text):
            alignment = 1.0
        elif ("Aligned? No" in text) or ("Aligned: No" in text):
            alignment = 0.0
        elif "Aligned?" in text or "Aligned:" in text:
            alignment = 0.5
        else:
            alignment = 0.0

        pain_points = 0.0
        if "Pain Points" in text:
            start = text.find("Pain Points")
            chunk = text[start : start + 2000]
            pain_points = min(1.0, max(0.0, chunk.count("\n-") / 5.0))
        desired = 0.0
        if "Desired Outcomes" in text:
            start = text.find("Desired Outcomes")
            chunk = text[start : start + 2000]
            desired = min(1.0, max(0.0, chunk.count("\n-") / 5.0))
        features = 0.0
        if "Micro-feature Suggestions" in text:
            try:
                section = text.split("Micro-feature Suggestions:", 1)[1]
                bullets = [
                    ln for ln in section.split("\n") if ln.strip().startswith("-")
                ]
                features = min(1.0, len(bullets) / 3.0)
            except Exception:
                features = 0.0

        return {
            "alignment": alignment,
            "pain_points": pain_points,
            "desired_outcomes": desired,
            "micro_features": features,
        }

    def aggregate_run_scores(self, insights: List[Dict]) -> Dict[str, Any]:
        per = [self.score_insight_heuristic(i.get("insights", "")) for i in insights]
        if not per:
            return {"overall": 0.0, "dimensions": {}}
        dims = {k: (sum(s[k] for s in per) / len(per)) for k in per[0].keys()}
        overall = sum(dims.values()) / len(dims)
        return {"overall": overall, "dimensions": dims}


__all__ = ["QualityEvaluator"]
