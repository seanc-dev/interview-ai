from typing import Dict, List


class CrossFunctionalSynthesizer:
    """Produces cross-functional synthesis and Markdown section."""

    def synthesize(self, insights: List[Dict]) -> Dict[str, List[str]]:
        points: List[str] = []
        for i in insights or []:
            text = i.get("insights", "")
            for line in text.splitlines():
                if line.strip().startswith("-"):
                    points.append(line.strip("- ").strip())
        return {
            "marketing": points[:3],
            "sales": points[3:6],
            "strategy": points[6:9],
            "gtm": points[9:12],
        }

    def to_markdown_section(self, structured: Dict[str, List[str]]) -> str:
        def bullets(items: List[str]) -> str:
            return "\n".join(f"  - {it}" for it in (items or [])[:5])

        return (
            "\n## Cross-Functional Insights\n\n"
            f"- Marketing:\n{bullets(structured.get('marketing'))}\n"
            f"- Sales:\n{bullets(structured.get('sales'))}\n"
            f"- Strategy:\n{bullets(structured.get('strategy'))}\n"
            f"- GTM:\n{bullets(structured.get('gtm'))}\n"
        )


class HypothesisStateTracker:
    def __init__(
        self,
        alignment_threshold: float = 0.6,
        min_frequency: int = 2,
        retire_threshold: float = 0.75,
    ) -> None:
        self.alignment_threshold = alignment_threshold
        self.min_frequency = min_frequency
        self.retire_threshold = retire_threshold

    def summarize(self, insights: List[Dict]) -> Dict[str, Dict]:
        stats: Dict[str, Dict[str, int]] = {}
        for i in insights or []:
            hyp = i.get("hypothesis") or i.get("hypothesis_label") or "Unknown"
            text = i.get("insights", "")
            aligned = 1 if ("Aligned? Yes" in text or "Aligned: Yes" in text) else 0
            entry = stats.setdefault(hyp, {"count": 0, "aligned": 0, "misaligned": 0})
            entry["count"] += 1
            if aligned:
                entry["aligned"] += 1
            else:
                entry["misaligned"] += 1

        promote, retire = [], []
        for hyp, v in stats.items():
            count = max(1, v["count"])
            align_ratio = v["aligned"] / count
            mis_ratio = v["misaligned"] / count
            if (
                v["count"] >= self.min_frequency
                and align_ratio >= self.alignment_threshold
            ):
                promote.append(hyp)
            if v["count"] <= self.min_frequency and mis_ratio >= self.retire_threshold:
                retire.append(hyp)
        return {"hypotheses": stats, "promote": promote, "retire": retire}


__all__ = ["CrossFunctionalSynthesizer", "HypothesisStateTracker"]
