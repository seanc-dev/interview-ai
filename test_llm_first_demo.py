#!/usr/bin/env python3
"""
Demo script to showcase the LLM-first analysis implementation.
This script demonstrates how the new LLM-driven classes work.
"""

import asyncio
import os
from dotenv import load_dotenv
from llm_interview_engine import (
    LLMInsightAnalyzer,
    LLMReportGenerator,
    LLMRoadmapGenerator,
    LLMProductEvolution,
)

# Load environment variables
load_dotenv()


async def demo_llm_first_analysis():
    """Demo the LLM-first analysis capabilities."""

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set your OpenAI API key before running this demo.")
        return False

    print("🚀 LLM-First Analysis Demo")
    print("=" * 50)

    # Sample insights for demonstration
    sample_insights = [
        {
            "insights": """Aligned? Yes

## Pain Points
- Overwhelm from poor boundaries
- Stress from time management issues
- Difficulty saying no to requests

## Desired Outcomes
- Better boundaries with work and personal life
- Time management tools and strategies
- Confidence to prioritize self-care"""
        },
        {
            "insights": """Aligned? No

## Pain Points
- Imposter syndrome in professional settings
- Fear of being judged for taking breaks
- Difficulty advocating for needs

## Desired Outcomes
- Confidence building tools
- Support groups for similar experiences
- Validation of their struggles"""
        },
    ]

    try:
        # 1. Demo LLMInsightAnalyzer
        print("\n📊 1. LLM-Driven Insight Analysis")
        print("-" * 30)

        analyzer = LLMInsightAnalyzer(api_key=api_key)

        # Analyze single insight
        result = await analyzer.analyze_single_insight_async(sample_insights[0])
        print(f"✅ Single insight analysis: {result['aligned']}")
        print(f"   Pain points: {len(result['pain_points'])} found")
        print(f"   Desired outcomes: {len(result['desired_outcomes'])} found")

        # Analyze multiple insights
        results = await analyzer.analyze_multiple_insights_async(sample_insights)
        print(f"✅ Multiple insights analysis: {len(results)} processed")

        # 2. Demo LLMReportGenerator
        print("\n📝 2. LLM-Driven Report Generation")
        print("-" * 30)

        report_gen = LLMReportGenerator(api_key=api_key)

        # Generate master report
        report = await report_gen.generate_master_report_async(
            results, "Wellness App Research"
        )
        print(f"✅ Master report generated: {len(report)} characters")
        print(f"   Report preview: {report[:200]}...")

        # Generate conversational analysis
        analysis = await report_gen.generate_conversational_analysis_async(results)
        print(f"✅ Conversational analysis: {len(analysis)} characters")

        # 3. Demo LLMRoadmapGenerator
        print("\n🗺️  3. LLM-Driven Roadmap Generation")
        print("-" * 30)

        roadmap_gen = LLMRoadmapGenerator(api_key=api_key)

        # Generate prioritized roadmap
        roadmap = await roadmap_gen.generate_prioritized_roadmap_async(
            results, "Wellness App"
        )
        print(f"✅ Prioritized roadmap generated: {len(roadmap)} characters")
        print(f"   Roadmap preview: {roadmap[:200]}...")

        # Synthesize features
        features = await roadmap_gen.synthesize_features_from_insights_async(results)
        print(f"✅ Features synthesized: {len(features.get('features', []))} features")

        # 4. Demo LLMProductEvolution
        print("\n🔄 4. LLM-Driven Product Evolution")
        print("-" * 30)

        evolution = LLMProductEvolution(api_key=api_key)

        # Evolve product sketch
        original_sketch = "A wellness app for busy professionals"
        evolved_sketch = await evolution.evolve_product_sketch_async(
            original_sketch, results
        )
        print(f"✅ Product sketch evolved: {len(evolved_sketch)} characters")
        print(f"   Evolved preview: {evolved_sketch[:200]}...")

        # Generate new hypotheses
        hypotheses = await evolution.generate_new_hypotheses_async(results)
        print(f"✅ New hypotheses generated: {len(hypotheses)} hypotheses")

        # 5. Demo End-to-End Pipeline
        print("\n🔗 5. End-to-End LLM Pipeline")
        print("-" * 30)

        # Full pipeline: insights → analysis → report → roadmap → evolution
        print("✅ Full LLM-first pipeline executed successfully!")
        print("   - Insights analyzed with LLM")
        print("   - Report generated conversationally")
        print("   - Roadmap prioritized intelligently")
        print("   - Product evolved based on insights")

        print("\n🎉 LLM-First Analysis Demo Completed Successfully!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


if __name__ == "__main__":
    asyncio.run(demo_llm_first_analysis())

