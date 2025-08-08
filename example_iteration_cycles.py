#!/usr/bin/env python3
"""
Example script demonstrating the iteration cycles feature.

This script shows how to use the AsyncIterativeResearchEngine to run
multiple cycles of research with automatic evolution between cycles.
"""

import asyncio
import os
from pathlib import Path
from llm_interview_engine import (
    AsyncIterativeResearchEngine,
    ProjectConfig,
    InterviewMode,
    ProblemHypothesis,
)


async def run_iteration_cycles_example():
    """Run an example of iteration cycles with evolution."""

    # Create a test configuration
    config = ProjectConfig(
        project_name="YGT Iteration Example",
        llm_model="gpt-4o",
        product_sketch="YGT is an emotionally intelligent AI companion that helps users manage overwhelm and set boundaries.",
        interview_modes=[
            InterviewMode(
                mode="recovery",
                persona_count=2,
                problem_hypotheses=[
                    ProblemHypothesis(
                        label="Overwhelm Regulation",
                        description="Users struggle to regulate overwhelm in high-stress situations",
                    ),
                    ProblemHypothesis(
                        label="Boundary Setting",
                        description="Users have difficulty setting and maintaining boundaries at work",
                    ),
                ],
            )
        ],
        output_format="markdown",
        version="v1",
    )

    # Save the config to a temporary file
    config_dir = Path("config/example/")
    config_dir.mkdir(exist_ok=True)

    config_dict = {
        "project_name": config.project_name,
        "llm_model": config.llm_model,
        "product_sketch": config.product_sketch,
        "interview_modes": [
            {
                "mode": mode.mode,
                "persona_count": mode.persona_count,
                "problem_hypotheses": [
                    {"label": h.label, "description": h.description}
                    for h in mode.problem_hypotheses
                ],
            }
            for mode in config.interview_modes
        ],
        "output_format": config.output_format,
        "version": config.version,
    }

    import json

    with open(config_dir / "ygt_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("🚀 Starting Iteration Cycles Example")
    print("=" * 50)
    print(f"📋 Project: {config.project_name}")
    print(f"🔄 Cycles: 3")
    print(f"⚡ Max Concurrent Interviews: 3")
    print(f"📁 Config Directory: {config_dir}")
    print("=" * 50)

    # Initialize the async iterative research engine with project name
    engine = AsyncIterativeResearchEngine(
        api_key=os.getenv("OPENAI_API_KEY"),
        config_dir=str(config_dir),
        cycles=3,
        evolution_enabled=True,
        max_concurrent_interviews=3,
        project_name="ygt_iteration_example",  # New project name parameter
    )

    try:
        # Run the iterative research process
        results = await engine.run_iterative_research_async()

        print("\n" + "=" * 50)
        print("✅ Iteration Cycles Example Completed!")
        print("=" * 50)

        # Print summary of results
        successful_cycles = [r for r in results if r["success"]]
        failed_cycles = [r for r in results if not r["success"]]

        print(f"📊 Total Cycles: {len(results)}")
        print(f"✅ Successful: {len(successful_cycles)}")
        print(f"❌ Failed: {len(failed_cycles)}")

        if successful_cycles:
            total_insights = sum(r["insights_count"] for r in successful_cycles)
            total_personas = sum(r["personas_generated"] for r in successful_cycles)
            evolved_cycles = sum(1 for r in successful_cycles if r["config_evolved"])

            print(f"💡 Total Insights: {total_insights}")
            print(f"👥 Total Personas: {total_personas}")
            print(f"🔄 Config Evolutions: {evolved_cycles}")

            # Show evolution details
            for i, result in enumerate(successful_cycles, 1):
                print(f"\n📈 Cycle {i} Summary:")
                print(f"   📊 Alignment Rate: {result['alignment_rate']:.1%}")
                print(f"   💡 Insights: {result['insights_count']}")
                print(f"   👥 Personas: {result['personas_generated']}")
                print(f"   ⏱️  Duration: {result['duration']:.1f}s")
                if result["config_evolved"]:
                    print(f"   🔄 Config Evolved: Yes")
                    print(f"   📋 New Version: {result['config_version']}")
                else:
                    print(f"   🔄 Config Evolved: No")

        # Show project structure info
        print(f"\n📁 Project Structure Created:")
        print(f"   📂 Project Directory: {engine.project_dir}")
        print(f"   📂 Config Directory: {engine.config_dir_path}")
        print(f"   📂 Runs Directory: {engine.runs_dir}")

        return results

    except Exception as e:
        print(f"❌ Error running iteration cycles: {str(e)}")
        return None


def main():
    """Main function to run the example."""
    print("🎯 Iteration Cycles Feature Demo")
    print("This demonstrates the 4-step cycle process:")
    print("1. Generate personas")
    print("2. Perform interviews")
    print("3. Aggregate insights")
    print("4. Evolve config (with each cycle)")
    print("Then repeat for specified number of cycles.\n")

    print("🆕 New Features:")
    print("- Config evolution with each cycle")
    print("- Project > Config > Run folder structure")
    print("- Master report with plain English improvements")
    print("- Roadmap for latest config version")
    print("- Persistent config evolution across runs")
    print("- Cost-effective gpt-4o-mini model\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   The example will run with mocked responses")
        print("   Set OPENAI_API_KEY to run with real API calls\n")

    # Run the async example
    results = asyncio.run(run_iteration_cycles_example())

    if results:
        print("\n🎉 Example completed successfully!")
        print("The iteration cycles feature is working correctly.")
        print("Check the outputs/ygt_iteration_example/ directory for:")
        print("  📊 Master reports with improvements tracking")
        print("  🗺️  Roadmaps for latest config version")
        print("  📁 Config evolution history")
    else:
        print("\n❌ Example failed. Check the error messages above.")


if __name__ == "__main__":
    main()
