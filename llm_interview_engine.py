#!/usr/bin/env python3
"""
LLM Interview Engine - A tool for running LLM-to-LLM research interviews
to evaluate problem/solution hypotheses for emotionally intelligent coaching products.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import openai
from dataclasses import dataclass, asdict
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProblemHypothesis:
    """
    Represents a problem hypothesis to be tested in LLM interviews.

    Attributes:
        label (str): A short identifier for the hypothesis
        description (str): Detailed description of the problem hypothesis
    """

    label: str
    description: str


@dataclass
class InterviewMode:
    """
    Represents an interview mode with its parameters and hypotheses.

    Attributes:
        mode (str): The interview mode identifier (e.g., "trauma_informed", "cognitive_behavioral")
        persona_count (int): Number of contrasting personas to generate (default: 3)
        problem_hypotheses (List[ProblemHypothesis]): List of hypotheses to test in this mode
    """

    mode: str
    persona_count: int = 3
    problem_hypotheses: List[ProblemHypothesis] = None

    def __post_init__(self):
        if self.problem_hypotheses is None:
            self.problem_hypotheses = []


@dataclass
class ProjectConfig:
    """
    Configuration for an LLM interview project.

    Attributes:
        project_name (str): Unique name for the project
        llm_model (str): OpenAI model to use for interviews (default: "gpt-4o")
        product_sketch (str): Internal product description (not shared with personas)
        interview_modes (List[InterviewMode]): List of interview modes and their configurations
        output_format (str): Output format for interviews ("markdown" or "json")
    """

    project_name: str
    llm_model: str = "gpt-4o"
    product_sketch: str = ""
    interview_modes: List[InterviewMode] = None
    output_format: str = "markdown"

    def __post_init__(self):
        if self.interview_modes is None:
            self.interview_modes = []


class LLMInterviewEngine:
    """
    Main engine for running LLM-to-LLM research interviews.

    This class provides functionality for:
    - Interactive CLI for project management
    - Multi-persona interview generation
    - Structured interview execution with three-phase process
    - Master report aggregation and analysis
    - Robust error handling with exponential backoff

    Example:
        engine = LLMInterviewEngine()
        engine.run_cli()
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM interview engine.

        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable or .env file.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # Load environment variables from .env file
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env file, environment variable, or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def run_cli(self):
        """
        Main CLI entry point for the interview engine.

        This method provides an interactive command-line interface that:
        1. Prompts user to choose between new, existing projects, or JSON config input
        2. Guides through project configuration
        3. Executes interview simulations
        4. Generates master reports
        """
        print("🤖 LLM Interview Engine")
        print("=" * 50)

        choice = self._prompt_project_choice()

        if choice == "new":
            config = self._create_new_project()
        elif choice == "json":
            config = self._load_json_config()
        else:
            config = self._load_existing_project()

        self._run_interviews(config)

    def _prompt_project_choice(self) -> str:
        """Prompt user to choose between new, existing project, or JSON config input"""
        while True:
            choice = (
                input(
                    "Run interviews on an existing project, start a new one, or paste JSON config? (existing/new/json): "
                )
                .strip()
                .lower()
            )
            if choice in ["existing", "new", "json"]:
                return choice
            print("Please enter 'existing', 'new', or 'json'")

    def _create_new_project(self) -> ProjectConfig:
        """Interactive creation of a new project"""
        print("\n📝 Creating New Project")
        print("-" * 30)

        project_name = input("Project name: ").strip()
        if not project_name:
            raise ValueError("Project name is required")

        llm_model = input("LLM model (default: gpt-4o): ").strip() or "gpt-4o"

        print("\nProduct sketch (multi-line, press Enter twice to finish):")
        product_sketch_lines = []
        while True:
            line = input()
            if line == "" and product_sketch_lines and product_sketch_lines[-1] == "":
                break
            product_sketch_lines.append(line)
        product_sketch = "\n".join(product_sketch_lines[:-1])  # Remove last empty line

        output_format = (
            input("Output format (markdown/json, default: markdown): ").strip()
            or "markdown"
        )
        if output_format not in ["markdown", "json"]:
            raise ValueError("Output format must be 'markdown' or 'json'")

        # Collect interview modes and hypotheses
        interview_modes = []
        while True:
            mode = input("\nInterview mode (or 'done' to finish): ").strip()
            if mode.lower() == "done":
                break

            persona_count = input(f"Persona count for '{mode}' (default: 3): ").strip()
            persona_count = int(persona_count) if persona_count else 3

            problem_hypotheses = []
            print(f"\nProblem hypotheses for '{mode}':")
            while True:
                label = input("Hypothesis label (or 'done' to finish): ").strip()
                if label.lower() == "done":
                    break

                description = input("Hypothesis description: ").strip()
                if not description:
                    print("Description is required")
                    continue

                problem_hypotheses.append(
                    ProblemHypothesis(label=label, description=description)
                )

            interview_modes.append(
                InterviewMode(
                    mode=mode,
                    persona_count=persona_count,
                    problem_hypotheses=problem_hypotheses,
                )
            )

        config = ProjectConfig(
            project_name=project_name,
            llm_model=llm_model,
            product_sketch=product_sketch,
            interview_modes=interview_modes,
            output_format=output_format,
        )

        # Save config
        self._save_config(config)

        return config

    def _load_json_config(self) -> ProjectConfig:
        """Load configuration from JSON input"""
        print("\n📋 Loading Configuration from JSON")
        print("-" * 40)
        print("Paste your JSON configuration below (press Enter twice when done):")

        json_lines = []
        while True:
            line = input()
            if line == "" and json_lines and json_lines[-1] == "":
                break
            json_lines.append(line)

        json_str = "\n".join(json_lines[:-1])  # Remove last empty line

        try:
            config_data = json.loads(json_str)
            config = self._dict_to_config(config_data)

            # Validate required fields
            if not config.project_name:
                raise ValueError("Project name is required in JSON config")

            print(
                f"\n✅ Successfully loaded configuration for project: {config.project_name}"
            )

            # Show summary
            self._show_project_summary(config)

            # Save config
            self._save_config(config)

            return config

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            print("Please check your JSON syntax and try again.")
            return self._create_new_project()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("Please check your configuration and try again.")
            return self._create_new_project()
        except Exception as e:
            print(f"❌ Unexpected error loading JSON config: {e}")
            print("Please check your configuration and try again.")
            return self._create_new_project()

    def _load_existing_project(self) -> ProjectConfig:
        """Load an existing project configuration"""
        print("\n📁 Loading Existing Project")
        print("-" * 30)

        # Find existing projects
        project_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]

        if not project_dirs:
            print("No existing projects found. Creating new project...")
            return self._create_new_project()

        print("Existing projects:")
        for i, project_dir in enumerate(project_dirs, 1):
            print(f"{i}. {project_dir.name}")

        while True:
            try:
                choice = int(input(f"\nSelect project (1-{len(project_dirs)}): ")) - 1
                if 0 <= choice < len(project_dirs):
                    project_dir = project_dirs[choice]
                    break
                print(f"Please enter a number between 1 and {len(project_dirs)}")
            except ValueError:
                print("Please enter a valid number")

        config_path = project_dir / "config.json"
        if not config_path.exists():
            print(f"Config file not found in {project_dir}. Creating new project...")
            return self._create_new_project()

        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Convert back to dataclass objects
        config = self._dict_to_config(config_data)

        # Show summary
        self._show_project_summary(config)

        # Ask what to do
        action = self._prompt_project_action()
        if action == "modify":
            config = self._modify_config(config)
        elif action == "variant":
            config = self._create_variant(config)

        return config

    def _show_project_summary(self, config: ProjectConfig):
        """Show summary of existing project"""
        print(f"\n📊 Project Summary: {config.project_name}")
        print("-" * 40)
        print(f"Model: {config.llm_model}")
        print(f"Output format: {config.output_format}")
        print(f"Interview modes: {len(config.interview_modes)}")

        total_interviews = sum(
            mode.persona_count * len(mode.problem_hypotheses)
            for mode in config.interview_modes
        )
        print(f"Total interviews: {total_interviews}")

        # Try to load master report
        master_report_path = self.output_dir / config.project_name / "master_report.md"
        if master_report_path.exists():
            print(f"Master report exists: {master_report_path}")

    def _prompt_project_action(self) -> str:
        """Prompt user for action on existing project"""
        while True:
            action = input("\nAction: reuse/modify/variant? ").strip().lower()
            if action in ["reuse", "modify", "variant"]:
                return action
            print("Please enter 'reuse', 'modify', or 'variant'")

    def _modify_config(self, config: ProjectConfig) -> ProjectConfig:
        """Modify existing configuration"""
        print("\n🔧 Modifying Configuration")
        print("-" * 30)

        # For now, just return the original config
        # TODO: Implement interactive modification
        return config

    def _create_variant(self, config: ProjectConfig) -> ProjectConfig:
        """Create a variant of existing configuration"""
        print("\n🔄 Creating Project Variant")
        print("-" * 30)

        variant_name = input(
            f"Variant name (default: {config.project_name}_variant): "
        ).strip()
        if not variant_name:
            variant_name = f"{config.project_name}_variant"

        config.project_name = variant_name
        return config

    def _save_config(self, config: ProjectConfig):
        """Save project configuration"""
        project_dir = self.output_dir / config.project_name
        project_dir.mkdir(exist_ok=True)

        config_path = project_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._config_to_dict(config), f, indent=2)

        print(f"✅ Configuration saved to {config_path}")

    def _config_to_dict(self, config: ProjectConfig) -> Dict:
        """Convert config to dictionary for JSON serialization"""
        data = asdict(config)
        # Convert InterviewMode objects
        data["interview_modes"] = []
        for mode in config.interview_modes:
            mode_data = asdict(mode)
            mode_data["problem_hypotheses"] = [
                asdict(h) for h in mode.problem_hypotheses
            ]
            data["interview_modes"].append(mode_data)
        return data

    def _dict_to_config(self, data: Dict) -> ProjectConfig:
        """Convert dictionary back to config object"""
        # Convert problem hypotheses and interview modes
        for mode_data in data["interview_modes"]:
            mode_data["problem_hypotheses"] = [
                ProblemHypothesis(**h) for h in mode_data["problem_hypotheses"]
            ]

        # Create InterviewMode objects
        interview_modes = []
        for mode_data in data["interview_modes"]:
            interview_modes.append(InterviewMode(**mode_data))

        # Create ProjectConfig with converted objects
        config_data = data.copy()
        config_data["interview_modes"] = interview_modes

        return ProjectConfig(**config_data)

    def _run_interviews(self, config: ProjectConfig):
        """Run the interview simulations"""
        print(f"\n🚀 Running Interviews for {config.project_name}")
        print("=" * 50)

        total_interviews = sum(
            mode.persona_count * len(mode.problem_hypotheses)
            for mode in config.interview_modes
        )

        print(f"Total interviews to run: {total_interviews}")
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Interview run cancelled.")
            return

        # Create project directory and run-specific subdirectory
        project_dir = self.output_dir / config.project_name
        project_dir.mkdir(exist_ok=True)
        
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = project_dir / f"run_{run_timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Initialize master report and roadmap paths (at project level)
        master_report_path = project_dir / "master_report.md"
        roadmap_path = project_dir / "roadmap.md"
        
        run_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": config.llm_model,
            "modes": [mode.mode for mode in config.interview_modes],
            "total_interviews": total_interviews,
            "run_timestamp": run_timestamp,
        }

        # Run interviews
        completed_interviews = 0
        failed_interviews = []
        all_insights = []

        for mode in config.interview_modes:
            for hypothesis in mode.problem_hypotheses:
                print(
                    f"\n📝 Running interviews for mode: {mode.mode}, hypothesis: {hypothesis.label}"
                )

                for persona_variant in range(1, mode.persona_count + 1):
                    try:
                        print(
                            f"  Persona variant {persona_variant}/{mode.persona_count}..."
                        )

                        # Generate and run interview
                        result = self._run_single_interview(
                            config, mode, hypothesis, persona_variant
                        )

                        # Save results to run directory
                        self._save_interview_results(
                            run_dir, mode, hypothesis, persona_variant, result
                        )

                        # Collect insights for aggregation
                        if "insight_summary" in result:
                            all_insights.append({
                                "mode": mode.mode,
                                "hypothesis": hypothesis.label,
                                "persona_variant": persona_variant,
                                "insights": result["insight_summary"]
                            })

                        completed_interviews += 1
                        print(
                            f"    ✅ Completed ({completed_interviews}/{total_interviews})"
                        )

                    except Exception as e:
                        error_msg = f"Failed interview for {mode.mode}/{hypothesis.label}/persona_{persona_variant}: {str(e)}"
                        logger.error(error_msg)
                        failed_interviews.append(error_msg)
                        print(f"    ❌ Failed: {str(e)}")

        # Update master report with aggregated insights
        self._update_master_report(
            master_report_path, run_metadata, completed_interviews, failed_interviews, all_insights
        )

        # Generate/update roadmap
        self._update_roadmap(roadmap_path, all_insights, run_metadata)

        print(f"\n🎉 Interview run completed!")
        print(f"✅ Successful: {completed_interviews}")
        print(f"❌ Failed: {len(failed_interviews)}")
        print(f"📁 Results saved to: {run_dir}")
        print(f"📊 Master report updated: {master_report_path}")
        print(f"🗺️  Roadmap updated: {roadmap_path}")

        if failed_interviews:
            print("\nFailed interviews:")
            for failure in failed_interviews:
                print(f"  - {failure}")

    def _run_single_interview(
        self,
        config: ProjectConfig,
        mode: InterviewMode,
        hypothesis: ProblemHypothesis,
        persona_variant: int,
    ) -> Dict:
        """Run a single interview simulation"""
        prompt = self._generate_interview_prompt(
            config, mode, hypothesis, persona_variant
        )

        # Call OpenAI API with retry logic
        response = self._call_openai_with_retry(prompt, config.llm_model)

        # Parse response based on output format
        if config.output_format == "markdown":
            return self._parse_markdown_response(response)
        else:
            return self._parse_json_response(response)

    def _call_openai_with_retry(
        self, prompt: str, model: str, max_retries: int = 3
    ) -> str:
        """Call OpenAI API with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert research interviewer conducting emotionally intelligent interviews.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                    temperature=0.7,
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                wait_time = 2**attempt  # Exponential backoff
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

    def _parse_markdown_response(self, response: str) -> Dict:
        """Parse markdown response into structured data"""
        sections = {"persona": "", "interview_transcript": "", "insight_summary": ""}

        current_section = None
        lines = response.split("\n")

        for line in lines:
            if line.startswith("## Persona"):
                current_section = "persona"
            elif line.startswith("## Interview Transcript"):
                current_section = "interview_transcript"
            elif line.startswith("## Insight Summary"):
                current_section = "insight_summary"
            elif current_section and line.strip():
                sections[current_section] += line + "\n"

        return sections

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response into structured data"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to markdown parsing if JSON is malformed
            logger.warning("JSON response malformed, falling back to markdown parsing")
            return self._parse_markdown_response(response)

    def _save_interview_results(
        self,
        project_dir: Path,
        mode: InterviewMode,
        hypothesis: ProblemHypothesis,
        persona_variant: int,
        result: Dict,
    ):
        """Save interview results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        persona_dir = project_dir / f"persona_variant_{persona_variant}"
        persona_dir.mkdir(exist_ok=True)

        # Save persona
        persona_file = persona_dir / f"persona_{timestamp}.md"
        with open(persona_file, "w") as f:
            f.write(f"# Persona for {mode.mode} - {hypothesis.label}\n\n")
            f.write(result.get("persona", ""))

        # Save interview transcript
        interview_file = persona_dir / f"interview_{timestamp}.md"
        with open(interview_file, "w") as f:
            f.write(f"# Interview Transcript for {mode.mode} - {hypothesis.label}\n\n")
            f.write(result.get("interview_transcript", ""))

        # Save insights
        insights_file = persona_dir / f"insights_{timestamp}.md"
        with open(insights_file, "w") as f:
            f.write(f"# Insight Summary for {mode.mode} - {hypothesis.label}\n\n")
            f.write(result.get("insight_summary", ""))

    def _update_master_report(
        self,
        master_report_path: Path,
        run_metadata: Dict,
        completed_interviews: int,
        failed_interviews: List[str],
        all_insights: List[Dict],
    ):
        """Update the master report with new run data and aggregated insights"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create or load existing report
        if master_report_path.exists():
            with open(master_report_path, "r") as f:
                existing_content = f.read()
        else:
            existing_content = "# Master Report\n\n"

        # Add new run entry
        new_entry = f"""
## Run: {timestamp}

**Metadata:**
- Model: {run_metadata['model']}
- Modes: {', '.join(run_metadata['modes'])}
- Total Interviews: {run_metadata['total_interviews']}
- Completed: {completed_interviews}
- Failed: {len(failed_interviews)}
- Run Directory: run_{run_metadata['run_timestamp']}

"""

        if failed_interviews:
            new_entry += "**Failed Interviews:**\n"
            for failure in failed_interviews:
                new_entry += f"- {failure}\n"
            new_entry += "\n"

        # Add aggregated insights summary
        if all_insights:
            new_entry += "**Key Insights Summary:**\n\n"
            
            # Group insights by mode
            mode_insights = {}
            for insight in all_insights:
                mode = insight['mode']
                if mode not in mode_insights:
                    mode_insights[mode] = []
                mode_insights[mode].append(insight)
            
            for mode, insights in mode_insights.items():
                new_entry += f"### {mode} Mode\n"
                
                # Group by hypothesis
                hypothesis_insights = {}
                for insight in insights:
                    hypothesis = insight['hypothesis']
                    if hypothesis not in hypothesis_insights:
                        hypothesis_insights[hypothesis] = []
                    hypothesis_insights[hypothesis].append(insight)
                
                for hypothesis, hypothesis_insights_list in hypothesis_insights.items():
                    new_entry += f"#### {hypothesis}\n"
                    
                    # Extract common themes
                    solution_fits = []
                    pain_points = []
                    micro_features = []
                    
                    for insight in hypothesis_insights_list:
                        insight_text = insight['insights']
                        
                        # Extract solution fit
                        if "Aligned? Yes" in insight_text:
                            solution_fits.append("✅ Aligned")
                        elif "Aligned? No" in insight_text:
                            solution_fits.append("❌ Misaligned")
                        
                        # Extract pain points
                        if "Pain Points:" in insight_text:
                            pain_section = insight_text.split("Pain Points:")[1].split("Desired Outcomes:")[0]
                            pain_points.extend([p.strip() for p in pain_section.split("-") if p.strip()])
                        
                        # Extract micro-features
                        if "Micro-feature Suggestions:" in insight_text:
                            features_section = insight_text.split("Micro-feature Suggestions:")[1]
                            micro_features.extend([f.strip() for f in features_section.split("\n") if f.strip() and not f.startswith("-")])
                    
                    # Add summary
                    if solution_fits:
                        aligned_count = solution_fits.count("✅ Aligned")
                        total_count = len(solution_fits)
                        new_entry += f"- **Solution Fit:** {aligned_count}/{total_count} personas aligned\n"
                    
                    if pain_points:
                        unique_pain_points = list(set(pain_points))[:3]  # Top 3 unique pain points
                        new_entry += f"- **Key Pain Points:** {', '.join(unique_pain_points)}\n"
                    
                    if micro_features:
                        unique_features = list(set(micro_features))[:3]  # Top 3 unique features
                        new_entry += f"- **Suggested Features:** {', '.join(unique_features)}\n"
                    
                    new_entry += "\n"
                
                new_entry += "\n"

        # Append to existing content
        updated_content = existing_content + new_entry

        # Write back to file
        with open(master_report_path, "w") as f:
            f.write(updated_content)

    def _update_roadmap(
        self,
        roadmap_path: Path,
        all_insights: List[Dict],
        run_metadata: Dict,
    ):
        """Generate or update the development roadmap based on interview insights"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create or load existing roadmap
        if roadmap_path.exists():
            with open(roadmap_path, "r") as f:
                existing_content = f.read()
        else:
            existing_content = "# Development Roadmap\n\n"

        # Generate new roadmap entry
        new_entry = f"""
## Run: {timestamp}

**Run Metadata:**
- Model: {run_metadata['model']}
- Modes: {', '.join(run_metadata['modes'])}
- Total Interviews: {run_metadata['total_interviews']}
- Run Directory: run_{run_metadata['run_timestamp']}

"""

        # Analyze insights and generate roadmap
        if all_insights:
            # Extract all micro-features and pain points
            all_micro_features = []
            all_pain_points = []
            solution_fit_scores = {}
            
            for insight in all_insights:
                insight_text = insight['insights']
                mode = insight['mode']
                hypothesis = insight['hypothesis']
                
                # Track solution fit by mode/hypothesis
                key = f"{mode}/{hypothesis}"
                if key not in solution_fit_scores:
                    solution_fit_scores[key] = {"aligned": 0, "total": 0}
                
                if "Aligned? Yes" in insight_text:
                    solution_fit_scores[key]["aligned"] += 1
                solution_fit_scores[key]["total"] += 1
                
                # Extract micro-features
                if "Micro-feature Suggestions:" in insight_text:
                    features_section = insight_text.split("Micro-feature Suggestions:")[1]
                    features = [f.strip() for f in features_section.split("\n") if f.strip() and not f.startswith("-")]
                    all_micro_features.extend(features)
                
                # Extract pain points
                if "Pain Points:" in insight_text:
                    pain_section = insight_text.split("Pain Points:")[1].split("Desired Outcomes:")[0]
                    pain_points = [p.strip() for p in pain_section.split("-") if p.strip()]
                    all_pain_points.extend(pain_points)
            
            # Generate prioritized roadmap
            new_entry += "### Prioritized Development Recommendations\n\n"
            
            # High Priority: Features with strong alignment
            high_priority_features = []
            medium_priority_features = []
            low_priority_features = []
            
            # Categorize features by alignment strength
            for key, scores in solution_fit_scores.items():
                alignment_rate = scores["aligned"] / scores["total"]
                if alignment_rate >= 0.8:
                    high_priority_features.append(key)
                elif alignment_rate >= 0.5:
                    medium_priority_features.append(key)
                else:
                    low_priority_features.append(key)
            
            # Add high priority recommendations
            if high_priority_features:
                new_entry += "#### 🔥 High Priority (Strong User Alignment)\n\n"
                for feature in high_priority_features:
                    mode, hypothesis = feature.split("/")
                    new_entry += f"- **{hypothesis}** ({mode} mode)\n"
                    new_entry += f"  - **Justification:** Strong user alignment across personas\n"
                    new_entry += f"  - **Success Measures:** User engagement, retention, positive feedback\n"
                    new_entry += f"  - **Timeline:** Next sprint\n\n"
            
            # Add medium priority recommendations
            if medium_priority_features:
                new_entry += "#### ⚡ Medium Priority (Moderate User Alignment)\n\n"
                for feature in medium_priority_features:
                    mode, hypothesis = feature.split("/")
                    new_entry += f"- **{hypothesis}** ({mode} mode)\n"
                    new_entry += f"  - **Justification:** Moderate user alignment, needs refinement\n"
                    new_entry += f"  - **Success Measures:** User testing, iteration based on feedback\n"
                    new_entry += f"  - **Timeline:** Next quarter\n\n"
            
            # Add low priority recommendations
            if low_priority_features:
                new_entry += "#### 📋 Low Priority (Weak User Alignment)\n\n"
                for feature in low_priority_features:
                    mode, hypothesis = feature.split("/")
                    new_entry += f"- **{hypothesis}** ({mode} mode)\n"
                    new_entry += f"  - **Justification:** Weak user alignment, needs significant rethinking\n"
                    new_entry += f"  - **Success Measures:** User research, concept validation\n"
                    new_entry += f"  - **Timeline:** Future consideration\n\n"
            
            # Add micro-feature suggestions
            if all_micro_features:
                unique_features = list(set(all_micro_features))
                new_entry += "### 🎯 Micro-Feature Suggestions\n\n"
                for i, feature in enumerate(unique_features[:10], 1):  # Top 10 features
                    new_entry += f"{i}. **{feature}**\n"
                new_entry += "\n"
            
            # Add pain point analysis
            if all_pain_points:
                unique_pain_points = list(set(all_pain_points))
                new_entry += "### 🚨 Key Pain Points to Address\n\n"
                for i, pain_point in enumerate(unique_pain_points[:5], 1):  # Top 5 pain points
                    new_entry += f"{i}. {pain_point}\n"
                new_entry += "\n"

        # Append to existing content
        updated_content = existing_content + new_entry

        # Write back to file
        with open(roadmap_path, "w") as f:
            f.write(updated_content)

    def _create_test_config(self) -> ProjectConfig:
        """Create a test configuration for testing purposes"""
        return ProjectConfig(
            project_name="TestProject",
            llm_model="gpt-4o",
            product_sketch="An AI-powered emotional regulation coaching app",
            interview_modes=[
                InterviewMode(
                    mode="trauma_informed",
                    persona_count=3,
                    problem_hypotheses=[
                        ProblemHypothesis(
                            label="emotional_overwhelm",
                            description="Users experience intense emotional overwhelm",
                        )
                    ],
                )
            ],
            output_format="markdown",
        )

    def _generate_interview_prompt(
        self,
        config: ProjectConfig,
        mode: InterviewMode,
        hypothesis: ProblemHypothesis,
        persona_variant: int,
    ) -> str:
        """Generate the interview prompt for a specific persona"""
        prompt = f"""[INTERNAL CONTEXT: Product sketch—do NOT share with persona]
"{config.product_sketch}"

=== ASSIGNMENT ===
Project: {config.project_name}
Interview Mode: {mode.mode}
Problem Hypothesis: {hypothesis.label}
Hypothesis Description: {hypothesis.description}
Persona Variant: {persona_variant} (one of three contrasting personas for this hypothesis)

=== PHASE 1: Persona Construction ===
Generate a fully fleshed, believable user persona that contrasts with the other variants along at least two meaningful axes (e.g., age/life stage, emotional baseline, coping strategy, readiness/resistance, internal conflict). Include:
- Placeholder name
- Demographics and context
- Emotional baseline and variability
- Internal conflicts or resistance tied to the hypothesis
- Existing coping behaviors or avoidance patterns
- Implicit goals and success definitions

=== PHASE 2: Interview Simulation ===
Conduct a semi-structured, emotionally intelligent interview of 7–10 questions, roleplaying two voices:
1. Interviewer: trauma-aware / mode-appropriate tone. Ask open-ended, exploratory questions about the persona's lived experience relevant to the hypothesis. Do NOT mention, pitch, validate, or justify the internal solution sketch.
2. Persona: respond with human-like nuance (hesitations, contradictions, emotional texture, subtext, partial truths, hopes, and concerns).

Focus on:
- Current experience with the problem
- Emotional triggers and friction
- Desired outcomes and what "success" feels like
- Language the persona uses (self-talk and external)
- Behavioral / emotional readiness or distress signals

Format the dialogue as a transcript with clear speaker labels.

=== PHASE 3: Insight Synthesis ===
Produce a structured insight summary including:
- Pain Points
- Desired Outcomes
- Language Patterns
- Success Signals
- Emotional Triggers / Friction
- Recommended Tone (how to speak to this persona)
- Personality / Cognitive Tendencies
- Solution Fit Assessment:
    * Aligned? (yes/no)
    * If aligned: why.
    * If misaligned: what would better serve this persona.
- Up to 3 organically emerging micro-feature suggestions.

Output all of the above in the specified {config.output_format} with explicit sections:
## Persona
## Interview Transcript
## Insight Summary"""

        return prompt


def main():
    """Main entry point"""
    try:
        engine = LLMInterviewEngine()
        engine.run_cli()
    except KeyboardInterrupt:
        print("\n\n👋 Interview run cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
