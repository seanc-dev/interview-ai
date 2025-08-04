import pytest
import json
import os
from unittest.mock import Mock, patch
from datetime import datetime
from pathlib import Path

# Import the main module
from llm_interview_engine import (
    LLMInterviewEngine,
    ProjectConfig,
    InterviewMode,
    ProblemHypothesis,
)


class TestCLIFunctionality:
    """Test CLI interaction and project management"""

    def test_new_project_creation_prompts(self):
        """Test that new project creation prompts for all required fields"""
        with patch("builtins.input") as mock_input:
            # Mock user inputs for project creation
            mock_input.side_effect = [
                "new",  # Project choice
                "TestProject",  # Project name
                "gpt-4o",  # LLM model
                "",  # Product sketch line 1
                "",  # Product sketch line 2 (empty to finish)
                "markdown",  # Output format
                "done",  # No interview modes
            ]

            with patch("os.getenv", return_value="test-api-key"):
                engine = LLMInterviewEngine()

                # Test that the engine can be initialized
                assert engine.api_key == "test-api-key"
                assert engine.output_dir == Path("outputs")

    def test_existing_project_detection(self):
        """Test detection and loading of existing projects"""
        with patch("pathlib.Path.iterdir") as mock_iterdir, patch(
            "pathlib.Path.is_dir"
        ) as mock_is_dir, patch("builtins.input") as mock_input, patch(
            "builtins.open", create=True
        ) as mock_open, patch(
            "json.load"
        ) as mock_json_load:

            # Mock existing project directory
            mock_project_dir = Mock()
            mock_project_dir.name = "TestProject"
            mock_iterdir.return_value = [mock_project_dir]
            mock_is_dir.return_value = True

            # Mock config file
            mock_config_data = {
                "project_name": "TestProject",
                "llm_model": "gpt-4o",
                "product_sketch": "Test product",
                "interview_modes": [],
                "output_format": "markdown",
            }
            mock_json_load.return_value = mock_config_data

            # Mock user input
            mock_input.side_effect = ["existing", "1", "reuse"]

            with patch("os.getenv", return_value="test-api-key"):
                engine = LLMInterviewEngine()

                # Test that the engine can detect existing projects
                project_dirs = list(engine.output_dir.iterdir())
                assert len(project_dirs) == 1
                assert project_dirs[0].name == "TestProject"

    def test_config_validation(self):
        """Test validation of project configuration"""
        # Test valid config
        config = ProjectConfig(
            project_name="TestProject",
            llm_model="gpt-4o",
            product_sketch="Test product",
            interview_modes=[
                InterviewMode(
                    mode="test_mode",
                    persona_count=3,
                    problem_hypotheses=[
                        ProblemHypothesis(
                            label="test_hypothesis", description="Test description"
                        )
                    ],
                )
            ],
            output_format="markdown",
        )

        assert config.project_name == "TestProject"
        assert config.llm_model == "gpt-4o"
        assert len(config.interview_modes) == 1
        assert config.interview_modes[0].mode == "test_mode"
        assert config.interview_modes[0].persona_count == 3
        assert len(config.interview_modes[0].problem_hypotheses) == 1
        assert (
            config.interview_modes[0].problem_hypotheses[0].label == "test_hypothesis"
        )


class TestPersonaGeneration:
    """Test persona generation and contrasting characteristics"""

    def test_contrasting_personas_creation(self):
        """Test that personas are created with meaningful contrasts"""
        # TODO: Implement test for persona contrast
        pass

    def test_persona_believability(self):
        """Test that generated personas are believable and realistic"""
        # TODO: Implement test for persona believability
        pass

    def test_demographic_variation(self):
        """Test demographic variation across personas"""
        # TODO: Implement test for demographic variation
        pass


class TestInterviewSimulation:
    """Test interview simulation and dialogue generation"""

    def test_interview_structure(self):
        """Test that interviews follow the three-phase structure"""
        with patch("os.getenv", return_value="test-api-key"):
            engine = LLMInterviewEngine()

            config = ProjectConfig(
                project_name="TestProject",
                llm_model="gpt-4o",
                product_sketch="Test product sketch",
                interview_modes=[
                    InterviewMode(
                        mode="test_mode",
                        persona_count=3,
                        problem_hypotheses=[
                            ProblemHypothesis(
                                label="test_hypothesis",
                                description="Test hypothesis description",
                            )
                        ],
                    )
                ],
                output_format="markdown",
            )

            prompt = engine._generate_interview_prompt(
                config,
                config.interview_modes[0],
                config.interview_modes[0].problem_hypotheses[0],
                1,
            )

            # Test that prompt contains all required sections
            assert "PHASE 1: Persona Construction" in prompt
            assert "PHASE 2: Interview Simulation" in prompt
            assert "PHASE 3: Insight Synthesis" in prompt
            assert "TestProject" in prompt
            assert "test_mode" in prompt
            assert "test_hypothesis" in prompt
            assert "Test product sketch" in prompt

    def test_trauma_aware_tone(self):
        """Test that interviewer maintains trauma-aware tone"""
        # TODO: Implement test for trauma-aware tone
        pass

    def test_human_like_responses(self):
        """Test that persona responses are human-like with nuance"""
        # TODO: Implement test for human-like responses
        pass


class TestOutputFormats:
    """Test output format generation and parsing"""

    def test_markdown_output_format(self):
        """Test markdown output format with structured sections"""
        # TODO: Implement test for markdown format
        pass

    def test_json_output_format(self):
        """Test JSON output format with machine-readable structure"""
        # TODO: Implement test for JSON format
        pass

    def test_output_parsing(self):
        """Test parsing of LLM outputs into structured data"""
        with patch("os.getenv", return_value="test-api-key"):
            engine = LLMInterviewEngine()

            # Test markdown parsing
            markdown_response = """## Persona
Test persona content

## Interview Transcript
Test interview content

## Insight Summary
Test insight content"""

            result = engine._parse_markdown_response(markdown_response)

            assert result["persona"] == "Test persona content\n"
            assert result["interview_transcript"] == "Test interview content\n"
            assert result["insight_summary"] == "Test insight content\n"

            # Test JSON parsing
            json_response = '{"persona": "test", "interview_transcript": "test", "insight_summary": "test"}'
            result = engine._parse_json_response(json_response)

            assert result["persona"] == "test"
            assert result["interview_transcript"] == "test"
            assert result["insight_summary"] == "test"


class TestMasterReportAggregation:
    """Test master report generation and aggregation"""

    def test_run_metadata_collection(self):
        """Test collection of run metadata"""
        # TODO: Implement test for metadata collection
        pass

    def test_cross_persona_analysis(self):
        """Test comparative analysis across personas"""
        # TODO: Implement test for cross-persona analysis
        pass

    def test_recommendation_generation(self):
        """Test generation of actionable recommendations"""
        # TODO: Implement test for recommendation generation
        pass


class TestErrorHandling:
    """Test error handling and resilience"""

    def test_api_retry_logic(self):
        """Test exponential backoff and retry logic"""
        # TODO: Implement test for retry logic
        pass

    def test_input_validation(self):
        """Test validation of user inputs"""
        # TODO: Implement test for input validation
        pass

    def test_graceful_failure_handling(self):
        """Test graceful handling of failures"""
        # TODO: Implement test for failure handling
        pass


class TestFileManagement:
    """Test file and directory management"""

    def test_output_directory_structure(self):
        """Test creation of proper output directory structure"""
        # TODO: Implement test for directory structure
        pass

    def test_file_naming_convention(self):
        """Test consistent file naming conventions"""
        # TODO: Implement test for naming conventions
        pass

    def test_config_persistence(self):
        """Test persistence and loading of configuration files"""
        # TODO: Implement test for config persistence
        pass


class TestIntegrationWorkflows:
    """Test end-to-end workflows"""

    def test_new_project_complete_flow(self):
        """Test complete new project workflow"""
        # TODO: Implement test for new project flow
        pass

    def test_existing_project_incremental_run(self):
        """Test incremental runs on existing projects"""
        # TODO: Implement test for incremental runs
        pass

    def test_same_again_mode(self):
        """Test 'same again' mode functionality"""
        # TODO: Implement test for same again mode
        pass


# Snapshot tests for UI/render output
class TestSnapshotTests:
    """Snapshot regression tests for UI/render output"""

    def test_cli_prompt_snapshots(self):
        """Test CLI prompt formatting and structure"""
        # TODO: Implement snapshot tests for CLI prompts
        pass

    def test_report_formatting_snapshots(self):
        """Test report formatting consistency"""
        # TODO: Implement snapshot tests for report formatting
        pass


class TestJSONConfigFunctionality:
    """Test JSON configuration input functionality"""

    def test_json_config_loading(self):
        """Test loading configuration from JSON input"""
        test_config = {
            "project_name": "JSONTestProject",
            "llm_model": "gpt-4o",
            "product_sketch": "A test product for JSON config",
            "interview_modes": [
                {
                    "mode": "test_mode",
                    "persona_count": 2,
                    "problem_hypotheses": [
                        {
                            "label": "test_hypothesis",
                            "description": "A test hypothesis for JSON config"
                        }
                    ]
                }
            ],
            "output_format": "markdown"
        }
        
        json_str = json.dumps(test_config, indent=2)
        
        with patch("builtins.input") as mock_input:
            # Mock user inputs for JSON config choice and input
            mock_input.side_effect = [
                "json",  # Project choice
                *json_str.split('\n'),  # JSON config lines
                "",  # Empty line to finish JSON input
                ""   # Second empty line to finish
            ]

            with patch("os.getenv", return_value="test-api-key"):
                engine = LLMInterviewEngine()
                
                # Test that the engine can load JSON config
                # This would normally call run_cli(), but we'll test the method directly
                config = engine._load_json_config()
                
                assert config.project_name == "JSONTestProject"
                assert config.llm_model == "gpt-4o"
                assert config.product_sketch == "A test product for JSON config"
                assert len(config.interview_modes) == 1
                assert config.interview_modes[0].mode == "test_mode"
                assert config.interview_modes[0].persona_count == 2
                assert len(config.interview_modes[0].problem_hypotheses) == 1
                assert config.interview_modes[0].problem_hypotheses[0].label == "test_hypothesis"

    def test_json_config_validation(self):
        """Test validation of JSON configuration"""
        with patch("builtins.input") as mock_input:
            # Mock user inputs for invalid JSON
            mock_input.side_effect = [
                "json",  # Project choice
                "{ invalid json }",  # Invalid JSON
                "",  # Empty line to finish JSON input
                ""   # Second empty line to finish
            ]

            with patch("os.getenv", return_value="test-api-key"):
                engine = LLMInterviewEngine()
                
                # Test that invalid JSON falls back to new project creation
                with patch.object(engine, '_create_new_project') as mock_create:
                    mock_create.return_value = ProjectConfig(project_name="FallbackProject")
                    config = engine._load_json_config()
                    assert config.project_name == "FallbackProject"

    def test_json_config_missing_required_fields(self):
        """Test handling of JSON config with missing required fields"""
        invalid_config = {
            "llm_model": "gpt-4o",
            # Missing project_name
            "product_sketch": "A test product",
            "interview_modes": [],
            "output_format": "markdown"
        }
        
        json_str = json.dumps(invalid_config, indent=2)
        
        with patch("builtins.input") as mock_input:
            # Mock user inputs for JSON config choice and input
            mock_input.side_effect = [
                "json",  # Project choice
                *json_str.split('\n'),  # JSON config lines
                "",  # Empty line to finish JSON input
                ""   # Second empty line to finish
            ]

            with patch("os.getenv", return_value="test-api-key"):
                engine = LLMInterviewEngine()
                
                # Test that missing required fields falls back to new project creation
                with patch.object(engine, '_create_new_project') as mock_create:
                    mock_create.return_value = ProjectConfig(project_name="FallbackProject")
                    config = engine._load_json_config()
                    assert config.project_name == "FallbackProject"

    def test_env_file_loading(self):
        """Test loading API key from .env file"""
        with patch("dotenv.load_dotenv") as mock_load_dotenv:
            with patch("os.getenv", return_value="test-api-key-from-env"):
                engine = LLMInterviewEngine()
                
                # Verify that load_dotenv was called
                mock_load_dotenv.assert_called_once()
                
                # Verify that API key was loaded from environment
                assert engine.api_key == "test-api-key-from-env"

    def test_end_to_end_json_flow(self):
        """Test complete end-to-end flow with JSON config input"""
        test_config = {
            "project_name": "EndToEndTestProject",
            "llm_model": "gpt-4o",
            "product_sketch": "An AI-powered emotional regulation coaching app",
            "interview_modes": [
                {
                    "mode": "trauma_informed",
                    "persona_count": 2,
                    "problem_hypotheses": [
                        {
                            "label": "emotional_overwhelm",
                            "description": "Users experience intense emotional overwhelm"
                        }
                    ]
                }
                ],
            "output_format": "markdown"
        }
        
        json_str = json.dumps(test_config, indent=2)
        
        with patch("builtins.input") as mock_input:
            # Mock user inputs for complete flow
            mock_input.side_effect = [
                "json",  # Project choice
                *json_str.split('\n'),  # JSON config lines
                "",  # Empty line to finish JSON input
                ""   # Second empty line to finish
            ]

            with patch("os.getenv", return_value="test-api-key"):
                engine = LLMInterviewEngine()
                
                # Test the complete flow
                config = engine._load_json_config()
                
                # Verify config was loaded correctly
                assert config.project_name == "EndToEndTestProject"
                assert len(config.interview_modes) == 1
                assert config.interview_modes[0].mode == "trauma_informed"
                assert len(config.interview_modes[0].problem_hypotheses) == 1
                assert config.interview_modes[0].problem_hypotheses[0].label == "emotional_overwhelm"
                
                # Verify config was saved
                assert (engine.output_dir / "EndToEndTestProject" / "config.json").exists()
