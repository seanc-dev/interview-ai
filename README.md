# LLM Interview Engine

A robust, interactive CLI tool for running LLM-to-LLM research interviews to evaluate problem/solution hypotheses for emotionally intelligent coaching products. The engine focuses on fidelity, contrast, and cumulative insight through structured persona generation and interview simulation.

## Features

- **Interactive CLI**: Guided project creation and management
- **Multi-Persona Generation**: Default 3 contrasting personas per hypothesis
- **Structured Interviews**: Three-phase interview process (Persona Construction, Interview Simulation, Insight Synthesis)
- **Master Report Aggregation**: Cumulative analysis across runs with pattern detection
- **Robust Error Handling**: Exponential backoff for API calls and graceful failure handling
- **Flexible Output Formats**: Markdown and JSON output options

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd interview-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

Run the interview engine:

```bash
python llm_interview_engine.py
```

The CLI will guide you through:

1. Choosing between new or existing projects
2. Configuring interview parameters
3. Running interview simulations
4. Generating master reports

### Project Structure

```
outputs/
├── Project_Name/
│   ├── config.json                 # Project configuration
│   ├── master_report.md            # Cumulative analysis
│   ├── persona_variant_1/
│   │   ├── persona_YYYYMMDD_HHMMSS.md
│   │   ├── interview_YYYYMMDD_HHMMSS.md
│   │   └── insights_YYYYMMDD_HHMMSS.md
│   ├── persona_variant_2/
│   └── persona_variant_3/
└── README.md
```

### Configuration Schema

The engine uses a JSON configuration format:

```json
{
  "project_name": "string",
  "llm_model": "string (default: gpt-4o)",
  "product_sketch": "string (internal context)",
  "interview_modes": [
    {
      "mode": "string",
      "persona_count": "integer (default: 3)",
      "problem_hypotheses": [
        {
          "label": "string",
          "description": "string"
        }
      ]
    }
  ],
  "output_format": "markdown|json"
}
```

### Interview Process

Each interview follows a three-phase structure:

#### Phase 1: Persona Construction

- Generates contrasting personas along demographic, emotional, and behavioral axes
- Includes demographics, emotional baseline, internal conflicts, and coping patterns

#### Phase 2: Interview Simulation

- 7-10 trauma-aware questions
- Human-like responses with emotional nuance
- Focus on lived experience and emotional triggers

#### Phase 3: Insight Synthesis

- Structured analysis of pain points, desired outcomes, and language patterns
- Solution fit assessment
- Micro-feature suggestions

### Master Report System

The master report (`master_report.md`) provides cumulative analysis across all runs:

#### Run Metadata

- Timestamps and model information
- Parameters and configuration used
- Persona identifiers and interview modes

#### Per-Persona Analysis

- Distilled insight summaries
- Key pain points and desired outcomes
- Language patterns and emotional triggers

#### Cross-Persona Patterns

- Common pain points across personas
- Divergent needs and contradictions
- Consensus success signals
- Risk identification (e.g., misaligned readiness)

#### Recommendations

- Hypothesis prioritization with confidence estimates
- Suggested micro-feature experiments
- Actionable insights and pivots

#### Change Tracking

- Append-only change log
- Incremental updates across runs
- Evolution tracking of insights

### CLI Options

#### New Project Creation

- Interactive prompts for all required fields
- Validation of inputs
- Automatic config file generation

#### Existing Project Management

- Load previous configurations
- View run summaries
- Modify parameters or create variants
- "Same again" mode for quick repetition

#### Interview Execution

- Batch processing with progress tracking
- Error resilience with exponential backoff
- Structured output generation

## Example Configuration

See `example_config.json` for a complete example with:

- Emotional regulation coaching product
- Trauma-informed and cognitive-behavioral interview modes
- Multiple problem hypotheses per mode

## Output Formats

### Markdown Format

Structured sections with clear headers:

```markdown
## Persona

[Persona description]

## Interview Transcript

[Q&A dialogue]

## Insight Summary

[Structured analysis]
```

### JSON Format

Machine-readable structure for programmatic analysis:

```json
{
  "persona": {...},
  "interview_transcript": [...],
  "insight_summary": {...}
}
```

## Error Handling

The engine includes robust error handling:

- Exponential backoff for API rate limits
- Graceful degradation on network issues
- Input validation and config verification
- Failure logging without batch abortion

## Development

### Running Tests

```bash
pytest test_llm_interview_engine.py
```

### Test Structure

- Unit tests for core functionality
- Integration tests for end-to-end workflows
- Snapshot tests for output consistency

## Contributing

1. Follow TDD principles
2. Add tests for new features
3. Update documentation
4. Ensure error handling coverage

## License

[Add your license information here]
