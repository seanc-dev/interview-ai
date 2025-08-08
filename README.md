# LLM Interview Engine

A robust, interactive CLI tool for running LLM-to-LLM research interviews to evaluate problem/solution hypotheses for emotionally intelligent coaching products. The engine focuses on fidelity, contrast, and cumulative insight through structured persona generation and interview simulation.

## Features

- **Interactive CLI**: Guided project creation and management
- **Multi-Persona Generation**: Default 3 contrasting personas per hypothesis
- **Structured Interviews**: Three-phase interview process (Persona Construction, Interview Simulation, Insight Synthesis)
- **Master Report Aggregation**: Cumulative analysis across runs with pattern detection
- **Robust Error Handling**: Exponential backoff for API calls and graceful failure handling
- **Flexible Output Formats**: Markdown and JSON output options
- **Version-Aware Iterative Research**: Support for versioned configs, outputs, and design logs
- **Design Log Evolution**: Automatic tracking of design decisions and insights across versions
- **Product Resonance Analysis**: Quantitative assessment of product-market fit and alignment
- **Iterative Research Cycles**: Automatic N-cycle research with product evolution
- **Non-Deterministic Interviews**: Unique personas and indirect hypothesis testing
- **Automatic Product Evolution**: AI-driven product sketch and hypothesis refinement

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

#### Iterative Research Mode (Recommended)

Run N cycles of research with automatic product evolution:

```bash
python llm_interview_engine.py --config-dir config/v1/ --cycles 3 --evolution-enabled
```

This will:

1. Load `ygt_config.json` from the specified directory
2. Run 3 interview cycles automatically
3. Generate unique personas for each cycle
4. Analyze insights to evolve product sketch and hypotheses
5. Save outputs to `outputs/v1/ProjectName/run_<TIMESTAMP>/`
6. Update design log with evolution history
7. Add resonance analysis to master report

#### Version-Aware Mode

Run with a specific config directory:

```bash
python llm_interview_engine.py --config-dir config/v1/
```

This will:

1. Load `ygt_config.json` from the specified directory
2. Run interviews using the configuration
3. Save outputs to `outputs/v1/ProjectName/run_<TIMESTAMP>/`
4. Update design log in the config directory
5. Add resonance analysis to master report

#### Legacy Mode

Run the interactive CLI:

```bash
python llm_interview_engine.py
```

The CLI will guide you through:

1. Choosing between new or existing projects
2. Configuring interview parameters
3. Running interview simulations
4. Generating master reports

### Project Structure

#### Iterative Research Structure (Recommended)

```
config/
├── v1/
│   ├── ygt_config.json            # Version 1 configuration
│   └── design_log.md              # Design evolution log
├── v2/
│   ├── ygt_config.json            # Version 2 configuration
│   └── design_log.md              # Design evolution log
└── ...

outputs/
├── v1/
│   └── ProjectName/
│       ├── master_report.md        # Cumulative analysis with resonance
│       ├── roadmap.md              # Development roadmap
│       ├── evolution_history.md    # Product evolution tracking
│       └── run_YYYYMMDD_HHMMSS/   # Individual run results
├── v2/
│   └── ProjectName/
│       ├── master_report.md
│       ├── roadmap.md
│       └── run_YYYYMMDD_HHMMSS/
└── ...
```

#### Version-Aware Structure

```
config/
├── v1/
│   ├── ygt_config.json            # Version 1 configuration
│   └── design_log.md              # Design evolution log
├── v2/
│   ├── ygt_config.json            # Version 2 configuration
│   └── design_log.md              # Design evolution log
└── ...

outputs/
├── v1/
│   └── ProjectName/
│       ├── master_report.md        # Cumulative analysis with resonance
│       ├── roadmap.md              # Development roadmap
│       └── run_YYYYMMDD_HHMMSS/   # Individual run results
├── v2/
│   └── ProjectName/
│       ├── master_report.md
│       ├── roadmap.md
│       └── run_YYYYMMDD_HHMMSS/
└── ...
```

#### Legacy Structure

```
outputs/
├── ProjectName/
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
- **Non-deterministic**: Each cycle generates unique personas using cycle-based seeding

#### Phase 2: Interview Simulation

- 7-10 trauma-aware questions
- Human-like responses with emotional nuance
- Focus on lived experience and emotional triggers
- **Indirect hypothesis testing**: Questions target underlying problems without revealing hypotheses

#### Phase 3: Insight Synthesis

- Structured analysis of pain points, desired outcomes, and language patterns
- Solution fit assessment
- Micro-feature suggestions
- **Evolution signals**: Identifies opportunities for product improvement

### Iterative Research Process

The engine supports automatic iterative research with product evolution:

#### 1. Cycle Execution

- Run N interview cycles automatically
- Generate unique personas per cycle
- Conduct non-deterministic interviews
- Collect and analyze insights

#### 2. Product Evolution

- Analyze insights for evolution signals
- Generate new product sketches based on feedback
- Create new hypotheses addressing identified gaps
- Validate evolution quality

#### 3. Evolution Tracking

- Maintain evolution history across cycles
- Track alignment improvements
- Document design decisions
- Generate evolution reports

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

#### Product Resonance Analysis (Version-Aware)

- Overall alignment rates across personas
- Mode-specific resonance metrics
- Tone and hypothesis validation
- Feature direction recommendations
- Risk mitigation strategies

#### Evolution Tracking (Iterative Research)

- Cycle-by-cycle improvement metrics
- Product sketch evolution history
- Hypothesis refinement tracking
- Convergence analysis

#### Recommendations

- Hypothesis prioritization with confidence estimates
- Suggested micro-feature experiments
- Actionable insights and pivots

#### Change Tracking

- Append-only change log
- Incremental updates across runs
- Evolution tracking of insights

### Design Log System

Each version maintains a design log (`design_log.md`) that tracks:

#### Run Summaries

- Timestamp and configuration details
- Modes and persona counts executed
- Hypotheses tested in each run

#### Key Insights

- Solution fit rates by hypothesis
- Common pain points identified
- Suggested micro-features
- Emerging themes and patterns

#### Product Evolution

- Product sketch critiques
- Recommendations for next version
- Evolution rationale and decisions

### Non-Deterministic Interview Features

#### Unique Persona Generation

- Cycle-based seeding ensures uniqueness
- Diverse demographic and psychological profiles
- Contrasting emotional and behavioral patterns

#### Indirect Hypothesis Testing

- Questions target underlying problems
- Hypotheses hidden from personas
- Natural conversation flow
- Diverse question templates

#### Interview Variety

- Different questions per cycle
- Varied emotional and challenge contexts
- Randomized persona characteristics

### CLI Options

#### Iterative Research Mode

```bash
python llm_interview_engine.py --config-dir config/v1/ --cycles 3 --evolution-enabled
```

#### Version-Aware Mode

```bash
python llm_interview_engine.py --config-dir config/v1/
```

#### Legacy Mode

```bash
python llm_interview_engine.py
```

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
- Cycle failure recovery in iterative mode

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest test_iterative_research_engine.py::TestIterativeResearchEngine
pytest test_iterative_research_engine.py::TestProductEvolutionEngine
pytest test_iterative_research_engine.py::TestNonDeterministicInterviewer
```

### Test Structure

- Unit tests for core functionality
- Integration tests for end-to-end workflows
- Snapshot tests for output consistency
- TDD approach with comprehensive coverage

## Contributing

1. Follow TDD principles
2. Add tests for new features
3. Update documentation
4. Ensure error handling coverage
5. Maintain non-deterministic interview quality

## License

[Add your license information here]
