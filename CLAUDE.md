# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making" from NeurIPS 2024. The system uses multiple Large Language Models (LLMs) to collaboratively solve medical questions through three adaptive difficulty levels: basic (expert arbitration), intermediate (expert collaboration), and advanced (multi-disciplinary teams).

## Current Status (Ultimate Branch)

### Performance Results (Latest - Updated)
- **Basic Mode**: 87.05% accuracy (195/224 samples) - exceptional performance
- **Intermediate Mode**: 78.67% accuracy (59/75 samples) with multi-agent collaboration
- **Advanced Mode**: 75.76% accuracy (25/33 samples) with MDT approach
- **Overall System**: 84.04% accuracy across all difficulty levels
- **Token Efficiency**: 8.67M total tokens (5.75M input + 2.93M output)

### Latest Improvements
- ✅ **Enhanced Evaluation System**: Multi-pattern answer extraction for diverse LLM response formats
- ✅ **Robust Parsing**: "Answer: X" pattern detection and parse error tracking
- ✅ **JSON-based Communication**: All processing modes use structured JSON responses
- ✅ **Enhanced Basic Mode**: 3-expert recruitment + arbitrator system (was single agent)
- ✅ **Word Limits**: Enforced response limits (50-300 words) for efficiency

## Development Setup

### Environment Setup
1. Install Python dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with API keys:
   - `openai_api_key=your_openai_api_key_here`
   - `genai_api_key=your_gemini_api_key_here`
3. Activate virtual environment if using one: `source venv/bin/activate`

### Running the System
Basic usage:
```bash
python3 main.py --dataset medqa --model gemini-2.5-flash-lite-preview-06-17 --difficulty adaptive --num_samples 1
```

Available models:
- `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-flash-lite-preview-06-17` (requires `genai_api_key`)
- `gpt-4o-mini`, `gpt-4.1-mini` (requires `openai_api_key`)

Difficulty modes:
- `adaptive`: System determines complexity automatically
- `basic`: 3-expert recruitment + arbitrator processing
- `intermediate`: Expert collaboration with multi-round debate
- `advanced`: Multi-disciplinary team approach

### Evaluation and Testing
```bash
# Evaluate system output and generate CSV reports
python evaluate_text_output.py

# Split test data by difficulty for analysis
python split_test_data.py

# Extract specific questions by ID for debugging
python extract_by_question_id.py
```

Key evaluation files:
- `output/`: JSON output files from system runs
- `evaluation/`: CSV evaluation reports with accuracy metrics
- `data/medqa/`: Test datasets split by difficulty level

**Important**: The `evaluate_text_output.py` script contains configurable input/output filenames at the top:
```python
input_filename = 'output/inter_json_adaptive_332samples.json'  # Update as needed
output_filename = 'evaluation/inter_json_adaptive_332samples.csv'  # Update as needed
```

## Architecture

### Core Components

1. **Agent Class** (`utils.py:53+`): Wrapper for LLM interactions supporting both OpenAI and Gemini models
2. **Group Class** (`utils.py:190+`): Manages collaborative medical expert teams
3. **Processing Pipeline** (`main.py`): Main execution loop with difficulty assessment and routing

### Processing Modes

- **Basic Processing** (`utils.py:540+`): 
  - **Expert Recruitment**: 3 independent medical specialists with equal authority
  - **Independent Analysis**: Each expert provides structured JSON response with reasoning
  - **Arbitrator Decision**: Medical arbitrator synthesizes expert opinions into final answer
  - **Performance**: 87.05% accuracy, highly token-efficient
  
- **Intermediate Processing** (`utils.py:756+`): 
  - **Expert Recruitment**: 3 experts with hierarchical relationships
  - **Multi-round Debate**: Collaborative discussion with adaptive participation
  - **JSON Communication**: Structured participation decisions and expert selection
  - **Moderated Decision**: Final moderator synthesizes team consensus
  - **Performance**: 78.67% accuracy with complex multi-agent interactions
  
- **Advanced Processing** (`utils.py:1084+`): 
  - **MDT Formation**: Multidisciplinary teams with specialized roles
  - **JSON Team Structure**: Structured team and member definitions
  - **Parallel Assessment**: Teams work independently then coordinate
  - **Overall Coordinator**: Final decision synthesis
  - **Performance**: 75.76% accuracy with comprehensive team approach

### Data Structure

- Input: JSONL files in `data/{dataset}/` (test.jsonl, train.jsonl)
- Output: JSON files in `output/` with format: `{model}_{dataset}_{difficulty}_{samples}samples.json`
- Each result includes: question, options, ground truth, model response, determined difficulty, and token usage

### System Controls (main.py)

```python
# Processing skip switches - control which difficulty levels to process
SKIP_BASIC = False          # Skip basic difficulty questions
SKIP_INTERMEDIATE = False   # Skip intermediate difficulty questions  
SKIP_ADVANCED = False       # Skip advanced difficulty questions

# Debug controls (utils.py)
SHOW_INTERACTION_TABLE = False  # Display agent interaction tables in intermediate mode
```

### Key Utility Functions

- `setup_model()`: Configures API clients based on model type
- `determine_difficulty()`: Uses LLM with JSON format to assess question complexity for adaptive mode
- `load_data()`: Loads test questions and exemplars from dataset files
- `create_question()`: Formats questions with randomized multiple choice options

### Evaluation System (`evaluate_text_output.py`)

Enhanced parsing system with multi-pattern answer extraction:

```python
def extract_final_answer_or_answer(text):
    # Handles multiple response formats:
    # 1. "Answer: C" patterns (common in majority_vote responses)
    # 2. "B) Normal hemoglobin..." patterns  
    # 3. "(A) Some answer" parentheses formats
    # 4. Parse error cases marked as 'X'
    # 5. Various structured answer formats
```

**Key Features**:
- **Multi-pattern Recognition**: Handles diverse LLM response formats including complex majority_vote responses
- **Parse Error Tracking**: Identifies and tracks parsing failures with 'X' marker
- **CSV Export**: Generates detailed evaluation reports with per-difficulty accuracy metrics
- **Answer Validation**: Robust extraction from malformed or incomplete responses

**Critical Parsing Improvements**:
- Early "Answer: X" detection for long majority_vote responses
- Enhanced word boundary handling for answer extraction
- Fallback patterns for parentheses formats like "(A)" and "B) text..."
- Support for various formatting styles and error cases

## Common Patterns

- **JSON-First Communication**: All agent interactions use structured JSON formats with regex parsing and fallbacks
- **Multi-layer Error Handling**: JSON parsing → regex extraction → text fallback → default responses
- **Temperature Control**: 0.0 for deterministic final decisions, 0.7 for creative expert collaboration
- **Token Efficiency**: Word limits enforced across all processing modes (50-300 words)
- **Comprehensive Tracking**: Token usage monitored for all agents, recruiters, and coordinators
- **Production-Ready Output**: Debug controls provide clean output for production vs verbose for development

## Development Priorities

### Completed ✅
- **Basic Mode Optimization**: Enhanced from single agent to 3-expert + arbitrator system
- **Evaluation System Enhancement**: Multi-pattern parsing for diverse response formats
- **JSON Communication**: Structured responses across all processing modes
- **Error Resilience**: Multi-layer parsing with comprehensive fallbacks
- **Performance Achievement**: 84.04% overall accuracy with robust token efficiency

### Current Focus
- **System Reliability**: Maintaining 84%+ accuracy across different datasets
- **Token Optimization**: Efficiency improvements while preserving performance
- **Response Format Standardization**: Enhanced JSON parsing and fallback mechanisms
- **Evaluation Robustness**: Handling edge cases in LLM response parsing

### Recent Achievements
- **Outstanding Basic Mode**: Achieved 87.05% accuracy through enhanced expert recruitment
- **Robust Evaluation**: Fixed parsing for complex majority_vote responses and diverse answer formats
- **System Reliability**: Comprehensive error handling for malformed LLM responses
- **Performance Tracking**: Detailed CSV reporting with per-difficulty accuracy metrics