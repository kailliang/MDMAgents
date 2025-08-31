# MDiAgents: Adaptive Collaboration of LLMs for Medical Decision-Making

This is the implementation for both the **MDMAgents_Lite** and **MDMAgents_Full** variants, developed and improved over the original NeurIPS 2024 paper. The system uses collaborative Large Language Models to solve medical questions through adaptive difficulty assessment, automatically routing questions through three processing modes: basic (expert arbitration), intermediate (expert collaboration), and advanced (multi-disciplinary teams).

- **MDMAgents_Lite**: Cost-effective variant using Gemini-2.5-Flash-Lite
- **MDMAgents_Full**: High-performance variant using Gemini-2.5-Flash

## Key Innovations
- **LangGraph Architecture**: Complete rewrite using LangGraph 0.6.6 for enhanced modularity and parallel processing
- **Parallel Processing**: 64% runtime improvement through concurrent agent execution in intermediate/advanced modes
- **Enhanced Architecture**: Optimized agent architecture and communication mechanisms for improved collaboration
- **JSON-Based Communication**: Implemented structured agent interactions with robust error handling and nested JSON unwrapping
- **Production-Ready**: Multi-pattern answer extraction with comprehensive fallback mechanisms and retry logic
- **Token Optimization**: Efficient processing with enforced token manage mechanism to provent context rot
- **Parse Error Elimination**: Advanced JSON parsing with retry mechanisms - eliminated all parse errors 

## 🏆 Performance Results

### MDMAgents_Lite
Cost-Effective Results with Gemini-2.5-Flash-Lite (LangGraph Implementation)
- **Overall Accuracy**: 87.05% across all difficulty levels (**+9.84%** improvement over **Original Paper Method**: 79.45% accuracy)
- **Basic Mode**: 84.29% accuracy with 3-expert + arbitrator system  
- **Intermediate Mode**: 91.67% accuracy with multi-agent collaboration 
- **Advanced Mode**: 92.86% accuracy with multi-disciplinary teams 
- **Runtime Efficiency**: 64% faster processing through parallel agent execution

### MDMAgents_Full
Exceptional Results with Gemini-2.5-Flash
- **Overall Accuracy**: 95.18% - breakthrough performance
- **Basic Mode**: 94.89% accuracy 
- **Intermediate Mode**: 100.00% accuracy - perfect performance
- **Advanced Mode**: 100.00% accuracy  - perfect performance
- **Token Efficiency**: 76.4% fewer tokens using Gemini-2.5-Flash-Lite
  
## ⭐ Ultimate Branch Achievements

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
cp .env.sample .env
# Edit .env and add your actual API keys
```

### Usage
```bash
python main.py --dataset medqa --model gemini-2.5-flash-lite --difficulty adaptive --num_samples 1
```

### Models & Modes
- **Models**: Gemini, OpenAI 
- **Modes**: `adaptive`, `basic`, `intermediate`, `advanced`

## 🧠 Processing Modes

**Basic**: 3 independent medical experts + arbitrator synthesis with retry mechanisms

**Intermediate**: 3 Expert collaboration with parallel multi-round debate 

**Advanced**: 3 Multi-disciplinary teams with coordinator 

## 📊 Evaluation
```bash
python3 evaluate_text_output.py  # Generate CSV reports
python3 split_test_data.py       # Split by difficulty
```

## 🏗️ Architecture
- **LangGraph System** (`main.py`): Production-ready entry point with LangGraph integration
- **Processing Modules** (`langgraph_*.py`): Modular processing modes with retry mechanisms
- **Core Infrastructure** (`langgraph_mdm.py`): LangGraph agent wrapper with real LLM calls

## 🔧 Configuration
```python
# main.py - Processing controls
SKIP_BASIC = False
SKIP_INTERMEDIATE = False  
SKIP_ADVANCED = False

# utils.py - Debug output
SHOW_INTERACTION_TABLE = False
SHOW_AGENT_INTERACTIONS = False  

```