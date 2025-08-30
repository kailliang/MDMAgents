# LangGraph-Based MDAgents Rewrite: Implementation Plan

## Overview

This document outlines the comprehensive plan for rewriting the MDMAgents system using LangGraph 0.6.6. The current system uses custom classes and manual orchestration to implement three processing modes (basic, intermediate, advanced). The new implementation will leverage LangGraph's state management, multi-agent orchestration, and control flow primitives for better scalability, maintainability, and extensibility.

## Current System Analysis

### Architecture Components
- **Agent Class**: Individual LLM wrapper supporting OpenAI and Gemini models with token tracking
- **Group Class**: Collection of agents with collaborative capabilities  
- **Processing Functions**: Three separate functions for basic, intermediate, and advanced processing
- **Token Management**: Global and per-agent token usage tracking
- **State Management**: Manual state passing between processing steps

### Processing Modes
1. **Basic Mode**: 3-expert recruitment + arbitrator system (87.05% accuracy)
2. **Intermediate Mode**: Expert collaboration with multi-round debate (78.67% accuracy)  
3. **Advanced Mode**: Multi-disciplinary teams with coordinator (75.76% accuracy)

### Key Features to Preserve
- High accuracy performance across all modes
- Token efficiency and tracking
- JSON-based communication between agents
- Adaptive difficulty assessment
- Multi-provider LLM support (OpenAI, Gemini)

## Target LangGraph Architecture

### Core Design Principles
- **State-First Design**: Use LangGraph's StateGraph for explicit state management
- **Agent Specialization**: Each agent type as a dedicated node with specific capabilities
- **Hierarchical Orchestration**: Supervisor pattern for coordinating specialized agents
- **Command-Based Control**: Use LangGraph's Command objects for dynamic routing
- **Memory Management**: Leverage checkpointing for conversation state persistence
- **⚠️ NON-MODIFICATION CONSTRAINT**: **DO NOT modify main.py and utils.py** - create new files for implementing LangGraph

### Proposed Graph Structure

```
Entry Point
    ↓
Difficulty Assessor Node
    ↓
Router Node (based on difficulty)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   Basic Graph   │ Intermediate    │  Advanced Graph │
│                 │    Graph        │                 │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                    ↓                    ↓
Result Synthesizer Node
    ↓
Output Formatter Node
```

## Implementation Stages

### Stage 1: Core Infrastructure Setup
**Goal**: Establish LangGraph foundation and state management
**Success Criteria**: 
- ✅ LangGraph StateGraph properly configured
- ✅ Custom state classes defined and working
- ✅ Basic message flow established
**Tests**: 
- ✅ State transitions work correctly
- ✅ Token tracking integrates properly
- ✅ Multi-provider LLM support maintained
**Status**: Complete

#### Key Components:
1. **State Schema Definition**
   ```python
   class MDMState(MessagesState):
       question: str
       difficulty: Literal["basic", "intermediate", "advanced"] 
       agents: List[Dict]
       token_usage: Dict
       final_decision: Optional[Dict]
       processing_stage: str
   ```

2. **Base Agent Node**
   - Wrapper for existing Agent class
   - Integration with LangGraph node patterns
   - Token usage tracking preservation

3. **Model Configuration**
   - Support for gemini-2.5-flash and gpt-4.1-mini  
   - Environment-based API key management
   - Provider-agnostic interface

### Stage 2: Difficulty Assessment & Routing
**Goal**: Implement adaptive difficulty assessment with dynamic routing
**Success Criteria**:
- Questions correctly classified by difficulty
- Routing to appropriate processing subgraph
- Fallback handling for classification failures
**Tests**:
- Known basic questions route to basic processing
- Known advanced questions route to advanced processing  
- Edge cases handle gracefully
**Status**: Not Started

#### Key Components:
1. **Difficulty Assessor Node**
   - LLM-based classification with JSON output
   - Confidence scoring for routing decisions
   - Fallback to intermediate for unclear cases

2. **Dynamic Router**
   - Command-based routing using LangGraph primitives
   - State updates before subgraph handoff
   - Error handling for malformed responses

### Stage 3: Basic Processing Graph  
**Goal**: Implement high-performance 3-expert + arbitrator system
**Success Criteria**:
- Maintains 87%+ accuracy on test set
- Token efficiency preserved or improved
- JSON communication structure maintained
**Tests**:
- End-to-end basic processing with sample questions
- Token usage within expected ranges
- Answer extraction and validation
**Status**: Not Started

#### Key Components:
1. **Expert Recruitment Node**
   - Parallel expert agent creation
   - Specialization assignment based on question domain
   - Independent expert analysis

2. **Expert Analysis Nodes** (3 parallel)
   - Structured JSON response generation
   - Medical reasoning documentation
   - Answer confidence scoring

3. **Arbitrator Node**
   - Expert response synthesis
   - Final decision with reasoning
   - Token-efficient processing

### Stage 4: Intermediate Processing Graph
**Goal**: Multi-agent collaboration with debate mechanism  
**Success Criteria**:
- Maintains 78%+ accuracy with collaborative processing
- Multi-round debate functionality preserved
- Dynamic participation decisions
**Tests**:
- Complex medical questions processed correctly
- Multi-round debate convergence
- Expert selection logic validation
**Status**: Not Started

#### Key Components:
1. **Collaborative Expert Nodes**
   - Hierarchical expert relationships
   - Debate participation decisions
   - Response synthesis capabilities

2. **Debate Coordinator**
   - Multi-round conversation management
   - Convergence detection
   - Expert selection for each round

3. **Consensus Builder**
   - Final team decision synthesis
   - Confidence aggregation
   - Reasoning documentation

### Stage 5: Advanced Processing Graph
**Goal**: Multi-disciplinary team approach with complex coordination
**Success Criteria**:
- Maintains 75%+ accuracy with MDT approach
- Team formation logic preserved
- Parallel assessment capabilities
**Tests**:
- Complex cases requiring multiple specialties
- Team coordination functionality
- Overall coordinator decision quality
**Status**: Not Started

#### Key Components:
1. **MDT Formation Node**
   - Dynamic team composition
   - Specialist role assignment
   - Parallel team initialization

2. **Parallel Team Assessment Nodes**
   - Independent team processing
   - Specialized perspective gathering
   - Team-specific reasoning

3. **Overall Coordinator**
   - Cross-team synthesis
   - Final decision integration
   - Comprehensive reasoning documentation

### Stage 6: Integration & Optimization
**Goal**: End-to-end system integration with performance optimization
**Success Criteria**:
- All processing modes integrated seamlessly
- Performance meets or exceeds current benchmarks
- System reliability and error handling robust
**Tests**:
- Full system test with representative dataset
- Performance regression testing
- Edge case and error handling validation
**Status**: Not Started

#### Key Components:
1. **Result Synthesizer**
   - Unified output format across all modes
   - Token usage aggregation  
   - Performance metrics collection

2. **Output Formatter**
   - JSON response standardization
   - Evaluation compatibility maintained
   - Debug information preservation

3. **Error Recovery System**
   - Graceful degradation for API failures
   - Retry logic with exponential backoff
   - Fallback processing modes

## Technical Implementation Details

### LangGraph Integration Patterns

#### State Management
```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import Command
from typing import Literal, List, Dict, Optional

class MDMState(MessagesState):
    question: str
    difficulty: Optional[str] = None
    experts: List[Dict] = []
    token_usage: Dict = {"input": 0, "output": 0}
    stage: str = "start"
    final_answer: Optional[Dict] = None
```

#### Agent Node Pattern
```python
def create_expert_node(specialty: str):
    async def expert_node(state: MDMState) -> Command:
        # Expert analysis logic
        response = await expert_agent.analyze(state.question)
        
        return Command(
            update={
                "experts": state.experts + [response],
                "token_usage": update_tokens(state.token_usage, response.usage)
            },
            goto="next_stage"
        )
    return expert_node
```

#### Supervisor Pattern Implementation
```python
def supervisor_node(state: MDMState) -> Command:
    if state.difficulty == "basic":
        return Command(goto="basic_processing")
    elif state.difficulty == "intermediate": 
        return Command(goto="intermediate_processing")
    else:
        return Command(goto="advanced_processing")
```

### Multi-Agent Communication

#### Handoff Tools
- Expert-to-expert knowledge transfer
- Cross-team consultation capabilities
- Dynamic agent recruitment

#### JSON Communication Protocol
- Structured message formats preserved
- Validation and error handling
- Backward compatibility with evaluation system

### Performance Considerations

#### Token Efficiency
- Maintain current token usage patterns
- Optimize for parallel processing where possible
- Implement smart caching for repeated operations

#### Latency Optimization  
- Parallel agent execution using LangGraph's Send API
- Connection pooling for API calls
- Intelligent batching strategies

#### Error Resilience
- Circuit breaker patterns for API failures
- Graceful degradation with reduced agent counts
- Comprehensive logging and monitoring

## Migration Strategy

### Phase 1: Parallel Development (Weeks 1-2)
- ✅ Build LangGraph infrastructure alongside current system (Stage 1 Complete)
- Implement processing graphs in new files only
- Validate core functionality and performance
- **CONSTRAINT**: Keep all LangGraph code in separate files (langgraph_*.py)

### Phase 2: Progressive Integration (Weeks 3-4)
- Create LangGraph processing graphs that mirror existing functionality
- Add new main entry point (langgraph_main.py) alongside existing main.py
- Test both systems in parallel for validation
- **CONSTRAINT**: Existing main.py and utils.py remain untouched

### Phase 3: Full Migration (Week 5)
- Switch to LangGraph as primary system via new entry point
- Keep legacy code available for comparison
- Performance tuning and optimization
- **CONSTRAINT**: Legacy files preserved for rollback if needed

### Phase 4: Enhancement (Week 6+)
- Leverage LangGraph-specific features in new modules
- Add new capabilities (memory, persistence, etc.)
- Explore advanced orchestration patterns
- **CONSTRAINT**: All enhancements in dedicated LangGraph files

## Testing Strategy

### Unit Testing
- Individual node functionality
- State transitions and updates
- Token usage tracking accuracy

### Integration Testing  
- End-to-end processing flows
- Cross-agent communication
- Error handling and recovery

### Performance Testing
- Accuracy benchmarks vs current system
- Token efficiency comparisons
- Latency and throughput measurements

### Regression Testing
- Existing evaluation datasets
- Edge case handling
- API failure scenarios

## Success Metrics

### Functional Requirements
- [ ] Accuracy: Basic ≥87%, Intermediate ≥78%, Advanced ≥75%
- [ ] Token Efficiency: Within 10% of current usage
- [ ] Response Time: No significant regression
- [ ] Error Handling: Robust failure recovery

### Technical Requirements  
- [ ] Code Maintainability: Improved modularity and testability
- [ ] Extensibility: Easy addition of new processing modes
- [ ] Monitoring: Comprehensive observability and debugging
- [ ] Documentation: Clear architecture and usage guides

## Risk Mitigation

### Technical Risks
- **LangGraph Learning Curve**: Mitigate with incremental implementation and thorough documentation
- **Performance Regression**: Address with comprehensive benchmarking and optimization
- **Integration Complexity**: Manage through staged migration and extensive testing

### Operational Risks
- **API Rate Limits**: Implement intelligent rate limiting and caching
- **Model Provider Changes**: Maintain provider-agnostic architecture
- **Data Privacy**: Ensure compliance with existing security protocols

## Future Enhancements

### LangGraph-Enabled Features
- **Memory Management**: Long-term conversation and case memory
- **Human-in-the-Loop**: Interactive expert consultation
- **Real-time Collaboration**: Live agent interaction monitoring
- **Advanced Orchestration**: Dynamic workflow adaptation

### System Capabilities
- **Multi-Modal Processing**: Image and document analysis
- **Streaming Responses**: Real-time response generation  
- **Distributed Processing**: Multi-node agent deployment
- **Advanced Analytics**: Deep performance and accuracy analysis

## Implementation Constraints

### File Modification Rules
**⚠️ CRITICAL CONSTRAINT**: 
- **DO NOT modify main.py or utils.py**
- **ALL LangGraph code must be in NEW files**
- Use naming pattern: `langgraph_*.py` for new modules
- Create `langgraph_main.py` as new entry point when ready

### File Structure
```
MDMAgents/
├── main.py                 # PRESERVE - original entry point
├── utils.py                # PRESERVE - original utilities  
├── langgraph_mdm.py        # ✅ NEW - core LangGraph infrastructure
├── langgraph_main.py       # NEW - LangGraph entry point (future)
├── langgraph_basic.py      # NEW - basic processing graph
├── langgraph_inter.py      # NEW - intermediate processing graph  
├── langgraph_advanced.py   # NEW - advanced processing graph
└── test_langgraph_*.py     # NEW - test files
```

### Development Guidelines
**NEVER**:
- Modify main.py or utils.py files
- Import/modify existing processing functions directly
- Mix LangGraph and legacy code in same file

**ALWAYS**:
- Create new files for LangGraph implementation
- Keep implementations cleanly separated
- Follow TDD approach with comprehensive tests
- Update plan documentation as progress is made

---

**Status**: Stage 1 Complete, Planning Updated  
**Next Phase**: Stage 2 Implementation  
**Timeline**: 6-week implementation with staged rollout  
**Owner**: Development Team  
**Last Updated**: 2025-08-30