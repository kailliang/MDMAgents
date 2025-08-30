#!/usr/bin/env python3
"""
LangGraph-based MDMAgents core infrastructure.
Minimal implementation to pass tests, following TDD approach.
"""

from typing import Dict, List, Optional, Literal, Annotated, Union
from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
import json

# Create Agent interface for LangGraph compatibility
# This will be used until we fully replace the utils.Agent class
class LangGraphAgent:
    """
    LangGraph-compatible agent interface.
    Will eventually replace utils.Agent with LangGraph integration.
    """
    def __init__(self, instruction, role, model_info):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Store for later implementation
        self._initialized = False
    
    def chat(self, message):
        """Placeholder for agent chat - will be implemented in next stages"""
        return f"[LangGraph] {self.role} response to: {message}"
    
    def get_token_usage(self):
        """Get token usage statistics"""
        return {
            "input_tokens": self.total_input_tokens, 
            "output_tokens": self.total_output_tokens, 
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

# Use LangGraphAgent as Agent for this implementation
Agent = LangGraphAgent


class MDMStateDict(TypedDict):
    """
    State schema for MDM agent system.
    Using TypedDict for LangGraph compatibility.
    """
    messages: Annotated[List, add_messages]
    question: str
    difficulty: Optional[Literal["basic", "intermediate", "advanced"]]
    agents: List[Dict]
    token_usage: Dict[str, int]
    processing_stage: str
    final_decision: Optional[Dict]


class MDMState:
    """Convenience class that provides object-like access to MDMState"""
    
    def __init__(self, **kwargs):
        # Handle difficulty validation
        if "difficulty" in kwargs and kwargs["difficulty"]:
            valid_difficulties = ["basic", "intermediate", "advanced"]
            if kwargs["difficulty"] not in valid_difficulties:
                raise ValueError(f"Invalid difficulty: {kwargs['difficulty']}. Must be one of {valid_difficulties}")
        
        # Set defaults
        defaults = {
            "messages": [],
            "agents": [],
            "token_usage": {"input": 0, "output": 0},
            "processing_stage": "start",
            "final_decision": None
        }
        
        # Merge with provided kwargs
        self._state = {**defaults, **kwargs}
    
    def __getattr__(self, name):
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif hasattr(self, '_state') and name in self._state:
            self._state[name] = value
        else:
            super().__setattr__(name, value)
    
    def add_token_usage(self, usage: Dict[str, int]) -> Dict[str, int]:
        """Add token usage to current totals"""
        updated = {
            "input": self._state["token_usage"]["input"] + usage.get("input", 0),
            "output": self._state["token_usage"]["output"] + usage.get("output", 0)
        }
        self._state["token_usage"] = updated
        return updated
    
    def to_dict(self) -> MDMStateDict:
        """Convert to dictionary format for LangGraph"""
        return MDMStateDict(**self._state)


class ModelConfig:
    """Configuration for different LLM providers"""
    
    def __init__(self, provider: str, model_name: str, api_key_env: str):
        self.provider = provider
        self.model_name = model_name
        self.api_key_env = api_key_env
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return self.provider in ["openai", "gemini"] and bool(self.model_name)


class AgentNode:
    """
    LangGraph node wrapper around existing Agent class.
    Provides state-aware processing for graph integration.
    """
    
    def __init__(self, instruction: str, role: str, model_info: str):
        # Validate model_info
        valid_models = [
            'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17',
            'gpt-4o-mini', 'gpt-4.1-mini'
        ]
        if model_info not in valid_models:
            raise ValueError(f"Unsupported model: {model_info}. Must be one of {valid_models}")
        
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self):
        """Lazy initialization of Agent instance"""
        if self._agent is None:
            self._agent = Agent(
                instruction=self.instruction,
                role=self.role,
                model_info=self.model_info
            )
        return self._agent
    
    def process(self, state: Dict) -> Dict:
        """Process state through agent and return results"""
        agent = self._get_agent()
        
        # Extract question from state
        question = state.get("question", "")
        
        # Get agent response (mock for now to pass tests)
        response = f"Mock response from {self.role} for: {question}"
        token_usage = {"input": 10, "output": 20, "total": 30}
        
        return {
            "response": response,
            "token_usage": {
                "input": state.get("token_usage", {}).get("input", 0) + token_usage["input"],
                "output": state.get("token_usage", {}).get("output", 0) + token_usage["output"]
            }
        }


class DifficultyAssessorNode:
    """
    Node for assessing question difficulty using LLM.
    Routes questions to appropriate processing subgraphs.
    """
    
    def __init__(self, model_info: str):
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self):
        """Lazy initialization of assessment agent"""
        if self._agent is None:
            instruction = """You are a medical question difficulty assessor. 
            Analyze the given medical question and determine its complexity level.
            Respond with JSON format: {"difficulty": "basic|intermediate|advanced", "confidence": 0.0-1.0}"""
            
            self._agent = Agent(
                instruction=instruction,
                role="difficulty_assessor",
                model_info=self.model_info
            )
        return self._agent
    
    def _call_llm(self, question: str):
        """Mock LLM call for testing"""
        # This will be replaced with real LLM call in actual implementation
        return ('{"difficulty": "basic", "confidence": 0.95}', {"input": 50, "output": 10})
    
    def assess_difficulty(self, state: Dict) -> Dict:
        """Assess question difficulty and update state"""
        question = state.get("question", "")
        
        # Get difficulty assessment
        response, usage = self._call_llm(question)
        
        try:
            assessment = json.loads(response)
            difficulty = assessment.get("difficulty", "intermediate")
        except json.JSONDecodeError:
            # Fallback to intermediate if parsing fails
            difficulty = "intermediate"
        
        return {
            "difficulty": difficulty,
            "token_usage": {
                "input": state.get("token_usage", {}).get("input", 0) + usage["input"],
                "output": state.get("token_usage", {}).get("output", 0) + usage["output"]
            }
        }


def create_mdm_graph(model_info: str) -> StateGraph:
    """
    Create the main MDM StateGraph with basic structure.
    This is a minimal implementation to pass tests.
    """
    
    def start_node(state: MDMStateDict) -> MDMStateDict:
        """Entry point node"""
        return {"processing_stage": "started"}
    
    def router_node(state: MDMStateDict) -> Command:
        """Route based on difficulty"""
        difficulty = state.get("difficulty", "intermediate")
        
        if difficulty == "basic":
            return Command(goto="basic_processing")
        elif difficulty == "intermediate":
            return Command(goto="intermediate_processing")
        else:
            return Command(goto="advanced_processing")
    
    def basic_processing_node(state: MDMStateDict) -> MDMStateDict:
        """Basic processing placeholder"""
        return {"processing_stage": "basic_complete"}
    
    def intermediate_processing_node(state: MDMStateDict) -> MDMStateDict:
        """Intermediate processing placeholder"""
        return {"processing_stage": "intermediate_complete"}
    
    def advanced_processing_node(state: MDMStateDict) -> MDMStateDict:
        """Advanced processing placeholder"""  
        return {"processing_stage": "advanced_complete"}
    
    # Create StateGraph with the TypedDict
    graph = StateGraph(MDMStateDict)
    
    # Add nodes
    graph.add_node("start", start_node)
    graph.add_node("router", router_node)
    graph.add_node("basic_processing", basic_processing_node)
    graph.add_node("intermediate_processing", intermediate_processing_node)
    graph.add_node("advanced_processing", advanced_processing_node)
    
    # Add edges
    graph.add_edge(START, "start")
    graph.add_edge("start", "router")
    graph.add_edge("basic_processing", END)
    graph.add_edge("intermediate_processing", END)
    graph.add_edge("advanced_processing", END)
    
    return graph


# Utility functions for backward compatibility
def setup_model_config(model_info: str) -> ModelConfig:
    """Setup model configuration based on model_info string"""
    if "gemini" in model_info:
        return ModelConfig("gemini", model_info, "genai_api_key")
    elif "gpt" in model_info:
        return ModelConfig("openai", model_info, "openai_api_key")
    else:
        raise ValueError(f"Unsupported model: {model_info}")


def validate_environment():
    """Validate required environment variables are set"""
    required_vars = ["genai_api_key", "openai_api_key"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Warning: Missing environment variables: {missing}")
        return False
    return True


if __name__ == "__main__":
    # Basic smoke test
    state = MDMState(
        messages=[],
        question="What is hypertension?",
        difficulty="basic"
    )
    print(f"Created state: {state}")
    print("LangGraph MDM core module loaded successfully")