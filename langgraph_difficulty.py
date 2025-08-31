#!/usr/bin/env python3
"""
LangGraph-based difficulty assessment and routing system.
Implements enhanced difficulty classification with LangGraph integration.
"""

import json
import re
import logging
from typing import Dict, Any, Literal
from langgraph.types import Command
from langgraph_mdm import LangGraphAgent, MDMStateDict

# Configure logging
logger = logging.getLogger(__name__)


class DifficultyAssessorNode:
    """
    Enhanced difficulty assessment node with real LLM integration.
    Replaces the mock implementation from langgraph_mdm.py.
    """
    
    def __init__(self, model_info: str):
        self.model_info = model_info
        self._agent = None
        
        # Define the prompt used in the original system
        self.difficulty_prompt_template = """Analyze the following medical query and determine its complexity level.

Medical Query:
{question}

**Difficulty Levels:**
- **basic**: a single medical agent can output an answer.
- **intermediate**: number of medical experts with different expertise should discuss and make final decision.
- **advanced**: multiple teams of clinicians from different departments need to collaborate with each other to make final decision.

Provide your assessment in the following JSON format:

{{
  "difficulty": "basic|intermediate|advanced",
  "confidence": 0.0-1.0
}}

**Requirements:**
- Return ONLY the JSON format, no other text
- Difficulty must be exactly one of: basic, intermediate, advanced
- Confidence should reflect your certainty in the assessment (0.0 = uncertain, 1.0 = very certain)
"""
    
    def _get_agent(self):
        """Lazy initialization of assessment agent"""
        if self._agent is None:
            # Validate model before creating agent
            valid_models = [
                'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-flash-lite-preview-06-17',
                'gpt-4o-mini', 'gpt-4.1-mini'
            ]
            if self.model_info not in valid_models:
                raise ValueError(f"Unsupported model: {self.model_info}. Must be one of {valid_models}")
            
            instruction = """You are a medical question difficulty assessor. 
            Analyze the given medical question and determine its complexity level.
            Always respond with valid JSON format containing difficulty and confidence scores."""
            
            self._agent = LangGraphAgent(
                instruction=instruction,
                role="difficulty_assessor",
                model_info=self.model_info
            )
        return self._agent
    
    def _call_llm(self, question: str) -> tuple[str, Dict[str, int]]:
        """Call LLM with difficulty assessment prompt"""
        logger.info(f"Starting difficulty assessment for question: {question[:50]}...")
        
        agent = self._get_agent()
        
        # Format the prompt with the question
        formatted_prompt = self.difficulty_prompt_template.format(question=question)
        logger.debug(f"Formatted prompt for difficulty assessment: {formatted_prompt[:100]}...")
        
        # Get current token usage before call
        usage_before = agent.get_token_usage()
        
        # Make real LLM call
        response = agent.chat(formatted_prompt)
        
        # Calculate token usage for this call
        usage_after = agent.get_token_usage()
        token_usage = {
            "input_tokens": usage_after["input_tokens"] - usage_before["input_tokens"], 
            "output_tokens": usage_after["output_tokens"] - usage_before["output_tokens"]
        }
        
        # Parse the response to extract difficulty and confidence for logging
        try:
            difficulty, confidence = self._parse_response(response)
            logger.info(f"Difficulty assessment completed - Level: {difficulty} (confidence: {confidence:.2f}) - Tokens used: {token_usage}")
        except Exception as e:
            logger.info(f"Difficulty assessment completed - Tokens used: {token_usage}")
            logger.debug(f"Could not parse difficulty for logging: {e}")
        
        logger.debug(f"Raw LLM response: {response}")
        
        return response, token_usage
    
    def _generate_mock_response(self, question: str) -> str:
        """Generate realistic difficulty assessment for testing"""
        question_lower = question.lower()
        
        # Simple heuristics for mock responses
        if any(keyword in question_lower for keyword in ['what is', 'define', 'definition', 'symptoms of']):
            return '{"difficulty": "basic", "confidence": 0.85}'
        elif any(keyword in question_lower for keyword in ['multiple', 'specialists', 'discuss', 'consultation']):
            return '{"difficulty": "intermediate", "confidence": 0.78}'
        elif any(keyword in question_lower for keyword in ['teams', 'departments', 'complex', 'multi-organ', 'coordination']):
            return '{"difficulty": "advanced", "confidence": 0.92}'
        else:
            # Fallback based on question length and complexity
            if len(question) < 50:
                return '{"difficulty": "basic", "confidence": 0.75}'
            elif len(question) < 120:
                return '{"difficulty": "intermediate", "confidence": 0.70}'
            else:
                return '{"difficulty": "advanced", "confidence": 0.80}'
    
    def assess_difficulty(self, state: MDMStateDict) -> Command:
        """
        Assess question difficulty and return Command object for routing.
        This is the main entry point for the difficulty assessment node.
        """
        question = state.get("question", "")
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        
        try:
            # Get LLM response
            response, usage = self._call_llm(question)
            
            # Parse JSON response
            difficulty, confidence = self._parse_response(response)
            
            # Update token usage
            updated_usage = {
                "input": current_usage["input"] + usage["input_tokens"],
                "output": current_usage["output"] + usage["output_tokens"]
            }
            
            # Determine routing destination
            routing_destination = f"{difficulty}_processing"
            
            # Return Command object with state updates and routing
            return Command(
                update={
                    "difficulty": difficulty,
                    "confidence": confidence,
                    "token_usage": updated_usage
                },
                goto=routing_destination
            )
            
        except Exception as e:
            print(f"Error in difficulty assessment: {e}")
            # Fallback to safe default
            return Command(
                update={
                    "difficulty": "intermediate",
                    "confidence": 0.5,
                    "token_usage": current_usage
                },
                goto="intermediate_processing"
            )
    
    def _parse_response(self, response: str) -> tuple[str, float]:
        """
        Parse LLM response with multi-layer fallback strategy.
        Returns (difficulty, confidence) tuple.
        """
        # Layer 1: JSON parsing
        try:
            data = json.loads(response)
            difficulty = data.get("difficulty", "").lower().strip()
            confidence = float(data.get("confidence", 0.0))
            
            if difficulty in ["basic", "intermediate", "advanced"]:
                return difficulty, min(max(confidence, 0.0), 1.0)  # Clamp confidence to [0,1]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        
        # Layer 2: Regex parsing for JSON-like structures
        json_match = re.search(r'\{\s*"difficulty"\s*:\s*"([^"]+)"\s*(?:,\s*"confidence"\s*:\s*([0-9.]+))?\s*\}', response, re.DOTALL)
        if json_match:
            difficulty = json_match.group(1).lower().strip()
            confidence_str = json_match.group(2)
            confidence = float(confidence_str) if confidence_str else 0.7  # Default confidence
            
            if difficulty in ["basic", "intermediate", "advanced"]:
                return difficulty, min(max(confidence, 0.0), 1.0)
        
        # Layer 3: Text-based parsing
        response_lower = response.lower()
        confidence = 0.6  # Lower confidence for text parsing
        
        if "basic" in response_lower:
            return "basic", confidence
        elif "intermediate" in response_lower:
            return "intermediate", confidence  
        elif "advanced" in response_lower:
            return "advanced", confidence
        
        # Layer 4: Complete fallback
        return "intermediate", 0.5


def difficulty_router(state: MDMStateDict) -> Literal["basic_processing", "intermediate_processing", "advanced_processing"]:
    """
    Router function for conditional edges.
    Routes based on difficulty assessment and confidence level.
    """
    difficulty = state.get("difficulty", "intermediate")
    confidence = state.get("confidence", 0.0)
    
    # If confidence is too low, route to intermediate (safe default)
    if confidence < 0.3:
        return "advanced_processing"
    
    # Route based on assessed difficulty
    if difficulty == "basic":
        return "basic_processing"
    elif difficulty == "advanced":
        return "advanced_processing"
    else:
        return "intermediate_processing"


def create_difficulty_assessment_graph():
    """
    Create a standalone difficulty assessment graph for testing.
    This can be integrated into the main MDM graph.
    """
    from langgraph.graph import StateGraph, START, END
    
    def difficulty_node(state: MDMStateDict) -> Command:
        """Node wrapper for difficulty assessment"""
        assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
        return assessor.assess_difficulty(state)
    
    def basic_placeholder(state: MDMStateDict) -> MDMStateDict:
        """Placeholder for basic processing"""
        return {"processing_stage": "basic_assessed"}
    
    def intermediate_placeholder(state: MDMStateDict) -> MDMStateDict:
        """Placeholder for intermediate processing"""
        return {"processing_stage": "intermediate_assessed"}
    
    def advanced_placeholder(state: MDMStateDict) -> MDMStateDict:
        """Placeholder for advanced processing"""
        return {"processing_stage": "advanced_assessed"}
    
    # Create graph
    from langgraph_mdm import MDMStateDict
    graph = StateGraph(MDMStateDict)
    
    # Add nodes
    graph.add_node("assess_difficulty", difficulty_node)
    graph.add_node("basic_processing", basic_placeholder)
    graph.add_node("intermediate_processing", intermediate_placeholder)  
    graph.add_node("advanced_processing", advanced_placeholder)
    
    # Add edges
    graph.add_edge(START, "assess_difficulty")
    graph.add_edge("basic_processing", END)
    graph.add_edge("intermediate_processing", END)
    graph.add_edge("advanced_processing", END)
    
    return graph


# Utility functions for integration testing
def validate_difficulty_assessment(question: str, expected_difficulty: str = None, model_info: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """
    Utility function to validate difficulty assessment for a given question.
    Used for integration testing and validation.
    """
    assessor = DifficultyAssessorNode(model_info=model_info)
    
    state = {
        "messages": [],
        "question": question,
        "token_usage": {"input": 0, "output": 0},
        "processing_stage": "start"
    }
    
    result = assessor.assess_difficulty(state)
    
    validation_result = {
        "question": question,
        "assessed_difficulty": result.update.get("difficulty"),
        "confidence": result.update.get("confidence"), 
        "routing_destination": result.goto,
        "token_usage": result.update.get("token_usage"),
        "matches_expected": result.update.get("difficulty") == expected_difficulty if expected_difficulty else None
    }
    
    return validation_result


if __name__ == "__main__":
    # Basic smoke test
    test_questions = [
        ("What is hypertension?", "basic"),
        ("A patient needs consultation from multiple specialists for complex treatment.", "intermediate"),
        ("Multi-organ failure requiring coordination between cardiology, nephrology, and ICU teams.", "advanced")
    ]
    
    print("Difficulty Assessment Smoke Test:")
    for question, expected in test_questions:
        result = validate_difficulty_assessment(question, expected)
        print(f"Question: {question[:50]}...")
        print(f"  Assessed: {result['assessed_difficulty']} (confidence: {result['confidence']:.2f})")
        print(f"  Expected: {expected}, Match: {result['matches_expected']}")
        print(f"  Routes to: {result['routing_destination']}")
        print()