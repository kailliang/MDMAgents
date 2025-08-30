#!/usr/bin/env python3
"""
Test suite for LangGraph-based difficulty assessment and routing.
Following TDD approach - tests written first to drive implementation.
"""

import pytest
import json
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_difficulty_assessor_node_basic_question():
    """Test difficulty assessment for basic medical questions"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock LLM response for basic question
    mock_response = '{"difficulty": "basic", "confidence": 0.95}'
    mock_usage = {"input_tokens": 50, "output_tokens": 15}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "What is hypertension?",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = assessor.assess_difficulty(state)
        
        assert result.update["difficulty"] == "basic"
        assert result.update["confidence"] == 0.95
        assert result.update["token_usage"]["input"] == 50
        assert result.update["token_usage"]["output"] == 15
        assert result.goto == "basic_processing"

def test_difficulty_assessor_node_intermediate_question():
    """Test difficulty assessment for intermediate medical questions"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock LLM response for intermediate question
    mock_response = '{"difficulty": "intermediate", "confidence": 0.87}'
    mock_usage = {"input_tokens": 75, "output_tokens": 18}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "A 45-year-old patient presents with chest pain and elevated troponins. Multiple specialists should discuss treatment options.",
            "token_usage": {"input": 100, "output": 50}
        }
        
        result = assessor.assess_difficulty(state)
        
        assert result.update["difficulty"] == "intermediate"
        assert result.update["confidence"] == 0.87
        assert result.update["token_usage"]["input"] == 175  # 100 + 75
        assert result.update["token_usage"]["output"] == 68   # 50 + 18
        assert result.goto == "intermediate_processing"

def test_difficulty_assessor_node_advanced_question():
    """Test difficulty assessment for advanced medical questions"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gpt-4.1-mini")
    
    # Mock LLM response for advanced question
    mock_response = '{"difficulty": "advanced", "confidence": 0.92}'
    mock_usage = {"input_tokens": 120, "output_tokens": 25}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Complex multi-organ failure requiring coordination between cardiology, nephrology, pulmonology, and ICU teams with conflicting treatment protocols.",
            "token_usage": {"input": 200, "output": 75}
        }
        
        result = assessor.assess_difficulty(state)
        
        assert result.update["difficulty"] == "advanced"
        assert result.update["confidence"] == 0.92
        assert result.update["token_usage"]["input"] == 320   # 200 + 120
        assert result.update["token_usage"]["output"] == 100  # 75 + 25
        assert result.goto == "advanced_processing"

def test_difficulty_assessor_json_parsing_fallback():
    """Test fallback when JSON parsing fails"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock malformed JSON response
    mock_response = 'This is a basic difficulty question with clear answer.'
    mock_usage = {"input_tokens": 60, "output_tokens": 20}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "What is diabetes?",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = assessor.assess_difficulty(state)
        
        # Should fallback to text parsing and find "basic"
        assert result.update["difficulty"] == "basic"
        assert result.update["confidence"] < 0.8  # Lower confidence for fallback
        assert result.goto == "basic_processing"

def test_difficulty_assessor_regex_fallback():
    """Test regex fallback when JSON is malformed but contains JSON-like structure"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock malformed JSON that can be regex parsed
    mock_response = 'Here is my assessment: {"difficulty": "intermediate"} based on analysis.'
    mock_usage = {"input_tokens": 80, "output_tokens": 30}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Complex cardiac case requiring specialist consultation",
            "token_usage": {"input": 50, "output": 25}
        }
        
        result = assessor.assess_difficulty(state)
        
        # Should extract difficulty from regex
        assert result.update["difficulty"] == "intermediate"
        assert result.update["confidence"] < 0.9  # Lower confidence for regex parsing
        assert result.goto == "intermediate_processing"

def test_difficulty_assessor_complete_fallback():
    """Test complete fallback when no difficulty can be parsed"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock completely unparseable response
    mock_response = 'This is a completely random response with no difficulty indicators.'
    mock_usage = {"input_tokens": 40, "output_tokens": 10}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Random medical question",
            "token_usage": {"input": 25, "output": 15}
        }
        
        result = assessor.assess_difficulty(state)
        
        # Should default to intermediate
        assert result.update["difficulty"] == "intermediate"
        assert result.update["confidence"] == 0.5  # Default confidence
        assert result.goto == "intermediate_processing"

def test_difficulty_router_with_confidence():
    """Test difficulty router considers confidence in routing decisions"""
    from langgraph_difficulty import difficulty_router
    
    # High confidence - should route to assessed difficulty
    high_confidence_state = {
        "difficulty": "basic",
        "confidence": 0.9,
        "messages": [],
        "question": "Test question"
    }
    
    result = difficulty_router(high_confidence_state)
    assert result == "basic_processing"
    
    # Low confidence - should route to intermediate (safe default)
    low_confidence_state = {
        "difficulty": "advanced", 
        "confidence": 0.3,
        "messages": [],
        "question": "Test question"
    }
    
    result = difficulty_router(low_confidence_state)
    assert result == "intermediate_processing"

def test_difficulty_router_missing_fields():
    """Test difficulty router handles missing state fields gracefully"""
    from langgraph_difficulty import difficulty_router
    
    # Missing difficulty
    state_no_difficulty = {
        "confidence": 0.8,
        "messages": [],
        "question": "Test question"
    }
    
    result = difficulty_router(state_no_difficulty)
    assert result == "intermediate_processing"  # Default fallback
    
    # Missing confidence
    state_no_confidence = {
        "difficulty": "basic",
        "messages": [],
        "question": "Test question"
    }
    
    result = difficulty_router(state_no_confidence) 
    assert result == "intermediate_processing"  # Low confidence fallback

def test_real_llm_integration():
    """Test integration with real LLM call (mocked for testing)"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock the Agent class that would be used for real LLM calls
    mock_agent = Mock()
    mock_agent.chat.return_value = None
    mock_agent.temp_responses.return_value = {0.0: '{"difficulty": "basic", "confidence": 0.85}'}
    mock_agent.get_token_usage.return_value = {
        "input_tokens": 65,
        "output_tokens": 22,
        "total_tokens": 87
    }
    
    with patch.object(assessor, '_get_agent', return_value=mock_agent):
        state = {
            "messages": [],
            "question": "What are the symptoms of pneumonia?",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = assessor._call_llm(state["question"])
        
        assert result[0] == '{"difficulty": "basic", "confidence": 0.85}'
        assert result[1]["input_tokens"] == 65
        assert result[1]["output_tokens"] == 22

def test_command_object_structure():
    """Test that Command objects are properly structured"""
    from langgraph_difficulty import DifficultyAssessorNode
    from langgraph.types import Command
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    mock_response = '{"difficulty": "intermediate", "confidence": 0.78}'
    mock_usage = {"input_tokens": 55, "output_tokens": 18}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Test medical question",
            "token_usage": {"input": 30, "output": 20}
        }
        
        result = assessor.assess_difficulty(state)
        
        # Verify Command object structure
        assert isinstance(result, Command)
        assert "difficulty" in result.update
        assert "confidence" in result.update
        assert "token_usage" in result.update
        assert result.goto is not None
        assert result.goto.endswith("_processing")

def test_model_compatibility():
    """Test compatibility with different model types"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    # Test Gemini model
    gemini_assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    assert gemini_assessor.model_info == "gemini-2.5-flash"
    
    # Test OpenAI model
    openai_assessor = DifficultyAssessorNode(model_info="gpt-4.1-mini")
    assert openai_assessor.model_info == "gpt-4.1-mini"
    
    # Test invalid model should raise error when trying to create agent
    invalid_assessor = DifficultyAssessorNode(model_info="invalid-model")
    with pytest.raises(ValueError, match="Unsupported model"):
        invalid_assessor._get_agent()

def test_token_usage_accumulation():
    """Test that token usage properly accumulates"""
    from langgraph_difficulty import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    mock_response = '{"difficulty": "basic", "confidence": 0.85}'
    mock_usage = {"input_tokens": 40, "output_tokens": 12}
    
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Test question",
            "token_usage": {"input": 100, "output": 50}  # Previous usage
        }
        
        result = assessor.assess_difficulty(state)
        
        # Verify token usage accumulates correctly
        assert result.update["token_usage"]["input"] == 140   # 100 + 40
        assert result.update["token_usage"]["output"] == 62   # 50 + 12

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])