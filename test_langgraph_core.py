#!/usr/bin/env python3
"""
Test suite for LangGraph-based MDMAgents core infrastructure.
Following TDD approach - write tests first, then implement.
"""

import pytest
from typing import Dict, List, Optional, Literal
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mdm_state_schema():
    """Test MDMState schema definition and basic functionality"""
    # Test will fail initially - this drives implementation
    from langgraph_mdm import MDMState
    
    # Basic state creation
    state = MDMState(
        messages=[],
        question="What is the diagnosis for chest pain?",
        difficulty="basic",
        agents=[],
        token_usage={"input": 0, "output": 0},
        processing_stage="start"
    )
    
    assert state.question == "What is the diagnosis for chest pain?"
    assert state.difficulty == "basic"
    assert state.processing_stage == "start"
    assert state.token_usage == {"input": 0, "output": 0}
    assert len(state.agents) == 0
    assert len(state.messages) == 0
    assert state.final_decision is None

def test_mdm_state_validation():
    """Test MDMState validation for required fields"""
    from langgraph_mdm import MDMState
    
    # Test difficulty validation
    with pytest.raises(ValueError):
        MDMState(
            messages=[],
            question="Test question",
            difficulty="invalid_difficulty"  # Should only accept basic/intermediate/advanced
        )

def test_agent_node_wrapper():
    """Test AgentNode wrapper around existing Agent class"""
    from langgraph_mdm import AgentNode
    
    # Mock the existing Agent class behavior - patch within langgraph_mdm
    with patch('langgraph_mdm.Agent') as MockAgent:
        mock_agent = Mock()
        mock_agent.chat.return_value = "Test response"
        mock_agent.get_token_usage.return_value = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        MockAgent.return_value = mock_agent
        
        # Create AgentNode wrapper
        agent_node = AgentNode(
            instruction="You are a medical expert",
            role="cardiologist", 
            model_info="gemini-2.5-flash"
        )
        
        # Test state processing
        state = {
            "messages": [],
            "question": "Test medical question",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = agent_node.process(state)
        
        assert "response" in result
        assert "token_usage" in result
        assert result["token_usage"]["input"] >= 10
        assert result["token_usage"]["output"] >= 20

def test_difficulty_assessor_node():
    """Test difficulty assessment node functionality"""
    from langgraph_mdm import DifficultyAssessorNode
    
    assessor = DifficultyAssessorNode(model_info="gemini-2.5-flash")
    
    # Mock LLM response for basic question
    with patch.object(assessor, '_call_llm') as mock_llm:
        mock_llm.return_value = ('{"difficulty": "basic", "confidence": 0.95}', {"input": 50, "output": 10})
        
        state = {
            "messages": [],
            "question": "What is hypertension?",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = assessor.assess_difficulty(state)
        
        assert result["difficulty"] == "basic"
        assert result["token_usage"]["input"] >= 50
        assert result["token_usage"]["output"] >= 10

def test_state_graph_construction():
    """Test basic StateGraph construction and configuration"""
    from langgraph_mdm import create_mdm_graph
    
    # This should create a basic graph structure
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    
    # Verify graph has required nodes
    assert graph is not None
    # More specific tests will be added as implementation progresses

def test_token_tracking_integration():
    """Test token usage tracking across nodes"""
    from langgraph_mdm import MDMState
    
    state = MDMState(
        messages=[],
        question="Test question",
        difficulty="basic",
        token_usage={"input": 100, "output": 50}
    )
    
    # Token updates should accumulate
    updated_usage = state.add_token_usage({"input": 25, "output": 15})
    
    assert updated_usage["input"] == 125
    assert updated_usage["output"] == 65

def test_model_configuration():
    """Test multi-provider model configuration"""
    from langgraph_mdm import ModelConfig
    
    # Test Gemini configuration
    gemini_config = ModelConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key_env="genai_api_key"
    )
    
    assert gemini_config.provider == "gemini"
    assert gemini_config.is_valid()
    
    # Test OpenAI configuration  
    openai_config = ModelConfig(
        provider="openai",
        model_name="gpt-4.1-mini", 
        api_key_env="openai_api_key"
    )
    
    assert openai_config.provider == "openai"
    assert openai_config.is_valid()

def test_error_handling():
    """Test error handling and graceful degradation"""
    from langgraph_mdm import AgentNode
    
    # Test with invalid model configuration
    with pytest.raises(ValueError, match="Unsupported model"):
        AgentNode(
            instruction="Test instruction",
            role="test_role",
            model_info="invalid_model"
        )

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])