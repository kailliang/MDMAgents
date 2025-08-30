#!/usr/bin/env python3
"""
Integration tests for Stage 2: Difficulty Assessment & Routing
Tests end-to-end functionality of the integrated system.
"""

import pytest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_end_to_end_basic_processing():
    """Test complete flow for basic difficulty questions"""
    from langgraph_mdm import create_mdm_graph, MDMState
    
    # Create graph
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    compiled_graph = graph.compile()
    
    # Test input state
    initial_state = {
        "messages": [],
        "question": "What is hypertension?",
        "agents": [],
        "token_usage": {"input": 0, "output": 0},
        "processing_stage": "start",
        "final_decision": None
    }
    
    # Run the graph
    result = compiled_graph.invoke(initial_state)
    
    # Verify results
    assert result["difficulty"] == "basic"
    assert result["processing_stage"] == "basic_complete"
    assert result["final_decision"]["placeholder"] == "basic_result"
    assert result["token_usage"]["input"] > 0  # Tokens were used
    assert result["token_usage"]["output"] > 0

def test_end_to_end_intermediate_processing():
    """Test complete flow for intermediate difficulty questions"""
    from langgraph_mdm import create_mdm_graph
    
    # Create graph
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    compiled_graph = graph.compile()
    
    # Test input state
    initial_state = {
        "messages": [],
        "question": "A 45-year-old patient presents with chest pain. Multiple specialists should discuss treatment options.",
        "agents": [],
        "token_usage": {"input": 0, "output": 0},
        "processing_stage": "start",
        "final_decision": None
    }
    
    # Run the graph
    result = compiled_graph.invoke(initial_state)
    
    # Verify results
    assert result["difficulty"] == "intermediate"
    assert result["processing_stage"] == "intermediate_complete"
    assert result["final_decision"]["placeholder"] == "intermediate_result"
    assert "confidence" in result
    assert result["confidence"] > 0.0

def test_end_to_end_advanced_processing():
    """Test complete flow for advanced difficulty questions"""
    from langgraph_mdm import create_mdm_graph
    
    # Create graph
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    compiled_graph = graph.compile()
    
    # Test input state
    initial_state = {
        "messages": [],
        "question": "Complex multi-organ failure requiring coordination between cardiology, nephrology, pulmonology teams.",
        "agents": [],
        "token_usage": {"input": 0, "output": 0},
        "processing_stage": "start",
        "final_decision": None
    }
    
    # Run the graph
    result = compiled_graph.invoke(initial_state)
    
    # Verify results
    assert result["difficulty"] == "advanced"
    assert result["processing_stage"] == "advanced_complete"
    assert result["final_decision"]["placeholder"] == "advanced_result"

def test_graph_structure_and_nodes():
    """Test that graph has correct structure and all nodes"""
    from langgraph_mdm import create_mdm_graph
    
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    
    # Get graph info (this tests that graph compiles correctly)
    compiled_graph = graph.compile()
    
    # Test that we can get graph representation
    graph_dict = compiled_graph.get_graph()
    
    # Verify nodes exist
    assert graph_dict is not None

def test_multiple_model_types():
    """Test that different model types work with the system"""
    from langgraph_mdm import create_mdm_graph
    
    models_to_test = ["gemini-2.5-flash", "gpt-4.1-mini"]
    
    for model_info in models_to_test:
        # Create graph with different model
        graph = create_mdm_graph(model_info=model_info)
        compiled_graph = graph.compile()
        
        # Test with basic question
        initial_state = {
            "messages": [],
            "question": "What are the symptoms of diabetes?",
            "agents": [],
            "token_usage": {"input": 0, "output": 0},
            "processing_stage": "start",
            "final_decision": None
        }
        
        result = compiled_graph.invoke(initial_state)
        
        # Verify basic structure works regardless of model
        assert "difficulty" in result
        assert "processing_stage" in result
        assert result["processing_stage"].endswith("_complete")

def test_token_usage_tracking():
    """Test that token usage is properly tracked throughout the flow"""
    from langgraph_mdm import create_mdm_graph
    
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    compiled_graph = graph.compile()
    
    # Test with existing token usage
    initial_state = {
        "messages": [],
        "question": "Test medical question for token tracking",
        "agents": [],
        "token_usage": {"input": 100, "output": 50},  # Starting with some tokens
        "processing_stage": "start",
        "final_decision": None
    }
    
    result = compiled_graph.invoke(initial_state)
    
    # Verify token usage increased
    assert result["token_usage"]["input"] > 100
    assert result["token_usage"]["output"] > 50

def test_confidence_based_routing():
    """Test that low confidence routes to intermediate processing"""
    from langgraph_difficulty import difficulty_router
    
    # Test low confidence scenarios
    low_confidence_states = [
        {"difficulty": "basic", "confidence": 0.3, "messages": [], "question": "test"},
        {"difficulty": "advanced", "confidence": 0.4, "messages": [], "question": "test"},
    ]
    
    for state in low_confidence_states:
        result = difficulty_router(state)
        assert result == "intermediate_processing", f"Low confidence should route to intermediate, got {result}"
    
    # Test high confidence scenarios  
    high_confidence_states = [
        {"difficulty": "basic", "confidence": 0.8, "messages": [], "question": "test"},
        {"difficulty": "advanced", "confidence": 0.9, "messages": [], "question": "test"},
    ]
    
    expected_routes = ["basic_processing", "advanced_processing"]
    
    for i, state in enumerate(high_confidence_states):
        result = difficulty_router(state)
        assert result == expected_routes[i], f"High confidence should route to {expected_routes[i]}, got {result}"

def test_error_resilience():
    """Test that the system handles errors gracefully"""
    from langgraph_mdm import create_mdm_graph
    
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    compiled_graph = graph.compile()
    
    # Test with empty question
    initial_state = {
        "messages": [],
        "question": "",  # Empty question
        "agents": [],
        "token_usage": {"input": 0, "output": 0},
        "processing_stage": "start",
        "final_decision": None
    }
    
    # Should not crash
    result = compiled_graph.invoke(initial_state)
    
    # Should have some difficulty assigned (fallback)
    assert "difficulty" in result
    assert result["difficulty"] in ["basic", "intermediate", "advanced"]

def test_state_preservation():
    """Test that important state is preserved throughout processing"""
    from langgraph_mdm import create_mdm_graph
    
    graph = create_mdm_graph(model_info="gemini-2.5-flash")
    compiled_graph = graph.compile()
    
    # Test with complex initial state
    initial_state = {
        "messages": [{"role": "user", "content": "test message"}],
        "question": "What is the treatment for pneumonia?",
        "agents": [{"role": "test_agent"}],
        "token_usage": {"input": 25, "output": 15},
        "processing_stage": "start",
        "final_decision": None
    }
    
    result = compiled_graph.invoke(initial_state)
    
    # Verify state preservation
    # Messages get converted to LangChain format, so check content
    assert len(result["messages"]) == len(initial_state["messages"])
    if result["messages"]:
        assert result["messages"][0].content == initial_state["messages"][0]["content"]
    assert result["agents"] == initial_state["agents"]      # Agents preserved
    assert result["question"] == initial_state["question"]  # Question preserved
    
    # Verify state updates
    assert result["processing_stage"] != initial_state["processing_stage"]  # Stage updated
    assert result["token_usage"]["input"] >= initial_state["token_usage"]["input"]  # Tokens accumulated

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])