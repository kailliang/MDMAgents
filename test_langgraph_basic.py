#!/usr/bin/env python3
"""
Test suite for LangGraph-based basic processing graph.
Following TDD approach - tests written first to drive implementation.

Basic processing implements 3-expert + arbitrator system:
1. Expert Recruitment - 3 independent medical specialists
2. Independent Expert Analysis - parallel processing with JSON responses  
3. Arbitrator Decision - synthesis of expert opinions

Target: 87%+ accuracy with token efficiency
"""

import pytest
import json
import sys
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_expert_recruitment_node():
    """Test expert recruitment creates 3 independent medical experts"""
    from langgraph_basic import ExpertRecruitmentNode
    
    recruiter = ExpertRecruitmentNode(model_info="gemini-2.5-flash")
    
    # Mock LLM response for expert recruitment
    mock_recruitment_response = '''
    {
      "experts": [
        {
          "id": 1,
          "role": "Cardiologist",
          "expertise_description": "Specializes in heart and cardiovascular system disorders",
          "hierarchy": "Independent"
        },
        {
          "id": 2,
          "role": "Pulmonologist", 
          "expertise_description": "Specializes in respiratory system diseases",
          "hierarchy": "Independent"
        },
        {
          "id": 3,
          "role": "Emergency Medicine Physician",
          "expertise_description": "Specializes in acute care and emergency medical situations", 
          "hierarchy": "Independent"
        }
      ]
    }
    '''
    
    mock_usage = {"input_tokens": 85, "output_tokens": 120}
    
    with patch.object(recruiter, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_recruitment_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "What is the primary treatment for acute myocardial infarction?",
            "token_usage": {"input": 0, "output": 0},
            "agents": []
        }
        
        result = recruiter.recruit_experts(state)
        
        # Verify Command structure
        assert "experts" in result.update
        assert len(result.update["experts"]) == 3
        
        # Verify expert structure
        experts = result.update["experts"]
        assert experts[0]["role"] == "Cardiologist"
        assert experts[1]["role"] == "Pulmonologist" 
        assert experts[2]["role"] == "Emergency Medicine Physician"
        assert all(expert["hierarchy"] == "Independent" for expert in experts)
        
        # Verify token tracking
        assert result.update["token_usage"]["input"] == 85
        assert result.update["token_usage"]["output"] == 120
        
        # Verify routing
        assert result.goto == "expert_analysis"

def test_expert_recruitment_fallback_parsing():
    """Test expert recruitment fallback when JSON parsing fails"""
    from langgraph_basic import ExpertRecruitmentNode
    
    recruiter = ExpertRecruitmentNode(model_info="gemini-2.5-flash")
    
    # Mock malformed response
    mock_response = "I recommend recruiting these experts for this medical case..."
    mock_usage = {"input_tokens": 60, "output_tokens": 45}
    
    with patch.object(recruiter, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Test medical question",
            "token_usage": {"input": 25, "output": 15},
            "agents": []
        }
        
        result = recruiter.recruit_experts(state)
        
        # Should use default experts
        experts = result.update["experts"]
        assert len(experts) == 3
        assert experts[0]["role"] == "General Internal Medicine Physician"
        assert experts[1]["role"] == "Emergency Medicine Physician"
        assert experts[2]["role"] == "Family Medicine Physician"
        assert result.goto == "expert_analysis"

def test_expert_analysis_node():
    """Test individual expert analysis with JSON response parsing"""
    from langgraph_basic import ExpertAnalysisNode
    
    # Test with cardiologist expert
    expert_data = {
        "id": 1,
        "role": "Cardiologist",
        "expertise_description": "Specializes in heart and cardiovascular system disorders",
        "hierarchy": "Independent"
    }
    
    analyzer = ExpertAnalysisNode(expert_data, model_info="gemini-2.5-flash")
    
    # Mock expert response
    mock_expert_response = '''
    {
      "reasoning": "The patient presents with classic symptoms of acute myocardial infarction including chest pain, elevated troponins, and ECG changes. Immediate percutaneous coronary intervention (PCI) is the gold standard treatment for STEMI within the therapeutic window.",
      "answer": "A) Immediate percutaneous coronary intervention (PCI)"
    }
    '''
    
    mock_usage = {"input_tokens": 150, "output_tokens": 85}
    
    with patch.object(analyzer, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_expert_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "A 55-year-old male presents with severe chest pain...",
            "token_usage": {"input": 200, "output": 100},
            "expert_responses": []
        }
        
        result = analyzer.analyze_question(state)
        
        # Verify expert response structure
        expert_responses = result.update["expert_responses"]
        assert len(expert_responses) == 1
        
        response = expert_responses[0]
        assert response["expert_id"] == 1
        assert response["role"] == "Cardiologist"
        assert "myocardial infarction" in response["reasoning"].lower()
        assert response["answer"].startswith("A)")
        
        # Verify token delta reporting
        assert "token_usage" not in result.update
        assert result.update["expert_token_delta"]["input"] == 150
        assert result.update["expert_token_delta"]["output"] == 85

def test_expert_analysis_json_parsing_fallback():
    """Test expert analysis fallback when JSON parsing fails"""
    from langgraph_basic import ExpertAnalysisNode
    
    expert_data = {
        "id": 2,
        "role": "Pulmonologist",
        "expertise_description": "Specializes in respiratory system diseases", 
        "hierarchy": "Independent"
    }
    
    analyzer = ExpertAnalysisNode(expert_data, model_info="gemini-2.5-flash")
    
    # Mock malformed response
    mock_response = "Based on the respiratory symptoms, I believe this is likely asthma. The answer is B."
    mock_usage = {"input_tokens": 120, "output_tokens": 60}
    
    with patch.object(analyzer, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Test question",
            "token_usage": {"input": 50, "output": 30},
            "expert_responses": []
        }
        
        result = analyzer.analyze_question(state)

        # Should create fallback response
        expert_responses = result.update["expert_responses"]
        response = expert_responses[0]

        assert result.update["expert_token_delta"]["input"] == mock_usage["input_tokens"] * 3
        assert result.update["expert_token_delta"]["output"] == mock_usage["output_tokens"] * 3

        assert response["expert_id"] == 2
        assert response["role"] == "Pulmonologist"
        assert "Unable to parse expert response" in response["reasoning"] or "asthma" in response["reasoning"].lower()
        assert "X)" in response["answer"] or "B" in response["answer"]

def test_sequential_expert_analysis():
    """Test sequential processing of all 3 experts in the subgraph"""
    from langgraph_basic import create_basic_processing_subgraph
    
    # Create subgraph to test expert processing
    subgraph = create_basic_processing_subgraph(model_info="gemini-2.5-flash")
    
    # Mock state with recruited experts
    state = {
        "messages": [],
        "question": "Medical question for analysis",
        "experts": [
            {"id": 1, "role": "Cardiologist", "expertise_description": "Heart specialist", "hierarchy": "Independent"},
            {"id": 2, "role": "Pulmonologist", "expertise_description": "Lung specialist", "hierarchy": "Independent"},
            {"id": 3, "role": "Emergency Medicine Physician", "expertise_description": "Emergency care", "hierarchy": "Independent"}
        ],
        "token_usage": {"input": 100, "output": 50},
        "expert_responses": []
    }
    
    # Should be able to compile and process
    compiled_subgraph = subgraph.compile()
    assert compiled_subgraph is not None

def test_arbitrator_node():
    """Test arbitrator synthesis of expert opinions"""
    from langgraph_basic import ArbitratorNode
    
    arbitrator = ArbitratorNode(model_info="gemini-2.5-flash")
    
    # Mock arbitrator response
    mock_arbitrator_response = '''
    {
      "analysis": "After reviewing all three expert opinions, there is consensus that immediate PCI is the gold standard treatment for STEMI. The cardiologist and emergency medicine physician both emphasize the time-critical nature, while the pulmonologist concurs with the cardiovascular approach. The evidence strongly supports immediate intervention.",
      "final_answer": "A) Immediate percutaneous coronary intervention (PCI)"
    }
    '''
    
    mock_usage = {"input_tokens": 200, "output_tokens": 95}
    
    with patch.object(arbitrator, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_arbitrator_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Treatment for acute myocardial infarction?",
            "expert_responses": [
                {
                    "expert_id": 1,
                    "role": "Cardiologist", 
                    "reasoning": "STEMI requires immediate intervention",
                    "answer": "A) Immediate PCI"
                },
                {
                    "expert_id": 2,
                    "role": "Pulmonologist",
                    "reasoning": "Cardiovascular emergency needs PCI", 
                    "answer": "A) Immediate PCI"
                },
                {
                    "expert_id": 3,
                    "role": "Emergency Medicine Physician",
                    "reasoning": "Time-critical intervention required",
                    "answer": "A) Immediate PCI"
                }
            ],
            "token_usage": {"input": 500, "output": 300}
        }
        
        result = arbitrator.make_final_decision(state)
        
        # Verify final decision structure
        final_decision = result.update["final_decision"]
        assert "analysis" in final_decision
        assert "final_answer" in final_decision
        assert final_decision["final_answer"].startswith("A)")
        assert "consensus" in final_decision["analysis"].lower()
        
        # Verify token tracking
        assert result.update["token_usage"]["input"] == 700   # 500 + 200
        assert result.update["token_usage"]["output"] == 395  # 300 + 95
        
        # Verify completion routing
        assert result.goto == "basic_complete"

def test_arbitrator_json_parsing_fallback():
    """Test arbitrator fallback when JSON parsing fails"""
    from langgraph_basic import ArbitratorNode
    
    arbitrator = ArbitratorNode(model_info="gemini-2.5-flash")
    
    # Mock malformed response
    mock_response = "After careful consideration, I believe the answer is A) Immediate PCI based on expert consensus."
    mock_usage = {"input_tokens": 150, "output_tokens": 40}
    
    with patch.object(arbitrator, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Test question",
            "expert_responses": [{"expert_id": 1, "role": "Test", "reasoning": "Test", "answer": "A) Test"}],
            "token_usage": {"input": 100, "output": 60}
        }
        
        result = arbitrator.make_final_decision(state)
        
        # Should create fallback response
        final_decision = result.update["final_decision"]
        assert "analysis" in final_decision
        assert "final_answer" in final_decision
        assert ("Unable to parse" in final_decision["analysis"] or 
                "careful consideration" in final_decision["analysis"])

def test_basic_processing_subgraph_construction():
    """Test basic processing subgraph structure and node connections"""
    from langgraph_basic import create_basic_processing_subgraph
    
    subgraph = create_basic_processing_subgraph(model_info="gemini-2.5-flash")
    
    # Verify subgraph compiles
    compiled_subgraph = subgraph.compile()
    assert compiled_subgraph is not None
    
    # Verify subgraph has correct structure
    graph_dict = compiled_subgraph.get_graph()
    assert graph_dict is not None

def test_basic_processing_end_to_end():
    """Test complete basic processing flow from recruitment to final decision"""
    from langgraph_basic import create_basic_processing_subgraph, ExpertRecruitmentNode, ExpertAnalysisNode, ArbitratorNode
    
    subgraph = create_basic_processing_subgraph(model_info="gemini-2.5-flash")
    compiled_subgraph = subgraph.compile()
    
    # Test state
    initial_state = {
        "messages": [],
        "question": "What is the treatment for acute myocardial infarction?\nA) Immediate PCI\nB) Medications only\nC) Surgery\nD) Wait and see",
        "token_usage": {"input": 0, "output": 0},
        "agents": [],
        "expert_responses": []
    }
    
    # Mock all LLM calls in the pipeline
    with patch.object(ExpertRecruitmentNode, '_call_llm') as mock_recruit, \
         patch.object(ExpertAnalysisNode, '_call_llm') as mock_analyze, \
         patch.object(ArbitratorNode, '_call_llm') as mock_arbitrate:
        
        # Mock recruitment response
        mock_recruit.return_value = ('''
        {
          "experts": [
            {"id": 1, "role": "Cardiologist", "expertise_description": "Heart specialist", "hierarchy": "Independent"},
            {"id": 2, "role": "Emergency Physician", "expertise_description": "Emergency care", "hierarchy": "Independent"},  
            {"id": 3, "role": "Internal Medicine", "expertise_description": "General medicine", "hierarchy": "Independent"}
          ]
        }
        ''', {"input_tokens": 80, "output_tokens": 100})
        
        # Mock expert analysis responses  
        mock_analyze.return_value = ('''
        {
          "reasoning": "Immediate PCI is gold standard for STEMI treatment",
          "answer": "A) Immediate PCI"
        }
        ''', {"input_tokens": 120, "output_tokens": 60})
        
        # Mock arbitrator response
        mock_arbitrate.return_value = ('''
        {
          "analysis": "All experts agree on immediate PCI as the correct treatment",
          "final_answer": "A) Immediate PCI"
        }
        ''', {"input_tokens": 180, "output_tokens": 80})
        
        # Run complete flow
        result = compiled_subgraph.invoke(initial_state)
        
        # Verify completion
        assert "final_decision" in result
        assert result["final_decision"]["final_answer"].startswith("A)")
        assert result["token_usage"]["input"] > 0
        assert result["token_usage"]["output"] > 0

def test_token_usage_accumulation_across_nodes():
    """Test that token usage properly accumulates across all basic processing nodes"""
    from langgraph_basic import ExpertRecruitmentNode, ArbitratorNode
    
    # Test token accumulation through multiple nodes
    recruiter = ExpertRecruitmentNode(model_info="gemini-2.5-flash")
    arbitrator = ArbitratorNode(model_info="gemini-2.5-flash")
    
    with patch.object(recruiter, '_call_llm') as mock_recruit, \
         patch.object(arbitrator, '_call_llm') as mock_arbitrate:
        
        mock_recruit.return_value = ('{"experts": []}', {"input_tokens": 50, "output_tokens": 75})
        mock_arbitrate.return_value = ('{"analysis": "test", "final_answer": "A) Test"}', {"input_tokens": 100, "output_tokens": 60})
        
        # Initial state with existing tokens
        state = {
            "messages": [], 
            "question": "Test",
            "token_usage": {"input": 200, "output": 150},
            "agents": [],
            "expert_responses": [{"expert_id": 1, "role": "Test", "reasoning": "Test", "answer": "A) Test"}]
        }
        
        # Process through recruitment
        result1 = recruiter.recruit_experts(state)
        assert result1.update["token_usage"]["input"] == 250   # 200 + 50
        assert result1.update["token_usage"]["output"] == 225  # 150 + 75
        
        # Update state and process through arbitrator
        state.update(result1.update)
        result2 = arbitrator.make_final_decision(state)
        assert result2.update["token_usage"]["input"] == 350   # 250 + 100  
        assert result2.update["token_usage"]["output"] == 285  # 225 + 60

def test_error_resilience_in_basic_processing():
    """Test error handling and graceful degradation in basic processing"""
    from langgraph_basic import create_basic_processing_subgraph, ExpertRecruitmentNode, ExpertAnalysisNode, ArbitratorNode
    
    subgraph = create_basic_processing_subgraph(model_info="gemini-2.5-flash")
    compiled_subgraph = subgraph.compile()
    
    # Test with minimal/invalid state
    problematic_state = {
        "messages": [],
        "question": "",  # Empty question
        "token_usage": {"input": 0, "output": 0},
        "agents": [],
        "expert_responses": []
    }
    
    # Should handle gracefully without crashing
    with patch.object(ExpertRecruitmentNode, '_call_llm') as mock_recruit, \
         patch.object(ExpertAnalysisNode, '_call_llm') as mock_analyze, \
         patch.object(ArbitratorNode, '_call_llm') as mock_arbitrate:
        
        mock_recruit.return_value = ("Invalid response", {"input_tokens": 10, "output_tokens": 5})
        mock_analyze.return_value = ("Invalid response", {"input_tokens": 5, "output_tokens": 3})
        mock_arbitrate.return_value = ("Invalid response", {"input_tokens": 8, "output_tokens": 4})
        
        result = compiled_subgraph.invoke(problematic_state)
        
        # Should still complete with fallback values
        assert "final_decision" in result
        assert result["token_usage"]["input"] >= 10
        assert result["token_usage"]["output"] >= 5

def test_model_compatibility():
    """Test basic processing works with different model types"""
    from langgraph_basic import ExpertRecruitmentNode, ArbitratorNode
    
    models_to_test = ["gemini-2.5-flash", "gpt-4.1-mini"]
    
    for model_info in models_to_test:
        # Test node creation doesn't fail
        recruiter = ExpertRecruitmentNode(model_info=model_info)
        arbitrator = ArbitratorNode(model_info=model_info)
        
        assert recruiter.model_info == model_info
        assert arbitrator.model_info == model_info

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
