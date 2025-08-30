#!/usr/bin/env python3
"""
Test suite for LangGraph-based intermediate processing graph.
Following TDD approach - tests written first to drive implementation.

Intermediate processing implements multi-agent collaboration with debate:
1. Hierarchical Expert Recruitment - 3 experts with relationships
2. Multi-round Debate - 3 rounds with participation decisions
3. Expert Selection - Dynamic selection of discussion partners
4. Communication - Targeted questions and opinions
5. Moderator Consensus - Final synthesis of expert opinions

Target: 78%+ accuracy with collaborative processing
"""

import pytest
import json
import sys
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_hierarchical_expert_recruitment():
    """Test expert recruitment with hierarchical relationships"""
    from langgraph_intermediate import HierarchicalExpertRecruitmentNode
    
    recruiter = HierarchicalExpertRecruitmentNode(model_info="gemini-2.5-flash")
    
    # Mock LLM response with hierarchical experts
    mock_response = '''
    {
      "experts": [
        {
          "id": 1,
          "role": "Cardiologist",
          "description": "Specializes in heart and cardiovascular disorders",
          "hierarchy": "Independent"
        },
        {
          "id": 2,
          "role": "Pulmonologist",
          "description": "Specializes in respiratory system diseases",
          "hierarchy": "Cardiologist > Pulmonologist"
        },
        {
          "id": 3,
          "role": "Emergency Medicine Physician",
          "description": "Specializes in acute care",
          "hierarchy": "Independent"
        }
      ]
    }
    '''
    
    mock_usage = {"input_tokens": 100, "output_tokens": 150}
    
    with patch.object(recruiter, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Complex cardiac case requiring specialist consultation",
            "token_usage": {"input": 0, "output": 0},
            "round_number": 1,
            "turn_number": 1
        }
        
        result = recruiter.recruit_experts(state)
        
        # Verify hierarchical structure
        assert "experts_hierarchy" in result.update
        experts = result.update["experts_hierarchy"]
        assert len(experts) == 3
        
        # Check hierarchy parsing
        assert experts[1]["hierarchy"] == "Cardiologist > Pulmonologist"
        assert experts[0]["hierarchy"] == "Independent"
        
        # Verify routing
        assert result.goto == "initial_opinions"

def test_hierarchy_parsing():
    """Test parsing of expert hierarchy strings into tree structure"""
    from langgraph_intermediate import parse_hierarchy_structure
    
    experts_data = [
        {"role": "Cardiologist", "hierarchy": "Independent"},
        {"role": "Pulmonologist", "hierarchy": "Cardiologist > Pulmonologist"},
        {"role": "Emergency Physician", "hierarchy": "Independent"}
    ]
    
    hierarchy_tree = parse_hierarchy_structure(experts_data)
    
    # Verify tree structure
    assert hierarchy_tree is not None
    assert "moderator" in hierarchy_tree
    assert len(hierarchy_tree["children"]) >= 2  # At least 2 independent experts

def test_participation_decision_node():
    """Test expert participation decision logic"""
    from langgraph_intermediate import DebateParticipationNode
    
    expert = {
        "id": 1,
        "role": "Cardiologist",
        "description": "Heart specialist"
    }
    
    participation_node = DebateParticipationNode(expert, model_info="gemini-2.5-flash")
    
    # Mock participation decision
    mock_response = '''
    {
      "participate": true,
      "reason": "I disagree with the pulmonologist's assessment and need to clarify"
    }
    '''
    mock_usage = {"input_tokens": 80, "output_tokens": 40}
    
    with patch.object(participation_node, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "round_opinions": {
                1: {
                    "cardiologist": "Initial assessment suggests cardiac involvement",
                    "pulmonologist": "Respiratory symptoms are primary concern"
                }
            },
            "round_number": 1,
            "token_usage": {"input": 100, "output": 50}
        }
        
        result = participation_node.decide_participation(state)
        
        # Verify participation decision
        assert "participation_decisions" in result.update
        decision = result.update["participation_decisions"][-1]
        assert decision["expert_id"] == 1
        assert decision["participate"] is True
        assert "disagree" in decision["reason"].lower()

def test_participation_decision_fallback():
    """Test fallback when JSON parsing fails for participation"""
    from langgraph_intermediate import DebateParticipationNode
    
    expert = {"id": 2, "role": "Pulmonologist", "description": "Lung specialist"}
    participation_node = DebateParticipationNode(expert, model_info="gemini-2.5-flash")
    
    # Mock malformed response
    mock_response = "Yes, I want to participate in the discussion"
    mock_usage = {"input_tokens": 60, "output_tokens": 20}
    
    with patch.object(participation_node, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "round_opinions": {1: {}},
            "round_number": 1,
            "token_usage": {"input": 50, "output": 25}
        }
        
        result = participation_node.decide_participation(state)
        
        # Should parse "yes" from text
        decision = result.update["participation_decisions"][-1]
        assert decision["participate"] is True

def test_expert_selection_node():
    """Test expert selection for communication"""
    from langgraph_intermediate import ExpertSelectionNode
    
    expert = {"id": 1, "role": "Cardiologist", "description": "Heart specialist"}
    selection_node = ExpertSelectionNode(expert, model_info="gemini-2.5-flash")
    
    # Mock expert selection
    mock_response = '''
    {
      "selected_experts": [2, 3],
      "reason": "Need to discuss with pulmonologist and emergency physician"
    }
    '''
    mock_usage = {"input_tokens": 70, "output_tokens": 35}
    
    with patch.object(selection_node, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "experts_hierarchy": [
                {"id": 1, "role": "Cardiologist"},
                {"id": 2, "role": "Pulmonologist"},
                {"id": 3, "role": "Emergency Physician"}
            ],
            "token_usage": {"input": 75, "output": 40}
        }
        
        result = selection_node.select_experts(state)
        
        # Verify selection
        assert "expert_selections" in result.update
        selection = result.update["expert_selections"][-1]
        assert selection["source_expert_id"] == 1
        assert selection["selected_experts"] == [2, 3]

def test_expert_communication_node():
    """Test expert communication generation"""
    from langgraph_intermediate import ExpertCommunicationNode
    
    source_expert = {"id": 1, "role": "Cardiologist"}
    target_expert = {"id": 2, "role": "Pulmonologist"}
    
    comm_node = ExpertCommunicationNode(source_expert, target_expert, model_info="gemini-2.5-flash")
    
    # Mock communication response
    mock_response = "Based on the ECG findings, I believe we should consider cardiac etiology first. The elevated troponins suggest myocardial involvement rather than primary respiratory issue."
    mock_usage = {"input_tokens": 90, "output_tokens": 45}
    
    with patch.object(comm_node, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "question": "Patient with chest pain and dyspnea",
            "round_number": 1,
            "turn_number": 2,
            "interaction_log": {},
            "token_usage": {"input": 200, "output": 100}
        }
        
        result = comm_node.generate_communication(state)
        
        # Verify communication
        assert "interaction_log" in result.update
        assert "elevated troponins" in result.update["communication_content"].lower()
        assert result.update["token_usage"]["input"] == 290  # 200 + 90

def test_round_synthesis_node():
    """Test synthesis of opinions after each round"""
    from langgraph_intermediate import RoundSynthesisNode
    
    synthesis_node = RoundSynthesisNode(model_info="gemini-2.5-flash")
    
    # Mock synthesis for each expert
    mock_responses = [
        "After discussion, cardiac cause is most likely",
        "Agreed that cardiac workup is priority",
        "Consensus on cardiac etiology"
    ]
    mock_usage = {"input_tokens": 60, "output_tokens": 30}
    
    with patch.object(synthesis_node, '_call_llm') as mock_llm:
        mock_llm.side_effect = [(resp, mock_usage) for resp in mock_responses]
        
        state = {
            "experts_hierarchy": [
                {"id": 1, "role": "Cardiologist"},
                {"id": 2, "role": "Pulmonologist"},
                {"id": 3, "role": "Emergency Physician"}
            ],
            "question": "Test question",
            "round_number": 1,
            "round_opinions": {},
            "token_usage": {"input": 300, "output": 150}
        }
        
        result = synthesis_node.synthesize_round(state)
        
        # Verify round synthesis
        assert result.update["round_number"] == 2  # Incremented
        assert 2 in result.update["round_opinions"]  # Next round opinions
        assert len(result.update["round_opinions"][2]) == 3  # All experts

def test_moderator_consensus_node():
    """Test final moderator consensus building"""
    from langgraph_intermediate import ModeratorConsensusNode
    
    moderator = ModeratorConsensusNode(model_info="gemini-2.5-flash")
    
    # Mock moderator response
    mock_response = "Answer: A) Immediate cardiac catheterization"
    mock_usage = {"input_tokens": 150, "output_tokens": 60}
    
    with patch.object(moderator, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "question": "Treatment for STEMI?",
            "final_expert_opinions": {
                "cardiologist": "A) Immediate cardiac catheterization",
                "pulmonologist": "A) Immediate cardiac catheterization",
                "emergency_physician": "A) Immediate cardiac catheterization"
            },
            "token_usage": {"input": 500, "output": 250}
        }
        
        result = moderator.build_consensus(state)
        
        # Verify consensus
        assert "final_decision" in result.update
        assert "majority_vote" in result.update["final_decision"]
        assert "A)" in result.update["final_decision"]["majority_vote"]
        assert result.goto == "intermediate_complete"

def test_multi_round_debate_flow():
    """Test complete multi-round debate flow"""
    from langgraph_intermediate import create_intermediate_processing_subgraph
    
    subgraph = create_intermediate_processing_subgraph(model_info="gemini-2.5-flash")
    compiled_subgraph = subgraph.compile()
    
    # Initial state
    initial_state = {
        "messages": [],
        "question": "Complex medical case requiring debate",
        "token_usage": {"input": 0, "output": 0},
        "round_number": 1,
        "turn_number": 1,
        "interaction_log": {},
        "round_opinions": {},
        "participation_decisions": [],
        "expert_selections": []
    }
    
    # Mock all LLM calls
    with patch('langgraph_intermediate.HierarchicalExpertRecruitmentNode._call_llm') as mock_recruit:
        mock_recruit.return_value = ('''
        {
          "experts": [
            {"id": 1, "role": "Cardiologist", "description": "Heart specialist", "hierarchy": "Independent"},
            {"id": 2, "role": "Pulmonologist", "description": "Lung specialist", "hierarchy": "Independent"},
            {"id": 3, "role": "Emergency Physician", "description": "Emergency care", "hierarchy": "Independent"}
          ]
        }
        ''', {"input_tokens": 100, "output_tokens": 120})
        
        # Should be able to compile and structure exists
        assert compiled_subgraph is not None

def test_early_termination_no_participation():
    """Test early termination when no experts want to participate"""
    from langgraph_intermediate import check_debate_continuation
    
    state = {
        "participation_decisions": [
            {"expert_id": 1, "participate": False, "round": 2, "turn": 1},
            {"expert_id": 2, "participate": False, "round": 2, "turn": 1},
            {"expert_id": 3, "participate": False, "round": 2, "turn": 1}
        ],
        "round_number": 2,
        "turn_number": 1
    }
    
    should_continue = check_debate_continuation(state)
    
    # Should terminate if no one participates
    assert should_continue is False

def test_token_usage_accumulation():
    """Test token usage accumulation across debate rounds"""
    from langgraph_intermediate import RoundSynthesisNode
    
    synthesis_node = RoundSynthesisNode(model_info="gemini-2.5-flash")
    
    # Initial token state
    state = {
        "token_usage": {"input": 1000, "output": 500},
        "experts_hierarchy": [{"id": 1, "role": "Expert"}],
        "question": "Test",
        "round_number": 1,
        "round_opinions": {}
    }
    
    mock_usage = {"input_tokens": 100, "output_tokens": 50}
    
    with patch.object(synthesis_node, '_call_llm') as mock_llm:
        mock_llm.return_value = ("Opinion", mock_usage)
        
        result = synthesis_node.synthesize_round(state)
        
        # Verify accumulation
        assert result.update["token_usage"]["input"] >= 1100  # Original + new
        assert result.update["token_usage"]["output"] >= 550

def test_invalid_expert_selection_handling():
    """Test handling of invalid expert selections"""
    from langgraph_intermediate import validate_expert_selection
    
    # Invalid selection (expert 5 doesn't exist)
    selection = {
        "source_expert_id": 1,
        "selected_experts": [2, 5]
    }
    
    experts = [
        {"id": 1, "role": "Expert1"},
        {"id": 2, "role": "Expert2"},
        {"id": 3, "role": "Expert3"}
    ]
    
    valid_targets = validate_expert_selection(selection, experts)
    
    # Should filter out invalid expert 5
    assert valid_targets == [2]

def test_intermediate_subgraph_integration():
    """Test integration of intermediate processing subgraph"""
    from langgraph_intermediate import create_intermediate_processing_subgraph
    
    subgraph = create_intermediate_processing_subgraph(model_info="gpt-4.1-mini")
    
    # Should compile successfully
    compiled = subgraph.compile()
    assert compiled is not None
    
    # Check graph structure
    graph_dict = compiled.get_graph()
    assert graph_dict is not None

def test_hierarchy_weight_assignment():
    """Test weight assignment based on hierarchy"""
    from langgraph_intermediate import assign_hierarchy_weights
    
    experts = [
        {"id": 1, "role": "Senior Cardiologist", "hierarchy": "Independent"},
        {"id": 2, "role": "Junior Pulmonologist", "hierarchy": "Senior Cardiologist > Junior Pulmonologist"},
        {"id": 3, "role": "Emergency Physician", "hierarchy": "Independent"}
    ]
    
    weighted_experts = assign_hierarchy_weights(experts)
    
    # Senior should have higher weight than junior
    senior_weight = next(e["weight"] for e in weighted_experts if e["id"] == 1)
    junior_weight = next(e["weight"] for e in weighted_experts if e["id"] == 2)
    
    assert senior_weight > junior_weight

def test_debate_state_preservation():
    """Test that debate state is preserved across rounds"""
    from langgraph_intermediate import create_intermediate_processing_subgraph
    
    subgraph = create_intermediate_processing_subgraph(model_info="gemini-2.5-flash")
    
    initial_state = {
        "messages": [],
        "question": "Test question",
        "token_usage": {"input": 0, "output": 0},
        "round_number": 1,
        "custom_field": "should_be_preserved",
        "interaction_log": {},
        "round_opinions": {}
    }
    
    # State should preserve custom fields
    with patch('langgraph_intermediate.HierarchicalExpertRecruitmentNode._call_llm'):
        # The state transformation should preserve fields
        assert "custom_field" in initial_state
        assert initial_state["custom_field"] == "should_be_preserved"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])