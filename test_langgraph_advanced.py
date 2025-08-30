#!/usr/bin/env python3
"""
Test suite for LangGraph-based advanced processing graph.
Following TDD approach - tests written first to drive implementation.

Advanced processing implements Multi-Disciplinary Team (MDT) approach:
1. MDT Formation - 3 teams with 3 members each (IAT, Specialist, FRDT)
2. Team Categorization - Initial Assessment, Specialist, Final Review
3. Internal Team Assessments - Lead-driven team discussions
4. Parallel Processing - Teams work independently
5. Overall Coordinator - Cross-team synthesis

Target: 75%+ accuracy with MDT approach
"""

import pytest
import json
import sys
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mdt_formation_node():
    """Test MDT formation creates 3 teams with proper structure"""
    from langgraph_advanced import MDTFormationNode
    
    formation = MDTFormationNode(model_info="gemini-2.5-flash")
    
    # Mock LLM response with MDT structure
    mock_response = '''
    {
      "teams": [
        {
          "team_id": 1,
          "team_name": "Initial Assessment Team (IAT)",
          "members": [
            {
              "member_id": 1,
              "role": "Emergency Medicine Physician (Lead)",
              "expertise_description": "Specializes in acute care and initial patient assessment"
            },
            {
              "member_id": 2,
              "role": "General Internal Medicine",
              "expertise_description": "Provides comprehensive medical evaluation"
            },
            {
              "member_id": 3,
              "role": "Nurse Practitioner",
              "expertise_description": "Assists with patient care coordination"
            }
          ]
        },
        {
          "team_id": 2,
          "team_name": "Cardiology Specialist Team",
          "members": [
            {
              "member_id": 1,
              "role": "Cardiologist (Lead)",
              "expertise_description": "Specializes in heart conditions"
            },
            {
              "member_id": 2,
              "role": "Cardiac Surgeon",
              "expertise_description": "Provides surgical expertise"
            },
            {
              "member_id": 3,
              "role": "Cardiac Nurse",
              "expertise_description": "Specialized cardiac care"
            }
          ]
        },
        {
          "team_id": 3,
          "team_name": "Final Review and Decision Team (FRDT)",
          "members": [
            {
              "member_id": 1,
              "role": "Chief Medical Officer (Lead)",
              "expertise_description": "Overall medical decision making"
            },
            {
              "member_id": 2,
              "role": "Quality Assurance Specialist",
              "expertise_description": "Ensures care quality standards"
            },
            {
              "member_id": 3,
              "role": "Patient Advocate",
              "expertise_description": "Represents patient interests"
            }
          ]
        }
      ]
    }
    '''
    
    mock_usage = {"input_tokens": 150, "output_tokens": 200}
    
    with patch.object(formation, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Complex multi-organ failure case",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = formation.form_teams(state)
        
        # Verify team structure
        assert "mdt_teams" in result.update
        teams = result.update["mdt_teams"]
        assert len(teams) == 3
        
        # Check team names
        team_names = [t["team_name"] for t in teams]
        assert any("IAT" in name or "Initial" in name for name in team_names)
        assert any("FRDT" in name or "Final" in name for name in team_names)
        
        # Check each team has 3 members
        for team in teams:
            assert len(team["members"]) == 3
            # Check for lead member
            roles = [m["role"] for m in team["members"]]
            assert any("Lead" in role for role in roles)
        
        # Verify routing
        assert result.goto == "categorize_teams"

def test_mdt_formation_fallback():
    """Test MDT formation with fallback when JSON parsing fails"""
    from langgraph_advanced import MDTFormationNode
    
    formation = MDTFormationNode(model_info="gemini-2.5-flash")
    
    # Mock malformed response
    mock_response = "I recommend forming three teams for this complex case..."
    mock_usage = {"input_tokens": 100, "output_tokens": 50}
    
    with patch.object(formation, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "messages": [],
            "question": "Test question",
            "token_usage": {"input": 0, "output": 0}
        }
        
        result = formation.form_teams(state)
        
        # Should use default teams
        teams = result.update["mdt_teams"]
        assert len(teams) == 3
        assert any("Initial Assessment" in t["team_name"] for t in teams)
        assert any("Specialist" in t["team_name"] for t in teams)
        assert any("Final Review" in t["team_name"] for t in teams)

def test_team_categorization():
    """Test categorization of teams into IAT, Specialist, and FRDT"""
    from langgraph_advanced import categorize_teams
    
    teams = [
        {"team_name": "Initial Assessment Team (IAT)", "members": []},
        {"team_name": "Cardiology Specialist Team", "members": []},
        {"team_name": "Final Review and Decision Team (FRDT)", "members": []}
    ]
    
    categorized = categorize_teams(teams)
    
    assert "initial" in categorized
    assert "specialist" in categorized
    assert "final_review" in categorized
    assert len(categorized["initial"]) == 1
    assert len(categorized["specialist"]) == 1
    assert len(categorized["final_review"]) == 1

def test_team_member_node():
    """Test individual team member creation and functionality"""
    from langgraph_advanced import TeamMemberNode
    
    member_info = {
        "role": "Cardiologist (Lead)",
        "expertise_description": "Specializes in heart conditions"
    }
    
    member = TeamMemberNode(member_info, model_info="gemini-2.5-flash")
    
    # Test member can provide input
    mock_response = "Based on the ECG findings, this appears to be acute myocardial infarction."
    mock_usage = {"input_tokens": 80, "output_tokens": 40}
    
    with patch.object(member, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "question": "Patient with chest pain",
            "token_usage": {"input": 100, "output": 50}
        }
        
        result = member.provide_assessment(state)
        
        assert "assessment" in result.update
        assert "myocardial infarction" in result.update["assessment"].lower()
        assert result.update["token_usage"]["input"] == 180

def test_team_assessment_node():
    """Test internal team assessment generation"""
    from langgraph_advanced import TeamAssessmentNode
    
    team = {
        "team_name": "Cardiology Specialist Team",
        "members": [
            {"role": "Cardiologist (Lead)", "expertise_description": "Heart specialist"},
            {"role": "Cardiac Surgeon", "expertise_description": "Surgical expertise"},
            {"role": "Cardiac Nurse", "expertise_description": "Patient care"}
        ]
    }
    
    assessment_node = TeamAssessmentNode(team, model_info="gemini-2.5-flash")
    
    # Mock team discussion
    mock_response = "Team consensus: Patient requires immediate cardiac catheterization."
    mock_usage = {"input_tokens": 120, "output_tokens": 60}
    
    with patch.object(assessment_node, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "question": "Acute coronary syndrome management",
            "token_usage": {"input": 200, "output": 100}
        }
        
        result = assessment_node.generate_assessment(state)
        
        assert "team_assessment" in result.update
        assert "cardiac catheterization" in result.update["team_assessment"].lower()

def test_lead_member_identification():
    """Test identification of lead member in team"""
    from langgraph_advanced import identify_lead_member
    
    members = [
        {"role": "General Physician", "expertise_description": "General care"},
        {"role": "Cardiologist (Lead)", "expertise_description": "Heart specialist"},
        {"role": "Nurse", "expertise_description": "Patient care"}
    ]
    
    lead = identify_lead_member(members)
    
    assert lead is not None
    assert "Lead" in lead["role"]
    assert lead["role"] == "Cardiologist (Lead)"

def test_lead_member_fallback():
    """Test fallback when no explicit lead member"""
    from langgraph_advanced import identify_lead_member
    
    members = [
        {"role": "Cardiologist", "expertise_description": "Heart specialist"},
        {"role": "Surgeon", "expertise_description": "Surgical care"},
        {"role": "Nurse", "expertise_description": "Patient care"}
    ]
    
    lead = identify_lead_member(members)
    
    # Should default to first member
    assert lead is not None
    assert lead["role"] == "Cardiologist"

def test_parallel_team_processing():
    """Test parallel processing of multiple teams"""
    from langgraph_advanced import process_teams_in_parallel
    
    teams = [
        {"team_id": 1, "team_name": "IAT", "assessment": None},
        {"team_id": 2, "team_name": "Specialist", "assessment": None},
        {"team_id": 3, "team_name": "FRDT", "assessment": None}
    ]
    
    state = {
        "mdt_teams": teams,
        "question": "Test question",
        "token_usage": {"input": 0, "output": 0}
    }
    
    # Mock assessments
    with patch('langgraph_advanced.TeamAssessmentNode.generate_assessment') as mock_assess:
        mock_assess.return_value.update = {
            "team_assessment": "Test assessment",
            "token_usage": {"input": 50, "output": 25}
        }
        
        results = process_teams_in_parallel(state)
        
        assert len(results) == 3
        assert all("assessment" in r for r in results)

def test_assessment_compilation():
    """Test compilation of team assessments into report"""
    from langgraph_advanced import compile_assessment_report
    
    assessments = {
        "initial": [{"team_name": "IAT", "assessment": "Initial assessment complete"}],
        "specialist": [{"team_name": "Cardiology", "assessment": "Cardiac evaluation done"}],
        "final_review": [{"team_name": "FRDT", "assessment": "Final review complete"}]
    }
    
    report = compile_assessment_report(assessments)
    
    assert "[Initial Assessments]" in report
    assert "[Specialist Team Assessments]" in report
    assert "[Final Review Team Decisions]" in report
    assert "Initial assessment complete" in report
    assert "Cardiac evaluation done" in report

def test_overall_coordinator_node():
    """Test overall coordinator synthesis"""
    from langgraph_advanced import OverallCoordinatorNode
    
    coordinator = OverallCoordinatorNode(model_info="gemini-2.5-flash")
    
    # Mock coordinator response
    mock_response = '''
    {
      "analysis": "After reviewing all MDT assessments, the consensus is immediate intervention",
      "final_answer": "A) Immediate cardiac catheterization"
    }
    '''
    mock_usage = {"input_tokens": 200, "output_tokens": 80}
    
    with patch.object(coordinator, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "question": "Treatment plan?",
            "compiled_report": "Combined MDT assessments...",
            "token_usage": {"input": 300, "output": 150}
        }
        
        result = coordinator.coordinate_decision(state)
        
        assert "coordinator_decision" in result.update
        assert "analysis" in result.update["coordinator_decision"]
        assert "final_answer" in result.update["coordinator_decision"]
        assert "A)" in result.update["coordinator_decision"]["final_answer"]
        assert result.goto == "advanced_complete"

def test_coordinator_json_parsing_fallback():
    """Test coordinator fallback when JSON parsing fails"""
    from langgraph_advanced import OverallCoordinatorNode
    
    coordinator = OverallCoordinatorNode(model_info="gemini-2.5-flash")
    
    # Mock non-JSON response
    mock_response = "The final answer is A) Immediate intervention based on all assessments"
    mock_usage = {"input_tokens": 150, "output_tokens": 40}
    
    with patch.object(coordinator, '_call_llm') as mock_llm:
        mock_llm.return_value = (mock_response, mock_usage)
        
        state = {
            "question": "Test",
            "compiled_report": "Test report",
            "token_usage": {"input": 100, "output": 50}
        }
        
        result = coordinator.coordinate_decision(state)
        
        # Should extract answer from text
        assert "coordinator_decision" in result.update
        decision = result.update["coordinator_decision"]
        assert "A)" in decision.get("final_answer", decision.get("raw_response", ""))

def test_advanced_subgraph_integration():
    """Test integration of advanced processing subgraph"""
    from langgraph_advanced import create_advanced_processing_subgraph
    
    subgraph = create_advanced_processing_subgraph(model_info="gpt-4.1-mini")
    
    # Should compile successfully
    compiled = subgraph.compile()
    assert compiled is not None
    
    # Check graph structure
    graph_dict = compiled.get_graph()
    assert graph_dict is not None

def test_token_usage_accumulation_mdt():
    """Test token usage accumulation across MDT processing"""
    from langgraph_advanced import MDTFormationNode
    
    formation = MDTFormationNode(model_info="gemini-2.5-flash")
    
    state = {
        "token_usage": {"input": 500, "output": 250},
        "messages": [],
        "question": "Test"
    }
    
    mock_usage = {"input_tokens": 150, "output_tokens": 100}
    
    with patch.object(formation, '_call_llm') as mock_llm:
        mock_llm.return_value = ('{"teams": []}', mock_usage)
        
        result = formation.form_teams(state)
        
        # Verify accumulation
        assert result.update["token_usage"]["input"] == 650  # 500 + 150
        assert result.update["token_usage"]["output"] == 350  # 250 + 100

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])