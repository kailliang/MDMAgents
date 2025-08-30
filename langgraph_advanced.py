#!/usr/bin/env python3
"""
LangGraph implementation of Advanced Processing Graph for MDMAgents.

Implements Multi-Disciplinary Team (MDT) approach with complex coordination:
1. MDT Formation - 3 teams with 3 members each (IAT, Specialist, FRDT)
2. Team Categorization - Initial Assessment, Specialist, Final Review
3. Internal Team Assessments - Lead-driven team discussions
4. Parallel Processing - Teams work independently
5. Overall Coordinator - Cross-team synthesis

Target: 75%+ accuracy with MDT approach
Based on existing process_advanced_query function in utils.py:1108+
"""

import json
import re
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph_mdm import LangGraphAgent


class AdvancedProcessingState(TypedDict, total=False):
    """Extended state for advanced processing with MDT mechanics"""
    # Core fields
    messages: List[Any]
    question: str
    token_usage: Dict[str, int]
    processing_stage: str
    final_decision: Optional[Dict]
    
    # Advanced processing specific
    mdt_teams: List[Dict]
    team_assessments: Dict
    initial_assessments: List[Dict]
    specialist_assessments: List[Dict]
    final_review_assessments: List[Dict]
    compiled_report: str
    coordinator_decision: Dict
    current_team: Optional[Dict]


class MDTFormationNode:
    """Node for forming Multi-Disciplinary Teams"""
    
    def __init__(self, model_info: str = "gemini-2.5-flash"):
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the MDT formation agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction="You are an experienced medical expert who organizes Multidisciplinary Teams (MDTs) for complex medical cases.",
                role="mdt_recruiter",
                model_info=self.model_info
            )
        return self._agent
    
    def _call_llm(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """Make LLM call and return response with token usage"""
        agent = self._get_agent()
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def form_teams(self, state: AdvancedProcessingState) -> Command:
        """Form 3 MDTs with 3 members each"""
        question = state["question"]
        
        num_teams = 3
        num_members_per_team = 3
        
        formation_prompt = f"""Question: {question}

You should organize {num_teams} MDTs with different specialties and each MDT should have {num_members_per_team} clinicians.

Return your recruitment plan in JSON format:

{{
  "teams": [
    {{
      "team_id": 1,
      "team_name": "Initial Assessment Team (IAT)",
      "members": [
        {{
          "member_id": 1,
          "role": "Emergency Medicine Physician (Lead)",
          "expertise_description": "Specializes in acute care and initial patient assessment"
        }},
        {{
          "member_id": 2,
          "role": "General Internal Medicine",
          "expertise_description": "Provides comprehensive medical evaluation"
        }},
        {{
          "member_id": 3,
          "role": "Nurse Practitioner",
          "expertise_description": "Assists with patient care coordination"
        }}
      ]
    }}
  ]
}}

You must include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT). 
Each team should have exactly {num_members_per_team} members with one designated as Lead.
Return only valid JSON without markdown code blocks."""
        
        response, token_usage = self._call_llm(formation_prompt)
        
        # Parse MDT response
        try:
            # Clean JSON response
            cleaned_response = self._clean_json_response(response)
            mdt_data = json.loads(cleaned_response)
            teams = mdt_data.get('teams', [])
            
            if len(teams) != num_teams:
                raise ValueError(f"Expected {num_teams} teams, got {len(teams)}")
                
        except (json.JSONDecodeError, ValueError):
            # Fallback to default teams
            teams = self._create_default_teams()
        
        # Update token usage
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "mdt_teams": teams,
                "token_usage": updated_usage,
                "processing_stage": "teams_formed"
            },
            goto="categorize_teams"
        )
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown and extracting JSON"""
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        # Find JSON content between braces
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            response = response[start_idx:end_idx+1]
        
        return response.strip()
    
    def _create_default_teams(self) -> List[Dict]:
        """Create default MDT structure when parsing fails"""
        return [
            {
                "team_id": 1,
                "team_name": "Initial Assessment Team (IAT)",
                "members": [
                    {"member_id": 1, "role": "Emergency Medicine Physician (Lead)", "expertise_description": "Acute care specialist"},
                    {"member_id": 2, "role": "General Internal Medicine", "expertise_description": "Comprehensive medical evaluation"},
                    {"member_id": 3, "role": "Nurse Practitioner", "expertise_description": "Patient care coordination"}
                ]
            },
            {
                "team_id": 2,
                "team_name": "Specialist Assessment Team",
                "members": [
                    {"member_id": 1, "role": "Specialist Physician (Lead)", "expertise_description": "Domain-specific expertise"},
                    {"member_id": 2, "role": "Consulting Specialist", "expertise_description": "Additional specialty input"},
                    {"member_id": 3, "role": "Clinical Pharmacist", "expertise_description": "Medication management"}
                ]
            },
            {
                "team_id": 3,
                "team_name": "Final Review and Decision Team (FRDT)",
                "members": [
                    {"member_id": 1, "role": "Chief Medical Officer (Lead)", "expertise_description": "Overall medical decision making"},
                    {"member_id": 2, "role": "Quality Assurance Specialist", "expertise_description": "Care quality standards"},
                    {"member_id": 3, "role": "Patient Advocate", "expertise_description": "Patient interests representation"}
                ]
            }
        ]


class TeamMemberNode:
    """Node representing an individual team member"""
    
    def __init__(self, member_info: Dict, model_info: str = "gemini-2.5-flash"):
        self.member_info = member_info
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the team member agent"""
        if self._agent is None:
            role = self.member_info.get("role", "medical team member")
            expertise = self.member_info.get("expertise_description", "medical expertise")
            self._agent = LangGraphAgent(
                instruction=f"You are a {role} who {expertise.lower()}.",
                role=role.lower(),
                model_info=self.model_info
            )
        return self._agent
    
    def _call_llm(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """Make LLM call and return response with token usage"""
        agent = self._get_agent()
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def provide_assessment(self, state: AdvancedProcessingState) -> Command:
        """Provide team member assessment"""
        question = state["question"]
        role = self.member_info.get("role", "team member")
        
        assessment_prompt = f"""As a {role}, provide your assessment for this medical case:

{question}

Provide your professional opinion in 100-200 words."""
        
        response, token_usage = self._call_llm(assessment_prompt)
        
        # Update token usage
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "assessment": response,
                "token_usage": updated_usage
            },
            goto="continue_team_assessment"
        )


class TeamAssessmentNode:
    """Node for internal team assessment generation"""
    
    def __init__(self, team: Dict, model_info: str = "gemini-2.5-flash"):
        self.team = team
        self.model_info = model_info
        self._lead_agent = None
    
    def _get_lead_agent(self) -> LangGraphAgent:
        """Get or create the team lead agent"""
        if self._lead_agent is None:
            lead_member = identify_lead_member(self.team.get("members", []))
            role = lead_member.get("role", "team lead")
            self._lead_agent = LangGraphAgent(
                instruction=f"You are a {role} leading a medical team discussion.",
                role=role.lower(),
                model_info=self.model_info
            )
        return self._lead_agent
    
    def _call_llm(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """Make LLM call and return response with token usage"""
        agent = self._get_lead_agent()
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def generate_assessment(self, state: AdvancedProcessingState) -> Command:
        """Generate team assessment through internal discussion"""
        question = state["question"]
        team_name = self.team.get("team_name", "Medical Team")
        
        # Format team members
        members_text = ""
        for member in self.team.get("members", []):
            role = member.get("role", "Team Member")
            expertise = member.get("expertise_description", "Medical expertise")
            members_text += f"- {role}: {expertise}\n"
        
        assessment_prompt = f"""You are leading the {team_name} discussion. Your team consists of:

{members_text}

Medical Case: {question}

Conduct an internal team discussion and provide the team's consensus assessment. 
Limit response to 200 words."""
        
        response, token_usage = self._call_llm(assessment_prompt)
        
        # Update token usage
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "team_assessment": response,
                "token_usage": updated_usage
            },
            goto="continue_parallel_processing"
        )


class OverallCoordinatorNode:
    """Node for overall coordinator synthesis"""
    
    def __init__(self, model_info: str = "gemini-2.5-flash"):
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the coordinator agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction="You are an experienced medical coordinator who reviews MDT assessments and makes final decisions.",
                role="overall_coordinator",
                model_info=self.model_info
            )
        return self._agent
    
    def _call_llm(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """Make LLM call and return response with token usage"""
        agent = self._get_agent()
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def coordinate_decision(self, state: AdvancedProcessingState) -> Command:
        """Coordinate final decision from all MDT assessments"""
        question = state["question"]
        compiled_report = state.get("compiled_report", "No report available")
        
        coordinator_prompt = f"""You are a medical coordinator. Review the following MDT investigations and provide your final decision in JSON format:

Combined MDT Assessments:
{compiled_report}

Question: {question}

Analyze all MDT assessments and provide your final decision in exactly this JSON format:

{{
  "analysis": "Your analysis of the MDT assessments and rationale in no more than 300 words",
  "final_answer": "X) Example Answer"
}}

Requirements:
- Consider all MDT assessments
- Final answer must correspond to one of the provided options
- Return ONLY the JSON, no other text
"""
        
        response, token_usage = self._call_llm(coordinator_prompt)
        
        # Parse coordinator response
        try:
            json_match = re.search(r'\{[^{}]*"analysis"\s*:[^{}]*"final_answer"\s*:\s*"[^"]*"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                coordinator_decision = json.loads(json_str)
            else:
                # Fallback - extract from text
                coordinator_decision = {
                    "analysis": response[:300] if response else "Unable to parse coordinator response",
                    "final_answer": self._extract_answer_from_text(response)
                }
        except json.JSONDecodeError:
            coordinator_decision = {
                "analysis": "JSON parsing error in coordinator response",
                "final_answer": "X) Parsing error",
                "raw_response": response
            }
        
        # Update token usage
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "coordinator_decision": coordinator_decision,
                "final_decision": coordinator_decision,  # For compatibility
                "token_usage": updated_usage,
                "processing_stage": "advanced_complete"
            },
            goto="advanced_complete"
        )
    
    def _extract_answer_from_text(self, text: str) -> str:
        """Extract answer from free text response"""
        # Look for patterns like "A)", "(A)", "Answer: A", etc.
        patterns = [
            r'\b([A-D])\)\s*',
            r'\(([A-D])\)',
            r'Answer:\s*([A-D])',
            r'answer\s+is\s+([A-D])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)}) Coordinator decision"
        
        return "X) Unable to extract answer"


# Utility functions

def categorize_teams(teams: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize teams into initial, specialist, and final review"""
    categorized = {
        "initial": [],
        "specialist": [],
        "final_review": []
    }
    
    for team in teams:
        team_name = team.get("team_name", "").lower()
        
        if "initial" in team_name or "iat" in team_name:
            categorized["initial"].append(team)
        elif "final" in team_name or "review" in team_name or "frdt" in team_name:
            categorized["final_review"].append(team)
        else:
            categorized["specialist"].append(team)
    
    return categorized


def identify_lead_member(members: List[Dict]) -> Dict:
    """Identify the lead member in a team"""
    # Look for explicit lead designation
    for member in members:
        role = member.get("role", "").lower()
        if "lead" in role:
            return member
    
    # Fallback to first member
    return members[0] if members else {}


def process_teams_in_parallel(state: AdvancedProcessingState) -> List[Dict]:
    """Process teams in parallel (simplified implementation)"""
    teams = state.get("mdt_teams", [])
    results = []
    
    for team in teams:
        # Simplified - would use actual parallel processing
        result = {
            "team_name": team.get("team_name", "Unknown Team"),
            "assessment": f"Assessment from {team.get('team_name', 'team')}"
        }
        results.append(result)
    
    return results


def compile_assessment_report(assessments: Dict[str, List[Dict]]) -> str:
    """Compile assessment report from categorized teams"""
    report = ""
    
    # Initial assessments
    report += "[Initial Assessments]\n"
    for idx, assessment in enumerate(assessments.get("initial", [])):
        team_name = assessment.get("team_name", f"Team {idx+1}")
        content = assessment.get("assessment", "No assessment available")
        report += f"Team {idx+1} - {team_name}:\n{content}\n\n"
    
    # Specialist assessments
    report += "[Specialist Team Assessments]\n"
    for idx, assessment in enumerate(assessments.get("specialist", [])):
        team_name = assessment.get("team_name", f"Team {idx+1}")
        content = assessment.get("assessment", "No assessment available")
        report += f"Team {idx+1} - {team_name}:\n{content}\n\n"
    
    # Final review assessments
    report += "[Final Review Team Decisions]\n"
    for idx, assessment in enumerate(assessments.get("final_review", [])):
        team_name = assessment.get("team_name", f"Team {idx+1}")
        content = assessment.get("assessment", "No assessment available")
        report += f"Team {idx+1} - {team_name}:\n{content}\n\n"
    
    return report


def advanced_processing_placeholder(state: AdvancedProcessingState) -> Dict[str, Any]:
    """Placeholder for advanced processing completion"""
    return {
        **state,
        "processing_stage": "advanced_complete",
        "final_decision": state.get("final_decision", {"placeholder": "advanced_result"})
    }


def create_advanced_processing_subgraph(model_info: str = "gemini-2.5-flash") -> StateGraph:
    """Create the advanced processing subgraph with MDT system"""
    
    # Create subgraph for advanced processing
    subgraph = StateGraph(AdvancedProcessingState)
    
    # Create node instances
    formation = MDTFormationNode(model_info=model_info)
    coordinator = OverallCoordinatorNode(model_info=model_info)
    
    # Simplified flow for initial implementation
    def categorize_teams_node(state: AdvancedProcessingState) -> Dict[str, Any]:
        """Categorize teams and prepare for parallel processing"""
        teams = state.get("mdt_teams", [])
        categorized = categorize_teams(teams)
        
        return {
            **state,
            "team_assessments": categorized,
            "processing_stage": "teams_categorized"
        }
    
    def parallel_team_processing(state: AdvancedProcessingState) -> Dict[str, Any]:
        """Process all teams and generate assessments"""
        teams = state.get("mdt_teams", [])
        
        # Simplified processing - generate mock assessments
        assessments = {
            "initial": [],
            "specialist": [],
            "final_review": []
        }
        
        for team in teams:
            team_name = team.get("team_name", "Unknown Team").lower()
            assessment = {
                "team_name": team.get("team_name", "Unknown Team"),
                "assessment": f"Team assessment from {team.get('team_name', 'team')}"
            }
            
            if "initial" in team_name or "iat" in team_name:
                assessments["initial"].append(assessment)
            elif "final" in team_name or "review" in team_name or "frdt" in team_name:
                assessments["final_review"].append(assessment)
            else:
                assessments["specialist"].append(assessment)
        
        # Compile report
        compiled_report = compile_assessment_report(assessments)
        
        return {
            **state,
            "team_assessments": assessments,
            "compiled_report": compiled_report,
            "processing_stage": "teams_processed"
        }
    
    # Add nodes to subgraph
    subgraph.add_node("form_teams", formation.form_teams)
    subgraph.add_node("categorize_teams", categorize_teams_node)
    subgraph.add_node("process_teams", parallel_team_processing)
    subgraph.add_node("coordinate_decision", coordinator.coordinate_decision)
    subgraph.add_node("advanced_complete", advanced_processing_placeholder)
    
    # Define edges
    subgraph.add_edge("form_teams", "categorize_teams")
    subgraph.add_edge("categorize_teams", "process_teams")
    subgraph.add_edge("process_teams", "coordinate_decision")
    subgraph.add_edge("coordinate_decision", "advanced_complete")
    
    # Set entry point
    subgraph.add_edge("__start__", "form_teams")
    
    return subgraph


if __name__ == "__main__":
    # Test advanced processing subgraph
    subgraph = create_advanced_processing_subgraph()
    compiled_subgraph = subgraph.compile()
    
    test_state = {
        "messages": [],
        "question": "Complex multi-organ failure requiring MDT coordination",
        "token_usage": {"input": 0, "output": 0}
    }
    
    print("Testing advanced processing subgraph...")
    result = compiled_subgraph.invoke(test_state)
    print(f"Coordinator decision: {result.get('coordinator_decision', {})}")
    print(f"Token usage: {result.get('token_usage', {})}")