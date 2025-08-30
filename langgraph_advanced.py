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
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph_mdm import LangGraphAgent


class AdvancedProcessingState(TypedDict, total=False):
    """Extended state for advanced processing with MDT mechanics"""
    # Core fields
    messages: List[Any]
    question: str
    answer_options: Optional[List[str]]
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
        answer_options = state.get("answer_options", [])
        compiled_report = state.get("compiled_report", "No report available")
        
        # Format answer options for display
        options_text = "\n".join(answer_options) if answer_options else "No options provided"
        
        coordinator_prompt = f"""You are a medical coordinator. Review the following MDT investigations and provide your final decision in JSON format:

Combined MDT Assessments:
{compiled_report}

Question: {question}

**Answer Options:**
{options_text}

Analyze all MDT assessments and provide your final decision in exactly this JSON format:

{{
  "analysis": "Your analysis of the MDT assessments and rationale in no more than 300 words",
  "final_answer": "X) Example Answer"
}}

**Requirements:**
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


# Async helper functions for real team interactions

async def process_team_internal_async(team: Dict, question: str, model_info: str) -> Dict[str, Any]:
    """Process a single team's internal discussion with authentic 4 LLM call MDT pattern"""
    team_name = team.get("team_name", "Medical Team")
    members = team.get("members", [])
    
    # Identify lead member and assistants
    lead_member = identify_lead_member(members)
    assist_members = [m for m in members if m != lead_member]
    
    # Create lead agent
    lead_role = lead_member.get("role", "team lead")
    lead_agent = LangGraphAgent(
        instruction=f"You are a {lead_role} leading a medical team discussion.",
        role=lead_role.lower(),
        model_info=model_info
    )
    
    # STEP 1: Lead coordination call (50 words) - Following old_utils.py:294-301
    delivery_prompt = f"""You are the lead of the medical group which aims to {team_name}. You have the following assistant clinicians who work for you:"""
    if assist_members:
        for a_mem in assist_members:
            delivery_prompt += f"\n{a_mem.get('role', 'Assistant')}"
    else:
        delivery_prompt += "\nYou are working independently or with a predefined protocol to address the goal."
    
    delivery_prompt += f"\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians (if any), or outline your approach. Strictly limit your response with no more than 50 words. \n Question: {question}"
    
    lead_delivery = lead_agent.chat(delivery_prompt)
    lead_delivery_usage = lead_agent.get_token_usage()
    
    # STEP 2: Parallel assistant investigations (100 words each) - Following old_utils.py:318-320
    investigations = []
    if assist_members:
        investigations = await gather_member_investigations_async(
            assist_members, lead_delivery, team_name, question, model_info
        )
    
    # STEP 3: Compile investigations - Following old_utils.py:322-327
    gathered_investigation = ""
    if investigations:
        for investigation_item in investigations:
            if len(investigation_item) >= 2:
                role, investigation = investigation_item[0], investigation_item[1]
                gathered_investigation += f"[{role}]\n{investigation}\n"
    else:
        gathered_investigation = lead_delivery
    
    # STEP 4: Lead final synthesis (100 words) - Following old_utils.py:330-332
    investigation_prompt = f"""The gathered investigation from your assistant clinicians (or your own initial assessment if working alone) is as follows:
{gathered_investigation}

Now, return your answer to the medical query among the option provided. Limit your response with no more than 100 words.

Question: {question}"""
    
    team_assessment = lead_agent.chat(investigation_prompt)
    lead_synthesis_usage = lead_agent.get_token_usage()
    
    # Calculate total token usage for this team (lead + assistants)
    total_tokens = {
        "input_tokens": lead_delivery_usage["input_tokens"] + lead_synthesis_usage["input_tokens"],
        "output_tokens": lead_delivery_usage["output_tokens"] + lead_synthesis_usage["output_tokens"]
    }
    
    # Add assistant token usage
    for investigation_item in investigations:
        if len(investigation_item) >= 3:  # (role, investigation, usage)
            _, _, assistant_usage = investigation_item
            total_tokens["input_tokens"] += assistant_usage["input_tokens"]
            total_tokens["output_tokens"] += assistant_usage["output_tokens"]
    
    return {
        "team_name": team_name,
        "assessment": team_assessment,
        "token_usage": total_tokens,
        "lead_delivery": lead_delivery,
        "investigations": investigations,
        "gathered_investigation": gathered_investigation
    }


async def gather_member_investigations_async(members: List[Dict], lead_delivery: str, 
                                           team_goal: str, question: str, model_info: str) -> List[tuple]:
    """Gather investigations from assistant members in parallel - Following old_utils.py:318-320"""
    
    async def get_member_investigation(member: Dict):
        role = member.get("role", "team member")
        expertise = member.get("expertise_description", "medical expertise")
        
        # Create member agent
        member_agent = LangGraphAgent(
            instruction=f"You are a {role} who {expertise.lower()}.",
            role=role.lower(),
            model_info=model_info
        )
        
        # Exact prompt pattern from old_utils.py:319
        investigation_prompt = f"""You are in a medical group where the goal is to {team_goal}. Your group lead is asking for the following investigations:
{lead_delivery}

Please remind your expertise and return your investigation summary that contains the core information. Strictly limit your response with no more than 100 words."""
        
        investigation = member_agent.chat(investigation_prompt)
        usage = member_agent.get_token_usage()
        
        return (role, investigation, usage)
    
    # Process all members in parallel
    tasks = [get_member_investigation(member) for member in members]
    return await asyncio.gather(*tasks)


def parallel_team_processing_sync(state: AdvancedProcessingState) -> Dict[str, Any]:
    """Sync wrapper for parallel team processing with real LLM calls"""
    
    teams = state.get("mdt_teams", [])
    question = state["question"]
    model_info = getattr(state, '_model_info', 'gemini-2.5-flash')
    
    # Process all teams in parallel (now that each team is just 1 LLM call)
    assessments = {
        "initial": [],
        "specialist": [],
        "final_review": []
    }
    
    total_token_usage = {"input_tokens": 0, "output_tokens": 0}
    
    # Use the parallel pattern from basic processing
    import asyncio
    
    async def _process_all_teams_async():
        team_tasks = [
            process_team_internal_async(team, question, model_info)
            for team in teams
        ]
        return await asyncio.gather(*team_tasks)
    
    def run_parallel_teams():
        return asyncio.run(_process_all_teams_async())
    
    try:
        team_results = run_parallel_teams()
    except RuntimeError:
        # Handle existing event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _process_all_teams_async())
            team_results = future.result()
    except Exception as e:
        print(f"Warning: Parallel processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing
        team_results = []
        for team in teams:
            try:
                def run_single_team():
                    return asyncio.run(process_team_internal_async(team, question, model_info))
                
                try:
                    result = run_single_team()
                    team_results.append(result)
                except RuntimeError:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, process_team_internal_async(team, question, model_info))
                        team_results.append(future.result())
            except Exception as team_error:
                print(f"Warning: Team {team.get('team_name', 'Unknown')} failed: {team_error}")
                team_results.append({
                    "team_name": team.get("team_name", "Error Team"),
                    "assessment": f"Processing error: {str(team_error)}",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}
                })
    
    # Process results
    for result in team_results:
        team_name = result["team_name"].lower()
        assessment_data = {
            "team_name": result["team_name"],
            "assessment": result["assessment"]
        }
        
        # Track token usage
        usage = result["token_usage"]
        total_token_usage["input_tokens"] += usage["input_tokens"]
        total_token_usage["output_tokens"] += usage["output_tokens"]
        
        # Categorize by team type
        if "initial" in team_name or "iat" in team_name:
            assessments["initial"].append(assessment_data)
        elif "final" in team_name or "review" in team_name or "frdt" in team_name:
            assessments["final_review"].append(assessment_data)
        else:
            assessments["specialist"].append(assessment_data)
    
    # Compile report
    compiled_report = compile_assessment_report(assessments)
    
    # Update state token usage
    current_usage = state.get("token_usage", {"input": 0, "output": 0})
    updated_usage = {
        "input": current_usage["input"] + total_token_usage["input_tokens"],
        "output": current_usage["output"] + total_token_usage["output_tokens"]
    }
    
    return {
        **state,
        "team_assessments": assessments,
        "compiled_report": compiled_report,
        "token_usage": updated_usage,
        "processing_stage": "teams_processed"
    }


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
        """Process all teams with real LLM interactions"""
        # Add model_info to state for the sync function
        state_with_model = {**state, '_model_info': model_info}
        return parallel_team_processing_sync(state_with_model)
    
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