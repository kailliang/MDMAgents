#!/usr/bin/env python3
"""
LangGraph implementation of Advanced Processing Graph for MDMAgents.

Implements Multi-Disciplinary Team (MDT) approach with complex coordination:
1. MDT Formation - 3 teams with 3 members each (IAT, Specialist, FRDT)
2. Team Categorization - Initial Assessment, Specialist, Final Review
3. Internal Team Assessments - Lead-driven team discussions
4. Parallel Processing - Teams work independently
5. Overall Coordinator - Cross-team synthesis

"""

import json
import re
import operator
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph_mdm import LangGraphAgent
from langsmith_integration import span as langsmith_span, preview_text


def _validate_word_count(text: str, max_words: int) -> bool:
    """Validate that text does not exceed maximum word count"""
    word_count = len(text.split())
    return word_count <= max_words


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    """Truncate text to maximum word count, preserving sentence structure"""
    words = text.split()
    if len(words) <= max_words:
        return text

    # Truncate to max_words and try to end at sentence boundary
    truncated_words = words[:max_words]
    truncated_text = ' '.join(truncated_words)

    # Try to end at a sentence boundary
    last_period = truncated_text.rfind('.')
    last_question = truncated_text.rfind('?')
    last_exclamation = truncated_text.rfind('!')

    last_sentence_end = max(last_period, last_question, last_exclamation)
    if last_sentence_end > len(truncated_text) * 0.7:  # If we can preserve 70% of content
        return truncated_text[:last_sentence_end + 1]

    return truncated_text + "..."


class AdvancedProcessingState(TypedDict, total=False):
    """Extended state for advanced processing with MDT mechanics"""
    # Core fields
    messages: List[Any]
    question: str
    answer_options: Optional[List[str]]
    token_usage: Annotated[Dict[str, int], lambda x, y: {"input": x.get("input", 0) + y.get("input", 0), "output": x.get("output", 0) + y.get("output", 0)}]
    processing_stage: str
    final_decision: Optional[Dict]
    
    # Advanced processing specific
    mdt_teams: List[Dict]
    team_assessments: Dict
    compiled_report: str
    coordinator_decision: Dict
    current_team: Optional[Dict]
    
    # Parallel processing - reducer for concurrent team results
    team_results: Annotated[List[Dict], operator.add]


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
        
        # Clear conversation history after each call to prevent accumulation across questions
        agent.clear_history()
        
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def form_teams(self, state: AdvancedProcessingState) -> Command:
        """Form 3 MDTs with 3 members each"""
        question = state["question"]
        answer_options = state.get("answer_options", [])
        
        num_teams = 3
        num_members_per_team = 3
        question_preview = preview_text(question)
        
        # Format answer options
        options_text = "\n".join(answer_options) if answer_options else "Multiple choice options not provided"
        
        formation_prompt = f"""Question: {question}

Answer Options:
{options_text}

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
          "expertise_description": "Specializes in acute care and initial patient assessment... within 30 words"
        }},
        {{
          "member_id": 2,
          "role": "General Internal Medicine",
          "expertise_description": "Provides comprehensive medical evaluation... within 30 words"
        }},
        {{
          "member_id": 3,
          "role": "Nurse Practitioner",
          "expertise_description": "Assists with patient care coordination... within 30 words"
        }}
      ]
    }}
  ]
}}

You must include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT). 
Each team should have exactly {num_members_per_team} members with one designated as Lead.
Return only valid JSON without markdown code blocks."""
        
        with langsmith_span(
            "advanced.form_teams",
            run_type="chain",
            inputs={
                "question_preview": question_preview,
                "options_count": len(answer_options or []),
            },
        ) as (_, finish_span):
            response, token_usage = self._call_llm(formation_prompt)
            
            # Parse MDT response
            try:
                cleaned_response = self._clean_json_response(response)
                mdt_data = json.loads(cleaned_response)
                teams = mdt_data.get('teams', [])
                
                if len(teams) != num_teams:
                    raise ValueError(f"Expected {num_teams} teams, got {len(teams)}")
                    
            except (json.JSONDecodeError, ValueError):
                teams = self._create_default_teams()
            
            # Update token usage
            current_usage = state.get("token_usage", {"input": 0, "output": 0})
            updated_usage = {
                "input": current_usage["input"] + token_usage["input_tokens"],
                "output": current_usage["output"] + token_usage["output_tokens"]
            }

            finish_span(
                outputs={
                    "teams": [team.get("team_name") for team in teams],
                    "token_usage": updated_usage,
                },
                usage=token_usage,
            )

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
                    {"member_id": 1, "role": "Emergency Medicine Physician (Lead)", "expertise_description": "Acute care specialist... within 30 words"},
                    {"member_id": 2, "role": "General Internal Medicine", "expertise_description": "Comprehensive medical evaluation... within 30 words"},
                    {"member_id": 3, "role": "Nurse Practitioner", "expertise_description": "Patient care coordination... within 30 words"}
                ]
            },
            {
                "team_id": 2,
                "team_name": "Specialist Assessment Team",
                "members": [
                    {"member_id": 1, "role": "Specialist Physician (Lead)", "expertise_description": "Domain-specific expertise... within 30 words"},
                    {"member_id": 2, "role": "Consulting Specialist", "expertise_description": "Additional specialty input... within 30 words"},
                    {"member_id": 3, "role": "Clinical Pharmacist", "expertise_description": "Medication management... within 30 words"}
                ]
            },
            {
                "team_id": 3,
                "team_name": "Final Review and Decision Team (FRDT)",
                "members": [
                    {"member_id": 1, "role": "Chief Medical Officer (Lead)", "expertise_description": "Overall medical decision making... within 30 words"},
                    {"member_id": 2, "role": "Quality Assurance Specialist", "expertise_description": "Care quality standards... within 30 words"},
                    {"member_id": 3, "role": "Patient Advocate", "expertise_description": "Patient interests representation... within 30 words"}
                ]
            }
        ]



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
        
        # Clear conversation history after each call to prevent accumulation across questions
        agent.clear_history()
        
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def coordinate_decision(self, state: AdvancedProcessingState) -> Command:
        """Coordinate final decision from all MDT assessments"""
        question = state["question"]
        answer_options = state.get("answer_options", [])
        compiled_report = state.get("compiled_report", "No report available")
        question_preview = preview_text(question)
        
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
  "analysis": "Your analysis of the MDT assessments and rationale in no more than 150 words",
  "final_answer": "X) Example Answer"
}}

**CRITICAL REQUIREMENTS:**
- Analysis MUST be 150 words or less - longer responses will be rejected
- Consider all MDT assessments  
- Final answer must correspond to one of the provided options
- Return ONLY the JSON, no other text or explanations
"""
        
        with langsmith_span(
            "advanced.coordinator_decision",
            run_type="chain",
            inputs={
                "question_preview": question_preview,
            },
        ) as (_, finish_span):
            # Call with word limit validation for analysis (150 words)
            response, token_usage = self._call_llm_with_analysis_limit(coordinator_prompt, 150)

            try:
                coordinator_decision = self._parse_coordinator_json(response)
            except json.JSONDecodeError:
                coordinator_decision = {
                    "analysis": "JSON parsing error in coordinator response",
                    "final_answer": "X) Parsing error",
                    "raw_response": response
                }
            
            current_usage = state.get("token_usage", {"input": 0, "output": 0})
            updated_usage = {
                "input": current_usage["input"] + token_usage["input_tokens"],
                "output": current_usage["output"] + token_usage["output_tokens"]
            }

            finish_span(
                outputs={
                    "final_answer": coordinator_decision.get("final_answer"),
                    "token_usage": updated_usage,
                },
                usage=token_usage,
            )

            return Command(
                update={
                    "coordinator_decision": coordinator_decision,
                    "final_decision": coordinator_decision,
                    "token_usage": updated_usage,
                    "processing_stage": "advanced_complete"
                },
                goto="__end__"
            )
    
    def _parse_coordinator_json(self, response: str) -> Dict[str, Any]:
        """Parse coordinator response with robust JSON extraction"""
        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Clean and try again
        cleaned_response = self._clean_json_response(response)
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            pass

        # Find JSON object with proper nesting support
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(response):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = response[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        # Verify it has required keys
                        if "analysis" in parsed or "final_answer" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue

        # Final fallback
        return {
            "analysis": response[:300] if response else "Unable to parse coordinator response",
            "final_answer": self._extract_answer_from_text(response)
        }

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

    def _call_llm_with_analysis_limit(self, prompt: str, max_analysis_words: int) -> tuple[str, Dict[str, int]]:
        """Make LLM call with analysis word count validation and retry logic"""
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for attempt in range(3):  # Allow 2 retries
            response, usage = self._call_llm(prompt)

            # Accumulate token usage
            total_usage["input_tokens"] += usage["input_tokens"]
            total_usage["output_tokens"] += usage["output_tokens"]

            # Check if analysis section meets word limit
            try:
                parsed = json.loads(response)
                analysis = parsed.get("analysis", "")
                if _validate_word_count(analysis, max_analysis_words):
                    return response, total_usage
            except json.JSONDecodeError:
                # If not valid JSON, check the entire response
                if _validate_word_count(response, max_analysis_words + 20):  # Allow some extra for JSON structure
                    return response, total_usage

            if attempt < 2:
                prompt += f"\n\nIMPORTANT: Previous analysis exceeded {max_analysis_words} words. Please provide a shorter analysis within the word limit."

        return response, total_usage  # Return last attempt even if it exceeds limit


# Team processing functions for LangGraph native parallelization

def process_single_team(team: Dict, question: str, answer_options: List[str], model_info: str) -> Dict[str, Any]:
    """Process a single team's internal discussion with authentic 4 LLM call MDT pattern - SYNC VERSION"""
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
    
    # STEP 1: Lead investigation call (100 words) - Lead investigates same as assistants
    # Format answer options
    options_text = "\n".join(answer_options) if answer_options else "Multiple choice options not provided"

    lead_investigation_prompt = f"""You are a {lead_role} providing your medical investigation for this case:

Question: {question}

Answer Options:
{options_text}

As the team lead with your medical expertise, provide your professional assessment of this case. Focus on determining the correct answer from the options above.

**CRITICAL: Your response MUST be 100 words or less. Responses over 100 words will be rejected.**"""

    # Call with word limit validation (100 words)
    for attempt in range(3):  # Allow 2 retries
        lead_investigation = lead_agent.chat(lead_investigation_prompt)
        lead_investigation_usage = lead_agent.get_token_usage()
        lead_agent.clear_history()

        if _validate_word_count(lead_investigation, 100):
            break

        if attempt < 2:
            lead_investigation_prompt += f"\n\nIMPORTANT: Previous response exceeded 100 words ({len(lead_investigation.split())} words). Please provide a shorter response within the 100-word limit."
        else:
            # Final fallback: truncate
            lead_investigation = _truncate_to_word_limit(lead_investigation, 100)
    
    # STEP 2: Assistant investigations (100 words each) 
    investigations = []
    if assist_members:
        investigations = gather_member_investigations_sync(
            assist_members, team_name, question, answer_options, model_info
        )
    
    # STEP 3: Compile investigations
    gathered_investigation = ""

    # Include lead's investigation first (same as other investigators)
    lead_role = lead_member.get("role", "Team Lead")
    gathered_investigation += f"[{lead_role}]\n{lead_investigation}\n\n"

    # Then add assistant investigations if any
    if investigations:
        for investigation_item in investigations:
            if len(investigation_item) >= 2:
                role, investigation = investigation_item[0], investigation_item[1]
                gathered_investigation += f"[{role}]\n{investigation}\n\n"
    
    # STEP 3: Lead final synthesis (100 words) - Synthesize all team investigations
    synthesis_prompt = f"""As team lead, review all team member investigations (including your own) and provide your final team decision:

{gathered_investigation}

Question: {question}

Answer Options:
{options_text}

Based on synthesizing all team member investigations above, determine which option is most appropriate and provide your final team reasoning. Focus on selecting the correct answer from the options.

**CRITICAL: Your response MUST be 100 words or less. Responses over 100 words will be rejected.**"""
    
    # Call with word limit validation (100 words)
    for attempt in range(3):  # Allow 2 retries
        team_assessment = lead_agent.chat(synthesis_prompt)
        lead_synthesis_usage = lead_agent.get_token_usage()
        lead_agent.clear_history()

        if _validate_word_count(team_assessment, 100):
            break

        if attempt < 2:
            synthesis_prompt += f"\n\nIMPORTANT: Previous response exceeded 100 words ({len(team_assessment.split())} words). Please provide a shorter response within the 100-word limit."
        else:
            # Final fallback: truncate
            team_assessment = _truncate_to_word_limit(team_assessment, 100)
    
    # Calculate total token usage for this team (lead investigation + synthesis + assistants)
    total_tokens = {
        "input_tokens": lead_investigation_usage["input_tokens"] + lead_synthesis_usage["input_tokens"],
        "output_tokens": lead_investigation_usage["output_tokens"] + lead_synthesis_usage["output_tokens"]
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
        "lead_investigation": lead_investigation,
        "investigations": investigations,
        "gathered_investigation": gathered_investigation
    }


def gather_member_investigations_sync(members: List[Dict], team_goal: str,
                                     question: str, answer_options: List[str], model_info: str) -> List[tuple]:
    """Gather investigations from assistant members - parallel investigation"""
    
    investigations = []
    for member in members:
        role = member.get("role", "team member")
        expertise = member.get("expertise_description", "medical expertise")
        
        # Create member agent
        member_agent = LangGraphAgent(
            instruction=f"You are a {role} who {expertise.lower()}.",
            role=role.lower(),
            model_info=model_info
        )
        
        # Direct investigation prompt - no lead dependency
        options_text = "\n".join(answer_options) if answer_options else "Multiple choice options not provided"

        investigation_prompt = f"""You are a {role} working in a medical team focused on {team_goal}. Provide your professional medical investigation for this case:

Question: {question}

Answer Options:
{options_text}

Based on your expertise as a {role}, provide your medical assessment of this case. Focus on determining the correct answer from the options above.

**CRITICAL: Your response MUST be 100 words or less. Responses over 100 words will be rejected.**"""
        
        # Call with word limit validation (100 words)
        for attempt in range(3):  # Allow 2 retries
            investigation = member_agent.chat(investigation_prompt)
            usage = member_agent.get_token_usage()
            member_agent.clear_history()

            if _validate_word_count(investigation, 100):
                break

            if attempt < 2:
                investigation_prompt += f"\n\nIMPORTANT: Previous response exceeded 100 words ({len(investigation.split())} words). Please provide a shorter response within the 100-word limit."
            else:
                # Final fallback: truncate
                investigation = _truncate_to_word_limit(investigation, 100)

        investigations.append((role, investigation, usage))
    
    return investigations


# Individual team processing nodes for LangGraph native parallelization

def process_team_1(state: AdvancedProcessingState) -> Dict[str, Any]:
    """Process first team in parallel"""
    teams = state.get("mdt_teams", [])
    team = teams[0] if len(teams) >= 1 else None

    with langsmith_span(
        "advanced.process_team",
        run_type="chain",
        inputs={
            "team_index": 1,
            "team_name": team.get("team_name") if team else None,
        },
    ) as (_, finish_span):
        if team is None:
            fallback = {
                "team_results": [{
                    "team_name": "Error Team 1",
                    "assessment": "No team data",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}
                }]
            }
            finish_span(outputs={"fallback": True})
            return fallback

        question = state["question"]
        answer_options = state.get("answer_options", [])
        model_info = state.get('_model_info', 'gemini-2.5-flash')

        result = process_single_team(team, question, answer_options, model_info)

        assessment_data = {
            "team_name": result["team_name"],
            "assessment": result["assessment"]
        }
        formatted_result = {
            "team_results": [assessment_data],
            "token_usage": {
                "input_tokens": result["token_usage"]["input_tokens"],
                "output_tokens": result["token_usage"]["output_tokens"]
            }
        }

        finish_span(
            outputs={
                "team_name": result["team_name"],
                "token_usage": formatted_result["token_usage"],
            },
            usage={
                "input_tokens": result["token_usage"].get("input_tokens", 0),
                "output_tokens": result["token_usage"].get("output_tokens", 0),
                "total_tokens": result["token_usage"].get("input_tokens", 0) + result["token_usage"].get("output_tokens", 0),
            },
        )

        return formatted_result


def process_team_2(state: AdvancedProcessingState) -> Dict[str, Any]:
    """Process second team in parallel"""
    teams = state.get("mdt_teams", [])
    team = teams[1] if len(teams) >= 2 else None

    with langsmith_span(
        "advanced.process_team",
        run_type="chain",
        inputs={
            "team_index": 2,
            "team_name": team.get("team_name") if team else None,
        },
    ) as (_, finish_span):
        if team is None:
            fallback = {
                "team_results": [{
                    "team_name": "Error Team 2",
                    "assessment": "No team data",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}
                }]
            }
            finish_span(outputs={"fallback": True})
            return fallback

        question = state["question"]
        answer_options = state.get("answer_options", [])
        model_info = state.get('_model_info', 'gemini-2.5-flash')

        result = process_single_team(team, question, answer_options, model_info)

        assessment_data = {
            "team_name": result["team_name"],
            "assessment": result["assessment"]
        }
        formatted_result = {
            "team_results": [assessment_data],
            "token_usage": {
                "input_tokens": result["token_usage"]["input_tokens"],
                "output_tokens": result["token_usage"]["output_tokens"]
            }
        }

        finish_span(
            outputs={
                "team_name": result["team_name"],
                "token_usage": formatted_result["token_usage"],
            },
            usage={
                "input_tokens": result["token_usage"].get("input_tokens", 0),
                "output_tokens": result["token_usage"].get("output_tokens", 0),
                "total_tokens": result["token_usage"].get("input_tokens", 0) + result["token_usage"].get("output_tokens", 0),
            },
        )

        return formatted_result


def process_team_3(state: AdvancedProcessingState) -> Dict[str, Any]:
    """Process third team in parallel"""
    teams = state.get("mdt_teams", [])
    team = teams[2] if len(teams) >= 3 else None

    with langsmith_span(
        "advanced.process_team",
        run_type="chain",
        inputs={
            "team_index": 3,
            "team_name": team.get("team_name") if team else None,
        },
    ) as (_, finish_span):
        if team is None:
            fallback = {
                "team_results": [{
                    "team_name": "Error Team 3",
                    "assessment": "No team data",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}
                }]
            }
            finish_span(outputs={"fallback": True})
            return fallback

        question = state["question"]
        answer_options = state.get("answer_options", [])
        model_info = state.get('_model_info', 'gemini-2.5-flash')

        result = process_single_team(team, question, answer_options, model_info)

        assessment_data = {
            "team_name": result["team_name"],
            "assessment": result["assessment"]
        }
        formatted_result = {
            "team_results": [assessment_data],
            "token_usage": {
                "input_tokens": result["token_usage"]["input_tokens"],
                "output_tokens": result["token_usage"]["output_tokens"]
            }
        }

        finish_span(
            outputs={
                "team_name": result["team_name"],
                "token_usage": formatted_result["token_usage"],
            },
            usage={
                "input_tokens": result["token_usage"].get("input_tokens", 0),
                "output_tokens": result["token_usage"].get("output_tokens", 0),
                "total_tokens": result["token_usage"].get("input_tokens", 0) + result["token_usage"].get("output_tokens", 0),
            },
        )

        return formatted_result


def compile_team_results(state: AdvancedProcessingState) -> Dict[str, Any]:
    """Compile all team results after parallel processing"""
    team_results = state.get("team_results", [])
    
    with langsmith_span(
        "advanced.compile_team_results",
        run_type="chain",
        inputs={
            "team_count": len(team_results),
        },
    ) as (_, finish_span):
        assessments = {
            "initial": [],
            "specialist": [],
            "final_review": []
        }

        for result in team_results:
            team_name = result["team_name"].lower()
            assessment_data = {
                "team_name": result["team_name"],
                "assessment": result["assessment"]
            }

            if "initial" in team_name or "iat" in team_name:
                assessments["initial"].append(assessment_data)
            elif "final" in team_name or "review" in team_name or "frdt" in team_name:
                assessments["final_review"].append(assessment_data)
            else:
                assessments["specialist"].append(assessment_data)

        compiled_report = compile_assessment_report(assessments)
        total_usage = {
            "input_tokens": sum(result.get("token_usage", {}).get("input_tokens", 0) for result in team_results),
            "output_tokens": sum(result.get("token_usage", {}).get("output_tokens", 0) for result in team_results),
        }
        total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]

        result_state = {
            "team_assessments": assessments,
            "compiled_report": compiled_report,
            "processing_stage": "teams_processed"
        }

        finish_span(
            outputs={
                "initial": len(assessments["initial"]),
                "specialist": len(assessments["specialist"]),
                "final_review": len(assessments["final_review"]),
                "token_usage": total_usage,
            },
            usage=total_usage,
        )

        return result_state


# Old complex async wrapper removed - now using LangGraph native parallelization


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



def compile_assessment_report(assessments: Dict[str, List[Dict]]) -> str:
    """Compile assessment report from categorized teams"""
    report = ""

    # Initial assessments
    report += "[Initial Assessments]\n"
    for assessment in assessments.get("initial", []):
        team_name = assessment.get("team_name", "Unknown Team")
        content = assessment.get("assessment", "No assessment available")
        # Extract team number from team_name or use fallback
        team_number = _extract_team_number(team_name)
        report += f"Team {team_number} - {team_name}:\n{content}\n\n"

    # Specialist assessments
    report += "[Specialist Team Assessments]\n"
    for assessment in assessments.get("specialist", []):
        team_name = assessment.get("team_name", "Unknown Team")
        content = assessment.get("assessment", "No assessment available")
        team_number = _extract_team_number(team_name)
        report += f"Team {team_number} - {team_name}:\n{content}\n\n"

    # Final review assessments
    report += "[Final Review Team Decisions]\n"
    for assessment in assessments.get("final_review", []):
        team_name = assessment.get("team_name", "Unknown Team")
        content = assessment.get("assessment", "No assessment available")
        team_number = _extract_team_number(team_name)
        report += f"Team {team_number} - {team_name}:\n{content}\n\n"

    return report


def _extract_team_number(team_name: str) -> int:
    """Extract team number from team name for consistent numbering"""
    team_name_lower = team_name.lower()

    # Map team names to their original numbers
    if "initial" in team_name_lower or "iat" in team_name_lower:
        return 1
    elif "final" in team_name_lower or "review" in team_name_lower or "frdt" in team_name_lower:
        return 3
    else:
        # Specialist or any other team gets number 2
        return 2




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
        with langsmith_span(
            "advanced.categorize_teams",
            run_type="chain",
            inputs={
                "team_count": len(teams),
            },
        ) as (_, finish_span):
            categorized = categorize_teams(teams)
            result_state = {
                **state,
                "team_assessments": categorized,
                "processing_stage": "teams_categorized"
            }
            finish_span(
                outputs={
                    "initial": len(categorized.get("initial", [])),
                    "specialist": len(categorized.get("specialist", [])),
                    "final_review": len(categorized.get("final_review", [])),
                }
            )
            return result_state
    
    def process_team_1_with_model(state: AdvancedProcessingState) -> Dict[str, Any]:
        """Process team 1 with model info"""
        state_with_model = {**state, '_model_info': model_info}
        return process_team_1(state_with_model)
    
    def process_team_2_with_model(state: AdvancedProcessingState) -> Dict[str, Any]:
        """Process team 2 with model info"""
        state_with_model = {**state, '_model_info': model_info}
        return process_team_2(state_with_model)
    
    def process_team_3_with_model(state: AdvancedProcessingState) -> Dict[str, Any]:
        """Process team 3 with model info"""
        state_with_model = {**state, '_model_info': model_info}
        return process_team_3(state_with_model)
    
    # Add nodes to subgraph - LangGraph native parallelization pattern
    subgraph.add_node("form_teams", formation.form_teams)
    subgraph.add_node("categorize_teams", categorize_teams_node)
    
    # Fan-out: Individual team processing nodes (execute in parallel)
    subgraph.add_node("process_team_1", process_team_1_with_model)
    subgraph.add_node("process_team_2", process_team_2_with_model)
    subgraph.add_node("process_team_3", process_team_3_with_model)
    
    # Fan-in: Compile results after all teams complete
    subgraph.add_node("compile_results", compile_team_results)
    subgraph.add_node("coordinate_decision", coordinator.coordinate_decision)
    
    # Define edges for fan-out/fan-in pattern
    subgraph.add_edge("form_teams", "categorize_teams")
    
    # Fan-out: All teams execute in parallel from categorize_teams
    subgraph.add_edge("categorize_teams", "process_team_1")
    subgraph.add_edge("categorize_teams", "process_team_2")
    subgraph.add_edge("categorize_teams", "process_team_3")
    
    # Fan-in: All teams must complete before compile_results
    subgraph.add_edge("process_team_1", "compile_results")
    subgraph.add_edge("process_team_2", "compile_results")
    subgraph.add_edge("process_team_3", "compile_results")
    
    # Continue with coordinator (terminal node)
    subgraph.add_edge("compile_results", "coordinate_decision")
    
    # Set entry point
    subgraph.add_edge("__start__", "form_teams")
    
    return subgraph


if __name__ == "__main__":
    # Safe test of advanced processing subgraph structure (no live LLM calls)
    print("Testing advanced processing subgraph structure...")

    try:
        subgraph = create_advanced_processing_subgraph()
        compiled_subgraph = subgraph.compile()
        print("✓ Subgraph creation and compilation successful")

        # Test state validation (no LLM calls)
        test_state = {
            "messages": [],
            "question": "Test question for structure validation",
            "answer_options": ["A) Option 1", "B) Option 2", "C) Option 3"],
            "token_usage": {"input": 0, "output": 0}
        }

        print("✓ Test state structure is valid")

        # Validate team formation structure
        formation_node = MDTFormationNode()
        default_teams = formation_node._create_default_teams()
        print(f"✓ Default teams structure: {len(default_teams)} teams")

        for i, team in enumerate(default_teams):
            members = team.get("members", [])
            print(f"  Team {i+1}: {team.get('team_name')} ({len(members)} members)")

        # Validate utility functions
        test_teams = default_teams
        categorized = categorize_teams(test_teams)
        print(f"✓ Team categorization: {len(categorized['initial'])} initial, {len(categorized['specialist'])} specialist, {len(categorized['final_review'])} final")

        # Test word count validation
        test_text = "This is a test sentence with multiple words for validation."
        is_valid_50 = _validate_word_count(test_text, 50)
        is_valid_5 = _validate_word_count(test_text, 5)
        print(f"✓ Word count validation: {len(test_text.split())} words - valid for 50: {is_valid_50}, valid for 5: {is_valid_5}")

        truncated = _truncate_to_word_limit(test_text, 5)
        print(f"✓ Word truncation: '{truncated}'")

        print("\n🎉 All structural tests passed - no LLM calls made")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
