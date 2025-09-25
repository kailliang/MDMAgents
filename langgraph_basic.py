#!/usr/bin/env python3
"""
LangGraph implementation of Basic Processing Graph for MDMAgents.

Implements the high-performance 3-expert + arbitrator system:
1. Expert Recruitment - 3 independent medical specialists with equal authority
2. Independent Expert Analysis - parallel processing with JSON responses
3. Arbitrator Decision - synthesis of expert opinions into final decision

"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from langgraph.graph import StateGraph
from langgraph.types import Command, Send
from langgraph_mdm import LangGraphAgent, MDMStateDict, LLMNodeMixin, extract_answer_from_text
from langsmith_integration import span as langsmith_span, preview_text

# Import consensus checking function from intermediate pipeline
def _check_consensus_from_list(responses: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Determine whether all experts currently agree on the same answer."""
    if not responses or len(responses) < 3:
        return False, None
    answers = {str(item.get("answer", "")).strip().upper() for item in responses if item.get("answer")}
    answers = {ans for ans in answers if ans}
    if len(answers) == 1:
        return True, next(iter(answers))
    return False, None


class ExpertRecruitmentNode(LLMNodeMixin):
    """Node for recruiting 3 independent medical experts with different specialties"""

    def __init__(self, model_info: str = "gemini-2.5-flash"):
        super().__init__(model_info)

    def _get_agent(self) -> LangGraphAgent:
        """Get or create the recruitment agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction="You are an experienced medical expert who recruits medical specialists to solve the given medical query.",
                role="recruiter",
                model_info=self.model_info
            )
        return self._agent
    
    def recruit_experts(self, state: MDMStateDict) -> Command:
        """Recruit 3 independent medical experts for the question"""
        question = state["question"]
        answer_options = state.get("answer_options", [])
        
        # Format answer options
        options_text = "\n".join(answer_options) if answer_options else "Multiple choice options not provided"
        question_preview = preview_text(question)

        num_experts_to_recruit = 3
        recruitment_prompt = f"""Question: {question}

Answer Options:
{options_text}

You need to recruit {num_experts_to_recruit} medical experts with different specialties but equal authority/weight to analyze this question. 

Please return your recruitment plan in JSON format:

{{
  "experts": [
    {{
      "id": 1,
      "role": "Cardiologist",
      "expertise_description": "Specializes in heart and cardiovascular system disorders",
      "hierarchy": "Independent"
    }},
    {{
      "id": 2,
      "role": "Pulmonologist", 
      "expertise_description": "Specializes in respiratory system diseases",
      "hierarchy": "Independent"
    }},
    {{
      "id": 3,
      "role": "Emergency Medicine Physician",
      "expertise_description": "Specializes in acute care and emergency medical situations", 
      "hierarchy": "Independent"
    }}
  ]
}}

All experts should be marked as "Independent" with equal authority. Return ONLY the JSON, no other text."""
        
        span_parent = state.get("langsmith_parent_run")
        with langsmith_span(
            "basic.recruit_experts",
            run_type="chain",
            inputs={
                "question_preview": question_preview,
                "options_count": len(answer_options or []),
            },
            parent=span_parent,
            require_parent=span_parent is not None,
        ) as (recruit_run, finish_span):
            response, token_usage = self._call_llm_with_retry(
                recruitment_prompt,
                max_attempts=3,
                required_fields=["experts"],
                parent_run=recruit_run,
            )
            
            # Parse JSON response for expert recruitment
            try:
                # Clean JSON response by removing markdown blocks
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                
                experts_data = json.loads(cleaned_response)
                recruited_experts = experts_data.get('experts', [])
                
                if len(recruited_experts) != num_experts_to_recruit:
                    raise ValueError(f"Expected {num_experts_to_recruit} experts, got {len(recruited_experts)}")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback to default experts
                recruited_experts = [
                    {"id": 1, "role": "General Internal Medicine Physician", "expertise_description": "Specializes in comprehensive adult medical care", "hierarchy": "Independent"},
                    {"id": 2, "role": "Emergency Medicine Physician", "expertise_description": "Specializes in acute care and emergency situations", "hierarchy": "Independent"},  
                    {"id": 3, "role": "Family Medicine Physician", "expertise_description": "Specializes in primary care across all age groups", "hierarchy": "Independent"}
                ]
            
            # Update token usage
            current_usage = state["token_usage"]
            updated_usage = {
                "input": current_usage["input"] + token_usage["input_tokens"],
                "output": current_usage["output"] + token_usage["output_tokens"]
            }

            finish_span(
                outputs={
                    "experts": [expert.get("role", "unknown") for expert in recruited_experts],
                    "token_usage": updated_usage,
                },
                usage=token_usage,
            )
            
            return Command(
                update={
                    "experts": recruited_experts,
                    "token_usage": updated_usage,
                    "processing_stage": "expert_recruitment_complete"
                },
                goto="expert_analysis"
            )


class ExpertAnalysisNode(LLMNodeMixin):
    """Node for individual expert analysis with JSON response parsing"""

    def __init__(self, expert_data: Dict[str, Any], model_info: str = "gemini-2.5-flash"):
        super().__init__(model_info)
        self.expert_data = expert_data

    def _get_agent(self) -> LangGraphAgent:
        """Get or create the expert agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction=f"You are a {self.expert_data['role']} who {self.expert_data['expertise_description'].lower()}. Your job is to analyze medical questions independently.",
                role=self.expert_data['role'],
                model_info=self.model_info
            )
        return self._agent
    
    def analyze_question(self, state: MDMStateDict, parent_run: Any = None) -> Command:
        """Analyze the medical question and provide structured JSON response"""
        question = state["question"]
        answer_options = state.get("answer_options", [])
        expert = self.expert_data
        question_preview = preview_text(question)
        
        # Format answer options for display
        options_text = "\n".join(answer_options) if answer_options else "No options provided"
        
        expert_prompt = f"""You are a {expert['role']}. Analyze the following multiple choice question and provide your response in exactly this JSON format:

{{
  "reasoning": "Your step-by-step medical analysis in no more than 100 words",
  "answer": "X) Example Answer"
}}

**Question:** {question}

**Answer Options:**
{options_text}

**Requirements:**
- Answer must correspond to one of the provided options
- Return ONLY the JSON, no other text
"""
        
        with langsmith_span(
            "basic.expert_analysis",
            run_type="chain",
            inputs={
                "expert_id": expert.get('id'),
                "expert_role": expert.get('role'),
                "question_preview": question_preview,
                "options_count": len(answer_options or []),
            },
            parent=parent_run,
        ) as (analysis_run, finish_span):
            response, token_usage = self._call_llm_with_retry(
                expert_prompt,
                max_attempts=3,
                required_fields=["reasoning", "answer"],
                parent_run=analysis_run,
            )
            
            # Parse expert response with improved multi-layer fallback
            expert_response = None
            
            try:
                # First try direct JSON parsing with cleanup
                cleaned = self._clean_json_response(response)
                parsed_json = json.loads(cleaned)
                if "reasoning" in parsed_json and "answer" in parsed_json:
                    expert_response = {
                        "expert_id": expert['id'],
                        "role": expert['role'],
                        "reasoning": parsed_json.get("reasoning", "").strip(),
                        "answer": parsed_json.get("answer", "").strip()
                    }
            except (json.JSONDecodeError, ValueError):
                pass
            
            # If direct parsing failed, try regex extraction
            if not expert_response:
                try:
                    # More flexible regex that handles nested JSON
                    json_match = re.search(r'\{.*?"reasoning"\s*:\s*"(.*?)".*?"answer"\s*:\s*"([^"]*)".*?\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed_json = json.loads(json_str)
                        expert_response = {
                            "expert_id": expert['id'],
                            "role": expert['role'],
                            "reasoning": parsed_json.get("reasoning", "").strip(),
                            "answer": parsed_json.get("answer", "").strip()
                        }
                except Exception:
                    pass
            
            # Final fallback - extract from text
            if not expert_response:
                expert_response = {
                    "expert_id": expert['id'],
                    "role": expert['role'],
                    "reasoning": response[:300] if len(response) > 0 else "Unable to parse expert response",
                    "answer": extract_answer_from_text(response, "Extracted answer")
                }
            
            # Prepare token delta so the caller can aggregate across experts safely
            usage_delta = {
                "input": token_usage["input_tokens"],
                "output": token_usage["output_tokens"]
            }
            current_usage = state.get("token_usage", {"input": 0, "output": 0})
            updated_usage = {
                "input": current_usage.get("input", 0) + usage_delta["input"],
                "output": current_usage.get("output", 0) + usage_delta["output"]
            }

            finish_span(
                outputs={
                    "expert_id": expert.get('id'),
                    "answer": expert_response.get("answer"),
                    "token_usage": updated_usage,
                },
                usage=token_usage,
            )
            
            return Command(
                update={
                    "expert_responses": [expert_response],
                    "expert_token_delta": usage_delta,
                },
                goto="collect_responses"
            )


def parallel_expert_analysis(state: MDMStateDict) -> List[Send]:
    """Create Send commands for parallel expert analysis using LangGraph Send API"""
    experts = state.get("experts", [])
    
    # Handle case where no experts are recruited yet
    if not experts:
        return []
    
    # Create Send command for each expert to process in parallel
    send_commands = []
    for expert in experts:
        send_commands.append(
            Send("expert_analysis_node", {
                **state,
                "current_expert": expert
            })
        )
    
    return send_commands


def expert_analysis_router(state: MDMStateDict) -> str:
    """Route to arbitrator when all expert responses are collected"""
    experts = state.get("experts", [])
    expert_responses = state.get("expert_responses", [])
    
    # Check if we have responses from all experts
    if len(expert_responses) >= len(experts):
        return "arbitrator"
    else:
        return "collect_responses"


class ArbitratorNode(LLMNodeMixin):
    """Node for arbitrator synthesis of expert opinions into final decision"""

    def __init__(self, model_info: str = "gemini-2.5-flash"):
        super().__init__(model_info)

    def _get_agent(self) -> LangGraphAgent:
        """Get or create the arbitrator agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction="You are a medical arbitrator who reviews multiple expert opinions and synthesizes the best final decision.",
                role="Medical Arbitrator",
                model_info=self.model_info
            )
        return self._agent
    
    def make_final_decision(self, state: MDMStateDict) -> Command:
        """Synthesize expert opinions and make final decision"""
        question = state["question"]
        answer_options = state.get("answer_options", [])
        expert_responses = state.get("expert_responses", [])
        question_preview = preview_text(question)

        # Format expert responses for arbitrator
        experts_summary = ""
        for response in expert_responses:
            experts_summary += f"Expert {response['expert_id']} ({response['role']}):\n"
            experts_summary += f"Reasoning: {response['reasoning']}\n"
            experts_summary += f"Answer: {response['answer']}\n\n"

        # Format answer options for display
        options_text = "\n".join(answer_options) if answer_options else "No options provided"

        arbitrator_prompt = f"""You are a medical arbitrator. Review the following expert opinions and provide your final decision in JSON format:

{experts_summary}

Question: {question}

**Answer Options:**
{options_text}

Analyze all expert opinions and provide your final decision in exactly this JSON format:

{{
  "analysis": "Your analysis of the expert opinions and rationale for final decision in no more than 150 words",
  "final_answer": "X) Example Answer"
}}

**Requirements:**
- Consider all expert opinions in your analysis
- Final answer must correspond to one of the provided options
- Return ONLY the JSON, no other text
"""

        span_parent = state.get("langsmith_parent_run")
        with langsmith_span(
            "basic.arbitrator",
            run_type="chain",
            inputs={
                "question_preview": question_preview,
                "experts": [resp.get("role") for resp in expert_responses],
            },
            parent=span_parent,
            require_parent=span_parent is not None,
        ) as (arbitration_run, finish_span):
            response, token_usage = self._call_llm_with_retry(
                arbitrator_prompt,
                max_attempts=3,
                required_fields=["analysis", "final_answer"],
                parent_run=arbitration_run,
            )

            # Parse arbitrator response with improved fallback
            final_decision_dict = None

            try:
                cleaned = self._clean_json_response(response)
                parsed = json.loads(cleaned)

                # Check if analysis contains nested JSON (the bug we found)
                if "analysis" in parsed and isinstance(parsed["analysis"], str):
                    if parsed["analysis"].strip().startswith('{'):
                        try:
                            nested = json.loads(parsed["analysis"])
                            if "analysis" in nested and "final_answer" in nested:
                                parsed = nested
                        except Exception:
                            pass

                if "analysis" in parsed and "final_answer" in parsed:
                    final_decision_dict = {
                        "analysis": parsed["analysis"],
                        "final_answer": parsed["final_answer"]
                    }
            except (json.JSONDecodeError, ValueError):
                pass

            if not final_decision_dict:
                try:
                    json_match = re.search(
                        r'\{.*?"analysis"\s*:\s*"(.*?)".*?"final_answer"\s*:\s*"([^"]*)".*?\}',
                        response,
                        re.DOTALL,
                    )
                    if json_match:
                        final_decision_dict = {
                            "analysis": json_match.group(1),
                            "final_answer": json_match.group(2)
                        }
                except Exception:
                    pass

            if not final_decision_dict:
                final_decision_dict = {
                    "analysis": response[:300] if len(response) > 0 else "Unable to parse arbitrator response",
                    "final_answer": extract_answer_from_text(response, "Final arbitrator decision")
                }

            # Update token usage
            current_usage = state["token_usage"]
            updated_usage = {
                "input": current_usage["input"] + token_usage["input_tokens"],
                "output": current_usage["output"] + token_usage["output_tokens"]
            }

            finish_span(
                outputs={
                    "final_answer": final_decision_dict.get("final_answer"),
                    "token_usage": updated_usage,
                },
                usage=token_usage,
            )

            return Command(
                update={
                    "final_decision": final_decision_dict,
                    "token_usage": updated_usage,
                    "processing_stage": "basic_complete"
                },
                goto="basic_complete"
            )


class ConsensusCheckNode:
    """Node to check if experts have reached consensus, bypassing arbitrator if they agree"""

    def check_consensus(self, state: MDMStateDict) -> Command:
        """Check if all experts agree, skip arbitrator if consensus exists"""
        expert_responses = state.get("expert_responses", [])
        question_preview = preview_text(state.get("question", ""))

        span_parent = state.get("langsmith_parent_run")
        with langsmith_span(
            "basic.consensus_check",
            run_type="chain",
            inputs={
                "question_preview": question_preview,
                "expert_count": len(expert_responses),
            },
            parent=span_parent,
            require_parent=span_parent is not None,
        ) as (consensus_run, finish_span):

            # Check for consensus among expert responses
            consensus, agreed_answer = _check_consensus_from_list(expert_responses)

            current_usage = state.get("token_usage", {"input": 0, "output": 0})

            if consensus and agreed_answer:
                # All experts agree - create final decision directly
                final_decision_dict = {
                    "analysis": f"All experts unanimously agreed on answer {agreed_answer}",
                    "final_answer": agreed_answer
                }

                finish_span(
                    outputs={
                        "consensus_found": True,
                        "agreed_answer": agreed_answer,
                        "token_usage": current_usage,
                    }
                )

                return Command(
                    update={
                        "final_decision": final_decision_dict,
                        "processing_stage": "consensus_complete",
                        "token_usage": current_usage
                    },
                    goto="basic_complete"
                )
            else:
                # No consensus - proceed to arbitrator
                finish_span(
                    outputs={
                        "consensus_found": False,
                        "token_usage": current_usage,
                    }
                )

                return Command(
                    update={
                        "processing_stage": "consensus_failed_proceeding_to_arbitrator"
                    },
                    goto="arbitrator"
                )


def basic_processing_placeholder(state: MDMStateDict) -> Dict[str, Any]:
    """Placeholder for basic processing completion"""
    return {
        **state,
        "processing_stage": "basic_complete",
        "final_decision": state.get("final_decision", {"placeholder": "basic_result"}),
        "expert_responses": state.get("expert_responses", [])
    }


def create_basic_processing_subgraph(model_info: str = "gemini-2.5-flash") -> StateGraph:
    """Create the basic processing subgraph with 3-expert + arbitrator system"""
    
    # Create subgraph for basic processing
    subgraph = StateGraph(MDMStateDict)
    
    # Create node instances
    recruiter = ExpertRecruitmentNode(model_info=model_info)
    arbitrator = ArbitratorNode(model_info=model_info)
    consensus_checker = ConsensusCheckNode()
    
    def process_all_experts(state: MDMStateDict) -> Dict[str, Any]:
        """Process all experts sequentially and collect responses"""
        experts = state.get("experts", [])
        expert_responses: List[Dict[str, Any]] = []
        running_usage = state.get("token_usage", {"input": 0, "output": 0}).copy()
        expert_roles = [expert.get("role") for expert in experts]

        span_parent = state.get("langsmith_parent_run")
        with langsmith_span(
            "basic.sequential_expert_processing",
            run_type="chain",
            inputs={
                "experts": expert_roles,
                "initial_tokens": running_usage,
            },
            parent=span_parent,
            require_parent=span_parent is not None,
        ) as (sequential_run, finish_span):
            for expert in experts:
                expert_node = ExpertAnalysisNode(expert, model_info)
                branch_state = {
                    **state,
                    "expert_responses": [],
                    "expert_token_delta": None,
                    "current_expert": expert,
                    "token_usage": running_usage,
                }
                result = expert_node.analyze_question(branch_state, parent_run=sequential_run)
                update = result.update or {}

                responses = update.get("expert_responses") or []
                expert_responses.extend(responses)

                delta = update.get("expert_token_delta") or {"input": 0, "output": 0}
                running_usage = {
                    "input": running_usage.get("input", 0) + delta.get("input", 0),
                    "output": running_usage.get("output", 0) + delta.get("output", 0),
                }

            finish_span(
                outputs={
                    "responses": len(expert_responses),
                    "token_usage": running_usage,
                }
            )

        return {
            **state,
            "expert_responses": expert_responses,
            "expert_token_delta": None,
            "token_usage": running_usage,
            "processing_stage": "expert_analysis_complete",
            "current_expert": None,
        }

    def parallel_expert_processing(state: MDMStateDict) -> Command:
        """Dispatch expert analysis in parallel using LangGraph Send API"""
        experts = state.get("experts", [])
        if not experts:
            return Command(
                update={
                    "expert_responses": [],
                    "expert_token_delta": None,
                    "processing_stage": "no_experts",
                    "current_expert": None,
                },
                goto="arbitrator",
            )

        try:
            base_state = {
                **state,
                "expert_token_delta": None,
                "processing_stage": "expert_analysis_pending",
            }
            send_commands = parallel_expert_analysis(base_state)
            return Command(
                update={
                    "expert_token_delta": None,
                    "processing_stage": "expert_analysis_pending",
                    "current_expert": None,
                },
                goto=send_commands,
            )
        except Exception as exc:
            print(f"Error in parallel expert analysis: {exc}")
            fallback_state = process_all_experts(state)
            return Command(update=fallback_state, goto="arbitrator")

    def expert_analysis_node(state: MDMStateDict) -> Command:
        """Execute expert analysis for a single expert"""
        expert = state.get("current_expert")
        if not expert:
            return Command(
                update={"expert_token_delta": None, "current_expert": None},
                goto="collect_responses",
            )

        expert_node = ExpertAnalysisNode(expert, model_info)
        return expert_node.analyze_question(state, parent_run=state.get("langsmith_parent_run"))

    def collect_expert_responses(state: MDMStateDict) -> Command:
        """Aggregate expert responses and advance when complete"""
        experts = state.get("experts", [])
        responses = state.get("expert_responses", [])
        total_expected = len(experts)

        span_parent = state.get("langsmith_parent_run")
        with langsmith_span(
            "basic.collect_expert_responses",
            run_type="chain",
            inputs={
                "responses_collected": len(responses),
                "expected_responses": total_expected,
            },
            parent=span_parent,
            require_parent=span_parent is not None,
        ) as (_collect_run, finish_span):
            finish_span(
                outputs={
                    "responses": len(responses),
                    "token_usage": state.get("token_usage", {"input": 0, "output": 0}),
                }
            )

        updated_state = {
            **state,
            "expert_responses": responses,
        }
        next_node = expert_analysis_router(updated_state)

        updates: Dict[str, Any] = {}
        goto: Any = ()

        if next_node == "arbitrator" and len(responses) >= len(experts):
            baseline_usage = state.get("token_usage", {"input": 0, "output": 0})
            delta_usage = state.get("expert_token_delta") or {"input": 0, "output": 0}
            final_usage = {
                "input": baseline_usage.get("input", 0) + delta_usage.get("input", 0),
                "output": baseline_usage.get("output", 0) + delta_usage.get("output", 0),
            }
            updates = {
                "token_usage": final_usage,
                "expert_token_delta": None,
                "processing_stage": "expert_analysis_complete",
            }
            goto = "consensus_check"

        return Command(update=updates, goto=goto)
    
    # Add nodes to subgraph
    subgraph.add_node("expert_recruitment", recruiter.recruit_experts)
    subgraph.add_node("expert_analysis", parallel_expert_processing)
    subgraph.add_node("expert_analysis_node", expert_analysis_node)
    subgraph.add_node("collect_responses", collect_expert_responses)
    subgraph.add_node("consensus_check", consensus_checker.check_consensus)
    subgraph.add_node("arbitrator", arbitrator.make_final_decision)
    subgraph.add_node("basic_complete", basic_processing_placeholder)

    # Define edges
    subgraph.add_edge("expert_recruitment", "expert_analysis")
    subgraph.add_edge("arbitrator", "basic_complete")

    # Set entry point
    subgraph.add_edge("__start__", "expert_recruitment")
    
    return subgraph


