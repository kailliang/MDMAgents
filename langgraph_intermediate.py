#!/usr/bin/env python3
"""
LangGraph implementation of the Intermediate Processing Graph for MDMAgents.

Pipeline requirements implemented:
1. Recruit three domain experts that match the problem needs.
2. Gather initial structured responses from each expert.
3. If consensus exists, complete immediately; otherwise run up to three structured debate rounds.
4. Resolve with a moderator who either acknowledges consensus or delivers a ruling.

Outputs:
- Per-expert responses are appended to `expert_responses` as list items of the
  shape: {"expert_id", "role", "answer", "reasoning", "round"}.
- Final output is stored in `final_decision` as {"answer", "reasoning"}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Annotated, TypedDict

from langgraph.graph import StateGraph
from langgraph.types import Command

from langgraph_mdm import LangGraphAgent, _merge_expert_responses
from langsmith_integration import preview_text, span as langsmith_span


def _accumulate_token_usage(
    left: Optional[Dict[str, int]], right: Optional[Dict[str, int]]
) -> Dict[str, int]:
    """Reducer used by LangGraph to accumulate token usage updates."""

    left = left or {"input": 0, "output": 0}
    right = right or {"input": 0, "output": 0}
    return {
        "input": left.get("input", 0) + right.get("input", 0),
        "output": left.get("output", 0) + right.get("output", 0),
    }


class IntermediateProcessingState(TypedDict, total=False):
    """State container used by the intermediate processing LangGraph."""

    messages: List[Any]
    question: str
    answer_options: List[str]
    experts: List[Dict[str, Any]]
    expert_responses: Annotated[List[Dict[str, Any]], _merge_expert_responses]
    token_usage: Annotated[Dict[str, int], _accumulate_token_usage]
    processing_stage: str
    round_number: int
    final_decision: Optional[Dict[str, str]]

@dataclass
class UsageDelta:
    """Helper dataclass to simplify token accounting."""

    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"input": self.input_tokens, "output": self.output_tokens}


def _clean_json_text(response: str) -> str:
    """Remove Markdown fences and surrounding whitespace from model output."""

    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# _merge_usage function removed - now using Annotated reducers for token tracking


def _parse_json_response(response: str) -> Dict[str, Any]:
    """Parse a JSON response with a permissive fallback."""

    cleaned = _clean_json_text(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Best-effort fallback into a minimal structure.
        answer_match = re.search(r"[\s\"\{\[]([A-E])\b", response, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else "A"
        reasoning = response.strip()
        return {"answer": answer, "reasoning": reasoning[:200]}


def _normalize_answer(letter: str) -> str:
    """Ensure answer letters are normalized to A-E."""

    if not letter:
        return ""
    match = re.match(r"^[A-E]", letter.strip().upper())
    return match.group(0) if match else ""


def _extract_expert_response(response: str) -> Dict[str, str]:
    """Return the structured expert response from raw model output."""

    parsed = _parse_json_response(response)
    answer = _normalize_answer(str(parsed.get("answer", "")))
    reasoning = str(parsed.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = response.strip()
    return {"answer": answer, "reasoning": reasoning}


def _check_consensus_from_list(responses: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Determine whether all experts currently agree on the same answer."""
    if not responses or len(responses) < 3:
        return False, None
    answers = {str(item.get("answer", "")).strip().upper() for item in responses if item.get("answer")}
    answers = {ans for ans in answers if ans}
    if len(answers) == 1:
        return True, next(iter(answers))
    return False, None


class ExpertRecruitmentNode:
    """Identify the three expertise areas required for the problem."""

    def __init__(self, model_info: str = "gemini-2.5-flash") -> None:
        self.model_info = model_info
        self._agent: Optional[LangGraphAgent] = None

    def _get_agent(self) -> LangGraphAgent:
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction=(
                    "You coordinate medical expert recruitment. Analyse the problem and"
                    " identify three distinct expertise areas that should collaborate."
                    " Return only valid JSON."
                ),
                role="intermediate_recruiter",
                model_info=self.model_info,
            )
        return self._agent

    def _call_llm(self, prompt: str) -> Tuple[str, UsageDelta]:
        agent = self._get_agent()
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        agent.clear_history()
        return response, UsageDelta(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def recruit_experts(self, state: IntermediateProcessingState) -> Command:
        question = state["question"]
        options = state.get("answer_options", [])
        options_text = "\n".join(options) if options else "No answer options provided"

        prompt = f"""
You are organising a three-expert medical review board.

Question:
{question}

Answer Choices:
{options_text}

1. Identify exactly three complementary expertise areas required.
2. Provide a one sentence description for each expert.
3. Responses must be returned as JSON with the schema:
{{
  "experts": [
    {{"id": 1, "expertise": "...", "description": "..."}},
    {{"id": 2, "expertise": "...", "description": "..."}},
    {{"id": 3, "expertise": "...", "description": "..."}}
  ]
}}
Return only the JSON object.
"""

        with langsmith_span(
            "intermediate.recruit_experts",
            run_type="chain",
            inputs={
                "question_preview": preview_text(question),
                "options_count": len(options),
            },
        ) as (_, finish_span):
            response, usage = self._call_llm(prompt)
            parsed = _parse_json_response(response)
            experts_raw = parsed.get("experts", []) if isinstance(parsed, dict) else []

            # Normalize to unified shape expected by other modules: use `role`
            experts: List[Dict[str, Any]] = []
            for idx, e in enumerate(experts_raw):
                eid = int(e.get("id", idx + 1))
                role = e.get("role") or e.get("expertise") or "Medical Specialist"
                experts.append(
                    {
                        "id": eid,
                        "role": str(role),
                        "description": e.get("description") or "General medical specialist",
                    }
                )

            if len(experts) != 3:
                experts = [
                    {
                        "id": idx + 1,
                        "role": fallback,
                        "description": "General medical specialist",
                    }
                    for idx, fallback in enumerate(
                        [
                            "Internal Medicine",
                            "Emergency Medicine",
                            "Family Medicine",
                        ]
                    )
                ]

            finish_span(
                outputs={
                    "experts": [exp.get("role") for exp in experts],
                },
                usage=usage.as_dict(),
            )

        return Command(
            update={
                "experts": experts,
                "token_usage": usage.as_dict(),
                "processing_stage": "experts_recruited",
            },
            goto="initial_responses",
        )


class ExpertResponseNode:
    """Collect structured responses from experts for the current round."""

    def __init__(self, model_info: str = "gemini-2.5-flash") -> None:
        self.model_info = model_info
        self._agent_cache: Dict[int, LangGraphAgent] = {}

    def _get_agent(self, expert: Dict[str, Any]) -> LangGraphAgent:
        expert_id = expert.get("id")
        if expert_id not in self._agent_cache:
            # Prefer role (unified shape); fallback to expertise
            role = expert.get("role") or expert.get("expertise") or "medical specialist"
            description = expert.get("description", "")
            self._agent_cache[expert_id] = LangGraphAgent(
                instruction=(
                    f"You are a {role} participating in a collaborative medical board. "
                    "Provide concise, structured answers in JSON and limit reasoning to 100 words."
                ),
                role=f"expert_{expert_id}",
                model_info=self.model_info,
            )
        return self._agent_cache[expert_id]

    def _call_expert(self, expert: Dict[str, Any], prompt: str) -> Tuple[str, UsageDelta]:
        agent = self._get_agent(expert)
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        agent.clear_history()
        return response, UsageDelta(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def provide_initial_responses(self, state: IntermediateProcessingState) -> Command:
        experts = state.get("experts", [])
        question = state["question"]
        options = state.get("answer_options", [])
        options_text = "\n".join(options) if options else "No answer options provided"

        responses_list: List[Dict[str, Any]] = []
        cumulative_usage = UsageDelta()

        with langsmith_span(
            "intermediate.initial_responses",
            run_type="chain",
            inputs={
                "experts": [expert.get("role") or expert.get("expertise") for expert in experts],
                "question_preview": preview_text(question),
            },
        ) as (_, finish_span):
            for expert in experts:
                expert_id = int(expert.get("id"))
                prompt = f"""
You are the {expert.get('role') or expert.get('expertise', 'medical expert')} for a medical board.

Question:
{question}

Answer Choices:
{options_text}

Provide your initial answer and reasoning. Respect the following rules:
- Limit reasoning to 100 words.
- Return JSON exactly as {{"answer": "A/B/C/D/E", "reasoning": "..."}}.
"""
                response, usage = self._call_expert(expert, prompt)
                cumulative_usage.input_tokens += usage.input_tokens
                cumulative_usage.output_tokens += usage.output_tokens
                parsed = _extract_expert_response(response)
                responses_list.append(
                    {
                        "expert_id": expert_id,
                        "role": expert.get("role") or expert.get("expertise", "expert"),
                        "answer": parsed.get("answer", ""),
                        "reasoning": parsed.get("reasoning", ""),
                        "round": 1,
                    }
                )

            consensus, answer = _check_consensus_from_list(responses_list)

            finish_span(
                outputs={
                    "consensus": consensus,
                    "answer": answer,
                },
                usage=cumulative_usage.as_dict(),
            )

        updates: Dict[str, Any] = {
            # Use list-based aggregator to avoid concurrent update issues
            "expert_responses": responses_list,
            "token_usage": cumulative_usage.as_dict(),
            "processing_stage": "initial_responses_collected",
        }

        # Check for initial consensus - if experts agree, no need for debate
        if consensus and answer:
            updates["final_decision"] = {"answer": answer, "reasoning": "initial_consensus"}
            return Command(update=updates, goto="intermediate_complete")

        # No consensus - proceed to structured debate rounds
        updates["round_number"] = 1
        return Command(update=updates, goto="debate_round")


class DebateRoundNode:
    """Run debate rounds with full information sharing among experts."""

    MAX_ROUNDS = 3

    def __init__(self, model_info: str = "gemini-2.5-flash") -> None:
        self.model_info = model_info
        self._agent_cache: Dict[int, LangGraphAgent] = {}

    def _get_agent(self, expert: Dict[str, Any]) -> LangGraphAgent:
        expert_id = expert.get("id")
        if expert_id not in self._agent_cache:
            role = expert.get("role") or expert.get("expertise", "medical specialist")
            self._agent_cache[expert_id] = LangGraphAgent(
                instruction=(
                    f"You are a {role} taking part in a structured debate."
                    " Read the other experts' arguments and, if needed, update"
                    " your answer. Always respond with JSON and limit reasoning to"
                    " 100 words."
                ),
                role=f"debater_{expert_id}",
                model_info=self.model_info,
            )
        return self._agent_cache[expert_id]

    def _call_expert(self, expert: Dict[str, Any], prompt: str) -> Tuple[str, UsageDelta]:
        agent = self._get_agent(expert)
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        agent.clear_history()
        return response, UsageDelta(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def conduct_debate_round(self, state: IntermediateProcessingState) -> Command:
        round_number = state.get("round_number", 1)
        experts = state.get("experts", [])
        previous_responses_list: List[Dict[str, Any]] = state.get("expert_responses", []) or []
        options = state.get("answer_options", [])
        question = state["question"]
        options_text = "\n".join(options) if options else "No answer options provided"

        if round_number > self.MAX_ROUNDS:
            return Command(goto="moderator")

        # Build latest view per expert from existing list, if any
        latest_by_id: Dict[int, Dict[str, Any]] = {}
        for item in previous_responses_list:
            try:
                eid = int(item.get("expert_id"))
            except (ValueError, TypeError):
                continue
            latest_by_id[eid] = item

        responses_this_round: List[Dict[str, Any]] = []
        cumulative_usage = UsageDelta()

        with langsmith_span(
            "intermediate.debate_round",
            run_type="chain",
            inputs={
                "round": round_number,
                "question_preview": preview_text(question),
            },
        ) as (_, finish_span):
            for expert in experts:
                expert_id = int(expert.get("id"))
                # Collect latest opinions from other experts for full information sharing
                others = [
                    v for k, v in latest_by_id.items() if int(k) != expert_id
                ]
                prompt = f"""
You are participating in debate round {round_number} as the {expert.get('role') or expert.get('expertise', 'expert')}.

Question:
{question}

Answer Choices:
{options_text}

Here are the most recent responses from your fellow experts (JSON list):
{json.dumps(others, indent=2)}

Update your answer if needed. Respond ONLY with JSON {{"answer": "A/B/C/D/E", "reasoning": "..."}}.
Limit reasoning to 100 words.
"""
                response, usage = self._call_expert(expert, prompt)
                cumulative_usage.input_tokens += usage.input_tokens
                cumulative_usage.output_tokens += usage.output_tokens
                parsed = _extract_expert_response(response)
                latest = {
                    "expert_id": expert_id,
                    "role": expert.get("role") or expert.get("expertise", "expert"),
                    "answer": parsed.get("answer", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "round": round_number,
                }
                responses_this_round.append(latest)
                latest_by_id[expert_id] = latest

            consensus, answer = _check_consensus_from_list(list(latest_by_id.values()))

            finish_span(
                outputs={
                    "consensus": consensus,
                    "answer": answer,
                    "round": round_number,
                },
                usage=cumulative_usage.as_dict(),
            )

        updates: Dict[str, Any] = {
            # Append this round's responses; parent graph merges via Annotated reducer
            "expert_responses": responses_this_round,
            "token_usage": cumulative_usage.as_dict(),
            "processing_stage": f"debate_round_{round_number}_complete",
        }

        if consensus and answer:
            updates["final_decision"] = {"answer": answer, "reasoning": "consensus"}
            return Command(update=updates, goto="intermediate_complete")

        if round_number >= self.MAX_ROUNDS:
            return Command(update=updates, goto="moderator")

        updates["round_number"] = round_number + 1
        return Command(update=updates, goto="debate_round")


class ModeratorNode:
    """Resolve the debate with consensus acknowledgement or a moderator ruling."""

    def __init__(self, model_info: str = "gemini-2.5-flash") -> None:
        self.model_info = model_info
        self._agent: Optional[LangGraphAgent] = None

    def _get_agent(self) -> LangGraphAgent:
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction=(
                    "You are the moderator of a medical expert panel. If the experts"
                    " do not agree, issue a final ruling with at most 200 words of"
                    " reasoning. Always respond with JSON."
                ),
                role="moderator",
                model_info=self.model_info,
            )
        return self._agent

    def _call_llm(self, prompt: str) -> Tuple[str, UsageDelta]:
        agent = self._get_agent()
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        agent.clear_history()
        return response, UsageDelta(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def make_final_decision(self, state: IntermediateProcessingState) -> Command:
        responses_list: List[Dict[str, Any]] = state.get("expert_responses", []) or []
        # Build latest answer per expert for consensus check
        latest_by_id: Dict[int, Dict[str, Any]] = {}
        for item in responses_list:
            try:
                eid = int(item.get("expert_id"))
            except (ValueError, TypeError):
                continue
            latest_by_id[eid] = item
        consensus, answer = _check_consensus_from_list(list(latest_by_id.values()))

        if consensus and answer:
            updates = {
                "final_decision": {"answer": answer, "reasoning": "consensus"},
                "processing_stage": "consensus_reached",
            }
            return Command(update=updates, goto="intermediate_complete")

        question = state["question"]
        options = state.get("answer_options", [])
        options_text = "\n".join(options) if options else "No answer options provided"

        prompt = f"""
You are the moderator for a medical multiple-choice problem.

Question:
{question}

Answer Choices:
{options_text}

Here are the latest structured answers from the three experts (JSON list):
{json.dumps(list(latest_by_id.values()), indent=2)}

Deliver the final answer and reasoning in JSON as {{"answer": "A/B/C/D/E", "reasoning": "..."}}.
Limit reasoning to 200 words.
"""

        with langsmith_span(
            "intermediate.moderator",
            run_type="chain",
            inputs={
                "question_preview": preview_text(question),
                "has_consensus": consensus,
            },
        ) as (_, finish_span):
            response, usage = self._call_llm(prompt)
            parsed = _extract_expert_response(response)
            if not parsed.get("answer"):
                parsed["answer"] = "A"

            finish_span(
                outputs={
                    "answer": parsed.get("answer"),
                },
                usage=usage.as_dict(),
            )

        updates = {
            "final_decision": parsed,
            "token_usage": usage.as_dict(),
            "processing_stage": "moderator_decision",
        }
        return Command(update=updates, goto="intermediate_complete")


def finalize_intermediate_processing(state: IntermediateProcessingState) -> IntermediateProcessingState:
    """Terminal node used to mark completion of the intermediate pipeline."""

    result = dict(state)
    result.setdefault("processing_stage", "intermediate_complete")
    return result


def create_intermediate_processing_subgraph(model_info: str = "gemini-2.5-flash") -> StateGraph:
    """Construct the LangGraph subgraph that implements the intermediate pipeline."""

    subgraph = StateGraph(IntermediateProcessingState)

    recruiter = ExpertRecruitmentNode(model_info=model_info)
    initial = ExpertResponseNode(model_info=model_info)
    debate = DebateRoundNode(model_info=model_info)
    moderator = ModeratorNode(model_info=model_info)

    subgraph.add_node("expert_recruitment", recruiter.recruit_experts)
    subgraph.add_node("initial_responses", initial.provide_initial_responses)
    subgraph.add_node("debate_round", debate.conduct_debate_round)
    subgraph.add_node("moderator", moderator.make_final_decision)
    subgraph.add_node("intermediate_complete", finalize_intermediate_processing)

    # Use only essential static edges; nodes handle routing via Command.goto
    subgraph.add_edge("__start__", "expert_recruitment")
    subgraph.add_edge("expert_recruitment", "initial_responses")
    # All other routing handled dynamically by Command.goto

    return subgraph
