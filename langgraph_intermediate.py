#!/usr/bin/env python3
"""
LangGraph implementation of Intermediate Processing Graph for MDMAgents.

Implements multi-agent collaboration with hierarchical debate system:
1. Hierarchical Expert Recruitment - 3 experts with relationships
2. Multi-round Debate - 3 rounds with participation decisions  
3. Expert Selection - Dynamic selection of discussion partners
4. Communication - Targeted questions and opinions (200 words)
5. Moderator Consensus - Final synthesis of expert opinions

Target: 78%+ accuracy with collaborative processing
Based on existing process_intermediate_query function in utils.py:757+
"""

import json
import re
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph_mdm import LangGraphAgent


class IntermediateProcessingState(TypedDict, total=False):
    """Extended state for intermediate processing with debate mechanics"""
    # Core fields
    messages: List[Any]
    question: str
    answer_options: Optional[List[str]]
    token_usage: Dict[str, int]
    processing_stage: str
    final_decision: Optional[Dict]
    
    # Intermediate processing specific
    experts_hierarchy: List[Dict]
    round_number: int
    interaction_log: Dict
    round_opinions: Dict
    participation_decisions: List[Dict]
    expert_selections: List[Dict]
    final_expert_opinions: Dict
    communication_content: Optional[str]
    current_expert: Optional[Dict]
    target_expert: Optional[Dict]


class HierarchicalExpertRecruitmentNode:
    """Node for recruiting experts with hierarchical relationships"""
    
    def __init__(self, model_info: str = "gemini-2.5-flash"):
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the recruitment agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction="You are an experienced medical expert who recruits a group of experts with diverse identity and hierarchical relationships.",
                role="hierarchical_recruiter",
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
    
    def recruit_experts(self, state: IntermediateProcessingState) -> Command:
        """Recruit 3 experts with hierarchical relationships"""
        question = state["question"]
        
        recruitment_prompt = f"""
Question: {question}

You need to recruit 3 medical experts with different specialties to discuss this question.
You should specify hierarchical relationships between experts (e.g., "Cardiologist > Pulmonologist" means Cardiologist has higher authority).

Please return your recruitment plan in JSON format:
{{
  "experts": [
    {{
      "id": 1,
      "role": "Cardiologist",
      "description": "Specializes in heart and cardiovascular disorders",
      "hierarchy": "Independent"
    }},
    {{
      "id": 2,
      "role": "Pulmonologist",
      "description": "Specializes in respiratory diseases",
      "hierarchy": "Cardiologist > Pulmonologist"
    }},
    {{
      "id": 3,
      "role": "Emergency Medicine Physician",
      "description": "Specializes in acute care",
      "hierarchy": "Independent"
    }}
  ]
}}

Return ONLY the JSON, no other text."""
        
        response, token_usage = self._call_llm(recruitment_prompt)
        
        # Parse JSON response
        try:
            # Clean JSON response
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            experts_data = json.loads(cleaned_response)
            recruited_experts = experts_data.get('experts', [])
            
            if len(recruited_experts) != 3:
                raise ValueError(f"Expected 3 experts, got {len(recruited_experts)}")
                
        except (json.JSONDecodeError, ValueError):
            # Fallback to default experts
            recruited_experts = [
                {"id": 1, "role": "General Internal Medicine Physician", "description": "Comprehensive adult care", "hierarchy": "Independent"},
                {"id": 2, "role": "Emergency Medicine Physician", "description": "Acute care specialist", "hierarchy": "Independent"},
                {"id": 3, "role": "Family Medicine Physician", "description": "Primary care specialist", "hierarchy": "Independent"}
            ]
        
        # Update token usage
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "experts_hierarchy": recruited_experts,
                "token_usage": updated_usage,
                "processing_stage": "expert_recruitment_complete"
            },
            goto="initial_opinions"
        )


class DebateParticipationNode:
    """Node for expert participation decisions"""
    
    def __init__(self, expert: Dict, model_info: str = "gemini-2.5-flash"):
        self.expert = expert
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the expert agent"""
        if self._agent is None:
            role = self.expert.get("role", "medical expert")
            desc = self.expert.get("description", "medical specialist")
            self._agent = LangGraphAgent(
                instruction=f"You are a {role} who {desc}. You collaborate with other experts.",
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
    
    async def _call_llm_async(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """Make async LLM call and return response with token usage"""
        # Wrap the sync call in asyncio.to_thread for true async behavior
        def sync_call():
            agent = self._get_agent()
            response = agent.chat(prompt)
            usage = agent.get_token_usage()
            return response, {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"]
            }
        
        return await asyncio.to_thread(sync_call)
    
    def decide_participation(self, state: IntermediateProcessingState) -> Command:
        """Decide whether to participate in current debate round"""
        round_num = state.get("round_number", 1)
        round_opinions = state.get("round_opinions", {}).get(round_num, {})
        
        # Format current opinions
        opinions_text = "\n".join([f"({k}): {v}" for k, v in round_opinions.items()])
        
        participation_prompt = f"""Given the opinions from other medical experts in your team, please indicate whether you want to participate in discussion.

Current Opinions:
{opinions_text}

Return your response in JSON format:
{{
  "participate": true/false,
  "reason": "brief explanation"
}}"""
        
        response, token_usage = self._call_llm(participation_prompt)
        
        # Parse participation decision
        try:
            decision_json = json.loads(response)
            participate = decision_json.get('participate', False)
            reason = decision_json.get('reason', 'No reason provided')
        except (json.JSONDecodeError, TypeError):
            # Fallback to text parsing
            participate = 'yes' in response.lower() or 'true' in response.lower()
            reason = response[:100] if response else "Parsed from text"
        
        # Update state with decision
        current_decisions = state.get("participation_decisions", [])
        new_decision = {
            "expert_id": self.expert["id"],
            "participate": participate,
            "reason": reason,
            "round": round_num
        }
        
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "participation_decisions": current_decisions + [new_decision],
                "token_usage": updated_usage
            },
            goto="check_participation"
        )


class ExpertSelectionNode:
    """Node for selecting which experts to communicate with"""
    
    def __init__(self, expert: Dict, model_info: str = "gemini-2.5-flash"):
        self.expert = expert
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the expert agent"""
        if self._agent is None:
            role = self.expert.get("role", "medical expert")
            self._agent = LangGraphAgent(
                instruction=f"You are a {role} selecting discussion partners.",
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
    
    def select_experts(self, state: IntermediateProcessingState) -> Command:
        """Select which experts to communicate with"""
        experts = state.get("experts_hierarchy", [])
        
        # Format expert list
        expert_list = "\n".join([f"{e['id']}. {e['role']}" for e in experts])
        
        selection_prompt = f"""Select which expert(s) you want to discuss with.

Available Experts:
{expert_list}

Return your response in JSON format:
{{
  "selected_experts": [2, 3],
  "reason": "Need to discuss specific points"
}}"""
        
        response, token_usage = self._call_llm(selection_prompt)
        
        # Parse selection
        try:
            selection_json = json.loads(response)
            selected = selection_json.get('selected_experts', [])
            reason = selection_json.get('reason', '')
        except (json.JSONDecodeError, TypeError):
            # Fallback parsing
            selected = [int(s) for s in re.findall(r'\d+', response) if 1 <= int(s) <= len(experts)]
            reason = "Parsed from text"
        
        # Update state
        current_selections = state.get("expert_selections", [])
        new_selection = {
            "source_expert_id": self.expert["id"],
            "selected_experts": selected,
            "reason": reason
        }
        
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "expert_selections": current_selections + [new_selection],
                "token_usage": updated_usage
            },
            goto="generate_communications"
        )


class ExpertCommunicationNode:
    """Node for generating expert-to-expert communications"""
    
    def __init__(self, source_expert: Dict, target_expert: Dict, model_info: str = "gemini-2.5-flash"):
        self.source_expert = source_expert
        self.target_expert = target_expert
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the source expert agent"""
        if self._agent is None:
            role = self.source_expert.get("role", "medical expert")
            self._agent = LangGraphAgent(
                instruction=f"You are a {role} communicating with colleagues.",
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
    
    def generate_communication(self, state: IntermediateProcessingState) -> Command:
        """Generate communication from source to target expert"""
        question = state["question"]
        target_role = self.target_expert.get("role", "colleague")
        
        comm_prompt = f"""Please provide your opinion/question for {target_role} regarding this medical case.

Question: {question}

Deliver your opinion to convince the other expert. Limit to 200 words."""
        
        response, token_usage = self._call_llm(comm_prompt)
        
        # Update interaction log
        round_num = state.get("round_number", 1)
        turn_num = state.get("turn_number", 1)
        interaction_log = state.get("interaction_log", {})
        
        # Initialize nested structure if needed
        if f"Round {round_num}" not in interaction_log:
            interaction_log[f"Round {round_num}"] = {}
        if f"Turn {turn_num}" not in interaction_log[f"Round {round_num}"]:
            interaction_log[f"Round {round_num}"][f"Turn {turn_num}"] = {}
        
        # Record interaction
        source_id = self.source_expert["id"]
        target_id = self.target_expert["id"]
        interaction_log[f"Round {round_num}"][f"Turn {turn_num}"][f"Expert {source_id} to {target_id}"] = response
        
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "interaction_log": interaction_log,
                "communication_content": response,
                "token_usage": updated_usage
            },
            goto="continue_debate"
        )


class RoundSynthesisNode:
    """Node for synthesizing opinions after each debate round"""
    
    def __init__(self, model_info: str = "gemini-2.5-flash"):
        self.model_info = model_info
    
    def _get_agent(self, role: str) -> LangGraphAgent:
        """Get agent for specific expert role"""
        return LangGraphAgent(
            instruction=f"You are a {role} reflecting on the discussion.",
            role=role.lower(),
            model_info=self.model_info
        )
    
    def _call_llm(self, prompt: str, role: str = "expert") -> tuple[str, Dict[str, int]]:
        """Make LLM call with specific role"""
        agent = self._get_agent(role)
        response = agent.chat(prompt)
        usage = agent.get_token_usage()
        return response, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"]
        }
    
    def synthesize_round(self, state: IntermediateProcessingState) -> Command:
        """Synthesize expert opinions after debate round"""
        experts = state.get("experts_hierarchy", [])
        question = state["question"]
        round_num = state.get("round_number", 1)
        
        # Collect updated opinions from each expert
        next_round_opinions = {}
        total_tokens = {"input": 0, "output": 0}
        
        for expert in experts:
            role = expert.get("role", "expert")
            
            synthesis_prompt = f"""Reflecting on the discussions in Round {round_num}, what is your current opinion on:
{question}

Limit your answer to 50 words."""
            
            response, token_usage = self._call_llm(synthesis_prompt, role)
            next_round_opinions[role.lower()] = response
            
            total_tokens["input"] += token_usage["input_tokens"]
            total_tokens["output"] += token_usage["output_tokens"]
        
        # Update round opinions
        round_opinions = state.get("round_opinions", {})
        round_opinions[round_num + 1] = next_round_opinions
        
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + total_tokens["input"],
            "output": current_usage["output"] + total_tokens["output"]
        }
        
        return Command(
            update={
                "round_opinions": round_opinions,
                "round_number": round_num + 1,
                "token_usage": updated_usage
            },
            goto="check_next_round"
        )


class ModeratorConsensusNode:
    """Node for final moderator consensus building"""
    
    def __init__(self, model_info: str = "gemini-2.5-flash"):
        self.model_info = model_info
        self._agent = None
    
    def _get_agent(self) -> LangGraphAgent:
        """Get or create the moderator agent"""
        if self._agent is None:
            self._agent = LangGraphAgent(
                instruction="You are a final medical decision maker who reviews all opinions and makes the final decision.",
                role="moderator",
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
    
    def build_consensus(self, state: IntermediateProcessingState) -> Command:
        """Build final consensus from expert opinions"""
        question = state["question"]
        answer_options = state.get("answer_options", [])
        final_opinions = state.get("final_expert_opinions", {})
        
        # Format expert opinions
        opinions_text = "\n".join([f"{k}: {v}" for k, v in final_opinions.items()])
        
        # Format answer options for display
        options_text = "\n".join(answer_options) if answer_options else "No options provided"
        
        consensus_prompt = f"""Given each expert's final answer, please review and make the final decision in JSON format.

Expert Opinions:
{opinions_text}

Question: {question}

**Answer Options:**
{options_text}

Provide your final decision in exactly this JSON format:

{{
  "majority_vote": "X) Example Answer",
  "reasoning": "Brief explanation of the consensus decision"
}}

**Requirements:**
- Final answer must correspond to one of the provided options
- Return ONLY the JSON, no other text
"""
        
        response, token_usage = self._call_llm(consensus_prompt)
        
        # Parse JSON response to extract clean answer
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"majority_vote"\s*:\s*"([^"]*)"[^{}]*\}', response, re.DOTALL)
            if json_match:
                majority_vote = json_match.group(1)
            else:
                # Fallback - try to parse as full JSON
                import json
                clean_response = response.strip()
                if clean_response.startswith('```json\n'):
                    clean_response = clean_response[8:-3].strip()  # Remove ```json\n and \n```
                
                parsed = json.loads(clean_response)
                majority_vote = parsed.get("majority_vote", "Parse error")
        except Exception as e:
            # Final fallback - extract any answer-like pattern
            answer_match = re.search(r'[A-E]\)\s*[^"]*', response)
            majority_vote = answer_match.group(0) if answer_match else "Parse error"
        
        # Create final decision
        final_decision = {
            "majority_vote": majority_vote,
            "raw_response": response,  # Keep full response for debugging
            "expert_count": len(final_opinions),
            "consensus_method": "moderator_synthesis"
        }
        
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + token_usage["input_tokens"],
            "output": current_usage["output"] + token_usage["output_tokens"]
        }
        
        return Command(
            update={
                "final_decision": final_decision,
                "token_usage": updated_usage,
                "processing_stage": "intermediate_complete"
            },
            goto="intermediate_complete"
        )


# Utility functions

def parse_hierarchy_structure(experts_data: List[Dict]) -> Dict:
    """Parse expert hierarchy into tree structure"""
    hierarchy_tree = {"moderator": "moderator", "children": []}
    
    for expert in experts_data:
        hierarchy = expert.get("hierarchy", "Independent")
        expert_id = expert.get("id", len(hierarchy_tree["children"]) + 1)
        
        if "independent" in hierarchy.lower():
            # Add as direct child of moderator
            hierarchy_tree["children"].append({
                "role": expert["role"],
                "id": expert_id,
                "children": []
            })
        else:
            # Parse hierarchical relationship
            parts = hierarchy.split(">")
            if len(parts) >= 2:
                parent = parts[0].strip()
                child = parts[1].strip()
                # Simplified - add as moderator child for now
                hierarchy_tree["children"].append({
                    "role": expert["role"],
                    "id": expert_id,
                    "parent": parent,
                    "children": []
                })
    
    return hierarchy_tree


def check_debate_continuation(state: IntermediateProcessingState) -> bool:
    """Check if debate should continue based on participation"""
    round_num = state.get("round_number", 1)
    turn_num = state.get("turn_number", 1)
    decisions = state.get("participation_decisions", [])
    
    # Get current round/turn decisions
    current_decisions = [
        d for d in decisions 
        if d.get("round") == round_num and d.get("turn") == turn_num
    ]
    
    # Check if any expert wants to participate
    any_participating = any(d.get("participate", False) for d in current_decisions)
    
    return any_participating


def validate_expert_selection(selection: Dict, experts: List[Dict]) -> List[int]:
    """Validate and filter expert selections"""
    selected = selection.get("selected_experts", [])
    valid_ids = [e["id"] for e in experts]
    
    # Filter out invalid selections
    valid_targets = [s for s in selected if s in valid_ids]
    
    return valid_targets


def assign_hierarchy_weights(experts: List[Dict]) -> List[Dict]:
    """Assign weights based on hierarchy relationships"""
    weighted_experts = []
    
    for expert in experts:
        hierarchy = expert.get("hierarchy", "Independent")
        
        # Simple weight assignment
        if "independent" in hierarchy.lower():
            weight = 1.0
        elif ">" in hierarchy:
            # Parent > Child means parent has higher weight
            parts = hierarchy.split(">")
            parent = parts[0].strip()
            child = parts[1].strip()
            
            # Check if this expert is parent or child
            if expert["role"].lower() in parent.lower():
                weight = 1.2  # Parent gets higher weight
            else:
                weight = 0.8  # Child gets lower weight
        else:
            weight = 1.0
        
        weighted_expert = {**expert, "weight": weight}
        weighted_experts.append(weighted_expert)
    
    return weighted_experts


def intermediate_processing_placeholder(state: IntermediateProcessingState) -> Dict[str, Any]:
    """Placeholder for intermediate processing completion"""
    return {
        **state,
        "processing_stage": "intermediate_complete",
        "final_decision": state.get("final_decision", {"placeholder": "intermediate_result"})
    }


def create_intermediate_processing_subgraph(model_info: str = "gemini-2.5-flash") -> StateGraph:
    """Create the intermediate processing subgraph with multi-round debate"""
    
    # Create subgraph for intermediate processing
    subgraph = StateGraph(IntermediateProcessingState)
    
    # Create node instances
    recruiter = HierarchicalExpertRecruitmentNode(model_info=model_info)
    moderator = ModeratorConsensusNode(model_info=model_info)
    synthesis = RoundSynthesisNode(model_info=model_info)
    
    # Simplified flow for initial implementation
    def collect_initial_opinions(state: IntermediateProcessingState) -> Dict[str, Any]:
        """Collect initial opinions from all experts"""
        experts = state.get("experts_hierarchy", [])
        question = state["question"]
        
        initial_opinions = {}
        total_tokens = {"input": 0, "output": 0}
        
        for expert in experts:
            # Mock opinion for now
            role = expert.get("role", "expert").lower()
            initial_opinions[role] = f"Initial assessment from {role}"
            # In real implementation, would call LLM here
        
        round_opinions = {1: initial_opinions}
        
        return {
            **state,
            "round_opinions": round_opinions,
            "round_number": 1
        }
    
    def parse_combined_decision(response: str) -> Dict[str, Any]:
        """Parse combined participation and selection JSON response"""
        try:
            data = json.loads(response.strip())
            return {
                'participate': data.get('participate', False),
                'reason': data.get('reason', ''),
                'selected_experts': data.get('selected_experts', []),
                'selection_reason': data.get('selection_reason', '')
            }
        except (json.JSONDecodeError, TypeError):
            # Fallback parsing
            should_participate = 'yes' in response.lower() or 'true' in response.lower() or '"participate": true' in response.lower()
            reason = response[:100] if response else "Parsed from text"
            
            # Try to extract expert IDs from text
            selected_experts = []
            if should_participate:
                import re
                # Look for numbers that could be expert IDs
                numbers = re.findall(r'\b([1-9])\b', response)
                selected_experts = [int(n) for n in numbers[:3]]  # Limit to reasonable number
            
            return {
                'participate': should_participate,
                'reason': reason,
                'selected_experts': selected_experts,
                'selection_reason': 'Extracted from text'
            }
    
    async def get_participation_and_selection_async(expert: Dict, assessment_str: str, 
                                                   expert_list: str, model_info: str) -> tuple[Dict[str, Any], Dict[str, int]]:
        """Get combined participation decision and expert selection"""
        participation_node = DebateParticipationNode(expert, model_info)
        
        combined_prompt = f"""Given the opinions from other medical experts in your team, decide:
1. Do you want to participate in this round's discussion?
2. If yes, which expert(s) do you want to communicate with?

Current Opinions:
{assessment_str}

Available Experts:
{expert_list}

Response in JSON format:
{{
  "participate": true/false,
  "reason": "brief explanation",
  "selected_experts": [1, 3],
  "selection_reason": "why I chose these experts"
}}

If you don't want to participate:
{{
  "participate": false,
  "reason": "I agree with current assessments", 
  "selected_experts": [],
  "selection_reason": ""
}}"""
        
        response, token_usage = await participation_node._call_llm_async(combined_prompt)
        decision = parse_combined_decision(response)
        return decision, token_usage
    
    async def generate_communication_async(source_expert: Dict, target_expert: Dict, 
                                         question: str, model_info: str) -> tuple[str, Dict[str, int]]:
        """Generate communication from source expert to target expert"""
        participation_node = DebateParticipationNode(source_expert, model_info)
        
        comm_prompt = f"""Please remind your medical expertise and then leave your opinion/question for an expert you chose (Agent {target_expert['id']}. {target_expert['role']}). 
You should deliver your opinion once you are confident enough and in a way to convince other expert. Limit your response with no more than 200 words.

Question: {question}"""
        
        return await participation_node._call_llm_async(comm_prompt)
    
    async def collect_expert_opinion_async(expert: Dict, round_num: int, question: str, 
                                         model_info: str) -> tuple[str, Dict[str, int]]:
        """Collect expert opinion after round discussions"""
        opinion_agent = LangGraphAgent(
            instruction=f"You are a {expert['role']} reflecting on the discussion.",
            role=expert['role'].lower(),
            model_info=model_info
        )
        
        opinion_prompt = f"Reflecting on the discussions in Round {round_num}, what is your current answer/opinion on the question: {question}\n Limit your answer within 50 words"
        
        def sync_call():
            response = opinion_agent.chat(opinion_prompt)
            usage = opinion_agent.get_token_usage()
            return response, {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"]
            }
        
        return await asyncio.to_thread(sync_call)
    
    async def collect_final_opinion_async(expert: Dict, question: str, 
                                        model_info: str) -> tuple[str, Dict[str, int]]:
        """Collect final expert opinion"""
        final_agent = LangGraphAgent(
            instruction=f"You are a {expert['role']} making final assessment.",
            role=expert['role'].lower(),
            model_info=model_info
        )
        
        final_prompt = f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\n limit your answer within 50 words."
        
        def sync_call():
            response = final_agent.chat(final_prompt)
            usage = final_agent.get_token_usage()
            return response, {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"]
            }
        
        return await asyncio.to_thread(sync_call)
    
    def multi_round_debate(state: IntermediateProcessingState) -> Dict[str, Any]:
        """Conduct multi-round debate with participation consent system - OPTIMIZED with parallel processing"""
        # Run the async version internally using existing event loop or create new one
        import asyncio
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to create a task and wait
                # For now, fall back to a simpler approach - run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _multi_round_debate_async(state))
                    return future.result()
            else:
                return loop.run_until_complete(_multi_round_debate_async(state))
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(_multi_round_debate_async(state))
    
    async def _multi_round_debate_async(state: IntermediateProcessingState) -> Dict[str, Any]:
        """Conduct multi-round debate with participation consent system - OPTIMIZED with parallel processing"""
        experts = state.get("experts_hierarchy", [])
        question = state["question"]
        current_round = state.get("round_number", 1)
        max_rounds = 3
        
        # Initialize tracking variables
        total_tokens = {"input": 0, "output": 0}
        interaction_log = state.get("interaction_log", {})
        round_opinions = state.get("round_opinions", {})
        
        # Prepare expert list for selection prompts
        expert_list = "\n".join([f"{e['id']}. {e['role']}" for e in experts])
        
        # Main debate loop for remaining rounds
        for round_num in range(current_round, max_rounds + 1):
            if round_num not in round_opinions:
                continue
                
            round_name = f"Round {round_num}"
            if round_name not in interaction_log:
                interaction_log[round_name] = {}
            
            # Current assessment for participation decisions
            current_assessment = round_opinions.get(round_num, {})
            assessment_str = "\n".join([f"({k}): {v}" for k, v in current_assessment.items()])
            
            # PARALLEL EXECUTION: Get all participation decisions + selections at once
            participation_tasks = [
                get_participation_and_selection_async(expert, assessment_str, expert_list, model_info)
                for expert in experts
            ]
            
            try:
                participation_results = await asyncio.gather(*participation_tasks)
            except Exception as e:
                print(f"Error in parallel participation: {e}")
                # Fallback to sequential processing
                participation_results = []
                for expert in experts:
                    result = await get_participation_and_selection_async(expert, assessment_str, expert_list, model_info)
                    participation_results.append(result)
            
            # Process participation results
            any_participated = False
            expert_decisions = {}
            
            for i, (decision, token_usage) in enumerate(participation_results):
                expert = experts[i]
                expert_decisions[expert['id']] = decision
                
                # Update token usage
                total_tokens["input"] += token_usage["input_tokens"]
                total_tokens["output"] += token_usage["output_tokens"]
                
                if decision['participate']:
                    any_participated = True
            
            # Generate communications in parallel for participating experts
            communication_tasks = []
            for expert in experts:
                decision = expert_decisions[expert['id']]
                if decision['participate'] and decision['selected_experts']:
                    # Create communication tasks for each selected target expert
                    for target_id in decision['selected_experts']:
                        if 1 <= target_id <= len(experts):
                            target_expert = next((e for e in experts if e["id"] == target_id), None)
                            if target_expert:
                                task = generate_communication_async(expert, target_expert, question, model_info)
                                communication_tasks.append((expert, target_expert, task))
            
            # PARALLEL EXECUTION: Execute all communications at once
            if communication_tasks:
                try:
                    # Extract just the tasks for parallel execution
                    task_list = [task_info[2] for task_info in communication_tasks]
                    communication_results = await asyncio.gather(*task_list)
                    
                    # Process communication results and log interactions
                    for i, (source_expert, target_expert, _) in enumerate(communication_tasks):
                        comm_response, comm_tokens = communication_results[i]
                        
                        # Update token usage
                        total_tokens["input"] += comm_tokens["input_tokens"]
                        total_tokens["output"] += comm_tokens["output_tokens"]
                        
                        # Log the interaction
                        source_key = f"Agent {source_expert['id']}"
                        target_key = f"Agent {target_expert['id']}"
                        if source_key not in interaction_log[round_name]:
                            interaction_log[round_name][source_key] = {}
                        interaction_log[round_name][source_key][target_key] = comm_response
                        
                except Exception as e:
                    print(f"Error in parallel communications: {e}")
                    # Fallback to sequential processing for communications
                    for source_expert, target_expert, task in communication_tasks:
                        try:
                            comm_response, comm_tokens = await task
                            total_tokens["input"] += comm_tokens["input_tokens"]
                            total_tokens["output"] += comm_tokens["output_tokens"]
                            
                            source_key = f"Agent {source_expert['id']}"
                            target_key = f"Agent {target_expert['id']}"
                            if source_key not in interaction_log[round_name]:
                                interaction_log[round_name][source_key] = {}
                            interaction_log[round_name][source_key][target_key] = comm_response
                        except Exception as task_error:
                            print(f"Error in individual communication task: {task_error}")
            
            # If no one participated in this round (after round 1), end debate
            if not any_participated and round_num > 1:
                break
                
            # PARALLEL EXECUTION: Collect updated opinions for next round  
            if round_num < max_rounds:
                opinion_tasks = [
                    collect_expert_opinion_async(expert, round_num, question, model_info)
                    for expert in experts
                ]
                
                try:
                    opinion_results = await asyncio.gather(*opinion_tasks)
                    next_round_opinions = {}
                    
                    for i, (opinion_response, usage) in enumerate(opinion_results):
                        expert = experts[i]
                        total_tokens["input"] += usage["input_tokens"]
                        total_tokens["output"] += usage["output_tokens"]
                        next_round_opinions[expert['role'].lower()] = opinion_response
                    
                    round_opinions[round_num + 1] = next_round_opinions
                    
                except Exception as e:
                    print(f"Error in parallel opinion collection: {e}")
                    # Fallback to sequential processing
                    next_round_opinions = {}
                    for expert in experts:
                        opinion_response, usage = await collect_expert_opinion_async(expert, round_num, question, model_info)
                        total_tokens["input"] += usage["input_tokens"]
                        total_tokens["output"] += usage["output_tokens"]
                        next_round_opinions[expert['role'].lower()] = opinion_response
                    round_opinions[round_num + 1] = next_round_opinions
        
        # PARALLEL EXECUTION: Collect final opinions from all experts
        final_opinion_tasks = [
            collect_final_opinion_async(expert, question, model_info)
            for expert in experts
        ]
        
        try:
            final_opinion_results = await asyncio.gather(*final_opinion_tasks)
            final_opinions = {}
            
            for i, (final_response, usage) in enumerate(final_opinion_results):
                expert = experts[i]
                total_tokens["input"] += usage["input_tokens"]
                total_tokens["output"] += usage["output_tokens"]
                final_opinions[expert['role']] = final_response
                
        except Exception as e:
            print(f"Error in parallel final opinion collection: {e}")
            # Fallback to sequential processing
            final_opinions = {}
            for expert in experts:
                final_response, usage = await collect_final_opinion_async(expert, question, model_info)
                total_tokens["input"] += usage["input_tokens"]
                total_tokens["output"] += usage["output_tokens"]
                final_opinions[expert['role']] = final_response
        
        # Update token usage
        current_usage = state.get("token_usage", {"input": 0, "output": 0})
        updated_usage = {
            "input": current_usage["input"] + total_tokens["input"],
            "output": current_usage["output"] + total_tokens["output"]
        }
        
        return {
            **state,
            "interaction_log": interaction_log,
            "round_opinions": round_opinions,
            "final_expert_opinions": final_opinions,
            "round_number": max_rounds,
            "token_usage": updated_usage,
            "processing_stage": "debate_complete"
        }
    
    # Add nodes to subgraph
    subgraph.add_node("recruit_experts", recruiter.recruit_experts)
    subgraph.add_node("initial_opinions", collect_initial_opinions)
    subgraph.add_node("debate_process", multi_round_debate)
    subgraph.add_node("moderator_consensus", moderator.build_consensus)
    subgraph.add_node("intermediate_complete", intermediate_processing_placeholder)
    
    # Define edges
    subgraph.add_edge("recruit_experts", "initial_opinions")
    subgraph.add_edge("initial_opinions", "debate_process")
    subgraph.add_edge("debate_process", "moderator_consensus")
    subgraph.add_edge("moderator_consensus", "intermediate_complete")
    
    # Set entry point
    subgraph.add_edge("__start__", "recruit_experts")
    
    return subgraph


if __name__ == "__main__":
    # Test intermediate processing subgraph
    subgraph = create_intermediate_processing_subgraph()
    compiled_subgraph = subgraph.compile()
    
    test_state = {
        "messages": [],
        "question": "Complex medical case requiring expert debate",
        "token_usage": {"input": 0, "output": 0},
        "round_number": 1,
        "interaction_log": {},
        "round_opinions": {}
    }
    
    print("Testing intermediate processing subgraph...")
    result = compiled_subgraph.invoke(test_state)
    print(f"Final decision: {result.get('final_decision', {})}")
    print(f"Token usage: {result.get('token_usage', {})}")