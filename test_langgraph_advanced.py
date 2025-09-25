import json

import langgraph_advanced
from langgraph_advanced import (
    MDTFormationNode,
    _safe_llm_call_with_json_analysis_limit,
    _safe_llm_call_with_truncation,
    process_single_team,
)


class DummyAgent:
    """Test double for LangGraphAgent that tracks cumulative token usage."""

    def __init__(self, deltas, responses=None):
        self._deltas = list(deltas)
        self._responses = responses or [f"response {idx}" for idx in range(len(self._deltas))]
        if len(self._responses) != len(self._deltas):
            raise ValueError("responses and deltas must be the same length")
        self._total_input = 0
        self._total_output = 0
        self.call_count = 0
        self.clear_count = 0

    def chat(self, prompt):
        if self.call_count >= len(self._deltas):
            raise RuntimeError("chat called more times than configured deltas")
        delta_input, delta_output = self._deltas[self.call_count]
        self._total_input += delta_input
        self._total_output += delta_output
        response = self._responses[self.call_count]
        self.call_count += 1
        return response

    def get_token_usage(self):
        return {
            "input_tokens": self._total_input,
            "output_tokens": self._total_output,
            "total_tokens": self._total_input + self._total_output,
        }

    def clear_history(self):
        self.clear_count += 1


def test_safe_llm_call_with_truncation_returns_delta_per_call():
    agent = DummyAgent([(10, 5), (7, 3)], responses=["resp1", "resp2"])

    first_response, first_usage = _safe_llm_call_with_truncation(agent, "prompt-1", 50)
    second_response, second_usage = _safe_llm_call_with_truncation(agent, "prompt-2", 50)

    assert first_response == "resp1"
    assert first_usage == {"input": 10, "output": 5, "total_tokens": 15}
    assert second_response == "resp2"
    assert second_usage == {"input": 7, "output": 3, "total_tokens": 10}
    assert agent.clear_count == 2


def test_safe_llm_call_with_json_analysis_limit_accumulates_delta():
    responses = [
        "not json",
        json.dumps({"analysis": "short", "final_answer": "A) Option"}),
    ]
    agent = DummyAgent([(5, 2), (7, 3)], responses=responses)

    parsed, usage = _safe_llm_call_with_json_analysis_limit(agent, "{}", max_analysis_words=25)

    assert parsed["final_answer"] == "A) Option"
    assert usage == {"input": 12, "output": 5, "total_tokens": 17}
    assert agent.clear_count == 2


def test_mdt_formation_node_call_llm_returns_incremental_usage():
    node = MDTFormationNode(model_info="stub-model")
    node._agent = DummyAgent([(3, 1), (4, 2)], responses=["{}", "{}"])

    _, first_usage = node._call_llm("prompt-1")
    _, second_usage = node._call_llm("prompt-2")

    assert first_usage == {"input": 3, "output": 1, "total_tokens": 4}
    assert second_usage == {"input": 4, "output": 2, "total_tokens": 6}
    assert node._agent.clear_count == 2


def test_process_single_team_aggregates_delta_usage(monkeypatch):
    call_plan = [
        ("lead investigation", {"input": 10, "output": 5, "total_tokens": 15}),
        ("assistant one", {"input": 4, "output": 2, "total_tokens": 6}),
        ("assistant two", {"input": 6, "output": 3, "total_tokens": 9}),
        ("lead synthesis", {"input": 8, "output": 4, "total_tokens": 12}),
    ]
    call_iter = iter(call_plan)

    def fake_safe_llm_call(agent, prompt, word_limit):
        try:
            return next(call_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise AssertionError("_safe_llm_call_with_truncation called too many times") from exc

    class StubAgent:
        def __init__(self, instruction, role, model_info):
            self.instruction = instruction
            self.role = role
            self.model_info = model_info

    monkeypatch.setattr(langgraph_advanced, "_safe_llm_call_with_truncation", fake_safe_llm_call)
    monkeypatch.setattr(langgraph_advanced, "LangGraphAgent", StubAgent)

    team = {
        "team_name": "Initial Assessment Team (IAT)",
        "members": [
            {"member_id": 1, "role": "Lead Physician"},
            {"member_id": 2, "role": "Assistant 1"},
            {"member_id": 3, "role": "Assistant 2"},
        ],
    }

    result = process_single_team(team, "Question?", ["A) Option"], "stub-model")

    assert result["token_usage"]["input_tokens"] == 10 + 4 + 6 + 8
    assert result["token_usage"]["output_tokens"] == 5 + 2 + 3 + 4
    assert result["team_name"] == "Initial Assessment Team (IAT)"
    assert len(result["investigations"]) == 2
