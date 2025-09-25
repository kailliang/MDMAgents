import contextlib

import langgraph_advanced as advanced


def test_compile_team_results_uses_aggregated_usage(monkeypatch):
    captures = {}

    @contextlib.contextmanager
    def fake_span(*args, **kwargs):
        def finish_span(*, outputs=None, usage=None, **finish_kwargs):
            captures["outputs"] = outputs
            captures["usage"] = usage
            captures["finish_kwargs"] = finish_kwargs

        yield (object(), finish_span)

    monkeypatch.setattr(advanced, "langsmith_span", fake_span)

    state = {
        "team_results": [
            {"team_name": "Initial Assessment Team", "assessment": "Assessment A"},
            {"team_name": "Specialist Team Bravo", "assessment": "Assessment B"},
            {"team_name": "Final Review Team", "assessment": "Assessment C"},
        ],
        "token_usage": {"input": 11, "output": 7},
    }

    result = advanced.compile_team_results(state)

    expected_total_tokens = 18

    assert result["token_usage"]["input"] == 11
    assert result["token_usage"]["output"] == 7
    assert result["token_usage"]["total_tokens"] == expected_total_tokens

    assert captures["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": expected_total_tokens,
    }

    assert captures["outputs"]["token_usage"] == {
        "input": 11,
        "output": 7,
        "total_tokens": expected_total_tokens,
    }
    assert "team_token_usage" not in captures["outputs"]

    categorized = result["team_assessments"]
    assert len(categorized["initial"]) == 1
    assert len(categorized["specialist"]) == 1
    assert len(categorized["final_review"]) == 1


def test_compile_team_results_uses_team_usage_when_state_missing(monkeypatch):
    captures = {}

    @contextlib.contextmanager
    def fake_span(*args, **kwargs):
        def finish_span(*, outputs=None, usage=None, **finish_kwargs):
            captures["outputs"] = outputs
            captures["usage"] = usage
            captures["finish_kwargs"] = finish_kwargs

        yield (object(), finish_span)

    monkeypatch.setattr(advanced, "langsmith_span", fake_span)

    state = {
        "team_results": [
            {
                "team_name": "Initial Assessment Team",
                "assessment": "Assessment A",
                "token_usage": {"input": 3, "output": 2},
            },
            {
                "team_name": "Specialist Team Bravo",
                "assessment": "Assessment B",
                "token_usage": {"input": 5, "output": 4},
            },
        ],
        "token_usage": {},
    }

    result = advanced.compile_team_results(state)

    expected_input = 8
    expected_output = 6
    expected_total = expected_input + expected_output

    assert result["token_usage"] == {
        "input": expected_input,
        "output": expected_output,
        "total_tokens": expected_total,
    }

    assert captures["usage"] == {
        "input_tokens": expected_input,
        "output_tokens": expected_output,
        "total_tokens": expected_total,
    }

    assert captures["outputs"]["team_token_usage"] == {
        "input": expected_input,
        "output": expected_output,
        "total_tokens": expected_total,
    }

    categorized = result["team_assessments"]
    assert categorized["initial"][0]["token_usage"] == {"input": 3, "output": 2}
    assert categorized["specialist"][0]["token_usage"] == {"input": 5, "output": 4}
