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

    categorized = result["team_assessments"]
    assert len(categorized["initial"]) == 1
    assert len(categorized["specialist"]) == 1
    assert len(categorized["final_review"]) == 1
