import pytest

from langsmith_integration import _normalize_usage


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            {"prompt_tokens": 10, "completion_tokens": 5},
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
        (
            {"input_tokens": 7, "output_tokens": 3},
            {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        ),
        (
            {"input": 12, "output": 8, "total_tokens": 25},
            {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 25},
        ),
        (
            {"total_tokens": 30},
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 30},
        ),
    ],
)
def test_normalize_usage(payload, expected):
    assert _normalize_usage(payload) == expected


def test_normalize_usage_missing_data():
    assert _normalize_usage(None) is None
    assert _normalize_usage({}) is None
