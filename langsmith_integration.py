#!/usr/bin/env python3
"""Utility helpers for optional LangSmith tracing.

This module activates LangSmith spans only when:
- The langsmith package is installed, and
- The ``langsmith_tracing`` (or ``LANGSMITH_TRACING``) env var is truthy, and
- A LangSmith API key is present.

All helpers no-op when tracing is disabled so the rest of the system can
import them unconditionally.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # Prefer LangSmith's native tracing context when available
    from langsmith.run_helpers import (
        get_current_run_tree as _ls_get_current_run_tree,
        tracing_context as _ls_tracing_context,
    )
except Exception:  # pragma: no cover - fallback when run_helpers missing
    _ls_get_current_run_tree = None

    def _ls_tracing_context(**_kwargs):  # type: ignore[return-type]
        return contextlib.nullcontext()


def _get_env(name: str) -> Optional[str]:
    """Return env var in snake_case or uppercase form, normalizing case."""

    value = os.getenv(name)
    upper_name = name.upper()

    if value is not None:
        if name != upper_name and os.getenv(upper_name) is None:
            os.environ[upper_name] = value
        return value

    upper_value = os.getenv(upper_name)
    if upper_value is not None:
        if name != upper_name and os.getenv(name) is None:
            os.environ[name] = upper_value
        return upper_value

    return None


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _normalize_usage(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert internal token usage dicts to LangSmith-compatible schema."""

    if not data:
        return None

    prompt = data.get("prompt_tokens")
    completion = data.get("completion_tokens")
    total = data.get("total_tokens")

    if prompt is None and completion is None:
        # Fall back to explicit input/output token keys
        prompt = data.get("input_tokens")
        completion = data.get("output_tokens")

    if prompt is None and completion is None:
        # Fall back to aggregated usage that omits the "_tokens" suffix
        prompt = data.get("input")
        completion = data.get("output")

    if prompt is None and completion is None and total is None:
        return None

    prompt_val = prompt or 0
    completion_val = completion or 0

    if total is None:
        total = prompt_val + completion_val

    return {
        "prompt_tokens": prompt_val,
        "completion_tokens": completion_val,
        "total_tokens": total,
    }


def preview_text(value: Optional[str], limit: int = 2000) -> Optional[str]:
    """Return text trimmed to the provided limit with ellipsis."""

    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


try:  # pragma: no cover - import guarded for optional dependency
    from langsmith import traceable as _native_traceable
    from langsmith.run_trees import RunTree as _RunTree
    from langsmith.wrappers import wrap_openai as _wrap_openai
except Exception as exc:  # pragma: no cover - fallback when not installed
    _native_traceable = None
    _RunTree = None
    _wrap_openai = None
    if _is_truthy(_get_env("langsmith_tracing")):
        logger.warning(
            "LangSmith tracing requested but langsmith package is unavailable: %s",
            exc,
        )
_enabled = False
if _native_traceable and _RunTree and _wrap_openai:
    if _is_truthy(_get_env("langsmith_tracing")):
        if _get_env("langsmith_api_key"):
            _enabled = True
            logger.info("LangSmith tracing enabled.")
        else:
            logger.warning(
                "LangSmith tracing requested but no langsmith_api_key was provided."
            )


def is_tracing_enabled() -> bool:
    """Return True when LangSmith tracing is active."""

    return _enabled


def traceable(*decorator_args, **decorator_kwargs):
    """Expose langsmith.traceable with graceful fallback when disabled."""

    if not _enabled or _native_traceable is None:
        # Support both bare decorator use and decorator factory usage
        if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
            return decorator_args[0]

        def _decorator(func):
            return func

        return _decorator

    return _native_traceable(*decorator_args, **decorator_kwargs)


def wrap_openai_client(client: Any) -> Any:
    """Wrap an OpenAI client with LangSmith instrumentation when enabled."""

    if _enabled and _wrap_openai is not None:
        try:
            return _wrap_openai(client)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to wrap OpenAI client for LangSmith: %s", exc)
    return client


@contextlib.contextmanager
def span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parent: Any = None,
    require_parent: bool = True,
):
    """Create a LangSmith run span context.

    Yields ``(run, finish)`` where ``finish(outputs=None, error=None)`` may be
    invoked to update the run before exit. The context automatically finalises the
    run if ``finish`` is not called explicitly.
    """

    if not _enabled or _RunTree is None:
        def _noop_finish(*_args, **_kwargs):
            return None

        yield (None, _noop_finish)
        return

    run_inputs = inputs or {}
    run_metadata = metadata or {}
    project = _get_env("langsmith_project")
    run_kwargs: Dict[str, Any] = {
        "name": name,
        "run_type": run_type,
        "inputs": run_inputs,
    }
    if project:
        run_kwargs["project_name"] = project
    if run_metadata:
        run_kwargs["metadata"] = run_metadata

    if parent is not None:
        parent_run = parent
    elif _ls_get_current_run_tree is not None:
        try:
            parent_run = _ls_get_current_run_tree()
        except Exception:  # pragma: no cover - safety net if LangSmith internals change
            parent_run = None
    else:
        parent_run = None

    # Avoid creating stray root runs unless explicitly requested
    if require_parent and parent_run is None:
        def _noop_finish(*_args, **_kwargs):
            return None
        yield (None, _noop_finish)
        return
    try:
        if parent_run:
            child_kwargs = {k: v for k, v in run_kwargs.items() if k != "project_name"}
            run = parent_run.create_child(**child_kwargs)
        else:
            run = _RunTree(**run_kwargs)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to create LangSmith span '%s': %s", name, exc)

        def _noop_finish(*_args, **_kwargs):
            return None

        yield (None, _noop_finish)
        return

    try:
        run.post()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to post LangSmith span '%s': %s", name, exc)

        def _noop_finish(*_args, **_kwargs):
            return None

        yield (None, _noop_finish)
        return

    finished = False

    def _finish(
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[BaseException] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        nonlocal finished
        if finished:
            return
        finished = True
        try:
            usage_payload = _normalize_usage(usage)
            # Attach usage as usage_metadata in outputs for LangSmith cost/latency
            final_outputs: Dict[str, Any] = dict(outputs or {})
            if usage_payload:
                final_outputs.setdefault(
                    "usage_metadata",
                    {
                        "input_tokens": usage_payload.get("prompt_tokens", 0),
                        "output_tokens": usage_payload.get("completion_tokens", 0),
                        "total_tokens": usage_payload.get("total_tokens", 0),
                    },
                )
            if error is not None:
                run.end(error=str(error))
            else:
                run.end(outputs=final_outputs)
            run.patch()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to finalise LangSmith span '%s': %s", name, exc)

    try:
        with _ls_tracing_context(parent=run):
            yield (run, _finish)
    except Exception as exc:
        _finish(error=exc)
        raise
    finally:
        if not finished:
            _finish()


__all__ = [
    "is_tracing_enabled",
    "traceable",
    "wrap_openai_client",
    "span",
    "preview_text",
]
