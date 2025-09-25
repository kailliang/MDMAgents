# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the LangGraph-driven entry point.
- Processing modules follow `langgraph_<topic>.py` (basic, intermediate, advanced, difficulty, integration, mdm). Extend these to add capabilities without renaming existing files.
- Integration helpers: `langsmith_integration.py` (optional tracing and reporting).
- Tests live at repo root as `test_*.py` with config in `pytest.ini`.
- Scripts: `evaluate_text_output.py` (report generation) and `split_test_data.py` (dataset management).
- Data and results: `data/` for inputs, `output/` for run artifacts; avoid committing sensitive data.

## Build, Test, and Development Commands
- `source .venv/bin/activate` activate virtualenv (uses `.python-version`).
- `pip install -r requirements.txt` install runtime and test dependencies.
- `python main.py --dataset medqa --model gemini-2.5-flash-lite --difficulty adaptive --num_samples 1` run a local sample.
- `pytest -q` run all tests; `pytest -k basic -q` run a subset.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation. Keep docstrings concise and describe expected state shape.
- Add type hints where helpful. Prefer private helpers with leading underscore.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, modules as `langgraph_*`.

## Testing Guidelines
- Framework: pytest (`asyncio_mode=auto`). Mark async tests with `@pytest.mark.asyncio`.
- Mirror module names: `test_langgraph_<topic>.py`. See `test_langsmith_integration.py` for patterns.
- Avoid real network/LLM calls; patch `_call_llm` in unit tests to keep runs deterministic.
- Add regression tests with bug fixes, focusing on node behavior and graph routing.
- Always run tests from the project virtualenv (`source .venv/bin/activate`) before submitting changes.

## Communication Norms
- Keep updates short and clear. Prefer plain language explanations.
- Keep language simple.
- Only use English for doctrings and comments.

## Commit & Pull Request Guidelines
- Commits: imperative mood, focused scope (e.g., "Fix token accumulation in basic flow").
- PRs: clear description, rationale, linked issues, and relevant CLI output (e.g., `pytest -q`). Note performance/behavior changes.
- Update `README.md` when flags or interfaces change. Do not include secrets or PHI in diffs.

## Security & Configuration Tips
- `.env` stores local secrets: `genai_api_key` (Gemini). Optional: `openai_api_key`, `LANGSMITH_*`.
- Keep `.env.sample` updated when introducing new variables.
- Never log or commit credentials or PHI. Review `output/` before sharing.

## LangSmith Tracing
- Enable by setting `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=ls_...`, optional `LANGSMITH_PROJECT` in `.env`.
- When disabled, integration no-ops so local runs remain lightweight. Enable to debug routing and token usage.
