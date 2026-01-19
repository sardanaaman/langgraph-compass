# Compass Development Guide

This file provides context for AI assistants working on this codebase.

## Project Overview

Compass is a LangGraph node that generates intelligent follow-up questions. It follows the same pattern as `SummarizationNode` from langmem - a callable class that integrates seamlessly into LangGraph workflows.

**Value Proposition**: Transform AI agents from reactive responders into proactive conversational partners by generating contextually relevant follow-ups that increase engagement, surface hidden needs, and drive business outcomes.

## Architecture

```
src/compass/
├── __init__.py      # Public API exports
├── node.py          # CompassNode - main entry point
├── triggers.py      # TriggerPolicy protocol and implementations
├── generator.py     # QuestionGenerator with strategy prompts
├── ranker.py        # SuggestionRanker for novelty filtering
└── helpers.py       # Integration utilities
```

### Key Design Decisions

1. **Node Pattern**: We follow LangGraph's node protocol (`__call__(state, config) -> dict`), not middleware
2. **Protocol-Based**: `TriggerPolicy` and `ExampleRetriever` are protocols for extensibility
3. **Auto-Detection**: Query/response extraction tries multiple common state keys
4. **Structured Output**: Generator attempts structured output, falls back to text parsing
5. **RunnableConfig Propagation**: Always pass config for LangSmith tracing

### Ecosystem Integrations

- **LangSmith**: Automatic tracing via RunnableConfig propagation
- **LangMem**: Personalization via custom ExampleRetriever implementations
- **Feedback Loop**: Users can log feedback via LangSmith's create_feedback API

## Development Commands

```bash
# Setup
uv sync --all-extras
uv run pre-commit install

# Run tests
uv run pytest

# Run tests without LLM (fast)
uv run pytest -m "not requires_llm"

# Lint
uv run ruff check src tests

# Type check
uv run mypy src
```

## Code Style

- Use Pydantic for structured data
- Prefer functional approaches for simple operations
- Always propagate `RunnableConfig` for LangSmith tracing
- Type hints on all public APIs
- Docstrings with examples for public functions

## Testing Philosophy

- Mock LLM calls in unit tests
- Use `@pytest.mark.requires_llm` for integration tests
- Test edge cases: empty inputs, missing keys, malformed data

## State Key Conventions

The node auto-detects these common keys:

**Query**: `query`, `question`, `input`, `user_input`, or first `HumanMessage`
**Response**: `final_response`, `response`, `output`, `answer`, or last `AIMessage`
**Messages**: `messages` (configurable via `messages_key`)

## Adding New Features

1. **New Trigger Policy**: Implement `TriggerPolicy` protocol in `triggers.py`
2. **New Strategy**: Add to `STRATEGY_PROMPTS` in `generator.py`
3. **New Helper**: Add to `helpers.py`, export from `__init__.py`

## Publishing

Releases are automated via GitHub Actions when you create a release:

1. Update version in `pyproject.toml` and `src/compass/__init__.py`
2. Commit and push to main
3. Create a GitHub Release with tag (e.g., `v1.0.1`)
4. GitHub Actions builds and publishes to PyPI automatically

Manual publishing (if needed):
```bash
uv build
uv publish
```
