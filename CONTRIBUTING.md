# Contributing to Compass

Thank you for your interest in contributing to Compass!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/amasardana/langgraph-compass.git
   cd langgraph-compass
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync --all-extras
   ```

4. **Set up pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run tests without LLM (fast)
uv run pytest -m "not requires_llm"

# Run specific test file
uv run pytest tests/test_node.py
```

### Code Quality

```bash
# Lint code
uv run ruff check src tests

# Auto-fix lint issues
uv run ruff check src tests --fix

# Format code
uv run ruff format src tests

# Type check
uv run mypy src
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
uv run pre-commit run --all-files
```

## Making Changes

1. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for any new functionality

4. **Run the full test suite** to ensure nothing is broken

5. **Commit your changes** (pre-commit hooks will run automatically)

6. **Push and create a Pull Request**

## Code Style

- Use type hints on all public APIs
- Write docstrings with examples for public functions
- Follow the existing patterns in the codebase
- Keep functions focused and minimal

## Adding New Features

- **New Trigger Policy**: Implement the `TriggerPolicy` protocol in `triggers.py`
- **New Strategy**: Add to `STRATEGY_PROMPTS` in `generator.py`
- **New Helper**: Add to `helpers.py` and export from `__init__.py`

## Questions?

Open an issue for discussion before making large changes.
