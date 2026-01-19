"""Pytest configuration and shared fixtures."""


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_llm: marks tests that require an actual LLM (deselect with '-m \"not requires_llm\"')",
    )
