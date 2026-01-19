"""Integration tests that verify the full Compass workflow.

These tests require an actual LLM and are marked with @pytest.mark.requires_llm.
Run with: uv run pytest -m requires_llm
Skip with: uv run pytest -m "not requires_llm"
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from compass import CompassNode, DefaultTriggerPolicy, get_compass_instruction


@pytest.mark.requires_llm
class TestCompassIntegration:
    """Integration tests that use a real LLM."""

    @pytest.fixture
    def model(self):
        """Get a real LLM model for testing."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-5-nano", temperature=0.7)

    def test_generates_real_suggestions(self, model):
        """Test that Compass generates actual suggestions with a real LLM."""
        compass = CompassNode(
            model=model,
            strategy="adaptive",
            max_suggestions=2,
        )

        state = {
            "query": "What is machine learning?",
            "final_response": (
                "Machine learning is a subset of artificial intelligence that enables "
                "systems to learn and improve from experience without being explicitly "
                "programmed. It focuses on developing algorithms that can access data "
                "and use it to learn for themselves. The process begins with observations "
                "or data, such as examples, direct experience, or instruction."
            ),
            "messages": [
                HumanMessage(content="What is machine learning?"),
                AIMessage(content="Machine learning is a subset of artificial intelligence..."),
            ],
        }

        result = compass(state)

        # Should return suggestions
        suggestions = result.get("compass_suggestions", [])
        assert len(suggestions) > 0
        assert len(suggestions) <= 2

        # Suggestions should be questions
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert suggestion.endswith("?")

    def test_respects_trigger_policy(self, model):
        """Test that trigger policy is respected."""
        compass = CompassNode(
            model=model,
            trigger=DefaultTriggerPolicy(min_response_length=1000),
        )

        state = {
            "query": "Hi",
            "final_response": "Hello! How can I help?",  # Too short
        }

        result = compass(state)
        assert result["compass_suggestions"] == []

    def test_different_strategies_produce_different_results(self, model):
        """Test that different strategies produce different types of questions."""
        state = {
            "query": "Explain photosynthesis",
            "final_response": (
                "Photosynthesis is the process by which plants convert light energy "
                "into chemical energy. It occurs in the chloroplasts of plant cells "
                "and involves the absorption of carbon dioxide and water to produce "
                "glucose and oxygen. The process requires sunlight and chlorophyll."
            ),
        }

        clarifying_compass = CompassNode(model=model, strategy="clarifying")
        exploratory_compass = CompassNode(model=model, strategy="exploratory")

        clarifying_result = clarifying_compass(state)
        exploratory_result = exploratory_compass(state)

        # Both should produce suggestions
        assert len(clarifying_result["compass_suggestions"]) > 0
        assert len(exploratory_result["compass_suggestions"]) > 0


class TestCompassWithMockedGraph:
    """Tests that verify Compass works in a graph-like flow without real LLM."""

    def test_full_flow_with_mock(self):
        """Test the full Compass flow with mocked components."""
        from unittest.mock import MagicMock

        # Create mock model
        mock_model = MagicMock()
        mock_model.with_structured_output.side_effect = AttributeError()
        mock_model.invoke.return_value = AIMessage(
            content="1. Would you like me to explain more about this topic?\n"
            "2. Interested in seeing some examples?"
        )

        compass = CompassNode(
            model=mock_model,
            max_suggestions=2,
        )

        # Simulate graph state after agent execution
        state = {
            "messages": [
                HumanMessage(content="What is Python?"),
                AIMessage(
                    content="Python is a high-level programming language known for "
                    "its simplicity and readability. It supports multiple programming "
                    "paradigms and has a vast ecosystem of libraries."
                ),
            ],
        }

        result = compass(state)

        # Verify suggestions were generated
        assert "compass_suggestions" in result
        assert len(result["compass_suggestions"]) == 2

    def test_helper_function_integration(self):
        """Test that helper functions work with Compass output."""
        state = {
            "compass_suggestions": ["Would you like me to explain more?"],
        }

        instruction = get_compass_instruction(state)

        assert "Would you like me to explain more?" in instruction
        assert "follow-up" in instruction.lower()

    def test_empty_suggestions_helper(self):
        """Test helper returns empty string when no suggestions."""
        state = {"compass_suggestions": []}

        instruction = get_compass_instruction(state)
        assert instruction == ""
