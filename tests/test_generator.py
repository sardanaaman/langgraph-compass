"""Tests for question generator."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from compass.generator import (
    DEFAULT_STARTERS,
    STRATEGY_PROMPTS,
    GeneratedQuestions,
    QuestionGenerator,
)


class TestStrategyPrompts:
    """Tests for strategy prompt definitions."""

    def test_all_strategies_defined(self):
        expected = {"adaptive", "clarifying", "exploratory", "deepening"}
        assert set(STRATEGY_PROMPTS.keys()) == expected

    def test_prompts_are_non_empty(self):
        for strategy, prompt in STRATEGY_PROMPTS.items():
            assert len(prompt) > 50, f"Strategy {strategy} has too short prompt"


class TestQuestionGenerator:
    """Tests for QuestionGenerator."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock language model that falls back to text parsing."""
        model = MagicMock()
        # Make structured output fail so we fall back to text parsing
        model.with_structured_output.side_effect = AttributeError()
        model.invoke.return_value = AIMessage(
            content="""1. Would you like me to explain more about this topic?
2. Interested in seeing some examples?
3. Should I dive deeper into the details?"""
        )
        return model

    def test_initialization_defaults(self, mock_model):
        generator = QuestionGenerator(mock_model)
        assert generator.strategy == "adaptive"
        assert generator.starters == DEFAULT_STARTERS

    def test_initialization_custom(self, mock_model):
        custom_starters = ["How about", "What if"]
        generator = QuestionGenerator(mock_model, strategy="clarifying", starters=custom_starters)
        assert generator.strategy == "clarifying"
        assert generator.starters == custom_starters

    def test_generate_parses_output(self, mock_model):
        generator = QuestionGenerator(mock_model)
        questions = generator.generate(
            query="What is Python?",
            response="Python is a programming language.",
            n=3,
        )
        assert len(questions) == 3
        assert all(q.endswith("?") for q in questions)

    def test_generate_with_history(self, mock_model):
        generator = QuestionGenerator(mock_model)
        questions = generator.generate(
            query="What is Python?",
            response="Python is a programming language.",
            history=["Previous question?"],
            n=3,
        )
        # Should still work, history is just passed to prompt
        assert len(questions) == 3

    def test_generate_with_examples(self, mock_model):
        generator = QuestionGenerator(mock_model)
        questions = generator.generate(
            query="What is Python?",
            response="Python is a programming language.",
            examples=["Example question 1?", "Example question 2?"],
            n=3,
        )
        assert len(questions) == 3

    def test_parse_questions_handles_numbered_list(self, mock_model):
        generator = QuestionGenerator(mock_model)
        content = """1. Would you like more details?
2. Interested in examples?
3. Should I explain further?"""
        questions = generator._parse_questions(content, 3)
        assert len(questions) == 3

    def test_parse_questions_handles_dashes(self, mock_model):
        generator = QuestionGenerator(mock_model)
        content = """- Would you like more details?
- Interested in examples?
- Should I explain further?"""
        questions = generator._parse_questions(content, 3)
        assert len(questions) == 3

    def test_parse_questions_respects_n_limit(self, mock_model):
        generator = QuestionGenerator(mock_model)
        content = """1. Would you like more details?
2. Interested in examples?
3. Should I explain further?
4. Want me to continue?"""
        questions = generator._parse_questions(content, 2)
        assert len(questions) == 2

    def test_structured_output_fallback(self, mock_model):
        """Test fallback when structured output fails."""
        # Make with_structured_output raise an error
        mock_model.with_structured_output.side_effect = AttributeError()

        generator = QuestionGenerator(mock_model)
        questions = generator.generate(
            query="Test?",
            response="Test response.",
            n=3,
        )

        # Should fall back to text parsing
        assert len(questions) == 3

    def test_structured_output_success(self):
        """Test when structured output works."""
        # Create a fresh mock that supports structured output
        mock_model = MagicMock()
        structured_model = MagicMock()
        structured_model.invoke.return_value = GeneratedQuestions(
            questions=[
                "Would you like more details?",
                "Interested in examples?",
            ]
        )
        mock_model.with_structured_output.return_value = structured_model

        generator = QuestionGenerator(mock_model)
        questions = generator.generate(
            query="Test?",
            response="Test response.",
            n=2,
        )

        assert len(questions) == 2
        assert questions[0] == "Would you like more details?"
