"""Tests for CompassNode."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from compass.node import CompassNode, ExampleRetriever
from compass.triggers import AlwaysTrigger, NeverTrigger


class MockExampleRetriever:
    """Mock example retriever for testing."""

    def __init__(self, examples: list[str]):
        self.examples = examples
        self.called_with: str | None = None

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        self.called_with = query
        return self.examples[:k]


class TestCompassNode:
    """Tests for CompassNode."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock language model."""
        model = MagicMock()
        model.with_structured_output.side_effect = AttributeError()
        model.invoke.return_value = AIMessage(
            content="""1. Would you like me to explain more?
2. Interested in seeing examples?
3. Should I go deeper?"""
        )
        return model

    def test_initialization_defaults(self, mock_model):
        node = CompassNode(mock_model)
        assert node.strategy == "adaptive"
        assert node.max_suggestions == 1
        assert node.output_key == "compass_suggestions"

    def test_returns_empty_when_trigger_false(self, mock_model):
        node = CompassNode(mock_model, trigger=NeverTrigger())
        result = node({"final_response": "A" * 100})
        assert result["compass_suggestions"] == []

    def test_returns_suggestions_when_trigger_true(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger())
        result = node(
            {
                "query": "What is Python?",
                "final_response": "Python is a programming language." * 10,
            }
        )
        assert len(result["compass_suggestions"]) == 1

    def test_respects_max_suggestions(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger(), max_suggestions=2)
        result = node(
            {
                "query": "What is Python?",
                "final_response": "Python is a programming language." * 10,
            }
        )
        assert len(result["compass_suggestions"]) <= 2

    def test_custom_output_key(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger(), output_key="my_suggestions")
        result = node(
            {
                "query": "Test?",
                "final_response": "Test response." * 10,
            }
        )
        assert "my_suggestions" in result

    def test_extracts_query_from_messages(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger())
        result = node(
            {
                "messages": [
                    HumanMessage(content="What is Python?"),
                    AIMessage(content="Python is a programming language." * 10),
                ],
            }
        )
        assert len(result["compass_suggestions"]) == 1

    def test_extracts_response_from_messages(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger())
        result = node(
            {
                "query": "What is Python?",
                "messages": [
                    AIMessage(content="Python is a programming language." * 10),
                ],
            }
        )
        assert len(result["compass_suggestions"]) == 1

    def test_explicit_query_key(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger(), query_key="user_question")
        result = node(
            {
                "user_question": "What is Python?",
                "final_response": "Python is a programming language." * 10,
            }
        )
        assert len(result["compass_suggestions"]) == 1

    def test_explicit_response_key(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger(), response_key="agent_output")
        result = node(
            {
                "query": "What is Python?",
                "agent_output": "Python is a programming language." * 10,
            }
        )
        assert len(result["compass_suggestions"]) == 1

    def test_returns_empty_when_no_query(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger())
        result = node({"final_response": "Some response." * 10})
        assert result["compass_suggestions"] == []

    def test_returns_empty_when_no_response(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger())
        result = node({"query": "What is Python?"})
        assert result["compass_suggestions"] == []

    def test_uses_example_retriever(self, mock_model):
        retriever = MockExampleRetriever(["Example 1?", "Example 2?"])
        node = CompassNode(mock_model, trigger=AlwaysTrigger(), example_retriever=retriever)
        node(
            {
                "query": "What is Python?",
                "final_response": "Python is a programming language." * 10,
            }
        )
        assert retriever.called_with == "What is Python?"

    def test_inject_into_messages(self, mock_model):
        node = CompassNode(mock_model, trigger=AlwaysTrigger(), inject_into_messages=True)
        result = node(
            {
                "query": "What is Python?",
                "final_response": "Python is a programming language." * 10,
            }
        )
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "compass_suggestions" in result["messages"][0].additional_kwargs

    def test_callable_with_config(self, mock_model):
        """Test that config is passed through to the model."""
        node = CompassNode(mock_model, trigger=AlwaysTrigger())
        config = {"configurable": {"thread_id": "test"}}
        result = node(
            {
                "query": "What is Python?",
                "final_response": "Python is a programming language." * 10,
            },
            config=config,
        )
        assert len(result["compass_suggestions"]) == 1


class TestExampleRetrieverProtocol:
    """Tests for ExampleRetriever protocol."""

    def test_mock_retriever_implements_protocol(self):
        retriever = MockExampleRetriever(["test"])
        assert isinstance(retriever, ExampleRetriever)

    def test_retriever_returns_examples(self):
        retriever = MockExampleRetriever(["Q1?", "Q2?", "Q3?"])
        result = retriever.retrieve("test query", k=2)
        assert len(result) == 2
        assert result == ["Q1?", "Q2?"]
