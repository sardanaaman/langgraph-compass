"""Tests for helper functions."""

from langchain_core.messages import AIMessage

from compass.helpers import (
    extract_previous_followups,
    format_suggestions_for_display,
    get_compass_instruction,
)


class TestGetCompassInstruction:
    """Tests for get_compass_instruction."""

    def test_returns_empty_for_no_suggestions(self):
        state = {"compass_suggestions": []}
        assert get_compass_instruction(state) == ""

    def test_returns_empty_for_missing_key(self):
        state = {}
        assert get_compass_instruction(state) == ""

    def test_returns_instruction_with_suggestion(self):
        state = {"compass_suggestions": ["Would you like more details?"]}
        result = get_compass_instruction(state)
        assert "Would you like more details?" in result
        assert "follow-up" in result.lower()

    def test_uses_first_suggestion_only(self):
        state = {"compass_suggestions": ["First?", "Second?", "Third?"]}
        result = get_compass_instruction(state)
        assert "First?" in result
        assert "Second?" not in result

    def test_custom_suggestion_key(self):
        state = {"my_suggestions": ["Custom suggestion?"]}
        result = get_compass_instruction(state, suggestion_key="my_suggestions")
        assert "Custom suggestion?" in result

    def test_custom_template(self):
        state = {"compass_suggestions": ["Test question?"]}
        template = "ASK THIS: {suggestion}"
        result = get_compass_instruction(state, instruction_template=template)
        assert result == "ASK THIS: Test question?"


class TestExtractPreviousFollowups:
    """Tests for extract_previous_followups."""

    def test_returns_empty_for_no_messages(self):
        assert extract_previous_followups([]) == []

    def test_extracts_followups_from_ai_messages(self):
        messages = [
            AIMessage(content="Here's info.\nWould you like more details?"),
        ]
        result = extract_previous_followups(messages)
        assert len(result) == 1
        assert "Would you like more details?" in result[0]

    def test_extracts_multiple_starters(self):
        messages = [
            AIMessage(content="Info.\nWould you like to know more?"),
            AIMessage(content="More info.\nShould I explain further?"),
            AIMessage(content="Details.\nInterested in examples?"),
            AIMessage(content="Examples.\nWant me to clarify?"),
        ]
        result = extract_previous_followups(messages)
        assert len(result) == 4

    def test_respects_max_history(self):
        messages = [
            AIMessage(content=f"Response {i}.\nWould you like option {i}?") for i in range(10)
        ]
        result = extract_previous_followups(messages, max_history=3)
        assert len(result) == 3
        # Should be the last 3
        assert "option 7" in result[0]
        assert "option 8" in result[1]
        assert "option 9" in result[2]

    def test_custom_starters(self):
        messages = [
            AIMessage(content="Info.\nCurious about more?"),
        ]
        result = extract_previous_followups(messages, starters=["Curious about"])
        assert len(result) == 1

    def test_ignores_non_string_content(self):
        messages = [
            AIMessage(content=["not", "a", "string"]),  # type: ignore
        ]
        result = extract_previous_followups(messages)
        assert len(result) == 0


class TestFormatSuggestionsForDisplay:
    """Tests for format_suggestions_for_display."""

    def test_returns_empty_for_no_suggestions(self):
        assert format_suggestions_for_display([]) == ""

    def test_formats_single_suggestion(self):
        result = format_suggestions_for_display(["Question?"])
        assert result == "Question?"

    def test_formats_multiple_suggestions(self):
        result = format_suggestions_for_display(["First?", "Second?"])
        assert result == "First?\nSecond?"

    def test_numbered_format(self):
        result = format_suggestions_for_display(["First?", "Second?"], numbered=True)
        assert result == "1. First?\n2. Second?"

    def test_with_prefix(self):
        result = format_suggestions_for_display(["Question?"], prefix="→ ")
        assert result == "→ Question?"

    def test_numbered_with_prefix(self):
        result = format_suggestions_for_display(["First?", "Second?"], numbered=True, prefix="  ")
        assert result == "  1. First?\n  2. Second?"
