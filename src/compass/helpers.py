"""Helper functions for integrating Compass into LangGraph workflows."""

from typing import Any


def get_compass_instruction(
    state: dict[str, Any],
    *,
    suggestion_key: str = "compass_suggestions",
    instruction_template: str | None = None,
) -> str:
    """Get an instruction to include in synthesis/response prompts.

    Use this to inject Compass suggestions into your agent's final
    response generation, making follow-ups feel organic.

    Args:
        state: The graph state containing suggestions.
        suggestion_key: Key where suggestions are stored.
        instruction_template: Custom template with {suggestion} placeholder.

    Returns:
        Instruction string to include in prompts, or empty string if no suggestions.

    Example:
        >>> state = {"compass_suggestions": ["Would you like more details?"]}
        >>> instruction = get_compass_instruction(state)
        >>> prompt = f"Answer the question. {instruction}"
    """
    suggestions = state.get(suggestion_key, [])
    if not suggestions:
        return ""

    template = instruction_template or (
        "At the end of your response, naturally include this follow-up question "
        'as a new paragraph: "{suggestion}" '
        "You may refine it slightly to flow with your response."
    )
    return template.format(suggestion=suggestions[0])


def extract_previous_followups(
    messages: list[Any],
    starters: list[str] | None = None,
    max_history: int = 5,
) -> list[str]:
    """Extract previous follow-up questions from message history.

    Scans messages for lines starting with common follow-up phrases
    to build a history for novelty filtering.

    Args:
        messages: List of message objects with 'content' attribute.
        starters: Starter phrases to look for.
        max_history: Maximum number of follow-ups to return.

    Returns:
        List of previous follow-up questions.

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> messages = [AIMessage(content="Here's info.\\nWould you like more details?")]
        >>> extract_previous_followups(messages)
        ['Would you like more details?']
    """
    default_starters = ["Would you like", "Want me to", "Interested in", "Should I"]
    starters = starters or default_starters
    followups: list[str] = []

    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue

        for line in content.split("\n"):
            line = line.strip()
            if any(line.startswith(s) for s in starters):
                followups.append(line)

    return followups[-max_history:]


def format_suggestions_for_display(
    suggestions: list[str],
    *,
    numbered: bool = False,
    prefix: str = "",
) -> str:
    """Format suggestions for display to users.

    Args:
        suggestions: List of suggestion strings.
        numbered: Whether to add numbers.
        prefix: Optional prefix for each line.

    Returns:
        Formatted string.
    """
    if not suggestions:
        return ""

    lines = []
    for i, suggestion in enumerate(suggestions, 1):
        if numbered:
            lines.append(f"{prefix}{i}. {suggestion}")
        else:
            lines.append(f"{prefix}{suggestion}")

    return "\n".join(lines)
