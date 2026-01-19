"""Trigger policies that determine when Compass should generate follow-up questions."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TriggerPolicy(Protocol):
    """Protocol for trigger policies.

    Implement this protocol to create custom logic for when
    follow-up questions should be generated.
    """

    def should_trigger(self, state: dict[str, Any]) -> bool:
        """Determine if follow-up generation should occur.

        Args:
            state: The current graph state dictionary.

        Returns:
            True if follow-ups should be generated, False otherwise.
        """
        ...


class DefaultTriggerPolicy:
    """Default policy with sensible skip conditions.

    This policy skips follow-up generation when:
    - Guardrails have fired (input or output)
    - Query classification indicates more info is needed
    - Response is too short to warrant follow-ups

    Example:
        >>> policy = DefaultTriggerPolicy(min_response_length=100)
        >>> policy.should_trigger({"final_response": "Short."})
        False

        >>> # Custom guardrail keys for your implementation
        >>> policy = DefaultTriggerPolicy(
        ...     guardrail_keys=["off_topic_triggered", "pii_detected"]
        ... )
    """

    # Default guardrail state keys to check
    DEFAULT_GUARDRAIL_KEYS = ["input_guardrail_fired", "output_guardrail_fired"]

    def __init__(
        self,
        *,
        skip_on_guardrail: bool = True,
        guardrail_keys: list[str] | None = None,
        skip_classifications: list[str] | None = None,
        require_agent_response: bool = True,
        min_response_length: int = 50,
        custom_skip_keys: list[str] | None = None,
    ) -> None:
        """Initialize the trigger policy.

        Args:
            skip_on_guardrail: Skip if guardrails fired.
            guardrail_keys: State keys to check for guardrail triggers.
                Defaults to ["input_guardrail_fired", "output_guardrail_fired"].
                Set to your own keys to match your guardrails implementation.
            skip_classifications: Query classifications to skip.
            require_agent_response: Require a substantive response.
            min_response_length: Minimum response length to trigger.
            custom_skip_keys: Additional state keys that, if truthy, skip generation.
        """
        self.skip_on_guardrail = skip_on_guardrail
        self.guardrail_keys = guardrail_keys or self.DEFAULT_GUARDRAIL_KEYS
        self.skip_classifications = skip_classifications or ["REQUIRES_MORE_INFORMATION"]
        self.require_agent_response = require_agent_response
        self.min_response_length = min_response_length
        self.custom_skip_keys = custom_skip_keys or []

    def should_trigger(self, state: dict[str, Any]) -> bool:
        """Check all conditions to determine if we should generate follow-ups."""
        # Skip if any guardrail key is truthy
        if self.skip_on_guardrail:
            for key in self.guardrail_keys:
                if state.get(key):
                    return False

        # Skip certain classifications
        if state.get("query_classification") in self.skip_classifications:
            return False

        # Check custom skip keys
        for key in self.custom_skip_keys:
            if state.get(key):
                return False

        # Require substantive response
        if self.require_agent_response:
            response = self._get_response(state)
            if len(response) < self.min_response_length:
                return False

        return True

    def _get_response(self, state: dict[str, Any]) -> str:
        """Extract response text from state, checking common keys."""
        # Check common response keys
        for key in ("final_response", "response", "output", "answer"):
            val = state.get(key)
            if isinstance(val, str):
                return val

        # Check messages for last AI message
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", "")
            if isinstance(content, str):
                return content

        return ""


class AlwaysTrigger:
    """Trigger policy that always generates follow-ups.

    Useful for testing or when you want follow-ups on every response.
    """

    def should_trigger(self, state: dict[str, Any]) -> bool:
        """Always returns True."""
        return True


class NeverTrigger:
    """Trigger policy that never generates follow-ups.

    Useful for temporarily disabling Compass.
    """

    def should_trigger(self, state: dict[str, Any]) -> bool:
        """Always returns False."""
        return False
