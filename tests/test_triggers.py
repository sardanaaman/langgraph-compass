"""Tests for trigger policies."""

from compass.triggers import AlwaysTrigger, DefaultTriggerPolicy, NeverTrigger, TriggerPolicy


class TestTriggerPolicyProtocol:
    """Test that trigger policies implement the protocol correctly."""

    def test_default_policy_implements_protocol(self):
        policy = DefaultTriggerPolicy()
        assert isinstance(policy, TriggerPolicy)

    def test_always_trigger_implements_protocol(self):
        policy = AlwaysTrigger()
        assert isinstance(policy, TriggerPolicy)

    def test_never_trigger_implements_protocol(self):
        policy = NeverTrigger()
        assert isinstance(policy, TriggerPolicy)


class TestDefaultTriggerPolicy:
    """Tests for DefaultTriggerPolicy."""

    def test_triggers_on_valid_state(self):
        policy = DefaultTriggerPolicy()
        state = {"final_response": "A" * 100}  # Long enough response
        assert policy.should_trigger(state) is True

    def test_skips_on_input_guardrail(self):
        policy = DefaultTriggerPolicy()
        state = {
            "final_response": "A" * 100,
            "input_guardrail_fired": True,
        }
        assert policy.should_trigger(state) is False

    def test_skips_on_output_guardrail(self):
        policy = DefaultTriggerPolicy()
        state = {
            "final_response": "A" * 100,
            "output_guardrail_fired": True,
        }
        assert policy.should_trigger(state) is False

    def test_skips_guardrails_when_disabled(self):
        policy = DefaultTriggerPolicy(skip_on_guardrail=False)
        state = {
            "final_response": "A" * 100,
            "input_guardrail_fired": True,
        }
        assert policy.should_trigger(state) is True

    def test_skips_on_classification(self):
        policy = DefaultTriggerPolicy()
        state = {
            "final_response": "A" * 100,
            "query_classification": "REQUIRES_MORE_INFORMATION",
        }
        assert policy.should_trigger(state) is False

    def test_custom_skip_classifications(self):
        policy = DefaultTriggerPolicy(skip_classifications=["CUSTOM_SKIP"])
        state = {
            "final_response": "A" * 100,
            "query_classification": "CUSTOM_SKIP",
        }
        assert policy.should_trigger(state) is False

    def test_skips_short_response(self):
        policy = DefaultTriggerPolicy(min_response_length=100)
        state = {"final_response": "Short response"}
        assert policy.should_trigger(state) is False

    def test_custom_skip_keys(self):
        policy = DefaultTriggerPolicy(custom_skip_keys=["my_skip_flag"])
        state = {
            "final_response": "A" * 100,
            "my_skip_flag": True,
        }
        assert policy.should_trigger(state) is False

    def test_response_from_messages(self):
        """Test extracting response from messages when no direct key."""
        from langchain_core.messages import AIMessage

        policy = DefaultTriggerPolicy()
        state = {
            "messages": [AIMessage(content="A" * 100)],
        }
        assert policy.should_trigger(state) is True

    def test_checks_multiple_response_keys(self):
        """Test that various response keys are checked."""
        policy = DefaultTriggerPolicy()

        for key in ("final_response", "response", "output", "answer"):
            state = {key: "A" * 100}
            assert policy.should_trigger(state) is True, f"Failed for key: {key}"


class TestAlwaysTrigger:
    """Tests for AlwaysTrigger."""

    def test_always_returns_true(self):
        policy = AlwaysTrigger()
        assert policy.should_trigger({}) is True
        assert policy.should_trigger({"input_guardrail_fired": True}) is True


class TestNeverTrigger:
    """Tests for NeverTrigger."""

    def test_always_returns_false(self):
        policy = NeverTrigger()
        assert policy.should_trigger({}) is False
        assert policy.should_trigger({"final_response": "A" * 1000}) is False
