"""CompassNode: LangGraph node for intelligent follow-up question generation."""

from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from compass.generator import QuestionGenerator, Strategy
from compass.helpers import extract_previous_followups
from compass.ranker import SuggestionRanker
from compass.triggers import DefaultTriggerPolicy, TriggerPolicy


@runtime_checkable
class ExampleRetriever(Protocol):
    """Protocol for retrieving example questions.

    Implement this to provide domain-specific example questions
    that inspire more relevant follow-ups.
    """

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve example questions relevant to the query."""
        ...


class CompassNode:
    """LangGraph node for generating intelligent follow-up questions.

    Integrates seamlessly into any LangGraph workflow, following the same
    pattern as SummarizationNode from langmem.

    Example:
        >>> from langgraph.graph import StateGraph, MessagesState, START, END
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> compass = CompassNode(model=ChatOpenAI())
        >>>
        >>> builder = StateGraph(MessagesState)
        >>> builder.add_node("agent", your_agent)
        >>> builder.add_node("compass", compass)
        >>> builder.add_edge(START, "agent")
        >>> builder.add_edge("agent", "compass")
        >>> builder.add_edge("compass", END)
        >>>
        >>> graph = builder.compile()
        >>> result = graph.invoke({"messages": [HumanMessage("Hello")]})
        >>> print(result.get("compass_suggestions"))
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        # Trigger configuration
        trigger: TriggerPolicy | None = None,
        # Generation configuration
        strategy: Strategy = "adaptive",
        max_suggestions: int = 1,
        starters: list[str] | None = None,
        # Output configuration
        output_key: str = "compass_suggestions",
        inject_into_messages: bool = False,
        # State field mapping
        query_key: str | None = None,
        response_key: str | None = None,
        messages_key: str = "messages",
        # Advanced
        example_retriever: ExampleRetriever | None = None,
        generate_candidates: int = 3,
    ) -> None:
        """Initialize the CompassNode.

        Args:
            model: Language model for generating questions.
            trigger: Policy determining when to generate (default: DefaultTriggerPolicy).
            strategy: Question generation strategy.
            max_suggestions: Maximum suggestions to return.
            starters: Approved starter phrases for questions.
            output_key: State key for storing suggestions.
            inject_into_messages: Also add suggestions to messages.
            query_key: State key for user query (auto-detected if None).
            response_key: State key for response (auto-detected if None).
            messages_key: State key for message history.
            example_retriever: Optional retriever for example questions.
            generate_candidates: Number of candidates to generate before ranking.
        """
        self.model = model
        self.trigger = trigger or DefaultTriggerPolicy()
        self.strategy = strategy
        self.max_suggestions = max_suggestions
        self.output_key = output_key
        self.inject_into_messages = inject_into_messages
        self.query_key = query_key
        self.response_key = response_key
        self.messages_key = messages_key
        self.example_retriever = example_retriever
        self.generate_candidates = generate_candidates

        self._generator = QuestionGenerator(
            model=model,
            strategy=strategy,
            starters=starters,
        )
        self._ranker = SuggestionRanker()

    def __call__(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Invoke the node - follows LangGraph node protocol.

        Args:
            state: Current graph state.
            config: LangGraph runnable config for tracing.

        Returns:
            State update with suggestions.
        """
        # Check trigger condition
        if not self.trigger.should_trigger(state):
            return {self.output_key: []}

        # Extract context from state
        query = self._get_query(state)
        response = self._get_response(state)

        if not query or not response:
            return {self.output_key: []}

        # Get previous follow-ups for novelty filtering
        messages = state.get(self.messages_key, [])
        previous = extract_previous_followups(messages, starters=self._generator.starters)

        # Retrieve examples if retriever is configured
        examples = None
        if self.example_retriever:
            examples = self.example_retriever.retrieve(query)

        # Generate candidate questions
        candidates = self._generator.generate(
            query=query,
            response=response,
            history=previous,
            examples=examples,
            n=self.generate_candidates,
            config=config,
        )

        # Rank and filter
        ranked = self._ranker.rank(candidates, previous)
        suggestions = ranked[: self.max_suggestions]

        # Build result
        result: dict[str, Any] = {self.output_key: suggestions}

        # Optionally inject into messages
        if self.inject_into_messages and suggestions:
            result[self.messages_key] = [
                AIMessage(
                    content="",
                    additional_kwargs={"compass_suggestions": suggestions},
                )
            ]

        return result

    def _get_query(self, state: dict[str, Any]) -> str:
        """Extract the user query from state."""
        # Use explicit key if configured
        if self.query_key and self.query_key in state:
            return str(state[self.query_key])

        # Check common query keys
        for key in ("query", "question", "input", "user_input"):
            val = state.get(key)
            if isinstance(val, str):
                return val

        # Fall back to first human message
        messages = state.get(self.messages_key, [])
        for msg in messages:
            if getattr(msg, "type", None) == "human":
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    return content

        return ""

    def _get_response(self, state: dict[str, Any]) -> str:
        """Extract the assistant response from state."""
        # Use explicit key if configured
        if self.response_key and self.response_key in state:
            return str(state[self.response_key])

        # Check common response keys
        for key in ("final_response", "response", "output", "answer"):
            val = state.get(key)
            if isinstance(val, str):
                return val

        # Fall back to last AI message
        messages = state.get(self.messages_key, [])
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    return content

        return ""
