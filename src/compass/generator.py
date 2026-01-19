"""Question generation with configurable strategies."""

from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

Strategy = Literal["adaptive", "clarifying", "exploratory", "deepening"]

STRATEGY_PROMPTS: dict[Strategy, str] = {
    "clarifying": (
        "Generate a clarifying follow-up question that helps ensure "
        "the user got what they needed. Focus on: missing details, ambiguities, "
        "or potential misunderstandings."
    ),
    "exploratory": (
        "Generate an exploratory follow-up question that opens "
        "new directions. Focus on: related topics, broader context, or "
        "alternative perspectives."
    ),
    "deepening": (
        "Generate a deepening follow-up question that goes deeper "
        "into the current topic. Focus on: underlying causes, specific examples, "
        "or detailed analysis."
    ),
    "adaptive": (
        "Based on the conversation, generate the most appropriate "
        "follow-up question. Choose between clarifying (if unclear), exploratory "
        "(if topic is exhausted), or deepening (if more detail would help)."
    ),
}

DEFAULT_STARTERS = [
    "Would you like me to",
    "Want me to",
    "Interested in",
    "Should I",
]


class GeneratedQuestions(BaseModel):
    """Structured output for generated questions."""

    questions: list[str] = Field(
        description="List of follow-up questions, each starting with an approved starter phrase"
    )


class QuestionGenerator:
    """Generates follow-up questions using configurable strategies.

    Example:
        >>> generator = QuestionGenerator(model, strategy="exploratory")
        >>> questions = generator.generate(
        ...     query="What is Python?",
        ...     response="Python is a programming language...",
        ...     n=3
        ... )
    """

    def __init__(
        self,
        model: BaseChatModel,
        strategy: Strategy = "adaptive",
        starters: list[str] | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model: The language model to use for generation.
            strategy: The question generation strategy.
            starters: Approved starter phrases for questions.
        """
        self.model = model
        self.strategy = strategy
        self.starters = starters or DEFAULT_STARTERS

    def generate(
        self,
        query: str,
        response: str,
        history: list[str] | None = None,
        examples: list[str] | None = None,
        n: int = 3,
        config: RunnableConfig | None = None,
    ) -> list[str]:
        """Generate candidate follow-up questions.

        Args:
            query: The user's original query.
            response: The assistant's response.
            history: Previous follow-ups to avoid repeating.
            examples: Example questions for inspiration.
            n: Number of questions to generate.
            config: LangGraph runnable config for tracing.

        Returns:
            List of generated follow-up questions.
        """
        messages = self._build_messages(query, response, history, examples, n)

        # Try structured output first, fall back to text parsing
        try:
            structured_model = self.model.with_structured_output(GeneratedQuestions)
            result = structured_model.invoke(messages, config=config)
            # Handle structured output result
            if isinstance(result, GeneratedQuestions):
                return list(result.questions[:n])
            # Fallback for dict response
            if isinstance(result, dict) and "questions" in result:
                return list(cast(list[str], result["questions"])[:n])
            raise AttributeError("Unexpected result type")
        except (AttributeError, NotImplementedError):
            # Model doesn't support structured output
            result = self.model.invoke(messages, config=config)
            content = result.content
            if isinstance(content, str):
                return self._parse_questions(content, n)
            return []

    def _build_messages(
        self,
        query: str,
        response: str,
        history: list[str] | None,
        examples: list[str] | None,
        n: int,
    ) -> list[SystemMessage | HumanMessage]:
        """Build the prompt messages for the model."""
        strategy_instruction = STRATEGY_PROMPTS[self.strategy]
        starters_str = ", ".join(f'"{s}"' for s in self.starters)

        system_prompt = f"""You are an expert at generating contextually relevant follow-up questions.

{strategy_instruction}

Requirements:
- Generate exactly {n} distinct follow-up questions
- Start each question with one of these phrases: {starters_str}
- Keep each question to one sentence
- Make questions actionable and specific
- Reference entities from the conversation when relevant
- Each question should offer genuine value to the user"""

        user_content_parts = [
            f"User's query: {query}",
            f"\nAssistant's response: {response[:1000]}{'...' if len(response) > 1000 else ''}",
        ]

        if history:
            user_content_parts.append(f"\nPrevious follow-ups to AVOID repeating: {history}")

        if examples:
            user_content_parts.append(
                f"\nExample questions for inspiration (adapt, don't copy): {examples}"
            )

        user_content_parts.append(f"\nGenerate {n} follow-up questions:")

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(user_content_parts)),
        ]

    def _parse_questions(self, content: str, n: int) -> list[str]:
        """Parse questions from unstructured text output."""
        lines = content.strip().split("\n")
        questions = []

        for line in lines:
            line = line.strip()
            # Remove common prefixes like "1.", "- ", etc.
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.)-] ").strip()
            elif line.startswith("-"):
                line = line[1:].strip()

            # Check if it starts with an approved starter
            if (
                any(line.startswith(starter) for starter in self.starters)
                or line
                and line.endswith("?")
            ):
                questions.append(line)

        return questions[:n]
