"""Evaluator protocol and built-in evaluators for Compass evaluation framework.

This module provides:
- Evaluator: Protocol for scoring Compass outputs
- ExactMatchEvaluator: Checks for exact matches with expected questions
- QuestionFormatEvaluator: Validates question formatting (ends with ?)
- StarterPhraseEvaluator: Checks for approved starter phrases
- LLMJudgeEvaluator: Uses LLM to score quality on a rubric

Example:
    >>> from compass.evals import ExactMatchEvaluator, Example
    >>> evaluator = ExactMatchEvaluator()
    >>> example = Example(id="1", query="Hi", response="Hello!", expected_followups=["How are you?"])
    >>> score = evaluator.evaluate(["How are you?"], example)
    >>> assert score == 1.0
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from compass.evals.models import Example


DEFAULT_LLM_JUDGE_RUBRIC = """Rate the follow-up questions on a scale of 1-5:

1 (Poor): Generic, irrelevant, or doesn't add value
2 (Fair): Somewhat relevant but vague or obvious
3 (Good): Relevant and specific to the conversation
4 (Very Good): Insightful, opens valuable directions
5 (Excellent): Perfectly anticipates user needs, highly actionable

Consider:
- Relevance to the original query and response
- Specificity (references entities from conversation)
- Actionability (user can clearly respond)
- Value-add (not just restating what was said)
"""


class LLMJudgeResponse(BaseModel):
    """Structured output for LLM judge evaluation."""

    score: int = Field(ge=1, le=5, description="Score from 1-5 based on the rubric")
    reasoning: str = Field(description="Brief explanation for the score")


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluating Compass outputs.

    Implement this protocol to create custom evaluators that score
    the quality of generated follow-up questions.

    Example:
        >>> class MyEvaluator:
        ...     name = "my_evaluator"
        ...     def evaluate(self, compass_output, example, config=None):
        ...         return 1.0 if compass_output else 0.0
    """

    @property
    def name(self) -> str:
        """Unique name for this evaluator."""
        ...

    def evaluate(
        self,
        compass_output: list[str],
        example: "Example",
        config: RunnableConfig | None = None,
    ) -> float:
        """Score the output.

        Args:
            compass_output: List of generated follow-up questions
            example: The evaluation example containing query, response, etc.
            config: Optional RunnableConfig for LangSmith tracing

        Returns:
            Score between 0.0 and 1.0
        """
        ...


class ExactMatchEvaluator:
    """Checks if any generated question exactly matches any expected question.

    Returns 1.0 if at least one generated question matches an expected question
    (case-insensitive), 0.0 otherwise.

    Example:
        >>> evaluator = ExactMatchEvaluator()
        >>> # Assuming example has expected_followups=["How can I help?"]
        >>> score = evaluator.evaluate(["how can i help?"], example)
        >>> assert score == 1.0
    """

    name: str = "exact_match"

    def evaluate(
        self,
        compass_output: list[str],
        example: "Example",
        config: RunnableConfig | None = None,
    ) -> float:
        """Check for exact matches between output and expected questions."""
        if not compass_output:
            return 0.0

        expected = example.expected_followups
        if not expected:
            return 0.0

        # Normalize for case-insensitive comparison
        normalized_output = {q.strip().lower() for q in compass_output}
        normalized_expected = {q.strip().lower() for q in expected}

        # Return 1.0 if any match exists
        if normalized_output & normalized_expected:
            return 1.0
        return 0.0


class QuestionFormatEvaluator:
    """Checks if all outputs are well-formed questions (end with ?).

    Returns the fraction of outputs that end with a question mark.

    Example:
        >>> evaluator = QuestionFormatEvaluator()
        >>> score = evaluator.evaluate(["How are you?", "Tell me more"], example)
        >>> assert score == 0.5  # 1 of 2 ends with ?
    """

    name: str = "question_format"

    def evaluate(
        self,
        compass_output: list[str],
        example: "Example",
        config: RunnableConfig | None = None,
    ) -> float:
        """Check if outputs end with question marks."""
        if not compass_output:
            return 0.0

        well_formed = sum(1 for q in compass_output if q.strip().endswith("?"))
        return well_formed / len(compass_output)


class StarterPhraseEvaluator:
    """Checks if questions start with approved phrases.

    Returns the fraction of outputs that start with one of the specified
    starter phrases (case-insensitive).

    Example:
        >>> evaluator = StarterPhraseEvaluator(starters=["Would you", "Should I"])
        >>> score = evaluator.evaluate(["Would you like more info?", "Here's a fact"], example)
        >>> assert score == 0.5  # 1 of 2 starts with approved phrase
    """

    name: str = "starter_phrase"

    def __init__(self, starters: list[str]) -> None:
        """Initialize with approved starter phrases.

        Args:
            starters: List of approved starter phrases (case-insensitive)
        """
        self.starters = [s.lower() for s in starters]

    def evaluate(
        self,
        compass_output: list[str],
        example: "Example",
        config: RunnableConfig | None = None,
    ) -> float:
        """Check if outputs start with approved phrases."""
        if not compass_output:
            return 0.0

        matches = 0
        for question in compass_output:
            q_lower = question.strip().lower()
            if any(q_lower.startswith(starter) for starter in self.starters):
                matches += 1

        return matches / len(compass_output)


class LLMJudgeEvaluator:
    """Uses an LLM to judge quality of follow-up questions.

    Scores quality on a 1-5 scale based on a rubric, then normalizes to 0-1.
    Propagates RunnableConfig for LangSmith tracing.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> model = ChatOpenAI(model="gpt-4o-mini")
        >>> evaluator = LLMJudgeEvaluator(model=model)
        >>> score = evaluator.evaluate(["Would you like more details?"], example)
        >>> assert 0.0 <= score <= 1.0

    Custom rubric example:
        >>> evaluator = LLMJudgeEvaluator(
        ...     model=model,
        ...     rubric="Rate specificity 1-5: 1=generic, 5=highly specific",
        ...     criteria=["specificity", "relevance"]
        ... )
    """

    name: str = "llm_judge"

    def __init__(
        self,
        model: BaseChatModel,
        rubric: str | None = None,
        criteria: list[str] | None = None,
    ) -> None:
        """Initialize the LLM judge evaluator.

        Args:
            model: The chat model to use for judging
            rubric: Custom rubric text (uses default if None)
            criteria: Optional list of specific criteria to evaluate
        """
        self.model = model
        self.rubric = rubric or DEFAULT_LLM_JUDGE_RUBRIC
        self.criteria = criteria

    def evaluate(
        self,
        compass_output: list[str],
        example: "Example",
        config: RunnableConfig | None = None,
    ) -> float:
        """Score the output using LLM judgment."""
        if not compass_output:
            return 0.0

        # Build the evaluation prompt
        criteria_text = ""
        if self.criteria:
            criteria_text = f"\n\nFocus on these criteria: {', '.join(self.criteria)}"

        questions_text = "\n".join(f"- {q}" for q in compass_output)

        system_prompt = f"""You are an expert evaluator of conversational AI follow-up questions.

{self.rubric}{criteria_text}

Evaluate the follow-up questions and provide a score from 1-5."""

        user_prompt = f"""Original Query: {example.query}

Agent Response: {example.response}

Generated Follow-up Questions:
{questions_text}

Rate the quality of these follow-up questions based on the rubric."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Try structured output first, fall back to text parsing
        try:
            structured_model = self.model.with_structured_output(LLMJudgeResponse)
            result = structured_model.invoke(messages, config=config)
            if isinstance(result, LLMJudgeResponse):
                # Normalize 1-5 score to 0-1
                return (result.score - 1) / 4
        except (AttributeError, TypeError):
            pass

        # Fallback: invoke without structured output and parse
        response = self.model.invoke(messages, config=config)
        content = str(response.content) if hasattr(response, "content") else str(response)

        # Try to extract a score from the response
        score = self._parse_score(content)
        return (score - 1) / 4 if score else 0.5  # Default to middle score if parsing fails

    def _parse_score(self, content: str) -> int | None:
        """Parse a numeric score from LLM response text."""
        import re

        # Look for patterns like "Score: 4" or "4/5" or just a standalone digit 1-5
        patterns = [
            r"score[:\s]+(\d)",
            r"(\d)\s*/\s*5",
            r"rating[:\s]+(\d)",
            r"\b([1-5])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    return score

        return None

