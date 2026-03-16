"""Pydantic data models for the Compass evaluation framework."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Example(BaseModel):
    """A single evaluation example.

    Represents an input case for evaluating Compass follow-up question generation.
    Contains the query/response pair and optional expected outputs for comparison.

    Example:
        >>> example = Example(
        ...     id="ex-001",
        ...     query="What is Python?",
        ...     response="Python is a programming language...",
        ...     expected_followups=["Want me to explain Python's key features?"],
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique identifier for this example")
    query: str = Field(description="User query/input", min_length=1)
    response: str = Field(description="Agent response to evaluate against", min_length=1)
    expected_followups: list[str] | None = Field(
        default=None,
        description="Expected follow-up questions (for exact/similarity matching)",
    )
    labels: dict[str, Any] | None = Field(
        default=None,
        description="Labels for classification (e.g., {'quality': 'good', 'strategy': 'clarifying'})",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context (conversation history, metadata)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata for filtering/grouping",
    )

    @field_validator("query", "response")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Validate that query and response are non-empty strings."""
        if not v.strip():
            raise ValueError("must be a non-empty string (not just whitespace)")
        return v


class ExampleResult(BaseModel):
    """Result of evaluating a single example.

    Captures the Compass output for an example along with scores from all
    evaluators and any errors that occurred during evaluation.

    Example:
        >>> result = ExampleResult(
        ...     example_id="ex-001",
        ...     query="What is Python?",
        ...     response="Python is a programming language...",
        ...     compass_output=["Want me to show some Python code examples?"],
        ...     expected=["Want me to explain Python's key features?"],
        ...     scores={"llm_judge": 0.8, "semantic_similarity": 0.75},
        ...     duration_ms=1250.5,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(description="ID of the evaluated example")
    query: str = Field(description="Original user query")
    response: str = Field(description="Original agent response")
    compass_output: list[str] = Field(description="Generated follow-up questions")
    expected: list[str] | None = Field(
        default=None, description="Expected follow-up questions (if provided)"
    )
    scores: dict[str, float] = Field(
        description="Evaluator name -> score mapping (scores typically 0.0-1.0)"
    )
    error: str | None = Field(
        default=None, description="Error message if evaluation failed"
    )
    duration_ms: float = Field(
        description="Time taken for Compass invocation in milliseconds", ge=0
    )


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all examples in an evaluation run.

    Provides summary statistics including success/failure counts and
    metric distributions across all evaluated examples.

    Example:
        >>> metrics = AggregatedMetrics(
        ...     total_examples=100,
        ...     successful_examples=98,
        ...     failed_examples=2,
        ...     metrics={"llm_judge_mean": 0.72, "success_rate": 0.98},
        ...     score_distribution={"llm_judge": [0.6, 0.7, 0.8, 0.9]},
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    total_examples: int = Field(description="Total number of examples evaluated", ge=0)
    successful_examples: int = Field(
        description="Number of examples that completed without error", ge=0
    )
    failed_examples: int = Field(
        description="Number of examples that errored during evaluation", ge=0
    )
    metrics: dict[str, float] = Field(
        description="Computed metric name -> value mapping"
    )
    score_distribution: dict[str, list[float]] = Field(
        description="Evaluator name -> list of all scores for distribution analysis"
    )

    @field_validator("failed_examples")
    @classmethod
    def validate_counts_consistent(cls, v: int, info: Any) -> int:
        """Validate that successful + failed = total."""
        if "total_examples" in info.data and "successful_examples" in info.data:
            total = info.data["total_examples"]
            successful = info.data["successful_examples"]
            if successful + v != total:
                raise ValueError(
                    f"successful_examples ({successful}) + failed_examples ({v}) "
                    f"must equal total_examples ({total})"
                )
        return v


class RunMetadata(BaseModel):
    """Metadata about an evaluation run.

    Captures configuration and timing information to enable reproducibility
    and comparison between evaluation runs.

    Example:
        >>> metadata = RunMetadata(
        ...     run_id="eval-2024-01-27-abc123",
        ...     timestamp=datetime.now(),
        ...     compass_version="1.0.1",
        ...     compass_config={"strategy": "adaptive", "max_suggestions": 3},
        ...     dataset_info={"name": "my_dataset", "size": 500},
        ...     duration_seconds=45.2,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Unique identifier for this evaluation run")
    timestamp: datetime = Field(description="When the evaluation run started")
    compass_version: str = Field(description="Version of Compass used for evaluation")
    compass_config: dict[str, Any] = Field(
        description="Configuration used for CompassNode (strategy, max_suggestions, etc.)"
    )
    dataset_info: dict[str, Any] = Field(
        description="Information about the dataset (name, source, size, etc.)"
    )
    duration_seconds: float = Field(
        description="Total duration of the evaluation run in seconds", ge=0
    )


class EvalResults(BaseModel):
    """Complete evaluation results combining metadata, per-example results, and aggregates.

    This is the top-level result object returned by EvalRunner.run() and saved
    to disk by reporters. Contains all information needed to analyze and
    reproduce an evaluation run.

    Example:
        >>> results = EvalResults(
        ...     metadata=run_metadata,
        ...     per_example=[result1, result2, result3],
        ...     aggregated=aggregated_metrics,
        ... )
        >>> results.model_dump_json(indent=2)  # Save to JSON
    """

    model_config = ConfigDict(extra="forbid")

    metadata: RunMetadata = Field(description="Metadata about the evaluation run")
    per_example: list[ExampleResult] = Field(
        description="Results for each evaluated example"
    )
    aggregated: AggregatedMetrics = Field(
        description="Aggregated metrics across all examples"
    )
