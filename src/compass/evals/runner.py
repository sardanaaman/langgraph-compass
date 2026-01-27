"""EvalRunner for orchestrating Compass evaluation runs.

This module provides the EvalRunner class that coordinates the evaluation
pipeline: loading examples from datasets, invoking Compass, running evaluators,
and computing aggregate metrics.

Example:
    >>> from langchain_openai import ChatOpenAI
    >>> from compass import CompassNode
    >>> from compass.evals import EvalRunner, MemoryDataset, ExactMatchEvaluator
    >>>
    >>> compass = CompassNode(model=ChatOpenAI())
    >>> runner = EvalRunner(compass=compass, evaluators=[ExactMatchEvaluator()])
    >>> dataset = MemoryDataset([
    ...     {"id": "1", "query": "Hello", "response": "Hi there!"}
    ... ])
    >>> results = runner.run(dataset)
    >>> print(results.aggregated.metrics)
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig

from compass import CompassNode, __version__
from compass.evals.datasets import Dataset
from compass.evals.evaluators import Evaluator
from compass.evals.metrics import (
    MeanScoreMetric,
    Metric,
    MetricRegistry,
    SuccessRateMetric,
)
from compass.evals.models import (
    AggregatedMetrics,
    EvalResults,
    Example,
    ExampleResult,
    RunMetadata,
)

if TYPE_CHECKING:
    pass


class EvalRunner:
    """Orchestrates evaluation runs for Compass.

    Coordinates the full evaluation pipeline: iterating over dataset examples,
    invoking CompassNode, running evaluators, and computing aggregate metrics.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from compass import CompassNode
        >>> from compass.evals import (
        ...     EvalRunner,
        ...     MemoryDataset,
        ...     ExactMatchEvaluator,
        ...     QuestionFormatEvaluator,
        ... )
        >>>
        >>> model = ChatOpenAI(model="gpt-4o-mini")
        >>> compass = CompassNode(model=model)
        >>> evaluators = [ExactMatchEvaluator(), QuestionFormatEvaluator()]
        >>> runner = EvalRunner(compass=compass, evaluators=evaluators)
        >>>
        >>> dataset = MemoryDataset([
        ...     {"id": "1", "query": "What is Python?", "response": "A language..."},
        ... ])
        >>> results = runner.run(dataset)
        >>> print(results.aggregated.metrics)
    """

    def __init__(
        self,
        compass: CompassNode,
        evaluators: list[Evaluator],
        metrics: list[Metric] | None = None,
        *,
        timeout_seconds: float = 30.0,
        continue_on_error: bool = True,
    ) -> None:
        """Initialize the EvalRunner.

        Args:
            compass: The CompassNode instance to evaluate.
            evaluators: List of evaluators to run on each Compass output.
            metrics: Optional list of metrics to compute. If None, uses default
                metrics (MeanScoreMetric for each evaluator + SuccessRateMetric).
            timeout_seconds: Maximum time allowed for each Compass invocation.
            continue_on_error: If True, capture errors per-example and continue.
                If False, raise exceptions immediately.
        """
        self.compass = compass
        self.evaluators = evaluators
        self.timeout_seconds = timeout_seconds
        self.continue_on_error = continue_on_error

        # Build metric registry
        self._registry = MetricRegistry()

        if metrics is not None:
            for metric in metrics:
                self._registry.register(metric)
        else:
            # Default metrics: mean score for each evaluator + success rate
            for evaluator in evaluators:
                self._registry.register(MeanScoreMetric(evaluator.name))
            self._registry.register(SuccessRateMetric())

    def run(
        self,
        dataset: Dataset,
        config: RunnableConfig | None = None,
    ) -> EvalResults:
        """Run evaluation on the dataset.

        Iterates over each example in the dataset, invokes Compass, runs all
        evaluators, and computes aggregate metrics.

        Args:
            dataset: The dataset of examples to evaluate.
            config: Optional RunnableConfig for LangSmith tracing.

        Returns:
            EvalResults containing metadata, per-example results, and aggregates.

        Example:
            >>> results = runner.run(dataset)
            >>> print(f"Success rate: {results.aggregated.metrics['success_rate']:.2%}")
        """
        run_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        run_start = time.perf_counter()

        per_example_results: list[ExampleResult] = []
        score_distribution: dict[str, list[float]] = defaultdict(list)

        for example in dataset:
            result = self._process_example(example, config)
            per_example_results.append(result)

            # Collect scores for distribution
            for eval_name, score in result.scores.items():
                score_distribution[eval_name].append(score)

        run_duration = time.perf_counter() - run_start

        # Compute aggregated metrics
        metrics = self._registry.compute_all(per_example_results)

        # Count successes/failures
        successful = sum(1 for r in per_example_results if r.error is None)
        failed = len(per_example_results) - successful

        aggregated = AggregatedMetrics(
            total_examples=len(per_example_results),
            successful_examples=successful,
            failed_examples=failed,
            metrics=metrics,
            score_distribution=dict(score_distribution),
        )

        # Build run metadata
        metadata = RunMetadata(
            run_id=run_id,
            timestamp=start_time,
            compass_version=__version__,
            compass_config=self._get_compass_config(),
            dataset_info=dataset.info,
            duration_seconds=run_duration,
        )

        return EvalResults(
            metadata=metadata,
            per_example=per_example_results,
            aggregated=aggregated,
        )

    def _process_example(
        self,
        example: Example,
        config: RunnableConfig | None,
    ) -> ExampleResult:
        """Process a single example through Compass and evaluators.

        Args:
            example: The example to process.
            config: Optional RunnableConfig for tracing.

        Returns:
            ExampleResult with output, scores, timing, and any error.
        """
        error: str | None = None
        compass_output: list[str] = []
        scores: dict[str, float] = {}

        # Invoke Compass with timing
        start = time.perf_counter()
        try:
            compass_output, duration_ms = self._invoke_compass(example, config)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            error = f"{type(e).__name__}: {e}"
            if not self.continue_on_error:
                raise

        # Run evaluators (only if no error during Compass invocation)
        if error is None:
            try:
                scores = self._evaluate_example(example, compass_output, config)
            except Exception as e:
                error = f"Evaluator error - {type(e).__name__}: {e}"
                if not self.continue_on_error:
                    raise

        return ExampleResult(
            example_id=example.id,
            query=example.query,
            response=example.response,
            compass_output=compass_output,
            expected=example.expected_followups,
            scores=scores,
            error=error,
            duration_ms=duration_ms,
        )

    def _invoke_compass(
        self,
        example: Example,
        config: RunnableConfig | None,
    ) -> tuple[list[str], float]:
        """Invoke Compass on an example and return output with timing.

        Args:
            example: The example to process.
            config: Optional RunnableConfig for tracing.

        Returns:
            Tuple of (list of generated follow-ups, duration in milliseconds).

        Raises:
            TimeoutError: If invocation exceeds timeout_seconds.
            Exception: Any exception from Compass (if continue_on_error=False).
        """
        # Build state dict for Compass
        state = self._build_state(example)

        start = time.perf_counter()

        # Note: Python's signal-based timeouts don't work well with threads.
        # For robust timeout handling, consider using concurrent.futures.
        # For now, we invoke directly and trust the model's timeout settings.
        result = self.compass(state, config)

        duration_ms = (time.perf_counter() - start) * 1000

        # Extract suggestions from result
        output_key = self.compass.output_key
        suggestions = result.get(output_key, [])

        # Ensure output is a list of strings
        if not isinstance(suggestions, list):
            suggestions = [suggestions] if suggestions else []
        suggestions = [str(s) for s in suggestions]

        return suggestions, duration_ms

    def _build_state(self, example: Example) -> dict[str, Any]:
        """Build a state dict from an example for Compass invocation.

        Args:
            example: The example to convert.

        Returns:
            State dict with query, response, and optionally messages.
        """
        state: dict[str, Any] = {
            "query": example.query,
            "response": example.response,
        }

        # Add context if provided (e.g., conversation history)
        if example.context:
            # If context contains messages, include them
            if "messages" in example.context:
                state["messages"] = example.context["messages"]
            # Merge other context items
            for key, value in example.context.items():
                if key != "messages" and key not in state:
                    state[key] = value

        return state

    def _evaluate_example(
        self,
        example: Example,
        compass_output: list[str],
        config: RunnableConfig | None,
    ) -> dict[str, float]:
        """Run all evaluators on a single example.

        Args:
            example: The evaluated example.
            compass_output: Generated follow-up questions from Compass.
            config: Optional RunnableConfig for tracing.

        Returns:
            Dict mapping evaluator names to scores.
        """
        scores: dict[str, float] = {}

        for evaluator in self.evaluators:
            score = evaluator.evaluate(compass_output, example, config)
            scores[evaluator.name] = score

        return scores

    def _get_compass_config(self) -> dict[str, Any]:
        """Extract configuration from the CompassNode for metadata.

        Returns:
            Dict with CompassNode configuration settings.
        """
        return {
            "strategy": self.compass.strategy,
            "max_suggestions": self.compass.max_suggestions,
            "output_key": self.compass.output_key,
            "inject_into_messages": self.compass.inject_into_messages,
            "generate_candidates": self.compass.generate_candidates,
        }

