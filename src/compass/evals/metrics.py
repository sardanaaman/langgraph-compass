"""Metrics for aggregating evaluation results.

This module provides a protocol for defining metrics, built-in metric
implementations, and a registry for managing metrics.

Example:
    >>> from compass.evals.metrics import MetricRegistry, MeanScoreMetric, SuccessRateMetric
    >>> registry = MetricRegistry()
    >>> registry.register(MeanScoreMetric("llm_judge"))
    >>> registry.register(SuccessRateMetric())
    >>> metrics = registry.compute_all(results)
    >>> print(metrics)
    {'llm_judge_mean': 0.75, 'success_rate': 0.98}
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from compass.evals.models import ExampleResult


@runtime_checkable
class Metric(Protocol):
    """Protocol for computing aggregate metrics from evaluation results.

    Implement this protocol to create custom metrics that aggregate
    scores across all evaluated examples.

    Example:
        >>> class MyCustomMetric:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_custom_metric"
        ...
        ...     def compute(self, results: list[ExampleResult]) -> float:
        ...         # Custom aggregation logic
        ...         return 0.5
    """

    @property
    def name(self) -> str:
        """Unique name for this metric.

        Returns:
            A string identifier used as the key in metric results.
        """
        ...

    def compute(self, results: list[ExampleResult]) -> float:
        """Compute the metric from all evaluation results.

        Args:
            results: List of evaluation results from all examples.

        Returns:
            The computed metric value, typically between 0.0 and 1.0.
        """
        ...


class MeanScoreMetric:
    """Computes the mean score for a specific evaluator across all results.

    Skips examples that have errors (where error is not None).

    Example:
        >>> metric = MeanScoreMetric("llm_judge")
        >>> metric.name
        'llm_judge_mean'
        >>> metric.compute(results)  # Returns average llm_judge score
        0.75
    """

    def __init__(self, evaluator_name: str) -> None:
        """Initialize the mean score metric.

        Args:
            evaluator_name: Name of the evaluator whose scores to average.
        """
        self._evaluator_name = evaluator_name

    @property
    def name(self) -> str:
        """Return the metric name as '{evaluator_name}_mean'."""
        return f"{self._evaluator_name}_mean"

    def compute(self, results: list[ExampleResult]) -> float:
        """Compute mean score, skipping examples with errors or missing scores.

        Args:
            results: List of evaluation results.

        Returns:
            Mean score for the evaluator, or 0.0 if no valid scores found.
        """
        if not results:
            return 0.0

        scores: list[float] = []
        for result in results:
            # Skip examples with errors
            if result.error is not None:
                continue
            # Skip examples missing this evaluator's score
            if self._evaluator_name not in result.scores:
                continue
            scores.append(result.scores[self._evaluator_name])

        if not scores:
            return 0.0

        return sum(scores) / len(scores)


class AccuracyMetric:
    """Computes fraction of examples scoring at or above a threshold.

    Skips examples that have errors (where error is not None).

    Example:
        >>> metric = AccuracyMetric("llm_judge", threshold=0.7)
        >>> metric.name
        'llm_judge_accuracy'
        >>> metric.compute(results)  # Fraction with llm_judge >= 0.7
        0.85
    """

    def __init__(self, evaluator_name: str, threshold: float = 0.5) -> None:
        """Initialize the accuracy metric.

        Args:
            evaluator_name: Name of the evaluator to check scores for.
            threshold: Minimum score to count as "accurate". Defaults to 0.5.
        """
        self._evaluator_name = evaluator_name
        self._threshold = threshold

    @property
    def name(self) -> str:
        """Return the metric name as '{evaluator_name}_accuracy'."""
        return f"{self._evaluator_name}_accuracy"

    def compute(self, results: list[ExampleResult]) -> float:
        """Compute fraction of examples meeting the threshold.

        Args:
            results: List of evaluation results.

        Returns:
            Fraction of non-error examples with score >= threshold,
            or 0.0 if no valid examples.
        """
        if not results:
            return 0.0

        passing = 0
        total_valid = 0

        for result in results:
            # Skip examples with errors
            if result.error is not None:
                continue
            # Skip examples missing this evaluator's score
            if self._evaluator_name not in result.scores:
                continue

            total_valid += 1
            if result.scores[self._evaluator_name] >= self._threshold:
                passing += 1

        if total_valid == 0:
            return 0.0

        return passing / total_valid


class SuccessRateMetric:
    """Computes the fraction of examples that completed without errors.

    Example:
        >>> metric = SuccessRateMetric()
        >>> metric.name
        'success_rate'
        >>> metric.compute(results)  # Fraction without errors
        0.98
    """

    @property
    def name(self) -> str:
        """Return the metric name 'success_rate'."""
        return "success_rate"

    def compute(self, results: list[ExampleResult]) -> float:
        """Compute fraction of examples without errors.

        Args:
            results: List of evaluation results.

        Returns:
            Fraction of examples where error is None, or 0.0 if empty.
        """
        if not results:
            return 0.0

        successful = sum(1 for r in results if r.error is None)
        return successful / len(results)


class MetricRegistry:
    """Registry for managing and computing metrics.

    Provides a centralized way to register metrics and compute all
    registered metrics at once.

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register(MeanScoreMetric("llm_judge"))
        >>> registry.register(SuccessRateMetric())
        >>> registry.list_metrics()
        ['llm_judge_mean', 'success_rate']
        >>> metrics = registry.compute_all(results)
        {'llm_judge_mean': 0.75, 'success_rate': 0.98}
    """

    def __init__(self) -> None:
        """Initialize an empty metric registry."""
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        """Register a metric with the registry.

        Args:
            metric: A metric instance implementing the Metric protocol.

        Example:
            >>> registry = MetricRegistry()
            >>> registry.register(MeanScoreMetric("exact_match"))
        """
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        """Retrieve a metric by name.

        Args:
            name: The name of the metric to retrieve.

        Returns:
            The registered metric instance.

        Raises:
            KeyError: If no metric with the given name is registered.

        Example:
            >>> registry = MetricRegistry()
            >>> registry.register(SuccessRateMetric())
            >>> metric = registry.get("success_rate")
        """
        if name not in self._metrics:
            raise KeyError(
                f"Metric '{name}' not found. "
                f"Available metrics: {self.list_metrics()}"
            )
        return self._metrics[name]

    def compute_all(self, results: list[ExampleResult]) -> dict[str, float]:
        """Compute all registered metrics.

        Args:
            results: List of evaluation results.

        Returns:
            Dictionary mapping metric names to computed values.

        Example:
            >>> registry = MetricRegistry()
            >>> registry.register(MeanScoreMetric("llm_judge"))
            >>> registry.register(SuccessRateMetric())
            >>> registry.compute_all(results)
            {'llm_judge_mean': 0.75, 'success_rate': 0.98}
        """
        return {name: metric.compute(results) for name, metric in self._metrics.items()}

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns:
            List of registered metric names in insertion order.

        Example:
            >>> registry = MetricRegistry()
            >>> registry.register(SuccessRateMetric())
            >>> registry.list_metrics()
            ['success_rate']
        """
        return list(self._metrics.keys())

