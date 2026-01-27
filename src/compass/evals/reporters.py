"""Report generators for Compass evaluation results.

Provides protocols and implementations for generating human-readable
and machine-readable reports from evaluation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from compass.evals.models import EvalResults, ExampleResult


@runtime_checkable
class Reporter(Protocol):
    """Protocol for generating evaluation reports.

    Reporters transform EvalResults into formatted output (e.g., Markdown, JSON)
    for analysis and sharing.

    Example:
        >>> class MyReporter:
        ...     def generate(self, results: EvalResults) -> str:
        ...         return f"Total examples: {results.aggregated.total_examples}"
        ...     def save(self, results: EvalResults, path: str | Path) -> None:
        ...         Path(path).write_text(self.generate(results))
    """

    def generate(self, results: EvalResults) -> str:
        """Generate report content as a string.

        Args:
            results: The evaluation results to report on.

        Returns:
            Formatted report content.
        """
        ...

    def save(self, results: EvalResults, path: str | Path) -> None:
        """Save report to a file.

        Args:
            results: The evaluation results to report on.
            path: File path to save the report to.
        """
        ...


def _calculate_percentiles(
    scores: list[float], percentiles: list[int]
) -> dict[str, float]:
    """Calculate percentiles from a list of scores without numpy.

    Args:
        scores: List of numeric scores.
        percentiles: List of percentile values to compute (e.g., [25, 50, 75, 95]).

    Returns:
        Dictionary mapping percentile labels (e.g., "P25") to values.

    Example:
        >>> _calculate_percentiles([1, 2, 3, 4, 5], [25, 50, 75])
        {'P25': 2.0, 'P50': 3.0, 'P75': 4.0}
    """
    if not scores:
        return {f"P{p}": 0.0 for p in percentiles}

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    result = {}

    for p in percentiles:
        # Linear interpolation method
        index = (p / 100) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        fraction = index - lower
        value = sorted_scores[lower] * (1 - fraction) + sorted_scores[upper] * fraction
        result[f"P{p}"] = round(value, 4)

    return result


def _get_primary_score(result: ExampleResult) -> float:
    """Get the primary score for sorting failing examples.

    Uses the first evaluator's score if available, otherwise 0.0.

    Args:
        result: An example result with scores.

    Returns:
        The primary score for ranking.
    """
    if not result.scores:
        return 0.0
    # Return the first score (or average if multiple)
    return sum(result.scores.values()) / len(result.scores)


class MarkdownReporter:
    """Generates human-readable Markdown reports from evaluation results.

    Creates a structured report with run summary, metrics tables,
    score distributions, failing examples, and error breakdowns.

    Example:
        >>> reporter = MarkdownReporter(include_examples=5)
        >>> report = reporter.generate(results)
        >>> reporter.save(results, "eval_report.md")
    """

    def __init__(
        self,
        include_examples: int = 10,
        include_distribution: bool = True,
    ) -> None:
        """Initialize the Markdown reporter.

        Args:
            include_examples: Maximum number of failing examples to include
                in the report, sorted by lowest score. Defaults to 10.
            include_distribution: Whether to include score distribution
                (percentiles) section. Defaults to True.
        """
        self.include_examples = include_examples
        self.include_distribution = include_distribution

    def generate(self, results: EvalResults) -> str:
        """Generate a Markdown report from evaluation results.

        Args:
            results: The evaluation results to report on.

        Returns:
            Complete Markdown report as a string.
        """
        sections = [
            self._generate_header(),
            self._generate_run_summary(results),
            self._generate_overall_metrics(results),
        ]

        if self.include_distribution:
            sections.append(self._generate_score_distribution(results))

        sections.append(self._generate_failing_examples(results))
        sections.append(self._generate_error_breakdown(results))

        return "\n".join(sections)

    def save(self, results: EvalResults, path: str | Path) -> None:
        """Save the Markdown report to a file.

        Args:
            results: The evaluation results to report on.
            path: File path to save the report to.
        """
        content = self.generate(results)
        Path(path).write_text(content, encoding="utf-8")

    def _generate_header(self) -> str:
        """Generate the report header."""
        return "# Compass Evaluation Report\n"

    def _generate_run_summary(self, results: EvalResults) -> str:
        """Generate the run summary section."""
        meta = results.metadata
        dataset_name = meta.dataset_info.get("name", "Unknown")
        dataset_size = meta.dataset_info.get("size", results.aggregated.total_examples)

        lines = [
            "## Run Summary",
            f"- **Run ID**: {meta.run_id}",
            f"- **Timestamp**: {meta.timestamp.isoformat()}",
            f"- **Compass Version**: {meta.compass_version}",
            f"- **Dataset**: {dataset_name} ({dataset_size} examples)",
            f"- **Duration**: {meta.duration_seconds:.1f} seconds",
            "",
        ]
        return "\n".join(lines)

    def _generate_overall_metrics(self, results: EvalResults) -> str:
        """Generate the overall metrics table."""
        lines = [
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        agg = results.aggregated
        # Add success rate first
        if agg.total_examples > 0:
            success_rate = agg.successful_examples / agg.total_examples * 100
            lines.append(f"| Success Rate | {success_rate:.1f}% |")

        # Add other metrics
        for name, value in sorted(agg.metrics.items()):
            if isinstance(value, float):
                lines.append(f"| {name} | {value:.4f} |")
            else:
                lines.append(f"| {name} | {value} |")

        lines.append("")
        return "\n".join(lines)

    def _generate_score_distribution(self, results: EvalResults) -> str:
        """Generate the score distribution section."""
        lines = ["## Score Distribution", ""]

        distributions = results.aggregated.score_distribution
        if not distributions:
            lines.append("*No score distributions available.*\n")
            return "\n".join(lines)

        percentiles = [25, 50, 75, 95]

        for evaluator_name, scores in sorted(distributions.items()):
            if not scores:
                continue
            pcts = _calculate_percentiles(scores, percentiles)
            lines.append(f"### {evaluator_name}")
            for label, value in pcts.items():
                lines.append(f"- **{label}**: {value:.4f}")
            lines.append("")

        return "\n".join(lines)

    def _generate_failing_examples(self, results: EvalResults) -> str:
        """Generate the top failing examples section."""
        lines = ["## Top Failing Examples", ""]

        if not results.per_example:
            lines.append("*No examples to display.*\n")
            return "\n".join(lines)

        # Sort by lowest score first
        sorted_examples = sorted(
            results.per_example, key=lambda r: _get_primary_score(r)
        )

        # Take up to include_examples
        failing = sorted_examples[: self.include_examples]

        if not failing:
            lines.append("*No failing examples.*\n")
            return "\n".join(lines)

        for result in failing:
            score = _get_primary_score(result)
            lines.append(f"### Example: {result.example_id} (Score: {score:.2f})")
            lines.append(f"**Query**: {result.query}")
            lines.append(f"**Response**: {result.response[:200]}...")
            lines.append(f"**Generated**: {result.compass_output}")
            if result.expected:
                lines.append(f"**Expected**: {result.expected}")
            lines.append(f"**Error**: {result.error or 'None'}")
            lines.append("")

        return "\n".join(lines)

    def _generate_error_breakdown(self, results: EvalResults) -> str:
        """Generate the error breakdown table."""
        lines = ["## Error Breakdown", ""]

        # Collect error types and counts
        error_counts: dict[str, int] = {}
        for result in results.per_example:
            if result.error:
                # Extract error type from message (e.g., "TimeoutError: ...")
                error_type = result.error.split(":")[0].strip()
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        if not error_counts:
            lines.append("*No errors encountered.*\n")
            return "\n".join(lines)

        lines.extend([
            "| Error Type | Count |",
            "|------------|-------|",
        ])

        for error_type, count in sorted(
            error_counts.items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {error_type} | {count} |")

        lines.append("")
        return "\n".join(lines)


class JSONReporter:
    """Generates machine-readable JSON reports from evaluation results.

    Uses Pydantic's model_dump_json() for serialization, ensuring
    proper handling of all data types including datetime.

    Example:
        >>> reporter = JSONReporter(indent=2)
        >>> json_str = reporter.generate(results)
        >>> reporter.save(results, "eval_results.json")
    """

    def __init__(self, indent: int = 2) -> None:
        """Initialize the JSON reporter.

        Args:
            indent: Number of spaces for JSON indentation.
                Defaults to 2 for readable output.
        """
        self.indent = indent

    def generate(self, results: EvalResults) -> str:
        """Generate a JSON report from evaluation results.

        Args:
            results: The evaluation results to report on.

        Returns:
            JSON string representation of the results.
        """
        return results.model_dump_json(indent=self.indent)

    def save(self, results: EvalResults, path: str | Path) -> None:
        """Save the JSON report to a file.

        Args:
            results: The evaluation results to report on.
            path: File path to save the report to.
        """
        content = self.generate(results)
        Path(path).write_text(content, encoding="utf-8")

