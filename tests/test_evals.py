"""Tests for the Compass evaluation framework."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from compass.evals.datasets import (
    CSVDataset,
    DatasetValidationError,
    IteratorDataset,
    JSONLDataset,
    MemoryDataset,
)
from compass.evals.evaluators import (
    ExactMatchEvaluator,
    QuestionFormatEvaluator,
    StarterPhraseEvaluator,
)
from compass.evals.metrics import (
    AccuracyMetric,
    MeanScoreMetric,
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
from compass.evals.reporters import JSONReporter, MarkdownReporter
from compass.evals.runner import EvalRunner


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_example() -> Example:
    """A valid example for testing."""
    return Example(
        id="test-001",
        query="What is Python?",
        response="Python is a programming language.",
        expected_followups=["Would you like examples?", "Should I explain more?"],
    )


@pytest.fixture
def sample_example_dict() -> dict:
    """A valid example as a dict."""
    return {
        "id": "test-002",
        "query": "How do I learn?",
        "response": "Start with basics.",
    }


@pytest.fixture
def sample_result(sample_example: Example) -> ExampleResult:
    """A valid example result for testing."""
    return ExampleResult(
        example_id=sample_example.id,
        query=sample_example.query,
        response=sample_example.response,
        compass_output=["Would you like examples?"],
        expected=sample_example.expected_followups,
        scores={"exact_match": 1.0, "question_format": 1.0},
        duration_ms=100.5,
    )


@pytest.fixture
def sample_results_list(sample_result: ExampleResult) -> list[ExampleResult]:
    """Multiple results for metric testing."""
    # Create a mix of successful and failed results
    results = [
        sample_result,
        ExampleResult(
            example_id="test-002",
            query="Q2",
            response="R2",
            compass_output=["Output"],
            scores={"exact_match": 0.5, "question_format": 1.0},
            duration_ms=50.0,
        ),
        ExampleResult(
            example_id="test-003",
            query="Q3",
            response="R3",
            compass_output=[],
            scores={"exact_match": 0.0, "question_format": 0.0},
            error="TestError: Something failed",
            duration_ms=10.0,
        ),
    ]
    return results


@pytest.fixture
def sample_run_metadata() -> RunMetadata:
    """Valid run metadata for testing."""
    return RunMetadata(
        run_id="run-001",
        timestamp=datetime.now(timezone.utc),
        compass_version="1.0.0",
        compass_config={"strategy": "adaptive", "max_suggestions": 3},
        dataset_info={"name": "test_dataset", "size": 10},
        duration_seconds=5.5,
    )


@pytest.fixture
def sample_aggregated_metrics() -> AggregatedMetrics:
    """Valid aggregated metrics for testing."""
    return AggregatedMetrics(
        total_examples=10,
        successful_examples=9,
        failed_examples=1,
        metrics={"exact_match_mean": 0.75, "success_rate": 0.9},
        score_distribution={"exact_match": [0.5, 0.75, 1.0]},
    )


@pytest.fixture
def sample_eval_results(
    sample_run_metadata: RunMetadata,
    sample_results_list: list[ExampleResult],
    sample_aggregated_metrics: AggregatedMetrics,
) -> EvalResults:
    """Complete eval results for testing reporters."""
    return EvalResults(
        metadata=sample_run_metadata,
        per_example=sample_results_list,
        aggregated=sample_aggregated_metrics,
    )


# ============================================================================
# Test Models
# ============================================================================


class TestExample:
    """Tests for the Example data model."""

    def test_valid_example(self, sample_example: Example) -> None:
        """Test creating a valid example with all fields."""
        assert sample_example.id == "test-001"
        assert sample_example.query == "What is Python?"
        assert len(sample_example.expected_followups) == 2

    def test_minimal_example(self) -> None:
        """Test creating an example with only required fields."""
        example = Example(id="min", query="Question?", response="Answer.")
        assert example.expected_followups is None
        assert example.labels is None
        assert example.context is None

    def test_empty_query_fails(self) -> None:
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError):
            Example(id="fail", query="", response="Valid response")

    def test_whitespace_only_query_fails(self) -> None:
        """Test that whitespace-only query raises validation error."""
        with pytest.raises(ValidationError):
            Example(id="fail", query="   ", response="Valid response")

    def test_empty_response_fails(self) -> None:
        """Test that empty response raises validation error."""
        with pytest.raises(ValidationError):
            Example(id="fail", query="Valid query", response="")

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            Example(
                id="fail",
                query="Q",
                response="R",
                unknown_field="not allowed",  # type: ignore
            )


class TestExampleResult:
    """Tests for the ExampleResult data model."""

    def test_valid_result(self, sample_result: ExampleResult) -> None:
        """Test creating a valid example result."""
        assert sample_result.example_id == "test-001"
        assert sample_result.duration_ms == 100.5
        assert len(sample_result.scores) == 2

    def test_result_with_error(self) -> None:
        """Test creating a result with an error."""
        result = ExampleResult(
            example_id="err-001",
            query="Q",
            response="R",
            compass_output=[],
            scores={},
            error="Timeout occurred",
            duration_ms=1000.0,
        )
        assert result.error == "Timeout occurred"

    def test_negative_duration_fails(self) -> None:
        """Test that negative duration raises validation error."""
        with pytest.raises(ValidationError):
            ExampleResult(
                example_id="fail",
                query="Q",
                response="R",
                compass_output=[],
                scores={},
                duration_ms=-1.0,
            )


class TestAggregatedMetrics:
    """Tests for the AggregatedMetrics data model."""

    def test_valid_aggregated_metrics(
        self, sample_aggregated_metrics: AggregatedMetrics
    ) -> None:
        """Test creating valid aggregated metrics."""
        assert sample_aggregated_metrics.total_examples == 10
        assert sample_aggregated_metrics.successful_examples == 9
        assert sample_aggregated_metrics.failed_examples == 1

    def test_counts_must_sum(self) -> None:
        """Test that successful + failed must equal total."""
        with pytest.raises(ValidationError):
            AggregatedMetrics(
                total_examples=10,
                successful_examples=5,
                failed_examples=3,  # 5+3 != 10
                metrics={},
                score_distribution={},
            )


class TestRunMetadata:
    """Tests for the RunMetadata data model."""

    def test_valid_metadata(self, sample_run_metadata: RunMetadata) -> None:
        """Test creating valid run metadata."""
        assert sample_run_metadata.run_id == "run-001"
        assert sample_run_metadata.compass_version == "1.0.0"

    def test_negative_duration_fails(self) -> None:
        """Test that negative duration raises validation error."""
        with pytest.raises(ValidationError):
            RunMetadata(
                run_id="fail",
                timestamp=datetime.now(timezone.utc),
                compass_version="1.0.0",
                compass_config={},
                dataset_info={},
                duration_seconds=-1.0,
            )


class TestEvalResults:
    """Tests for the EvalResults data model."""

    def test_valid_eval_results(self, sample_eval_results: EvalResults) -> None:
        """Test creating valid eval results."""
        assert sample_eval_results.metadata.run_id == "run-001"
        assert len(sample_eval_results.per_example) == 3

    def test_serialization_roundtrip(self, sample_eval_results: EvalResults) -> None:
        """Test that results can be serialized and deserialized."""
        json_str = sample_eval_results.model_dump_json()
        parsed = EvalResults.model_validate_json(json_str)
        assert parsed.metadata.run_id == sample_eval_results.metadata.run_id



# ============================================================================
# Test Datasets
# ============================================================================


class TestMemoryDataset:
    """Tests for the MemoryDataset class."""

    def test_from_dicts(self, sample_example_dict: dict) -> None:
        """Test creating dataset from dict list."""
        dataset = MemoryDataset([sample_example_dict])
        assert len(dataset) == 1
        examples = list(dataset)
        assert examples[0].id == "test-002"

    def test_from_examples(self, sample_example: Example) -> None:
        """Test creating dataset from Example objects."""
        dataset = MemoryDataset([sample_example])
        assert len(dataset) == 1

    def test_mixed_input(self, sample_example: Example, sample_example_dict: dict) -> None:
        """Test creating dataset from mixed Example and dict."""
        dataset = MemoryDataset([sample_example, sample_example_dict])
        assert len(dataset) == 2

    def test_validation_error(self) -> None:
        """Test that invalid examples raise DatasetValidationError."""
        with pytest.raises(DatasetValidationError) as exc_info:
            MemoryDataset([{"id": "bad", "query": "", "response": "R"}])
        assert "index 0" in str(exc_info.value)

    def test_len_and_iter(self, sample_example_dict: dict) -> None:
        """Test length and iteration."""
        dataset = MemoryDataset([sample_example_dict, sample_example_dict])
        assert len(dataset) == 2
        count = sum(1 for _ in dataset)
        assert count == 2

    def test_info_property(self, sample_example: Example) -> None:
        """Test the info property."""
        dataset = MemoryDataset([sample_example])
        info = dataset.info
        assert info["source"] == "memory"
        assert info["num_examples"] == 1


class TestJSONLDataset:
    """Tests for the JSONLDataset class."""

    def test_load_valid_file(self, sample_example_dict: dict) -> None:
        """Test loading a valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_example_dict) + "\n")
            f.write(json.dumps(sample_example_dict) + "\n")
            f.flush()
            path = f.name

        try:
            dataset = JSONLDataset(path)
            assert len(dataset) == 2
        finally:
            Path(path).unlink()

    def test_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            JSONLDataset("/nonexistent/path.jsonl")

    def test_invalid_json(self) -> None:
        """Test that invalid JSON raises DatasetValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(DatasetValidationError) as exc_info:
                JSONLDataset(path)
            assert "line 1" in str(exc_info.value)
        finally:
            Path(path).unlink()

    def test_skips_empty_lines(self, sample_example_dict: dict) -> None:
        """Test that empty lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_example_dict) + "\n")
            f.write("\n")  # Empty line
            f.write(json.dumps(sample_example_dict) + "\n")
            f.flush()
            path = f.name

        try:
            dataset = JSONLDataset(path)
            assert len(dataset) == 2
        finally:
            Path(path).unlink()


class TestCSVDataset:
    """Tests for the CSVDataset class."""

    def test_load_valid_file(self) -> None:
        """Test loading a valid CSV file."""
        csv_content = "id,query,response\n1,What is X?,X is Y.\n2,How?,Like this."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            path = f.name

        try:
            dataset = CSVDataset(path)
            assert len(dataset) == 2
            examples = list(dataset)
            assert examples[0].query == "What is X?"
        finally:
            Path(path).unlink()

    def test_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CSVDataset("/nonexistent/path.csv")

    def test_column_mapping(self) -> None:
        """Test custom column mapping."""
        csv_content = "example_id,user_input,agent_output\n1,Q,R\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            path = f.name

        try:
            dataset = CSVDataset(
                path,
                column_mapping={
                    "id": "example_id",
                    "query": "user_input",
                    "response": "agent_output",
                },
            )
            assert len(dataset) == 1
        finally:
            Path(path).unlink()

    def test_missing_required_column(self) -> None:
        """Test that missing required column raises error."""
        csv_content = "id,query\n1,What?\n"  # Missing 'response'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            path = f.name

        try:
            with pytest.raises(DatasetValidationError) as exc_info:
                CSVDataset(path)
            assert "response" in str(exc_info.value)
        finally:
            Path(path).unlink()


class TestIteratorDataset:
    """Tests for the IteratorDataset class."""

    def test_from_generator(self, sample_example_dict: dict) -> None:
        """Test creating dataset from a generator."""

        def gen():
            for i in range(3):
                yield {**sample_example_dict, "id": str(i)}

        dataset = IteratorDataset(gen())
        assert len(dataset) == 3

    def test_length_hint(self, sample_example_dict: dict) -> None:
        """Test using length hint before materialization."""

        def gen():
            for i in range(5):
                yield {**sample_example_dict, "id": str(i)}

        dataset = IteratorDataset(gen(), length=5)
        # Should use hint before materialization
        assert len(dataset) == 5
        assert dataset.info["materialized"] is False

    def test_materialization(self, sample_example_dict: dict) -> None:
        """Test that iterating materializes the dataset."""

        def gen():
            for i in range(2):
                yield {**sample_example_dict, "id": str(i)}

        dataset = IteratorDataset(gen())
        list(dataset)  # Materialize
        assert dataset.info["materialized"] is True

    def test_multiple_iterations(self, sample_example_dict: dict) -> None:
        """Test that multiple iterations work after materialization."""

        def gen():
            yield sample_example_dict

        dataset = IteratorDataset(gen())
        first_pass = list(dataset)
        second_pass = list(dataset)
        assert first_pass == second_pass


# ============================================================================
# Test Evaluators
# ============================================================================


class TestExactMatchEvaluator:
    """Tests for the ExactMatchEvaluator class."""

    def test_exact_match_returns_1(self, sample_example: Example) -> None:
        """Test that exact match returns 1.0."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate(["Would you like examples?"], sample_example)
        assert score == 1.0

    def test_no_match_returns_0(self, sample_example: Example) -> None:
        """Test that no match returns 0.0."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate(["Something completely different"], sample_example)
        assert score == 0.0

    def test_case_insensitive(self, sample_example: Example) -> None:
        """Test that matching is case-insensitive."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate(["WOULD YOU LIKE EXAMPLES?"], sample_example)
        assert score == 1.0

    def test_empty_output_returns_0(self, sample_example: Example) -> None:
        """Test that empty output returns 0.0."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate([], sample_example)
        assert score == 0.0

    def test_no_expected_returns_0(self) -> None:
        """Test that missing expected followups returns 0.0."""
        evaluator = ExactMatchEvaluator()
        example = Example(id="1", query="Q", response="R")  # No expected
        score = evaluator.evaluate(["Some question?"], example)
        assert score == 0.0

    def test_name_attribute(self) -> None:
        """Test the evaluator name attribute."""
        evaluator = ExactMatchEvaluator()
        assert evaluator.name == "exact_match"


class TestQuestionFormatEvaluator:
    """Tests for the QuestionFormatEvaluator class."""

    def test_all_questions_valid(self, sample_example: Example) -> None:
        """Test that all questions ending in ? returns 1.0."""
        evaluator = QuestionFormatEvaluator()
        score = evaluator.evaluate(
            ["How are you?", "What's next?"], sample_example
        )
        assert score == 1.0

    def test_partial_valid(self, sample_example: Example) -> None:
        """Test partial question format compliance."""
        evaluator = QuestionFormatEvaluator()
        score = evaluator.evaluate(
            ["Is this a question?", "This is not"],
            sample_example,
        )
        assert score == 0.5

    def test_no_questions_valid(self, sample_example: Example) -> None:
        """Test that no valid questions returns 0.0."""
        evaluator = QuestionFormatEvaluator()
        score = evaluator.evaluate(
            ["No question here", "Neither here"],
            sample_example,
        )
        assert score == 0.0

    def test_empty_output(self, sample_example: Example) -> None:
        """Test that empty output returns 0.0."""
        evaluator = QuestionFormatEvaluator()
        score = evaluator.evaluate([], sample_example)
        assert score == 0.0


class TestStarterPhraseEvaluator:
    """Tests for the StarterPhraseEvaluator class."""

    def test_matches_starter(self, sample_example: Example) -> None:
        """Test matching starter phrases."""
        evaluator = StarterPhraseEvaluator(starters=["Would you", "Should I"])
        score = evaluator.evaluate(
            ["Would you like to know more?"],
            sample_example,
        )
        assert score == 1.0

    def test_partial_match(self, sample_example: Example) -> None:
        """Test partial starter phrase matching."""
        evaluator = StarterPhraseEvaluator(starters=["Would you"])
        score = evaluator.evaluate(
            ["Would you like to know?", "How about this?"],
            sample_example,
        )
        assert score == 0.5

    def test_case_insensitive(self, sample_example: Example) -> None:
        """Test case-insensitive matching."""
        evaluator = StarterPhraseEvaluator(starters=["would you"])
        score = evaluator.evaluate(
            ["WOULD YOU like examples?"],
            sample_example,
        )
        assert score == 1.0

    def test_empty_output(self, sample_example: Example) -> None:
        """Test empty output returns 0.0."""
        evaluator = StarterPhraseEvaluator(starters=["Would"])
        score = evaluator.evaluate([], sample_example)
        assert score == 0.0


# ============================================================================
# Test Metrics
# ============================================================================


class TestMeanScoreMetric:
    """Tests for the MeanScoreMetric class."""

    def test_computes_mean(self, sample_results_list: list[ExampleResult]) -> None:
        """Test computing mean score across results."""
        metric = MeanScoreMetric("exact_match")
        # Results have scores 1.0, 0.5, 0.0 (but 0.0 has error)
        # Only 1.0 and 0.5 should count -> mean = 0.75
        mean = metric.compute(sample_results_list)
        assert mean == 0.75

    def test_skips_errors(self, sample_results_list: list[ExampleResult]) -> None:
        """Test that examples with errors are skipped."""
        metric = MeanScoreMetric("question_format")
        # Results have scores 1.0, 1.0, 0.0 (but 0.0 has error)
        # Only 1.0 and 1.0 should count -> mean = 1.0
        mean = metric.compute(sample_results_list)
        assert mean == 1.0

    def test_empty_results(self) -> None:
        """Test computing mean on empty results."""
        metric = MeanScoreMetric("exact_match")
        assert metric.compute([]) == 0.0

    def test_missing_evaluator(self, sample_result: ExampleResult) -> None:
        """Test computing mean for missing evaluator returns 0."""
        metric = MeanScoreMetric("nonexistent")
        assert metric.compute([sample_result]) == 0.0

    def test_name_format(self) -> None:
        """Test the metric name format."""
        metric = MeanScoreMetric("llm_judge")
        assert metric.name == "llm_judge_mean"


class TestAccuracyMetric:
    """Tests for the AccuracyMetric class."""

    def test_computes_accuracy(self, sample_results_list: list[ExampleResult]) -> None:
        """Test computing accuracy above threshold."""
        metric = AccuracyMetric("exact_match", threshold=0.5)
        # Scores are 1.0, 0.5 (error excluded) -> 2/2 >= 0.5
        accuracy = metric.compute(sample_results_list)
        assert accuracy == 1.0

    def test_partial_accuracy(self, sample_results_list: list[ExampleResult]) -> None:
        """Test partial accuracy with higher threshold."""
        metric = AccuracyMetric("exact_match", threshold=0.75)
        # Only 1.0 >= 0.75 -> 1/2 = 0.5
        accuracy = metric.compute(sample_results_list)
        assert accuracy == 0.5

    def test_empty_results(self) -> None:
        """Test computing accuracy on empty results."""
        metric = AccuracyMetric("exact_match")
        assert metric.compute([]) == 0.0

    def test_name_format(self) -> None:
        """Test the metric name format."""
        metric = AccuracyMetric("llm_judge")
        assert metric.name == "llm_judge_accuracy"


class TestSuccessRateMetric:
    """Tests for the SuccessRateMetric class."""

    def test_computes_success_rate(
        self, sample_results_list: list[ExampleResult]
    ) -> None:
        """Test computing success rate."""
        metric = SuccessRateMetric()
        # 2 successful, 1 error -> 2/3
        rate = metric.compute(sample_results_list)
        assert abs(rate - 2 / 3) < 0.001

    def test_all_successful(self, sample_result: ExampleResult) -> None:
        """Test 100% success rate."""
        metric = SuccessRateMetric()
        rate = metric.compute([sample_result, sample_result])
        assert rate == 1.0

    def test_empty_results(self) -> None:
        """Test computing success rate on empty results."""
        metric = SuccessRateMetric()
        assert metric.compute([]) == 0.0

    def test_name_attribute(self) -> None:
        """Test the metric name attribute."""
        metric = SuccessRateMetric()
        assert metric.name == "success_rate"


class TestMetricRegistry:
    """Tests for the MetricRegistry class."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving metrics."""
        registry = MetricRegistry()
        metric = SuccessRateMetric()
        registry.register(metric)
        retrieved = registry.get("success_rate")
        assert retrieved is metric

    def test_get_missing_raises(self) -> None:
        """Test that getting missing metric raises KeyError."""
        registry = MetricRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_metrics(self) -> None:
        """Test listing registered metrics."""
        registry = MetricRegistry()
        registry.register(SuccessRateMetric())
        registry.register(MeanScoreMetric("exact_match"))
        names = registry.list_metrics()
        assert "success_rate" in names
        assert "exact_match_mean" in names

    def test_compute_all(self, sample_results_list: list[ExampleResult]) -> None:
        """Test computing all registered metrics."""
        registry = MetricRegistry()
        registry.register(SuccessRateMetric())
        registry.register(MeanScoreMetric("exact_match"))
        results = registry.compute_all(sample_results_list)
        assert "success_rate" in results
        assert "exact_match_mean" in results
        assert results["exact_match_mean"] == 0.75



# ============================================================================
# Test Runner
# ============================================================================


class TestEvalRunner:
    """Tests for the EvalRunner class."""

    def test_run_with_mock_compass(self, sample_example_dict: dict) -> None:
        """Test running evaluation with mocked Compass."""
        # Create mock CompassNode
        mock_compass = MagicMock()
        mock_compass.strategy = "adaptive"
        mock_compass.max_suggestions = 3
        mock_compass.output_key = "suggestions"
        mock_compass.inject_into_messages = False
        mock_compass.generate_candidates = False
        mock_compass.return_value = {"suggestions": ["Is there more?"]}

        runner = EvalRunner(
            compass=mock_compass,
            evaluators=[QuestionFormatEvaluator()],
        )
        dataset = MemoryDataset([sample_example_dict])
        results = runner.run(dataset)

        assert results.aggregated.total_examples == 1
        assert results.aggregated.successful_examples == 1
        assert mock_compass.call_count == 1

    def test_handles_errors_per_example(self, sample_example_dict: dict) -> None:
        """Test that errors are captured per example when continue_on_error=True."""
        mock_compass = MagicMock()
        mock_compass.strategy = "adaptive"
        mock_compass.max_suggestions = 3
        mock_compass.output_key = "suggestions"
        mock_compass.inject_into_messages = False
        mock_compass.generate_candidates = False
        mock_compass.side_effect = ValueError("LLM error")

        runner = EvalRunner(
            compass=mock_compass,
            evaluators=[QuestionFormatEvaluator()],
            continue_on_error=True,
        )
        dataset = MemoryDataset([sample_example_dict])
        results = runner.run(dataset)

        assert results.aggregated.failed_examples == 1
        assert results.per_example[0].error is not None
        assert "ValueError" in results.per_example[0].error

    def test_raises_on_error_when_disabled(self, sample_example_dict: dict) -> None:
        """Test that errors are raised when continue_on_error=False."""
        mock_compass = MagicMock()
        mock_compass.side_effect = RuntimeError("Fatal error")

        runner = EvalRunner(
            compass=mock_compass,
            evaluators=[QuestionFormatEvaluator()],
            continue_on_error=False,
        )
        dataset = MemoryDataset([sample_example_dict])

        with pytest.raises(RuntimeError, match="Fatal error"):
            runner.run(dataset)

    def test_computes_metrics(self, sample_example_dict: dict) -> None:
        """Test that metrics are computed correctly."""
        mock_compass = MagicMock()
        mock_compass.strategy = "adaptive"
        mock_compass.max_suggestions = 3
        mock_compass.output_key = "suggestions"
        mock_compass.inject_into_messages = False
        mock_compass.generate_candidates = False
        mock_compass.return_value = {"suggestions": ["What else?"]}

        evaluator = QuestionFormatEvaluator()
        runner = EvalRunner(
            compass=mock_compass,
            evaluators=[evaluator],
        )
        dataset = MemoryDataset([sample_example_dict, sample_example_dict])
        results = runner.run(dataset)

        # Default metrics include mean score for each evaluator + success rate
        assert "question_format_mean" in results.aggregated.metrics
        assert "success_rate" in results.aggregated.metrics
        assert results.aggregated.metrics["question_format_mean"] == 1.0
        assert results.aggregated.metrics["success_rate"] == 1.0

    def test_custom_metrics(self, sample_example_dict: dict) -> None:
        """Test using custom metrics."""
        mock_compass = MagicMock()
        mock_compass.strategy = "adaptive"
        mock_compass.max_suggestions = 3
        mock_compass.output_key = "suggestions"
        mock_compass.inject_into_messages = False
        mock_compass.generate_candidates = False
        mock_compass.return_value = {"suggestions": ["Question?"]}

        custom_metric = AccuracyMetric("question_format", threshold=0.9)
        runner = EvalRunner(
            compass=mock_compass,
            evaluators=[QuestionFormatEvaluator()],
            metrics=[custom_metric],
        )
        dataset = MemoryDataset([sample_example_dict])
        results = runner.run(dataset)

        assert "question_format_accuracy" in results.aggregated.metrics
        # success_rate should not be present as we specified custom metrics
        assert "success_rate" not in results.aggregated.metrics

    def test_result_contains_metadata(self, sample_example_dict: dict) -> None:
        """Test that results contain proper metadata."""
        mock_compass = MagicMock()
        mock_compass.strategy = "adaptive"
        mock_compass.max_suggestions = 3
        mock_compass.output_key = "suggestions"
        mock_compass.inject_into_messages = False
        mock_compass.generate_candidates = False
        mock_compass.return_value = {"suggestions": ["Question?"]}

        runner = EvalRunner(
            compass=mock_compass,
            evaluators=[QuestionFormatEvaluator()],
        )
        dataset = MemoryDataset([sample_example_dict])
        results = runner.run(dataset)

        assert results.metadata.run_id is not None
        assert results.metadata.compass_config["strategy"] == "adaptive"
        assert results.metadata.dataset_info["source"] == "memory"



# ============================================================================
# Test Reporters
# ============================================================================


class TestMarkdownReporter:
    """Tests for the MarkdownReporter class."""

    def test_generate_report(self, sample_eval_results: EvalResults) -> None:
        """Test generating a Markdown report."""
        reporter = MarkdownReporter()
        report = reporter.generate(sample_eval_results)

        assert "# Compass Evaluation Report" in report
        assert "## Run Summary" in report
        assert "## Overall Metrics" in report
        assert "## Score Distribution" in report
        assert "## Top Failing Examples" in report
        assert sample_eval_results.metadata.run_id in report

    def test_save_to_file(self, sample_eval_results: EvalResults) -> None:
        """Test saving report to file."""
        reporter = MarkdownReporter()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name

        try:
            reporter.save(sample_eval_results, path)
            content = Path(path).read_text()
            assert "# Compass Evaluation Report" in content
        finally:
            Path(path).unlink()

    def test_include_examples_limit(self, sample_eval_results: EvalResults) -> None:
        """Test limiting number of examples in report."""
        reporter = MarkdownReporter(include_examples=1)
        report = reporter.generate(sample_eval_results)
        # Should only include 1 example, not all 3
        example_count = report.count("### Example:")
        assert example_count == 1

    def test_disable_distribution(self, sample_eval_results: EvalResults) -> None:
        """Test disabling distribution section."""
        reporter = MarkdownReporter(include_distribution=False)
        report = reporter.generate(sample_eval_results)
        assert "## Score Distribution" not in report


class TestJSONReporter:
    """Tests for the JSONReporter class."""

    def test_generate_json(self, sample_eval_results: EvalResults) -> None:
        """Test generating JSON report."""
        reporter = JSONReporter()
        json_str = reporter.generate(sample_eval_results)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "per_example" in parsed
        assert "aggregated" in parsed

    def test_save_to_file(self, sample_eval_results: EvalResults) -> None:
        """Test saving JSON report to file."""
        reporter = JSONReporter()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            reporter.save(sample_eval_results, path)
            content = Path(path).read_text()
            parsed = json.loads(content)
            assert parsed["metadata"]["run_id"] == sample_eval_results.metadata.run_id
        finally:
            Path(path).unlink()

    def test_roundtrip(self, sample_eval_results: EvalResults) -> None:
        """Test JSON serialization roundtrip."""
        reporter = JSONReporter()
        json_str = reporter.generate(sample_eval_results)

        # Parse back and validate
        parsed_results = EvalResults.model_validate_json(json_str)
        assert parsed_results.metadata.run_id == sample_eval_results.metadata.run_id
        assert len(parsed_results.per_example) == len(sample_eval_results.per_example)

    def test_custom_indent(self, sample_eval_results: EvalResults) -> None:
        """Test custom indentation."""
        reporter = JSONReporter(indent=4)
        json_str = reporter.generate(sample_eval_results)
        # With indent=4, lines should be more indented
        assert "    " in json_str  # 4 spaces