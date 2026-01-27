"""Compass Evaluation Framework.

Provides tools for measuring the quality of follow-up question generation.
"""

from compass.evals.datasets import (
    CSVDataset,
    Dataset,
    DatasetValidationError,
    IteratorDataset,
    JSONLDataset,
    MemoryDataset,
)
from compass.evals.evaluators import (
    Evaluator,
    ExactMatchEvaluator,
    LLMJudgeEvaluator,
    QuestionFormatEvaluator,
    StarterPhraseEvaluator,
)
from compass.evals.metrics import (
    AccuracyMetric,
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
from compass.evals.reporters import JSONReporter, MarkdownReporter, Reporter
from compass.evals.runner import EvalRunner
from compass.evals.cli import main as eval_main

__all__ = [
    # Data models
    "Example",
    "ExampleResult",
    "AggregatedMetrics",
    "RunMetadata",
    "EvalResults",
    # Datasets
    "Dataset",
    "DatasetValidationError",
    "MemoryDataset",
    "JSONLDataset",
    "CSVDataset",
    "IteratorDataset",
    # Evaluators
    "Evaluator",
    "ExactMatchEvaluator",
    "QuestionFormatEvaluator",
    "StarterPhraseEvaluator",
    "LLMJudgeEvaluator",
    # Metrics
    "Metric",
    "MeanScoreMetric",
    "AccuracyMetric",
    "SuccessRateMetric",
    "MetricRegistry",
    # Reporters
    "Reporter",
    "MarkdownReporter",
    "JSONReporter",
    # Runner
    "EvalRunner",
    # CLI
    "eval_main",
]
