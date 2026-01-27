# Compass Evals Framework — Design Document

## Overview

The Compass Evals Framework provides a modular, extensible system for measuring the quality of follow-up question generation over time. It enables users to:

1. **Define datasets** — Load examples from various sources (memory, files, iterators)
2. **Run evaluations** — Execute Compass against each example with configurable settings
3. **Compute metrics** — Score outputs using built-in or custom evaluators
4. **Generate reports** — Produce human-readable summaries and machine-readable artifacts

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EvalRunner                                      │
│  Orchestrates the evaluation pipeline                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│   │   Dataset    │───▶│   Compass    │───▶│  Evaluator   │───▶│  Metrics  │ │
│   │   (input)    │    │   (invoke)   │    │   (score)    │    │ (aggregate)│ │
│   └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│          │                   │                   │                   │       │
│          ▼                   ▼                   ▼                   ▼       │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         EvalResults                                   │  │
│   │  - per_example: list[ExampleResult]                                   │  │
│   │  - aggregated: AggregatedMetrics                                      │  │
│   │  - metadata: RunMetadata                                              │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│                            ┌──────────────────┐                              │
│                            │    Reporters     │                              │
│                            │  (Markdown/JSON) │                              │
│                            └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Models (`models.py`)

```python
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime

class Example(BaseModel):
    """A single evaluation example."""
    id: str = Field(description="Unique identifier for this example")
    query: str = Field(description="User query/input")
    response: str = Field(description="Agent response to evaluate against")
    expected_followups: list[str] | None = Field(
        default=None,
        description="Expected follow-up questions (for exact/similarity matching)"
    )
    labels: dict[str, Any] | None = Field(
        default=None,
        description="Labels for classification (e.g., {'quality': 'good', 'strategy': 'clarifying'})"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context (conversation history, metadata)"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata for filtering/grouping"
    )

class ExampleResult(BaseModel):
    """Result of evaluating a single example."""
    example_id: str
    query: str
    response: str
    compass_output: list[str]  # Generated follow-ups
    expected: list[str] | None
    scores: dict[str, float]  # Evaluator name -> score
    error: str | None = None
    duration_ms: float

class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all examples."""
    total_examples: int
    successful_examples: int
    failed_examples: int
    metrics: dict[str, float]  # Metric name -> value
    score_distribution: dict[str, list[float]]  # Evaluator -> all scores

class RunMetadata(BaseModel):
    """Metadata about the evaluation run."""
    run_id: str
    timestamp: datetime
    compass_version: str
    compass_config: dict[str, Any]
    dataset_info: dict[str, Any]
    duration_seconds: float

class EvalResults(BaseModel):
    """Complete evaluation results."""
    metadata: RunMetadata
    per_example: list[ExampleResult]
    aggregated: AggregatedMetrics
```

### 2. Dataset Abstraction (`datasets.py`)

```python
from typing import Protocol, Iterator, runtime_checkable

@runtime_checkable
class Dataset(Protocol):
    """Protocol for evaluation datasets."""
    
    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples."""
        ...
    
    def __len__(self) -> int:
        """Return number of examples."""
        ...
    
    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        ...

# Built-in implementations
class MemoryDataset:
    """Dataset from in-memory Python objects."""
    def __init__(self, examples: list[Example | dict]) -> None: ...

class JSONLDataset:
    """Dataset from JSONL file."""
    def __init__(self, path: str | Path) -> None: ...

class CSVDataset:
    """Dataset from CSV file with column mapping."""
    def __init__(
        self, 
        path: str | Path,
        column_mapping: dict[str, str] | None = None
    ) -> None: ...

class IteratorDataset:
    """Dataset from any iterator/generator."""
    def __init__(
        self, 
        iterator: Iterator[Example | dict],
        length: int | None = None
    ) -> None: ...
```

**Validation**: Each dataset validates examples on load and provides helpful error messages:

```python
# Example error message
ValidationError: Example at index 3 is invalid:
  - 'query' field is required but missing
  - 'response' must be a non-empty string, got: ""
```

### 3. Evaluators (`evaluators.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluating Compass outputs."""

    @property
    def name(self) -> str:
        """Unique name for this evaluator."""
        ...

    def evaluate(
        self,
        compass_output: list[str],
        example: Example,
        config: RunnableConfig | None = None,
    ) -> float:
        """Score the output. Returns 0.0-1.0."""
        ...

# Built-in evaluators
class ExactMatchEvaluator:
    """Checks if any generated question exactly matches expected."""
    name = "exact_match"

class SemanticSimilarityEvaluator:
    """Computes semantic similarity using embeddings."""
    def __init__(self, embeddings: Embeddings, threshold: float = 0.8) -> None: ...
    name = "semantic_similarity"

class LLMJudgeEvaluator:
    """Uses an LLM to judge quality on a rubric."""
    def __init__(
        self,
        model: BaseChatModel,
        rubric: str | None = None,
        criteria: list[str] | None = None,
    ) -> None: ...
    name = "llm_judge"

class StarterPhraseEvaluator:
    """Checks if questions start with approved phrases."""
    def __init__(self, starters: list[str]) -> None: ...
    name = "starter_phrase"

class QuestionFormatEvaluator:
    """Checks if outputs are well-formed questions."""
    name = "question_format"
```

**LLM Judge Rubric Example**:
```python
rubric = """
Rate the follow-up question on a scale of 1-5:

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
```

### 4. Metrics (`metrics.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Metric(Protocol):
    """Protocol for computing aggregate metrics."""

    @property
    def name(self) -> str:
        """Unique name for this metric."""
        ...

    def compute(self, results: list[ExampleResult]) -> float:
        """Compute the metric from all results."""
        ...

class MetricRegistry:
    """Registry for metrics with built-in defaults."""

    def register(self, metric: Metric) -> None: ...
    def get(self, name: str) -> Metric: ...
    def compute_all(self, results: list[ExampleResult]) -> dict[str, float]: ...

# Built-in metrics
class MeanScoreMetric:
    """Average score across all examples for a given evaluator."""
    def __init__(self, evaluator_name: str) -> None: ...

class AccuracyMetric:
    """Percentage of examples scoring above threshold."""
    def __init__(self, evaluator_name: str, threshold: float = 0.5) -> None: ...

class SuccessRateMetric:
    """Percentage of examples that didn't error."""
    name = "success_rate"

class ScoreDistributionMetric:
    """Returns percentiles (p25, p50, p75, p95) for an evaluator."""
    def __init__(self, evaluator_name: str) -> None: ...
```

### 5. Eval Runner (`runner.py`)

```python
class EvalRunner:
    """Orchestrates evaluation runs."""

    def __init__(
        self,
        compass: CompassNode,
        evaluators: list[Evaluator],
        metrics: list[Metric] | None = None,
        *,
        timeout_seconds: float = 30.0,
        continue_on_error: bool = True,
    ) -> None:
        """
        Args:
            compass: The CompassNode to evaluate
            evaluators: List of evaluators to run on each output
            metrics: List of metrics to compute (uses defaults if None)
            timeout_seconds: Timeout per example
            continue_on_error: If True, log errors and continue; if False, raise
        """
        ...

    def run(
        self,
        dataset: Dataset,
        config: RunnableConfig | None = None,
    ) -> EvalResults:
        """Run evaluation on the dataset."""
        ...

    def _invoke_compass(self, example: Example) -> tuple[list[str], float]:
        """Invoke Compass and return (output, duration_ms)."""
        ...

    def _evaluate_example(
        self,
        example: Example,
        output: list[str]
    ) -> dict[str, float]:
        """Run all evaluators on a single example."""
        ...
```

**Error Handling**: Errors are captured per-example, not raised:

```python
ExampleResult(
    example_id="ex-42",
    compass_output=[],
    error="TimeoutError: Compass invocation exceeded 30s",
    scores={},
    duration_ms=30000.0,
)
```

### 6. Reporters (`reporters.py`)

```python
from typing import Protocol

class Reporter(Protocol):
    """Protocol for generating reports."""

    def generate(self, results: EvalResults) -> str:
        """Generate report content."""
        ...

    def save(self, results: EvalResults, path: str | Path) -> None:
        """Save report to file."""
        ...

class MarkdownReporter:
    """Generates human-readable Markdown reports."""

    def __init__(
        self,
        include_examples: int = 10,  # Top N failing examples
        include_distribution: bool = True,
    ) -> None: ...

class JSONReporter:
    """Generates machine-readable JSON artifacts."""

    def __init__(self, indent: int = 2) -> None: ...
```

**Markdown Report Structure**:
```markdown
# Compass Evaluation Report

## Run Summary
- **Run ID**: eval-2024-01-27-abc123
- **Timestamp**: 2024-01-27T14:30:00Z
- **Compass Version**: 1.0.1
- **Dataset**: my_dataset.jsonl (500 examples)
- **Duration**: 45.2 seconds

## Overall Metrics
| Metric | Value |
|--------|-------|
| Success Rate | 98.4% |
| Mean LLM Judge Score | 0.72 |
| Exact Match Accuracy | 0.15 |
| Semantic Similarity (>0.8) | 0.68 |

## Score Distribution (LLM Judge)
- P25: 0.60
- P50 (Median): 0.75
- P75: 0.85
- P95: 0.95

## Top Failing Examples
### Example: ex-42 (Score: 0.20)
**Query**: What is the capital of France?
**Response**: The capital of France is Paris...
**Generated**: ["Would you like me to explain more?"]
**Expected**: ["Interested in learning about Paris landmarks?"]
**Error**: None

## Error Breakdown
| Error Type | Count |
|------------|-------|
| TimeoutError | 5 |
| ValidationError | 3 |
```

### 7. CLI (`cli.py`)

```python
# Usage:
# compass-eval --dataset data.jsonl --output results/ --evaluator llm_judge

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Compass evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset file")
    parser.add_argument("--output", default="./eval_results", help="Output directory")
    parser.add_argument("--evaluator", action="append", help="Evaluators to use")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for Compass")
    parser.add_argument("--plugin", action="append", help="Import path for custom plugins")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout per example")
    ...
```

**Plugin Loading**:
```bash
# Load custom evaluator from user code
compass-eval --dataset data.jsonl --plugin myproject.evals:MyCustomEvaluator
```

## Extension Points

### Custom Dataset Loader

```python
from compass.evals import Dataset, Example

class MyDatabaseDataset:
    """Load examples from a database."""

    def __init__(self, connection_string: str, query: str):
        self.conn = connect(connection_string)
        self.query = query
        self._examples = self._load()

    def _load(self) -> list[Example]:
        rows = self.conn.execute(self.query)
        return [
            Example(
                id=row["id"],
                query=row["user_query"],
                response=row["agent_response"],
                expected_followups=json.loads(row["expected"]),
            )
            for row in rows
        ]

    def __iter__(self):
        return iter(self._examples)

    def __len__(self):
        return len(self._examples)

    @property
    def info(self):
        return {"source": "database", "query": self.query}
```

### Custom Evaluator

```python
from compass.evals import Evaluator, Example

class DomainRelevanceEvaluator:
    """Check if follow-ups are relevant to our domain."""

    name = "domain_relevance"

    def __init__(self, domain_keywords: list[str]):
        self.keywords = set(kw.lower() for kw in domain_keywords)

    def evaluate(
        self,
        compass_output: list[str],
        example: Example,
        config: RunnableConfig | None = None,
    ) -> float:
        if not compass_output:
            return 0.0

        # Score based on keyword presence
        scores = []
        for question in compass_output:
            words = set(question.lower().split())
            overlap = len(words & self.keywords)
            scores.append(min(overlap / 3, 1.0))  # Cap at 1.0

        return sum(scores) / len(scores)
```

### Custom Metric

```python
from compass.evals import Metric, ExampleResult

class StrategyDistributionMetric:
    """Track distribution of question strategies."""

    name = "strategy_distribution"

    def compute(self, results: list[ExampleResult]) -> dict[str, float]:
        strategies = {"clarifying": 0, "exploratory": 0, "deepening": 0}

        for result in results:
            for question in result.compass_output:
                if "clarify" in question.lower():
                    strategies["clarifying"] += 1
                elif "explore" in question.lower():
                    strategies["exploratory"] += 1
                else:
                    strategies["deepening"] += 1

        total = sum(strategies.values()) or 1
        return {k: v / total for k, v in strategies.items()}
```

## Example Usage

### Basic Evaluation

```python
from langchain_openai import ChatOpenAI
from compass import CompassNode
from compass.evals import (
    EvalRunner,
    JSONLDataset,
    LLMJudgeEvaluator,
    MarkdownReporter,
)

# 1. Load dataset
dataset = JSONLDataset("my_eval_data.jsonl")

# 2. Create Compass
model = ChatOpenAI(model="gpt-4o-mini")
compass = CompassNode(model=model, strategy="adaptive")

# 3. Set up evaluators
evaluators = [
    LLMJudgeEvaluator(model=model),
]

# 4. Run evaluation
runner = EvalRunner(compass=compass, evaluators=evaluators)
results = runner.run(dataset)

# 5. Generate report
reporter = MarkdownReporter()
reporter.save(results, "eval_report.md")

print(f"Mean LLM Judge Score: {results.aggregated.metrics['llm_judge_mean']:.2f}")
```

### Tracking Regressions

```python
# Compare two runs
from compass.evals import EvalResults

baseline = EvalResults.model_validate_json(Path("results/baseline.json").read_text())
current = EvalResults.model_validate_json(Path("results/current.json").read_text())

# Compare metrics
for metric_name in baseline.aggregated.metrics:
    baseline_val = baseline.aggregated.metrics[metric_name]
    current_val = current.aggregated.metrics.get(metric_name, 0)
    change = ((current_val - baseline_val) / baseline_val) * 100 if baseline_val else 0
    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
    print(f"{metric_name}: {baseline_val:.2f} → {current_val:.2f} ({arrow} {abs(change):.1f}%)")
```

## Interpreting Results

### What Makes a "Good" Compass Output?

| Metric | Good | Concerning | Action |
|--------|------|------------|--------|
| **LLM Judge Mean** | > 0.7 | < 0.5 | Review rubric, check strategy |
| **Semantic Similarity** | > 0.6 | < 0.4 | May need better examples |
| **Success Rate** | > 95% | < 90% | Check for timeouts, errors |
| **Question Format** | > 99% | < 95% | Check starter phrases |

### Tracking Regressions

1. **Establish baseline**: Run eval on a stable dataset before changes
2. **Set thresholds**: Define acceptable regression (e.g., < 5% drop)
3. **Automate**: Add eval to CI/CD pipeline
4. **Alert**: Fail builds if metrics drop below threshold

### Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Low LLM Judge scores | Generic questions | Add domain examples via ExampleRetriever |
| High similarity to previous | Novelty filter too weak | Adjust ranker threshold |
| Many timeouts | Model too slow | Use faster model for evals |
| Format errors | Wrong starters | Update starter phrases |

## File Structure

```
src/compass/evals/
├── __init__.py          # Public exports
├── models.py            # Pydantic data models
├── datasets.py          # Dataset protocol and loaders
├── evaluators.py        # Evaluator protocol and built-ins
├── metrics.py           # Metric protocol and registry
├── runner.py            # EvalRunner orchestration
├── reporters.py         # Markdown and JSON reporters
└── cli.py               # CLI entrypoint
```

## Dependencies

No new dependencies required for core functionality:
- `pydantic` (already in project)
- `langchain-core` (already in project)

Optional dependencies for specific evaluators:
- `langchain-openai` for embeddings (already in project)

## Open Questions

1. **Async support?** — Should `EvalRunner.run()` support async for parallel execution?
   - Recommendation: Start sync, add async later if needed

2. **Caching?** — Should we cache Compass outputs for re-evaluation with different evaluators?
   - Recommendation: Yes, add optional caching layer

3. **Streaming progress?** — Should runner emit progress events?
   - Recommendation: Add optional callback for progress reporting

