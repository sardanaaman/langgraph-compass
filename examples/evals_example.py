"""End-to-end example of the Compass evaluation framework.

This example demonstrates:
1. Creating a custom dataset loader
2. Running evaluations with multiple evaluators
3. Generating Markdown and JSON reports
4. Interpreting the results

Run with: python examples/evals_example.py

Note: Uses a mock model so no API keys are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock

# ============================================================================
# 1. CUSTOM DATASET LOADER
# ============================================================================


class SyntheticDataset:
    """Generate synthetic examples for testing the evaluation framework.

    This demonstrates how to create a custom dataset loader that implements
    the Dataset protocol: __iter__, __len__, and info property.

    Use cases:
    - Testing the evaluation pipeline without real data
    - Generating edge cases for stress testing
    - Prototyping before collecting real examples
    """

    def __init__(self, num_examples: int = 5, seed: int | None = None) -> None:
        """Initialize the synthetic dataset.

        Args:
            num_examples: Number of synthetic examples to generate.
            seed: Optional random seed for reproducibility.
        """
        self._num_examples = num_examples
        self._seed = seed
        self._examples = self._generate_examples()

    def _generate_examples(self) -> list[dict[str, Any]]:
        """Generate synthetic query-response pairs with expected followups."""
        from compass.evals import Example

        # Predefined topics and patterns for realistic examples
        topics = [
            {
                "query": "What is Python?",
                "response": "Python is a high-level programming language known for its readability and versatility.",
                "expected": ["Would you like to see some Python code examples?"],
            },
            {
                "query": "How does machine learning work?",
                "response": "Machine learning uses algorithms to find patterns in data and make predictions.",
                "expected": ["Should I explain supervised vs unsupervised learning?"],
            },
            {
                "query": "What are microservices?",
                "response": "Microservices are an architectural style where applications are built as independent services.",
                "expected": ["Would you like to compare this with monolithic architecture?"],
            },
            {
                "query": "Explain containerization",
                "response": "Containerization packages applications with their dependencies for consistent deployment.",
                "expected": ["Are you interested in learning about Docker or Kubernetes?"],
            },
            {
                "query": "What is an API?",
                "response": "An API (Application Programming Interface) allows different software systems to communicate.",
                "expected": ["Should I show you how to make an API call?"],
            },
        ]

        examples = []
        for i in range(self._num_examples):
            topic = topics[i % len(topics)]
            examples.append(
                Example(
                    id=f"synthetic-{i + 1:03d}",
                    query=topic["query"],
                    response=topic["response"],
                    expected_followups=topic["expected"],
                    metadata={"source": "synthetic", "index": i},
                )
            )

        return examples

    def __iter__(self) -> Iterator:
        """Iterate over examples."""
        return iter(self._examples)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self._examples)

    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata (required by Dataset protocol)."""
        return {
            "source": "synthetic",
            "num_examples": self._num_examples,
            "seed": self._seed,
            "name": "SyntheticDataset",
        }


# ============================================================================
# 2. MOCK MODEL FOR TESTING (no API keys needed)
# ============================================================================


def create_mock_model() -> MagicMock:
    """Create a mock ChatModel that returns predictable follow-up questions.

    This allows the example to run without API keys while still demonstrating
    the full evaluation pipeline.
    """
    mock = MagicMock()

    # Define the mock responses for structured output
    def mock_invoke(messages, config=None):
        # Return a mock AIMessage with follow-up questions
        mock_response = MagicMock()
        mock_response.content = (
            '["Would you like to learn more about this topic?", '
            '"Should I provide some practical examples?"]'
        )
        return mock_response

    mock.invoke = mock_invoke
    mock.with_structured_output = MagicMock(return_value=mock)

    return mock


# ============================================================================
# 3. MAIN EVALUATION FLOW
# ============================================================================


def main():
    """Run the complete evaluation pipeline."""
    # Import evals components
    from compass import CompassNode
    from compass.evals import (
        EvalRunner,
        ExactMatchEvaluator,
        JSONReporter,
        MarkdownReporter,
        QuestionFormatEvaluator,
    )

    print("=" * 60)
    print("Compass Evaluation Framework - End-to-End Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Create the synthetic dataset
    # -------------------------------------------------------------------------
    print("\n📊 Step 1: Creating synthetic dataset...")
    dataset = SyntheticDataset(num_examples=5)
    print(f"   Created dataset with {len(dataset)} examples")
    print(f"   Dataset info: {dataset.info}")

    # -------------------------------------------------------------------------
    # Step 2: Create CompassNode with mock model
    # -------------------------------------------------------------------------
    print("\n🧭 Step 2: Setting up CompassNode with mock model...")
    mock_model = create_mock_model()
    compass = CompassNode(
        model=mock_model,
        strategy="adaptive",  # Uses context-aware prompting
        max_suggestions=2,
    )
    print(f"   Strategy: {compass.strategy}")
    print(f"   Max suggestions: {compass.max_suggestions}")

    # -------------------------------------------------------------------------
    # Step 3: Set up evaluators
    # -------------------------------------------------------------------------
    print("\n📏 Step 3: Configuring evaluators...")
    evaluators = [
        # Checks if generated questions match expected ones exactly
        ExactMatchEvaluator(),
        # Checks if outputs are properly formatted as questions
        QuestionFormatEvaluator(),
    ]
    print(f"   Evaluators: {[e.name for e in evaluators]}")

    # -------------------------------------------------------------------------
    # Step 4: Run the evaluation
    # -------------------------------------------------------------------------
    print("\n🏃 Step 4: Running evaluation...")
    runner = EvalRunner(
        compass=compass,
        evaluators=evaluators,
        timeout_seconds=30.0,
        continue_on_error=True,  # Don't stop on individual failures
    )
    results = runner.run(dataset)
    print(f"   Completed in {results.metadata.duration_seconds:.2f} seconds")
    print(f"   Successful: {results.aggregated.successful_examples}/{results.aggregated.total_examples}")

    # -------------------------------------------------------------------------
    # Step 5: Generate reports
    # -------------------------------------------------------------------------
    print("\n📝 Step 5: Generating reports...")

    # Create output directory
    output_dir = Path(__file__).parent / "eval_results"
    output_dir.mkdir(exist_ok=True)

    # Generate Markdown report (human-readable)
    md_reporter = MarkdownReporter(include_examples=5, include_distribution=True)
    md_path = output_dir / "eval_report.md"
    md_reporter.save(results, md_path)
    print(f"   ✅ Markdown report: {md_path}")

    # Generate JSON report (machine-readable)
    json_reporter = JSONReporter(indent=2)
    json_path = output_dir / "eval_results.json"
    json_reporter.save(results, json_path)
    print(f"   ✅ JSON report: {json_path}")

    # -------------------------------------------------------------------------
    # Step 6: Print summary with interpretation guidance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    agg = results.aggregated
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)

    # Success rate
    if agg.total_examples > 0:
        success_rate = agg.successful_examples / agg.total_examples
        print(f"{'Success Rate':<30} {success_rate:>9.1%}")

    # Individual metrics
    for name, value in sorted(agg.metrics.items()):
        print(f"{name:<30} {value:>10.4f}")

    # -------------------------------------------------------------------------
    # Interpretation guidance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📖 HOW TO INTERPRET THESE RESULTS:")
    print("-" * 60)
    print("""
• success_rate: Percentage of examples that ran without errors.
  - Good: > 95%, Concerning: < 90%

• exact_match_mean: How often generated questions match expected ones.
  - This is typically LOW (0.1-0.2) since exact matching is strict.
  - Use semantic similarity for more meaningful comparison.

• question_format_mean: Fraction of outputs ending with '?'.
  - Should be > 0.95 for well-formed questions.

NEXT STEPS:
1. Review failing examples in the Markdown report
2. Add LLMJudgeEvaluator for quality assessment (requires API key)
3. Create a real dataset from production conversations
4. Set up CI/CD integration to track regressions
""")

    print(f"\n✨ Reports saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
