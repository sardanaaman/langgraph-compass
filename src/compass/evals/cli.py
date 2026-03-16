"""CLI entrypoint for running Compass evaluations.

This module provides a command-line interface for running evaluations on datasets
using the Compass evaluation framework.

Usage:
    python -m compass.evals.cli --dataset data.jsonl --output results/
    compass-eval --dataset data.jsonl --evaluator exact_match --evaluator question_format

Example:
    $ compass-eval --dataset examples/sample.jsonl --output ./results \\
        --evaluator exact_match --evaluator llm_judge \\
        --model gpt-4o-mini --strategy adaptive
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compass.evals.evaluators import Evaluator


def load_plugin(import_path: str) -> Any:
    """Load a plugin class from an import path.

    Args:
        import_path: Import path in format "module.path:ClassName"
            e.g., "myproject.evals:MyEvaluator"

    Returns:
        The loaded class.

    Raises:
        ValueError: If the import path format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Invalid plugin format: '{import_path}'. "
            "Expected 'module.path:ClassName' (e.g., 'myproject.evals:MyEvaluator')"
        )

    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_evaluators(
    evaluator_names: list[str], model: Any | None = None
) -> list[Evaluator]:
    """Create evaluator instances from names.

    Args:
        evaluator_names: List of evaluator names (exact_match, question_format, llm_judge)
        model: LLM model instance for evaluators that need it (e.g., llm_judge)

    Returns:
        List of evaluator instances.

    Raises:
        ValueError: If an unknown evaluator name is provided.
    """
    from compass.evals.evaluators import (
        ExactMatchEvaluator,
        LLMJudgeEvaluator,
        QuestionFormatEvaluator,
    )

    evaluators: list[Evaluator] = []

    for name in evaluator_names:
        name_lower = name.lower().strip()
        if name_lower == "exact_match":
            evaluators.append(ExactMatchEvaluator())
        elif name_lower == "question_format":
            evaluators.append(QuestionFormatEvaluator())
        elif name_lower == "llm_judge":
            if model is None:
                raise ValueError("llm_judge evaluator requires a model (--model)")
            evaluators.append(LLMJudgeEvaluator(model=model))
        else:
            raise ValueError(
                f"Unknown evaluator: '{name}'. "
                "Available: exact_match, question_format, llm_judge"
            )

    return evaluators


def load_dataset(path: str) -> Any:
    """Load a dataset based on file extension.

    Args:
        path: Path to the dataset file.

    Returns:
        A Dataset instance.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    from compass.evals.datasets import CSVDataset, JSONLDataset

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = path_obj.suffix.lower()
    if ext == ".jsonl":
        return JSONLDataset(path)
    elif ext == ".csv":
        return CSVDataset(path)
    else:
        raise ValueError(
            f"Unsupported file extension: '{ext}'. Supported: .jsonl, .csv"
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="compass-eval",
        description="Run Compass evaluation on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  compass-eval --dataset data.jsonl --output results/
  compass-eval --dataset data.csv --evaluator exact_match --evaluator llm_judge
  compass-eval --dataset data.jsonl --plugin myproject.evals:CustomEvaluator
        """,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset file (.jsonl or .csv)",
    )
    parser.add_argument(
        "--output",
        default="./eval_results",
        help="Output directory for reports (default: ./eval_results)",
    )
    parser.add_argument(
        "--evaluator",
        action="append",
        default=[],
        dest="evaluators",
        help="Evaluator to use (can specify multiple). Options: exact_match, question_format, llm_judge",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model for Compass and LLM judge (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--strategy",
        default="adaptive",
        help="Compass strategy (default: adaptive)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per example in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        dest="plugins",
        help="Import path for custom plugin (e.g., 'myproject.evals:MyEvaluator')",
    )

    return parser.parse_args(args)



def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        parsed = parse_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    try:
        # Import required modules (may not exist yet if parallel tasks not complete)
        from langchain_openai import ChatOpenAI

        from compass import CompassNode
        from compass.evals.reporters import JSONReporter, MarkdownReporter
        from compass.evals.runner import EvalRunner

        # Create output directory
        output_dir = Path(parsed.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        print(f"Loading dataset: {parsed.dataset}")
        dataset = load_dataset(parsed.dataset)
        print(f"  Loaded {len(dataset)} examples")

        # Create model
        print(f"Creating model: {parsed.model}")
        model = ChatOpenAI(model=parsed.model)

        # Create CompassNode
        print(f"Creating CompassNode with strategy: {parsed.strategy}")
        compass = CompassNode(model=model, strategy=parsed.strategy)

        # Build evaluator list
        evaluators: list[Evaluator] = []

        # Add built-in evaluators
        if parsed.evaluators:
            print(f"Adding evaluators: {', '.join(parsed.evaluators)}")
            evaluators.extend(create_evaluators(parsed.evaluators, model=model))

        # Load plugins
        if parsed.plugins:
            print(f"Loading plugins: {', '.join(parsed.plugins)}")
            for plugin_path in parsed.plugins:
                plugin_class = load_plugin(plugin_path)
                # Instantiate plugin (assume no-arg constructor for evaluators)
                evaluators.append(plugin_class())

        # Use default evaluators if none specified
        if not evaluators:
            print("No evaluators specified, using defaults: exact_match, question_format")
            evaluators = create_evaluators(["exact_match", "question_format"])

        # Create runner
        print(f"Running evaluation (timeout: {parsed.timeout}s per example)...")
        runner = EvalRunner(
            compass=compass,
            evaluators=evaluators,
            timeout_seconds=parsed.timeout,
        )

        # Run evaluation
        results = runner.run(dataset)

        # Save reports
        print(f"\nSaving reports to: {output_dir}")

        # Markdown report
        md_reporter = MarkdownReporter()
        md_path = output_dir / "report.md"
        md_reporter.save(results, md_path)
        print(f"  Markdown report: {md_path}")

        # JSON report
        json_reporter = JSONReporter()
        json_path = output_dir / "results.json"
        json_reporter.save(results, json_path)
        print(f"  JSON report: {json_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total examples: {results.aggregated.total_examples}")
        print(f"Successful: {results.aggregated.successful_examples}")
        print(f"Failed: {results.aggregated.failed_examples}")
        print(f"Duration: {results.metadata.duration_seconds:.2f}s")
        print("\nMetrics:")
        for name, value in results.aggregated.metrics.items():
            print(f"  {name}: {value:.4f}")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        print("Make sure all dependencies are installed.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

