"""Dataset protocol and built-in loaders for evaluation data.

This module provides a protocol for evaluation datasets and several built-in
implementations for loading data from various sources.

Example:
    >>> from compass.evals.datasets import MemoryDataset, JSONLDataset
    >>>
    >>> # From in-memory data
    >>> dataset = MemoryDataset([
    ...     {"id": "1", "query": "What is Python?", "response": "A programming language..."},
    ...     {"id": "2", "query": "How do I learn?", "response": "Start with basics..."},
    ... ])
    >>> len(dataset)
    2
    >>>
    >>> # From JSONL file
    >>> dataset = JSONLDataset("examples.jsonl")
    >>> for example in dataset:
    ...     print(example.query)
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from compass.evals.models import Example


class DatasetValidationError(Exception):
    """Raised when dataset validation fails with details about the issue."""

    def __init__(self, message: str, index: int | None = None, line: int | None = None):
        self.index = index
        self.line = line
        super().__init__(message)


def _format_validation_errors(errors: list[ErrorDetails]) -> str:
    """Format Pydantic validation errors into a human-readable string."""
    lines = []
    for error in errors:
        loc = ".".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Unknown error")
        if loc:
            lines.append(f"  - '{loc}': {msg}")
        else:
            lines.append(f"  - {msg}")
    return "\n".join(lines)


def _validate_example(
    data: Example | dict[str, Any], index: int | None = None, line: int | None = None
) -> Example:
    """Validate and convert data to an Example.

    Args:
        data: Either an Example object or a dict to validate.
        index: Optional index for error messages (0-based).
        line: Optional line number for error messages (1-based).

    Returns:
        A validated Example object.

    Raises:
        DatasetValidationError: If validation fails.
    """
    if isinstance(data, Example):
        return data

    try:
        return Example.model_validate(data)
    except ValidationError as e:
        location = ""
        if line is not None:
            location = f" at line {line}"
        elif index is not None:
            location = f" at index {index}"

        error_details = _format_validation_errors(e.errors())
        raise DatasetValidationError(
            f"Example{location} is invalid:\n{error_details}",
            index=index,
            line=line,
        ) from e


@runtime_checkable
class Dataset(Protocol):
    """Protocol for evaluation datasets.

    Datasets provide iteration over Example objects and support
    len() for determining the number of examples.

    Example:
        >>> class MyDataset:
        ...     def __iter__(self) -> Iterator[Example]:
        ...         yield Example(id="1", query="...", response="...")
        ...     def __len__(self) -> int:
        ...         return 1
        ...     @property
        ...     def info(self) -> dict[str, Any]:
        ...         return {"source": "custom"}
        >>>
        >>> isinstance(MyDataset(), Dataset)
        True
    """

    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples in the dataset."""
        ...

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        ...

    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata.

        Returns:
            A dictionary containing dataset information such as source,
            path, number of examples, etc.
        """
        ...


class MemoryDataset:
    """Dataset from in-memory Python objects.

    Validates and converts dicts to Example objects on initialization.

    Example:
        >>> dataset = MemoryDataset([
        ...     {"id": "1", "query": "Hello", "response": "Hi there"},
        ...     Example(id="2", query="Bye", response="Goodbye"),
        ... ])
        >>> len(dataset)
        2
        >>> dataset.info
        {'source': 'memory', 'num_examples': 2}

    Args:
        examples: List of Example objects or dicts to convert.

    Raises:
        DatasetValidationError: If any example fails validation.
    """

    def __init__(self, examples: list[Example | dict[str, Any]]) -> None:
        """Initialize with a list of examples."""
        self._examples: list[Example] = []
        for i, item in enumerate(examples):
            self._examples.append(_validate_example(item, index=i))

    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples."""
        return iter(self._examples)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self._examples)

    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        return {"source": "memory", "num_examples": len(self._examples)}


class JSONLDataset:
    """Dataset from a JSONL (JSON Lines) file.

    Each line in the file should be a valid JSON object matching
    the Example schema.

    Example:
        >>> dataset = JSONLDataset("examples.jsonl")
        >>> for example in dataset:
        ...     print(example.id)

    Args:
        path: Path to the JSONL file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetValidationError: If any line fails to parse or validate.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize by loading and validating the JSONL file."""
        self._path = Path(path)
        self._examples: list[Example] = []

        if not self._path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self._path}")

        self._load()

    def _load(self) -> None:
        """Load and validate all examples from the file."""
        with self._path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetValidationError(
                        f"Invalid JSON at line {line_num}: {e}",
                        line=line_num,
                    ) from e

                self._examples.append(_validate_example(data, line=line_num))

    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples."""
        return iter(self._examples)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self._examples)

    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        return {
            "source": "jsonl",
            "path": str(self._path),
            "num_examples": len(self._examples),
        }


# Default column names for CSV files
DEFAULT_CSV_COLUMNS = {
    "id": "id",
    "query": "query",
    "response": "response",
    "expected_followups": "expected_followups",
}


class CSVDataset:
    """Dataset from a CSV file with configurable column mapping.

    The expected_followups column should contain a JSON array string.

    Example:
        >>> # With default columns: id, query, response, expected_followups
        >>> dataset = CSVDataset("examples.csv")
        >>>
        >>> # With custom column mapping
        >>> dataset = CSVDataset("data.csv", column_mapping={
        ...     "id": "example_id",
        ...     "query": "user_input",
        ...     "response": "agent_output",
        ... })

    Args:
        path: Path to the CSV file.
        column_mapping: Optional mapping from Example fields to CSV column names.
            Keys are Example field names, values are CSV column names.
            Default: {"id": "id", "query": "query", "response": "response",
                      "expected_followups": "expected_followups"}

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetValidationError: If any row fails to parse or validate.
    """

    def __init__(
        self,
        path: str | Path,
        column_mapping: dict[str, str] | None = None,
    ) -> None:
        """Initialize by loading and validating the CSV file."""
        self._path = Path(path)
        self._column_mapping = {**DEFAULT_CSV_COLUMNS, **(column_mapping or {})}
        self._examples: list[Example] = []

        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        self._load()

    def _load(self) -> None:
        """Load and validate all examples from the file."""
        with self._path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # Validate required columns exist
            if reader.fieldnames is None:
                raise DatasetValidationError("CSV file is empty or has no header")

            missing_cols = []
            for field, col_name in self._column_mapping.items():
                if field in ("id", "query", "response") and col_name not in reader.fieldnames:
                    missing_cols.append(f"'{col_name}' (for {field})")

            if missing_cols:
                raise DatasetValidationError(
                    f"CSV is missing required columns: {', '.join(missing_cols)}"
                )

            for line_num, row in enumerate(reader, start=2):  # Header is line 1
                data = self._row_to_dict(row, line_num)
                self._examples.append(_validate_example(data, line=line_num))

    def _row_to_dict(self, row: dict[str, str], line_num: int) -> dict[str, Any]:
        """Convert a CSV row to an Example-compatible dict."""
        data: dict[str, Any] = {}

        # Required fields
        data["id"] = row.get(self._column_mapping["id"], "")
        data["query"] = row.get(self._column_mapping["query"], "")
        data["response"] = row.get(self._column_mapping["response"], "")

        # Optional: expected_followups (JSON string)
        followups_col = self._column_mapping.get("expected_followups", "expected_followups")
        followups_str = row.get(followups_col, "").strip()
        if followups_str:
            try:
                data["expected_followups"] = json.loads(followups_str)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(
                    f"Invalid JSON in '{followups_col}' column at line {line_num}: {e}",
                    line=line_num,
                ) from e

        return data

    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples."""
        return iter(self._examples)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self._examples)

    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        return {
            "source": "csv",
            "path": str(self._path),
            "num_examples": len(self._examples),
            "column_mapping": self._column_mapping,
        }


class IteratorDataset:
    """Dataset from any iterator or generator.

    Materializes the iterator on first access to support len() and
    multiple iterations.

    Example:
        >>> def example_generator():
        ...     for i in range(3):
        ...         yield {"id": str(i), "query": f"Q{i}", "response": f"R{i}"}
        >>>
        >>> dataset = IteratorDataset(example_generator())
        >>> len(dataset)
        3
        >>>
        >>> # With known length (for progress bars before materialization)
        >>> dataset = IteratorDataset(example_generator(), length=3)

    Args:
        iterator: An iterator yielding Example objects or dicts.
        length: Optional length hint (used before first access materializes data).

    Raises:
        DatasetValidationError: If any yielded item fails validation.
    """

    def __init__(
        self,
        iterator: Iterator[Example | dict[str, Any]],
        length: int | None = None,
    ) -> None:
        """Initialize with an iterator."""
        self._iterator: Iterator[Example | dict[str, Any]] | None = iterator
        self._length_hint = length
        self._examples: list[Example] | None = None

    def _materialize(self) -> None:
        """Materialize the iterator into a list of validated examples."""
        if self._examples is not None:
            return  # Already materialized

        if self._iterator is None:
            self._examples = []
            return

        self._examples = []
        for i, item in enumerate(self._iterator):
            self._examples.append(_validate_example(item, index=i))

        # Release the iterator
        self._iterator = None

    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples (materializes on first call)."""
        self._materialize()
        return iter(self._examples)  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Return number of examples (materializes on first call if no length hint)."""
        if self._examples is not None:
            return len(self._examples)

        if self._length_hint is not None:
            return self._length_hint

        # Must materialize to get length
        self._materialize()
        return len(self._examples)  # type: ignore[arg-type]

    @property
    def info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        if self._examples is not None:
            num_examples = len(self._examples)
        elif self._length_hint is not None:
            num_examples = self._length_hint
        else:
            num_examples = None

        return {
            "source": "iterator",
            "num_examples": num_examples,
            "materialized": self._examples is not None,
        }

