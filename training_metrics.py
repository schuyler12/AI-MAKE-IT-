"""Utilities for analyzing training metrics stored as JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any



def _as_float(value: Any, field_name: str, line_number: int) -> float:
    """Convert a metric field to float, raising ValueError with context."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: field {field_name!r} must be numeric"
        ) from exc


def analyze_training_jsonl(
    jsonl_path: str, output_graph: str = "training_metrics.png"
) -> list[dict]:
    """Analyze a JSONL training log and save a metrics graph.

    Each JSONL row should contain ``step``, ``loss``, ``learning_rate``,
    ``correct``, and ``total``. The optional ``failed`` field defaults to 0.
    Rows with missing required fields, invalid JSON, non-numeric metrics, or
    non-positive totals are skipped.

    Args:
        jsonl_path: Path to the JSONL training log.
        output_graph: Path where the generated PNG graph should be saved.

    Returns:
        A list of dictionaries containing the calculated metrics for each valid
        row in the input file.
    """
    required_fields = ("step", "loss", "learning_rate", "correct", "total")
    metrics: list[dict] = []

    with Path(jsonl_path).open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(row, dict) or any(
                field not in row for field in required_fields
            ):
                continue

            try:
                step = _as_float(row["step"], "step", line_number)
                loss = _as_float(row["loss"], "loss", line_number)
                learning_rate = _as_float(
                    row["learning_rate"], "learning_rate", line_number
                )
                correct = _as_float(row["correct"], "correct", line_number)
                total = _as_float(row["total"], "total", line_number)
                failed = _as_float(row.get("failed", 0), "failed", line_number)
            except ValueError:
                continue

            if total <= 0:
                continue

            accuracy = correct / total
            error_rate = 1 - accuracy
            fail_rate = failed / total

            metrics.append(
                {
                    "step": step,
                    "loss": loss,
                    "learning_rate": learning_rate,
                    "correct": correct,
                    "total": total,
                    "failed": failed,
                    "accuracy": accuracy,
                    "error_rate": error_rate,
                    "fail_rate": fail_rate,
                }
            )

    _plot_metrics(metrics, output_graph)
    return metrics


def _plot_metrics(metrics: list[dict], output_graph: str) -> None:
    """Generate and save a line graph for calculated training metrics."""
    output_path = Path(output_graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))

    if metrics:
        steps = [metric["step"] for metric in metrics]
        series = (
            ("Accuracy", "accuracy"),
            ("Error rate", "error_rate"),
            ("Fail rate", "fail_rate"),
            ("Loss", "loss"),
            ("Learning rate", "learning_rate"),
        )
        for label, key in series:
            plt.plot(
                steps, [metric[key] for metric in metrics], marker="o", label=label
            )

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a JSONL training log and save a metrics graph."
    )
    parser.add_argument("jsonl_path", help="Path to the JSONL training log.")
    parser.add_argument(
        "--output",
        default="training_metrics.png",
        help="Path for the generated graph image (default: training_metrics.png).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = analyze_training_jsonl(args.jsonl_path, args.output)
    print(f"Processed {len(metrics)} valid rows. Graph saved to {args.output}")


if __name__ == "__main__":
    main()
