"""Report writers for walk-forward experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from algotrader.training.experiment import WalkForwardExperimentResult


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if pd.isna(value):
        return None
    return value


def to_json_safe(value: Any) -> Any:
    """Convert nested values into valid, stable JSON primitives."""

    return _to_jsonable(value)


def build_experiment_summary(
    result: WalkForwardExperimentResult,
    *,
    symbol: str,
    dataset_rows: int,
    feature_count: int,
    model_backend: str | None = None,
) -> dict[str, Any]:
    """Create a compact top-level summary for the experiment."""

    fold_summaries = result.fold_summaries
    summary: dict[str, Any] = {
        "symbol": symbol,
        "dataset_rows": int(dataset_rows),
        "feature_count": int(feature_count),
        "fold_count": int(len(fold_summaries)),
        "prediction_rows": int(len(result.test_predictions)),
        "model_backend": model_backend,
    }

    if not fold_summaries.empty:
        numeric_means = fold_summaries.select_dtypes(include=["number"]).mean(numeric_only=True)
        summary.update({f"mean_{key}": _to_jsonable(value) for key, value in numeric_means.items()})
        summary["best_fold_sharpe"] = _to_jsonable(fold_summaries["sharpe"].max())
        summary["worst_fold_drawdown"] = _to_jsonable(fold_summaries["max_drawdown"].min())

    return summary


def write_experiment_reports(
    result: WalkForwardExperimentResult,
    output_dir: str | Path,
    *,
    summary: dict[str, Any],
) -> dict[str, Path]:
    """Write CSV and JSON artifacts for an experiment run."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    fold_summary_path = destination / "fold_summaries.csv"
    predictions_path = destination / "test_predictions.csv"
    summary_path = destination / "summary.json"

    result.fold_summaries.to_csv(fold_summary_path, index=False)
    result.test_predictions.to_csv(predictions_path, index=True)
    summary_path.write_text(
        json.dumps(to_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "fold_summaries": fold_summary_path,
        "test_predictions": predictions_path,
        "summary": summary_path,
    }


def _format_distribution_lines(distribution: pd.Series, mapping: dict[Any, str] | None = None) -> list[str]:
    lines: list[str] = []
    for key, value in distribution.items():
        label = mapping.get(key, str(key)) if mapping is not None else str(key)
        lines.append(f"  {label}: {100 * float(value):.2f}%")
    if not lines:
        lines.append("  none")
    return lines


def format_test_terminal_summary(
    summary: dict[str, Any],
    dataset: pd.DataFrame,
) -> str:
    """Render the small set of headline diagnostics for terminal output."""

    label_distribution = dataset["label"].value_counts(normalize=True).sort_index()
    hit_reason_distribution = dataset["hit_reason"].value_counts(normalize=True)

    lines = [
        f"Symbol: {summary['symbol']}",
        f"Mean Total Return: {100 * float(summary.get('mean_total_return', 0.0)):.2f}%",
        f"Mean Sharpe: {float(summary.get('mean_sharpe', 0.0)):.2f}",
        f"Mean Trade Count: {float(summary.get('mean_trade_count', 0.0)):.2f}",
        f"Mean Max Drawdown: {100 * float(summary.get('mean_max_drawdown', 0.0)):.2f}%",
        "Label Distribution:",
        *_format_distribution_lines(label_distribution, mapping={0: "Flat", 1: "Long"}),
        "Hit-Reason Distribution:",
        *_format_distribution_lines(hit_reason_distribution),
    ]
    return "\n".join(lines)


def format_ablation_results_table(results: pd.DataFrame) -> str:
    """Render an aligned ablation comparison table for terminal output."""

    if results.empty:
        return "No ablation results."

    display = results.copy()
    display["mean_total_return"] = display["mean_total_return"].map(lambda value: f"{100 * float(value):.2f}%")
    display["mean_sharpe"] = display["mean_sharpe"].map(lambda value: f"{float(value):.3f}")
    display["mean_max_drawdown"] = display["mean_max_drawdown"].map(lambda value: f"{100 * float(value):.2f}%")
    display["mean_trade_count"] = display["mean_trade_count"].map(lambda value: f"{float(value):.2f}")
    display["feature_count"] = display["feature_count"].map(lambda value: str(int(value)))

    columns = [
        ("variant", "Variant"),
        ("mean_sharpe", "Sharpe"),
        ("mean_total_return", "Return"),
        ("mean_max_drawdown", "Drawdown"),
        ("mean_trade_count", "Trades"),
        ("feature_count", "Features"),
    ]
    widths = {
        key: max(len(header), display[key].astype(str).map(len).max())
        for key, header in columns
    }
    header_line = "  ".join(header.ljust(widths[key]) for key, header in columns)
    separator_line = "  ".join("-" * widths[key] for key, _ in columns)
    row_lines = [
        "  ".join(str(row[key]).ljust(widths[key]) for key, _ in columns)
        for row in display.to_dict(orient="records")
    ]
    return "\n".join([header_line, separator_line, *row_lines])
