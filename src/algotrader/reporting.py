"""Report writers for walk-forward experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from algotrader.training.experiment import WalkForwardExperimentResult


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


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
        json.dumps({key: _to_jsonable(value) for key, value in summary.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "fold_summaries": fold_summary_path,
        "test_predictions": predictions_path,
        "summary": summary_path,
    }
