"""Artifact-driven debugging metrics for trained runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from algotrader.ingestion import load_ohlcv_csv
from algotrader.training.artifacts import load_model_artifact, load_training_manifest
from algotrader.training.dataset import build_training_dataset


def _rounded_pct_map(series: pd.Series) -> dict[str, float]:
    return {str(key): round(float(value) * 100, 2) for key, value in series.items()}


def compute_debug_metrics(
    *,
    input_csv: str | Path,
    model_dir: str | Path = "models/latest",
    reports_dir: str | Path = "reports/latest",
) -> dict[str, Any]:
    """Compute label mix, fold Sharpe chronology, and feature importances."""

    price_frame = load_ohlcv_csv(input_csv)
    dataset = build_training_dataset(price_frame)
    manifest, fold_manifest = load_training_manifest(model_dir)
    reports_path = Path(reports_dir)
    fold_summaries = pd.read_csv(reports_path / "fold_summaries.csv")

    label_distribution = dataset.data["label"].value_counts(normalize=True).sort_index()
    hit_reason_distribution = dataset.data["hit_reason"].value_counts(normalize=True)

    sharpe_records = []
    if "fold" in fold_summaries.columns and "sharpe" in fold_summaries.columns:
        fold_summary_subset = fold_summaries[["fold", "sharpe"]].merge(
            fold_manifest[["fold", "test_start", "test_end"]],
            on="fold",
            how="left",
        )
        for row in fold_summary_subset.sort_values("fold").itertuples(index=False):
            sharpe_records.append(
                {
                    "fold": int(row.fold),
                    "test_start": None if pd.isna(row.test_start) else pd.Timestamp(row.test_start).isoformat(),
                    "test_end": None if pd.isna(row.test_end) else pd.Timestamp(row.test_end).isoformat(),
                    "sharpe": None if pd.isna(row.sharpe) else round(float(row.sharpe), 4),
                }
            )

    feature_columns = manifest["feature_columns"]
    model_backends: dict[str, int] = {}
    fold_importances: list[np.ndarray] = []
    representative_fold: dict[str, Any] | None = None

    for row in fold_manifest.itertuples(index=False):
        model = load_model_artifact(Path(model_dir) / row.model_file)
        backend = getattr(model, "_algotrader_backend", type(model).__name__)
        model_backends[str(backend)] = model_backends.get(str(backend), 0) + 1

        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fold_importances.append(np.asarray(importances, dtype=float))
            if representative_fold is None:
                ranked = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
                representative_fold = {
                    "model_file": row.model_file,
                    "top_features": [
                        {"feature": feature, "importance": round(float(value), 6)}
                        for feature, value in ranked[:10]
                    ],
                }

    result: dict[str, Any] = {
        "dataset_rows": int(len(dataset.data)),
        "label_distribution_pct": {
            "flat_0": round(100 * float(label_distribution.get(0, 0.0)), 2),
            "long_1": round(100 * float(label_distribution.get(1, 0.0)), 2),
        },
        "hit_reason_pct": _rounded_pct_map(hit_reason_distribution),
        "fold_count": int(len(fold_manifest)),
        "fold_sharpes": sharpe_records,
        "model_backends": model_backends,
    }

    if fold_importances:
        avg_importance = np.vstack(fold_importances).mean(axis=0)
        ranked_avg = sorted(zip(feature_columns, avg_importance), key=lambda item: item[1], reverse=True)
        result["avg_feature_importances"] = [
            {"feature": feature, "importance": round(float(value), 6)}
            for feature, value in ranked_avg
        ]
    else:
        result["avg_feature_importances"] = None

    result["representative_fold"] = representative_fold
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute debugging metrics for a trained algotrader run.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to normalized OHLCV CSV")
    parser.add_argument("--model-dir", type=Path, default=Path("models/latest"), help="Directory containing saved fold models")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/latest"), help="Directory containing fold summary reports")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = compute_debug_metrics(
        input_csv=args.input_csv,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(metrics, indent=2))
