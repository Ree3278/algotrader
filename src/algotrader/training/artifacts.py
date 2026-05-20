"""Serialization helpers for trained fold artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_model_artifact(model: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(model, handle)


def load_model_artifact(path: str | Path) -> Any:
    source = Path(path)
    with source.open("rb") as handle:
        return pickle.load(handle)


def save_training_manifest(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    fold_manifest: pd.DataFrame,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = destination / "manifest.json"
    folds_path = destination / "fold_manifest.csv"

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    fold_manifest.to_csv(folds_path, index=False)
    return {"manifest": manifest_path, "fold_manifest": folds_path}


def load_training_manifest(model_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    source = Path(model_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(source / "fold_manifest.csv")
    for column in (
        "train_start",
        "train_end",
        "calibration_start",
        "calibration_end",
        "test_start",
        "test_end",
    ):
        if column in folds.columns:
            folds[column] = pd.to_datetime(folds[column], utc=True, errors="coerce")
    return manifest, folds
