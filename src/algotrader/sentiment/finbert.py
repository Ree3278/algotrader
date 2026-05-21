"""FinBERT scoring helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"


def score_news_with_finbert(
    frame: pd.DataFrame,
    *,
    model_name: str = DEFAULT_FINBERT_MODEL,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Score news rows with FinBERT probabilities."""

    if frame.empty:
        result = frame.copy()
        result["p_positive"] = []
        result["p_negative"] = []
        result["p_neutral"] = []
        return result

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("transformers is required for FinBERT scoring. Install the `nlp` extra.") from exc

    scored = frame.copy()
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)
    if "summary" in scored.columns:
        scored["summary"] = scored["summary"].fillna("").astype(str)
    else:
        scored["summary"] = ""
    texts = (scored["headline"].fillna("").astype(str) + " " + scored["summary"]).str.strip().tolist()

    classifier = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
        truncation=True,
    )
    predictions = classifier(texts, batch_size=batch_size)

    p_positive: list[float] = []
    p_negative: list[float] = []
    p_neutral: list[float] = []

    for prediction in predictions:
        score_map = {item["label"].lower(): float(item["score"]) for item in prediction}
        p_positive.append(score_map.get("positive", 0.0))
        p_negative.append(score_map.get("negative", 0.0))
        p_neutral.append(score_map.get("neutral", 0.0))

    scored["p_positive"] = p_positive
    scored["p_negative"] = p_negative
    scored["p_neutral"] = p_neutral
    return scored
