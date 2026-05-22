from __future__ import annotations

import pandas as pd

from algotrader.reporting import format_ablation_results_table, to_json_safe


def test_to_json_safe_serializes_non_finite_floats_stably() -> None:
    payload = {
        "positive_inf": float("inf"),
        "negative_inf": float("-inf"),
        "nested": [{"value": float("inf")}],
    }

    serialized = to_json_safe(payload)

    assert serialized["positive_inf"] == "Infinity"
    assert serialized["negative_inf"] == "-Infinity"
    assert serialized["nested"][0]["value"] == "Infinity"


def test_format_ablation_results_table_renders_aligned_headers() -> None:
    results = pd.DataFrame(
        [
            {
                "variant": "price_plus_regime",
                "mean_sharpe": 0.331,
                "mean_total_return": 0.022,
                "mean_max_drawdown": -0.115,
                "mean_trade_count": 29.37,
                "feature_count": 14,
            }
        ]
    )

    table = format_ablation_results_table(results)

    assert "Variant" in table
    assert "Sharpe" in table
    assert "Return" in table
    assert "price_plus_regime" in table
