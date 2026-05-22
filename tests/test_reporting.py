from __future__ import annotations

from algotrader.reporting import to_json_safe


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
