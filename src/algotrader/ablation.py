"""Feature ablation runner for price, regime, and sentiment variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from algotrader.ingestion import save_ohlcv_csv
from algotrader.pipeline import (
    TestPipelineConfig,
    _add_shared_data_args,
    _add_shared_model_args,
    _load_or_fetch_frames,
    _settings_from_args,
    run_pipeline,
)


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_vix: bool
    use_sentiment: bool


ABLATION_VARIANTS = (
    AblationVariant(name="price_only", use_vix=False, use_sentiment=False),
    AblationVariant(name="price_plus_regime", use_vix=True, use_sentiment=False),
    AblationVariant(name="price_plus_regime_plus_sentiment", use_vix=True, use_sentiment=True),
)


def _save_timeseries_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)


def _materialize_local_inputs(
    base_config: TestPipelineConfig,
    output_dir: Path,
) -> tuple[TestPipelineConfig, dict[str, Path | None]]:
    price_frame, vix_frame, sentiment_frame = _load_or_fetch_frames(base_config)
    input_root = output_dir / "inputs"
    price_csv = input_root / f"{base_config.symbol.lower()}_daily.csv"
    vix_csv = input_root / "vix_daily.csv" if vix_frame is not None else None
    sentiment_csv = input_root / "sentiment_daily.csv" if sentiment_frame is not None else None

    save_ohlcv_csv(price_frame, price_csv)
    if vix_frame is not None and vix_csv is not None:
        save_ohlcv_csv(vix_frame, vix_csv)
    if sentiment_frame is not None and sentiment_csv is not None:
        _save_timeseries_csv(sentiment_frame, sentiment_csv)

    local_config = replace(
        base_config,
        input_csv=price_csv,
        vix_input_csv=vix_csv,
        sentiment_features_csv=sentiment_csv,
        fetch_yfinance=False,
        fetch_alpha_vantage=False,
    )
    return local_config, {"input_csv": price_csv, "vix_csv": vix_csv, "sentiment_csv": sentiment_csv}


def _top_result_lines(results: pd.DataFrame) -> list[str]:
    lines = ["Ablation results:"]
    for row in results.sort_values(by="mean_sharpe", ascending=False).itertuples(index=False):
        lines.append(
            "  "
            f"{row.variant}: "
            f"sharpe={row.mean_sharpe:.3f}, "
            f"return={100 * row.mean_total_return:.2f}%, "
            f"dd={100 * row.mean_max_drawdown:.2f}%, "
            f"trades={row.mean_trade_count:.2f}, "
            f"features={row.feature_count}"
        )
    return lines


def run_feature_ablation(
    base_config: TestPipelineConfig,
    *,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run the three ablation variants under a fixed label configuration."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    local_base_config, materialized_inputs = _materialize_local_inputs(base_config, destination)

    rows: list[dict[str, object]] = []
    for variant in ABLATION_VARIANTS:
        if variant.use_vix and local_base_config.vix_input_csv is None:
            raise ValueError("VIX input is required for regime ablation variants")
        if variant.use_sentiment and local_base_config.sentiment_features_csv is None:
            raise ValueError("Sentiment input is required for sentiment ablation variant")

        run_root = destination / "runs" / variant.name
        run_config = replace(
            local_base_config,
            vix_input_csv=local_base_config.vix_input_csv if variant.use_vix else None,
            sentiment_features_csv=local_base_config.sentiment_features_csv if variant.use_sentiment else None,
            model_dir=run_root / "models",
            output_dir=run_root / "reports",
        )

        result = run_pipeline(run_config)
        rows.append(
            {
                "variant": variant.name,
                "feature_count": int(result.summary["feature_count"]),
                "dataset_rows": int(result.summary["dataset_rows"]),
                "fold_count": int(result.summary["fold_count"]),
                "mean_total_return": float(result.summary.get("mean_total_return", 0.0)),
                "mean_sharpe": float(result.summary.get("mean_sharpe", 0.0)),
                "mean_trade_count": float(result.summary.get("mean_trade_count", 0.0)),
                "mean_max_drawdown": float(result.summary.get("mean_max_drawdown", 0.0)),
                "mean_profit_factor": float(result.summary.get("mean_profit_factor", 0.0)),
                "mean_win_rate": float(result.summary.get("mean_win_rate", 0.0)),
                "model_dir": str(run_config.model_dir),
                "reports_dir": str(run_config.output_dir),
            }
        )

    results = pd.DataFrame(rows).sort_values(by="mean_sharpe", ascending=False).reset_index(drop=True)
    csv_path = destination / "ablation_results.csv"
    json_path = destination / "ablation_results.json"
    summary_path = destination / "ablation_summary.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(results.to_json(orient="records", indent=2), encoding="utf-8")
    summary_payload = {
        "materialized_inputs": {key: None if value is None else str(value) for key, value in materialized_inputs.items()},
        "best_by_mean_sharpe": None if results.empty else results.iloc[0].to_dict(),
        "variants": json.loads(results.to_json(orient="records")),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return results, {"csv": csv_path, "json": json_path, "summary": summary_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run price/regime/sentiment feature ablations.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ablation/latest"),
        help="Directory for ablation outputs and per-variant artifacts",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TestPipelineConfig:
    return TestPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv,
        vix_input_csv=args.vix_csv,
        sentiment_features_csv=args.sentiment_features_csv,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        settings=_settings_from_args(args),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    results, paths = run_feature_ablation(_config_from_args(args), output_dir=args.output_dir)
    for line in _top_result_lines(results):
        print(line)
    print(f"results_csv={paths['csv']}")
    print(f"results_json={paths['json']}")
    print(f"summary_json={paths['summary']}")
