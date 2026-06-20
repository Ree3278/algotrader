from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "runtime_data"
REPORTS_DIR = RUNTIME_DIR / "reports"
MODELS_DIR = RUNTIME_DIR / "models" / "latest"
DATA_DIR = RUNTIME_DIR / "data"
RESEARCH_DIR = RUNTIME_DIR / "research"


CORE_PRICE_FACTORS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "ATR_14",
    "RSI_14",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "BB_upper",
    "BB_lower",
    "BB_pct_b",
    "volume_zscore_20d",
]


st.set_page_config(
    page_title="SPY ML Trading Research Demo",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fa;
        color: #17202a;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7dde5;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #4d5d6c;
    }
    .section-note {
        border-left: 4px solid #2764c5;
        background: #ffffff;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 1.25rem 0;
        color: #2b3440;
    }
    .artifact-ok {
        color: #166534;
        font-weight: 700;
    }
    .artifact-missing {
        color: #991b1b;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.{digits}f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if np.isinf(float(value)):
        return "inf"
    return f"{float(value):,.{digits}f}"


def intish(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(round(float(value))):,}"


def compact_date(value: object) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "n/a"
    return timestamp.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def load_runtime() -> dict[str, object]:
    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))

    fold_summaries = pd.read_csv(REPORTS_DIR / "fold_summaries.csv")
    predictions = pd.read_csv(REPORTS_DIR / "test_predictions.csv", parse_dates=["Date"])
    spy = pd.read_csv(DATA_DIR / "spy_daily.csv", parse_dates=["Date"])
    vix = pd.read_csv(DATA_DIR / "vix_daily.csv", parse_dates=["Date"])
    fold_manifest = pd.read_csv(MODELS_DIR / "fold_manifest.csv")
    ablation = pd.read_csv(RESEARCH_DIR / "ablation_results.csv")
    label_sweep = pd.read_csv(RESEARCH_DIR / "label_sweep_results.csv")

    for frame in (fold_summaries, predictions, spy, vix, fold_manifest, ablation, label_sweep):
        frame.columns = [column.strip() for column in frame.columns]

    predictions["Date"] = pd.to_datetime(predictions["Date"], utc=True, errors="coerce")
    spy["Date"] = pd.to_datetime(spy["Date"], utc=True, errors="coerce")
    vix["Date"] = pd.to_datetime(vix["Date"], utc=True, errors="coerce")

    return {
        "summary": summary,
        "manifest": manifest,
        "fold_summaries": fold_summaries,
        "predictions": predictions.dropna(subset=["Date"]),
        "spy": spy.dropna(subset=["Date"]),
        "vix": vix.dropna(subset=["Date"]),
        "fold_manifest": fold_manifest,
        "ablation": ablation,
        "label_sweep": label_sweep,
    }


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics, strict=True):
        column.metric(label, value, help=help_text)


def callout(text: str) -> None:
    st.markdown(f"<div class='section-note'>{text}</div>", unsafe_allow_html=True)


def readable_feature_name(feature: str) -> str:
    overrides = {
        "return_1d": "1-day return",
        "return_5d": "5-day return",
        "volatility_20d": "20-day volatility",
        "price_above_sma_200": "Price above 200-day SMA",
        "ATR_14": "14-day ATR",
        "RSI_14": "14-day RSI",
        "MACD_line": "MACD line",
        "MACD_signal": "MACD signal",
        "MACD_hist": "MACD histogram",
        "BB_upper": "Bollinger upper band",
        "BB_lower": "Bollinger lower band",
        "BB_pct_b": "Bollinger percent B",
        "volume_zscore_20d": "20-day volume z-score",
        "vix_zscore_60d": "60-day VIX z-score",
        "price_to_sma_200": "Price to 200-day SMA",
        "sma_200_slope_20d": "20-day 200-SMA slope",
        "sma_50_above_sma_200": "50-day SMA above 200-day SMA",
    }
    return overrides.get(feature, feature.replace("_", " "))


def feature_table(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        if feature in CORE_PRICE_FACTORS:
            group = "12-factor price indicator matrix"
        elif "sma" in feature or "price_above" in feature or "price_to" in feature:
            group = "Regime and trend state"
        elif "vix" in feature:
            group = "Volatility regime context"
        else:
            group = "Additional feature"
        rows.append({"Feature": feature, "Display name": readable_feature_name(feature), "Group": group})
    return pd.DataFrame(rows)


def artifact_status(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        exists = path.exists()
        rows.append(
            {
                "Artifact": str(path.relative_to(ROOT)),
                "Status": "Ready" if exists else "Missing",
                "Size": f"{path.stat().st_size / 1024:,.1f} KB" if exists and path.is_file() else "n/a",
            }
        )
    return pd.DataFrame(rows)


def render_sidebar(manifest: dict[str, object], fold_manifest: pd.DataFrame) -> None:
    experiment_name = manifest.get("experiment_name", "walk-forward experiment")
    created_at = compact_date(manifest.get("created_at"))
    referenced_models = fold_manifest["model_file"].dropna().nunique() if "model_file" in fold_manifest else 0

    st.sidebar.title("Demo Controls")
    st.sidebar.caption("Static Streamlit Cloud bundle. No API keys or live downloads required.")
    st.sidebar.markdown(f"**Experiment**  \n{experiment_name}")
    st.sidebar.markdown(f"**Created**  \n{created_at}")
    st.sidebar.markdown(f"**Fold models referenced**  \n{referenced_models}")
    st.sidebar.divider()
    st.sidebar.markdown("**Recruiter signals**")
    st.sidebar.markdown(
        "- Leakage-aware validation\n"
        "- Friction-aware backtesting\n"
        "- Feature engineering depth\n"
        "- XGBoost benchmark artifacts"
    )


def overview_tab(data: dict[str, object]) -> None:
    summary: dict[str, object] = data["summary"]  # type: ignore[assignment]
    manifest: dict[str, object] = data["manifest"]  # type: ignore[assignment]
    fold_summaries: pd.DataFrame = data["fold_summaries"]  # type: ignore[assignment]
    predictions: pd.DataFrame = data["predictions"]  # type: ignore[assignment]
    spy: pd.DataFrame = data["spy"]  # type: ignore[assignment]

    split_config = manifest["experiment_config"]["split_config"]
    backtest_config = manifest["experiment_config"]["backtest_config"]
    label_config = manifest["experiment_spec"]["labeler"]["config"]

    callout(
        "This demo is positioned as a research engineering artifact: fixed SPY daily data, "
        "saved fold models, purged walk-forward evaluation, triple-barrier labels, and "
        "transaction-cost-aware long/flat backtests."
    )

    metric_row(
        [
            ("Model backend", str(summary.get("model_backend", "n/a")).upper(), "Saved benchmark backend"),
            ("Walk-forward folds", intish(summary.get("fold_count")), "Out-of-sample annual test folds"),
            ("Feature columns", intish(summary.get("feature_count")), "Inputs used by the saved model"),
            ("Prediction rows", intish(summary.get("prediction_rows")), "Out-of-sample prediction records"),
        ]
    )
    metric_row(
        [
            ("Mean fold return", pct(summary.get("mean_total_return"), 2), "Average fold strategy return"),
            ("Mean Sharpe", num(summary.get("mean_sharpe"), 2), "Annualized fold Sharpe from net returns"),
            ("Mean drawdown", pct(summary.get("mean_max_drawdown"), 2), "Average fold max drawdown"),
            ("Mean win rate", pct(summary.get("mean_win_rate"), 1), "Winning closed trades"),
        ]
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Out-of-Sample Fold Performance")
        chart = fold_summaries.set_index("fold")[["total_return", "benchmark_total_return", "sharpe"]]
        st.line_chart(chart, height=320)
    with right:
        st.subheader("Research Design")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Train window", f"{split_config['train_size']} bars"],
                    ["Embargo", f"{split_config['embargo_size']} bars"],
                    ["Max label horizon", f"{split_config['max_label_horizon']} bars"],
                    ["Test window", f"{split_config['test_size']} bars"],
                    ["Profit / stop barrier", f"{label_config['profit_target_atr']} ATR / {label_config['stop_loss_atr']} ATR"],
                    ["Costs", f"{backtest_config['commission_bps']} bps commission + {backtest_config['slippage_bps']} bps slippage"],
                ],
                columns=["Component", "Setting"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("SPY Daily Close")
        price_view = spy[["Date", "close"]].set_index("Date")
        st.line_chart(price_view, height=260)
    with right:
        st.subheader("Model Probability Distribution")
        probability_bins = pd.cut(predictions["probability_long"], bins=np.linspace(0, 1, 21), include_lowest=True)
        probability_hist = probability_bins.value_counts().sort_index()
        probability_hist.index = probability_hist.index.astype(str)
        st.bar_chart(probability_hist, height=260)


def validation_tab(data: dict[str, object]) -> None:
    manifest: dict[str, object] = data["manifest"]  # type: ignore[assignment]
    fold_manifest: pd.DataFrame = data["fold_manifest"]  # type: ignore[assignment]
    fold_summaries: pd.DataFrame = data["fold_summaries"]  # type: ignore[assignment]

    split_config = manifest["experiment_config"]["split_config"]
    callout(
        "Leakage control is handled before model training: each fold trains on a trailing window, "
        "purges samples whose label horizon can overlap the test period, and applies a 10-bar embargo "
        "between training and testing."
    )

    metric_row(
        [
            ("Embargo", f"{split_config['embargo_size']} bars", "Gap between training and test windows"),
            ("Purge horizon", f"{split_config['max_label_horizon']} bars", "Triple-barrier max holding window"),
            ("Train window", f"{split_config['train_size']} bars", "Rolling model training sample"),
            ("Test window", f"{split_config['test_size']} bars", "Out-of-sample evaluation sample"),
        ]
    )

    timeline = fold_manifest.copy()
    date_columns = ["train_start", "train_end", "calibration_start", "calibration_end", "test_start", "test_end"]
    for column in date_columns:
        if column in timeline:
            timeline[column] = timeline[column].map(compact_date)

    st.subheader("Fold Timeline Audit")
    st.dataframe(
        timeline[
            [
                "fold",
                "train_start",
                "train_end",
                "calibration_start",
                "calibration_end",
                "test_start",
                "test_end",
                "threshold_selection_mode",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Fold Stability")
    st.line_chart(fold_summaries.set_index("fold")[["selected_threshold", "calibration_exposure", "exposure"]], height=300)


def features_tab(data: dict[str, object]) -> None:
    manifest: dict[str, object] = data["manifest"]  # type: ignore[assignment]
    predictions: pd.DataFrame = data["predictions"]  # type: ignore[assignment]
    label_sweep: pd.DataFrame = data["label_sweep"]  # type: ignore[assignment]

    profile = manifest["experiment_spec"]["profile"]
    feature_columns = list(profile["feature_columns"])
    label_config = manifest["experiment_spec"]["labeler"]["config"]

    callout(
        "The model uses 12 core price indicators plus causal regime/trend state and VIX context. "
        "Labels come from triple-barrier return rules with profit, stop, and timeout outcomes."
    )

    metric_row(
        [
            ("Core price factors", intish(len([feature for feature in feature_columns if feature in CORE_PRICE_FACTORS])), None),
            ("Total features", intish(len(feature_columns)), None),
            ("Profit target", f"{label_config['profit_target_atr']} ATR", None),
            ("Stop loss", f"{label_config['stop_loss_atr']} ATR", None),
        ]
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Feature Matrix")
        st.dataframe(feature_table(feature_columns), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Out-of-Sample Label Mix")
        label_mix = predictions["label"].map({0: "Flat", 1: "Long"}).value_counts(normalize=True).rename("Share")
        st.bar_chart(label_mix, height=260)
        st.subheader("Threshold Regime Mix")
        regime_mix = predictions["threshold_regime"].value_counts(normalize=True).rename("Share")
        st.bar_chart(regime_mix, height=260)

    st.subheader("Triple-Barrier Label Sweep")
    sweep_view = label_sweep.sort_values("mean_sharpe", ascending=False).copy()
    sweep_view["mean_total_return"] = sweep_view["mean_total_return"].map(lambda value: pct(value, 2))
    sweep_view["mean_max_drawdown"] = sweep_view["mean_max_drawdown"].map(lambda value: pct(value, 2))
    sweep_view["mean_sharpe"] = sweep_view["mean_sharpe"].map(lambda value: num(value, 2))
    st.dataframe(
        sweep_view[
            [
                "run_slug",
                "profit_target_atr",
                "stop_loss_atr",
                "max_holding_bars",
                "mean_total_return",
                "mean_sharpe",
                "mean_max_drawdown",
                "label_long_pct",
                "hit_profit_target_pct",
                "hit_stop_loss_pct",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def backtest_tab(data: dict[str, object]) -> None:
    summary: dict[str, object] = data["summary"]  # type: ignore[assignment]
    manifest: dict[str, object] = data["manifest"]  # type: ignore[assignment]
    fold_summaries: pd.DataFrame = data["fold_summaries"]  # type: ignore[assignment]
    ablation: pd.DataFrame = data["ablation"]  # type: ignore[assignment]

    backtest_config = manifest["experiment_config"]["backtest_config"]
    round_trip_bps = 2 * (backtest_config["commission_bps"] + backtest_config["slippage_bps"])

    callout(
        "Backtests use net returns after commission and slippage. Signals are converted into a "
        "long/flat event simulation with one open position at a time, turnover, exposure, drawdown, "
        "win rate, and profit factor tracked by fold."
    )

    metric_row(
        [
            ("Commission", f"{backtest_config['commission_bps']} bps", "Applied per trade side"),
            ("Slippage", f"{backtest_config['slippage_bps']} bps", "Applied per trade side"),
            ("Round trip cost", f"{round_trip_bps:g} bps", "Entry plus exit estimated friction"),
            ("Mean turnover", num(summary.get("mean_turnover"), 1), "Average fold entry and exit turnover"),
        ]
    )
    metric_row(
        [
            ("Mean exposure", pct(summary.get("mean_exposure"), 1), "Average long allocation"),
            ("Mean trades", num(summary.get("mean_trade_count"), 1), "Average closed trades per fold"),
            ("Best fold Sharpe", num(summary.get("best_fold_sharpe"), 2), "Best out-of-sample fold"),
            ("Worst drawdown", pct(summary.get("worst_fold_drawdown"), 1), "Worst single fold drawdown"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Friction-Aware Metrics by Fold")
        st.line_chart(fold_summaries.set_index("fold")[["trade_count", "turnover", "exposure"]], height=300)
    with right:
        st.subheader("Risk and Return by Fold")
        st.line_chart(fold_summaries.set_index("fold")[["total_return", "max_drawdown", "win_rate"]], height=300)

    st.subheader("Benchmark Experiment Comparison")
    ablation_view = ablation.copy()
    for column in ("mean_total_return", "mean_max_drawdown", "mean_win_rate"):
        ablation_view[column] = ablation_view[column].map(lambda value: pct(value, 2))
    ablation_view["mean_sharpe"] = ablation_view["mean_sharpe"].map(lambda value: num(value, 2))
    st.dataframe(
        ablation_view[
            [
                "variant",
                "feature_count",
                "fold_count",
                "mean_total_return",
                "mean_sharpe",
                "mean_trade_count",
                "mean_max_drawdown",
                "mean_win_rate",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def artifacts_tab(data: dict[str, object]) -> None:
    manifest: dict[str, object] = data["manifest"]  # type: ignore[assignment]
    fold_manifest: pd.DataFrame = data["fold_manifest"]  # type: ignore[assignment]

    model_files = [MODELS_DIR / model_file for model_file in fold_manifest["model_file"].dropna().unique()]
    calibrator_files = [
        MODELS_DIR / calibrator_file
        for calibrator_file in fold_manifest["calibrator_file"].dropna().unique()
        if str(calibrator_file) != "nan"
    ]
    required_paths = [
        REPORTS_DIR / "summary.json",
        REPORTS_DIR / "fold_summaries.csv",
        REPORTS_DIR / "test_predictions.csv",
        DATA_DIR / "spy_daily.csv",
        DATA_DIR / "vix_daily.csv",
        MODELS_DIR / "manifest.json",
        MODELS_DIR / "fold_manifest.csv",
        RESEARCH_DIR / "ablation_results.csv",
        RESEARCH_DIR / "label_sweep_results.csv",
        *model_files,
        *calibrator_files,
    ]

    ready_count = sum(path.exists() for path in required_paths)
    callout(
        "This page checks whether the static Streamlit Cloud bundle includes the data, reports, "
        "research tables, manifest, and saved fold-level model files needed to reproduce the demo narrative."
    )
    metric_row(
        [
            ("Required artifacts", intish(len(required_paths)), None),
            ("Ready artifacts", intish(ready_count), None),
            ("Runtime bundle", f"{sum(path.stat().st_size for path in RUNTIME_DIR.rglob('*') if path.is_file()) / (1024 ** 2):.1f} MB", None),
            ("Experiment", str(manifest.get("model_backend", "xgboost")).upper(), None),
        ]
    )

    status = artifact_status(required_paths)
    status["Status"] = status["Status"].map(
        lambda value: "<span class='artifact-ok'>Ready</span>"
        if value == "Ready"
        else "<span class='artifact-missing'>Missing</span>"
    )
    st.markdown(status.to_html(escape=False, index=False), unsafe_allow_html=True)


def main() -> None:
    data = load_runtime()
    manifest: dict[str, object] = data["manifest"]  # type: ignore[assignment]
    fold_manifest: pd.DataFrame = data["fold_manifest"]  # type: ignore[assignment]
    summary: dict[str, object] = data["summary"]  # type: ignore[assignment]

    render_sidebar(manifest, fold_manifest)

    st.title("SPY ML Trading Research Demo")
    st.caption(
        "Entry-level data science portfolio dashboard focused on leakage control, "
        "feature engineering, XGBoost benchmarking, and friction-aware evaluation."
    )

    tabs = st.tabs(
        [
            "Executive Summary",
            "Leakage Control",
            "Features & Labels",
            "Backtest Friction",
            "Deployment Artifacts",
        ]
    )
    with tabs[0]:
        overview_tab(data)
    with tabs[1]:
        validation_tab(data)
    with tabs[2]:
        features_tab(data)
    with tabs[3]:
        backtest_tab(data)
    with tabs[4]:
        artifacts_tab(data)

    st.divider()
    st.caption(
        f"Static bundle: {summary.get('symbol', 'SPY')} daily OHLCV, VIX context, "
        f"{summary.get('fold_count', 'n/a')} walk-forward folds, "
        f"{summary.get('prediction_rows', 'n/a')} out-of-sample predictions."
    )


if __name__ == "__main__":
    main()
