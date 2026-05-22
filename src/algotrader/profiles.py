"""Composable model-profile factory built from feature blocks."""

from __future__ import annotations

from dataclasses import dataclass

from algotrader.training.dataset import (
    DEFAULT_FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    TREND_STATE_FEATURE_COLUMNS,
    VOL_STATE_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    feature_columns: tuple[str, ...]
    requires_vix: bool = False
    requires_sentiment: bool = False


@dataclass(frozen=True)
class ModelProfile:
    name: str
    blocks: tuple[FeatureBlock, ...]

    @property
    def feature_columns(self) -> list[str]:
        ordered: list[str] = []
        for block in self.blocks:
            for column in block.feature_columns:
                if column not in ordered:
                    ordered.append(column)
        return ordered

    @property
    def requires_vix(self) -> bool:
        return any(block.requires_vix for block in self.blocks)

    @property
    def requires_sentiment(self) -> bool:
        return any(block.requires_sentiment for block in self.blocks)

    @property
    def block_names(self) -> list[str]:
        return [block.name for block in self.blocks]


FEATURE_BLOCKS = {
    "price_only": FeatureBlock("price_only", tuple(DEFAULT_FEATURE_COLUMNS)),
    "regime": FeatureBlock("regime", tuple(REGIME_FEATURE_COLUMNS), requires_vix=True),
    "trend_state": FeatureBlock("trend_state", tuple(TREND_STATE_FEATURE_COLUMNS)),
    "vol_state": FeatureBlock("vol_state", tuple(VOL_STATE_FEATURE_COLUMNS)),
    "sentiment": FeatureBlock("sentiment", tuple(SENTIMENT_FEATURE_COLUMNS), requires_sentiment=True),
}

PROFILE_PRESETS = {
    "price_only": ("price_only",),
    "price_plus_regime": ("price_only", "regime"),
    "price_plus_regime_plus_trend_state": ("price_only", "regime", "trend_state"),
    "price_plus_regime_plus_trend_state_plus_vol_state": ("price_only", "regime", "trend_state", "vol_state"),
    "price_plus_regime_plus_sentiment": ("price_only", "regime", "sentiment"),
}


def build_model_profile(
    *,
    name: str | None = None,
    block_names: list[str] | tuple[str, ...] | None = None,
) -> ModelProfile:
    """Build a model profile from a preset name or explicit block list."""

    if (name is None) == (block_names is None):
        raise ValueError("Provide exactly one of name or block_names")

    resolved_block_names = PROFILE_PRESETS[name] if name is not None else tuple(block_names or ())
    missing = [block_name for block_name in resolved_block_names if block_name not in FEATURE_BLOCKS]
    if missing:
        raise ValueError(f"Unknown feature blocks: {missing}")
    blocks = tuple(FEATURE_BLOCKS[block_name] for block_name in resolved_block_names)
    profile_name = name or "custom_" + "_plus_".join(resolved_block_names)
    return ModelProfile(name=profile_name, blocks=blocks)


def list_profile_names() -> list[str]:
    return list(PROFILE_PRESETS)
