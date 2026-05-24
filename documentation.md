# Algotrader Modular Documentation

This document explains how the codebase is organized, how the modular pieces fit together, and how to reuse the framework for future experiments.

## Goal

The project is structured so that a trading experiment is assembled from reusable parts instead of hardcoded pipeline branches.

The main idea is:

- data comes from standardized sources
- features are grouped into reusable blocks
- blocks are combined into profiles
- profiles, labels, model config, decision policy, and evaluation rules are wrapped into an `ExperimentSpec`
- runners like `train`, `test`, `ablation`, and `holdout` consume the same spec

That lets you build a new experiment by changing configuration and composition, not by rewriting pipeline code.

## High-Level Architecture

Main modules:

- [src/algotrader/ingestion](/Users/reezhan/Documents/algotrader/src/algotrader/ingestion)
- [src/algotrader/features](/Users/reezhan/Documents/algotrader/src/algotrader/features)
- [src/algotrader/labels](/Users/reezhan/Documents/algotrader/src/algotrader/labels)
- [src/algotrader/profiles.py](/Users/reezhan/Documents/algotrader/src/algotrader/profiles.py:1)
- [src/algotrader/thresholds.py](/Users/reezhan/Documents/algotrader/src/algotrader/thresholds.py:1)
- [src/algotrader/specs.py](/Users/reezhan/Documents/algotrader/src/algotrader/specs.py:1)
- [src/algotrader/experiment_registry.py](/Users/reezhan/Documents/algotrader/src/algotrader/experiment_registry.py:1)
- [src/algotrader/training](/Users/reezhan/Documents/algotrader/src/algotrader/training)
- [src/algotrader/backtest](/Users/reezhan/Documents/algotrader/src/algotrader/backtest)
- [src/algotrader/reporting.py](/Users/reezhan/Documents/algotrader/src/algotrader/reporting.py:1)
- [src/algotrader/pipeline.py](/Users/reezhan/Documents/algotrader/src/algotrader/pipeline.py:1)
- [src/algotrader/ablation.py](/Users/reezhan/Documents/algotrader/src/algotrader/ablation.py:1)
- [src/algotrader/sweep.py](/Users/reezhan/Documents/algotrader/src/algotrader/sweep.py:1)
- [src/algotrader/holdout.py](/Users/reezhan/Documents/algotrader/src/algotrader/holdout.py:1)
- [src/algotrader/settings.py](/Users/reezhan/Documents/algotrader/src/algotrader/settings.py:1)

## Core Building Blocks

### 1. Data and Ingestion

Responsibility:

- fetch raw market data
- normalize it into a common OHLCV format
- optionally load companion data like VIX or sentiment

Examples:

- `yfinance` daily fetch
- Alpha Vantage daily fetch
- local normalized CSV load

Important rule:

- everything downstream expects normalized frames with a time index and predictable column names

### 2. Feature Blocks

Feature blocks live in [profiles.py](/Users/reezhan/Documents/algotrader/src/algotrader/profiles.py:1).

A feature block is one coherent family of inputs.

Current blocks:

- `price_only`
- `regime`
- `trend_state`
- `atr_percentile`
- `vol_state`
- `sentiment`

Each block knows:

- its name
- its feature columns
- whether it requires VIX
- whether it requires sentiment

You can inspect the block registry through:

- `build_feature_block(name)`
- `list_feature_block_names()`

### 3. Model Profiles

A profile is just a composition of feature blocks.

Examples:

- `price_only`
- `price_plus_regime`
- `price_plus_regime_plus_trend_state`
- `price_plus_regime_plus_sentiment`

A `ModelProfile` exposes:

- `feature_columns`
- `requires_vix`
- `requires_sentiment`
- `block_names`

This is how feature engineering becomes modular. You do not hand-write feature column lists in the pipeline anymore.

### 4. Labels

Labeling logic lives in [labels](/Users/reezhan/Documents/algotrader/src/algotrader/labels).

Current labeler:

- triple-barrier long/flat labeling

The label output includes:

- `label`
- `entry_index`
- `exit_index`
- `event_end_index`
- `hit_reason`
- `entry_price`
- `exit_price`
- `realized_return`

This is important because the backtest is event-driven and depends on these fields.

### 5. Decision Policies

Decision-policy logic is split between:

- threshold regime definitions in [thresholds.py](/Users/reezhan/Documents/algotrader/src/algotrader/thresholds.py:1)
- threshold selection logic in [training/experiment.py](/Users/reezhan/Documents/algotrader/src/algotrader/training/experiment.py:1)

Current threshold policies:

- `global`
- `trend_regime`
- `trend_regime_constrained`
- `trend_vix_regime`

Decision policy is more than “pick a threshold.” It also includes:

- probability calibration mode
- optional exposure cap
- threshold selection objective
- soft penalty settings

### 6. Evaluation

Evaluation logic includes:

- purged walk-forward CV
- fold calibration
- test-time event backtest
- holdout evaluation
- ablation comparison
- label sweeps

Current evaluators are implemented as runners rather than as separate classes, but they are all driven by the same experiment-spec model.

## ExperimentSpec

The main abstraction is `ExperimentSpec` in [specs.py](/Users/reezhan/Documents/algotrader/src/algotrader/specs.py:1).

It bundles:

- `profile`
- `labeler`
- `model`
- `decision_policy`
- `evaluation`

That means one spec fully describes a strategy experiment.

Conceptually:

```python
ExperimentSpec(
    name="my_experiment",
    profile=...,
    labeler=...,
    model=...,
    decision_policy=...,
    evaluation=...,
)
```

Important methods:

- `build_walk_forward_config()`
- `to_dict()`

Important constructors:

- `build_experiment_spec(...)`
- `build_experiment_spec_from_dict(...)`

The manifest now stores the full experiment spec, which makes saved runs reproducible.

## Experiment Registry

Named reusable experiments live in [experiment_registry.py](/Users/reezhan/Documents/algotrader/src/algotrader/experiment_registry.py:1).

This is the preferred entry point for future research.

Current examples:

- `price_only`
- `price_plus_regime`
- `price_plus_regime_plus_trend_state`
- `price_plus_regime_plus_sentiment`
- `price_plus_regime_plus_trend_state_plus_regime_thresholding`
- `price_plus_regime_plus_trend_state_plus_regime_thresholding_plus_soft_objective`

Helpers:

- `build_registered_experiment(name)`
- `list_experiment_names()`

Why this matters:

- new experiments can be added in one file
- train/test/ablation can all reference the same name
- the same experiment identity can be reused across future cycles

## Settings vs Specs vs Registry

These three layers are different.

### `settings.py`

Use settings for:

- default paths
- default labels
- default split sizes
- default model hyperparameters
- default threshold grid and calibration defaults

Think of settings as the project-wide default environment.

### `specs.py`

Use specs for:

- assembling a full experiment from reusable parts

Think of specs as the actual strategy definition.

### `experiment_registry.py`

Use the registry for:

- named reusable strategy definitions

Think of the registry as the catalog of experiments you want to run repeatedly.

## How the Pipeline Resolves Experiments

The pipeline can run from three levels of specificity.

### 1. Explicit `experiment_spec`

This is the strongest override.

Used mainly internally by ablation or custom code.

### 2. Named `experiment_name`

If present, the pipeline loads the spec from the registry.

This is now the preferred CLI path.

### 3. Profile plus settings fallback

If no explicit spec or experiment name is given, the pipeline builds a spec from:

- selected profile
- current settings
- CLI overrides

This preserves backward compatibility.

## Common Workflows

### Run a Named Experiment

Train:

```bash
uv run --extra research algotrader-train \
  --experiment price_plus_regime_plus_trend_state_plus_regime_thresholding
```

Test:

```bash
uv run --extra research algotrader-test \
  --experiment price_plus_regime_plus_trend_state_plus_regime_thresholding
```

### Run a Profile-Driven Experiment

```bash
uv run --extra research algotrader-train \
  --profile price_plus_regime_plus_trend_state
```

This is still supported, but named experiments are cleaner when the decision policy matters too.

### Run Ablation

```bash
uv run --extra research algotrader-ablation
```

Ablation variants now resolve to experiment specs instead of building ad hoc logic inside the runner.

### Run Label Sweep

```bash
uv run --extra research algotrader-label-sweep
```

The sweep changes label settings while keeping the rest of the experiment pipeline stable.

### Run Holdout

```bash
uv run --extra research algotrader-holdout
```

Holdout evaluates one final untouched segment using the same experiment assembly path.

## How To Add a New Feature Block

1. Add the new columns to the feature engineering path.
   File:
   [technicals.py](/Users/reezhan/Documents/algotrader/src/algotrader/features/technicals.py:1)

2. Expose the new column names in the dataset constants if needed.
   File:
   [dataset.py](/Users/reezhan/Documents/algotrader/src/algotrader/training/dataset.py:1)

3. Register a new `FeatureBlock` in [profiles.py](/Users/reezhan/Documents/algotrader/src/algotrader/profiles.py:1).

4. Optionally create or update a profile preset that uses the block.

5. If this should be reusable as a named experiment, add a registry entry in [experiment_registry.py](/Users/reezhan/Documents/algotrader/src/algotrader/experiment_registry.py:1).

## How To Add a New Experiment

Best path:

1. Decide whether you only need a new profile or a full new experiment.
2. If only feature composition changes, create a new profile preset.
3. If threshold policy, calibration, or objective also change, create a new registry entry.

Example pattern:

```python
"my_new_experiment": build_experiment_spec(
    settings=DEFAULT_SETTINGS,
    name="my_new_experiment",
    profile_name="price_plus_regime_plus_trend_state",
    threshold_policy_name="trend_regime",
    threshold_selection_objective_name="soft_risk_adjusted",
    calibration_return_weight=2.0,
    calibration_exposure_target=0.70,
    calibration_exposure_penalty=1.5,
)
```

Then run:

```bash
uv run --extra research algotrader-train --experiment my_new_experiment
uv run --extra research algotrader-test --experiment my_new_experiment
```

## How To Think About Reuse

The intended reuse model is:

- reuse ingestion across symbols and sources
- reuse feature blocks across profiles
- reuse profiles across multiple decision policies
- reuse labelers across multiple experiments
- reuse decision policies across multiple profiles
- reuse evaluation runners across all experiments

That means future work should mostly be one of:

- add a feature block
- add a profile preset
- add a decision policy
- add a named experiment

and not:

- fork pipeline code
- copy runner code
- hardcode new variants in multiple places

## Current Recommended Boundaries

If you are extending the project, use these boundaries:

- `features`: how raw inputs become columns
- `profiles`: how columns are grouped into reusable blocks
- `labels`: how targets are defined
- `thresholds`: how probabilities become regime-aware decisions
- `specs`: how one full experiment is assembled
- `experiment_registry`: what named experiments exist
- `pipeline` / `ablation` / `sweep` / `holdout`: how experiments are executed

