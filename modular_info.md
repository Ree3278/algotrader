Refactor around six plug-in layers.

DataSource
responsibility: produce normalized time series
examples:
YFinanceOHLCVSource
AlphaVantageOHLCVSource
CsvOHLCVSource
CsvSentimentSource
output: standard frames only
FeatureBlock
responsibility: add one coherent feature family
examples:
PriceBlock
RegimeBlock
TrendStateBlock
SentimentBlock
ATRPercentileBlock
output: dataframe in, dataframe out
current profiles.py should become composition of FeatureBlocks, not hardcoded column bundles
Labeler
responsibility: build targets and event metadata
examples:
TripleBarrierLabeler
later MetaLabeler, ForwardReturnLabeler
output:
label
hit_reason
entry/exit metadata
ModelSpec
responsibility: define learner + calibration
examples:
XGBoostSpec
HistGradientBoostingSpec
should own:
fit
predict_proba
optional probability calibration
DecisionPolicy
responsibility: convert probabilities into actions
examples:
GlobalThresholdPolicy
TrendRegimeThresholdPolicy
later ExposureAwarePolicy
should own:
regime assignment
threshold search objective
hard caps / soft penalties
Evaluator
responsibility: score a strategy design
examples:
WalkForwardEvaluator
HoldoutEvaluator
AblationEvaluator
LabelSweepEvaluator
What The Wrapper Should Look Like

Have one experiment spec that wires those blocks together:

ExperimentSpec(
    data=CsvOHLCVSource(...),
    aux_data=[CsvVIXSource(...), CsvSentimentSource(...)],
    feature_blocks=[PriceBlock(), RegimeBlock(), TrendStateBlock()],
    labeler=TripleBarrierLabeler(...),
    model=XGBoostSpec(...),
    decision_policy=TrendRegimeThresholdPolicy(...),
    evaluator=WalkForwardEvaluator(...),
)
That becomes the only object the runners consume.

Then:

train.py runs one ExperimentSpec
ablation.py compares several ExperimentSpecs
sweep.py varies one component, usually labeler
holdout.py reuses the exact same ExperimentSpec with a different evaluator
Practical Refactor Plan

I would do it in this order:

Extract interfaces and dataclasses
DataSource
FeatureBlock
Labeler
ModelSpec
DecisionPolicy
Evaluator
ExperimentSpec
Move current logic behind those interfaces
keep behavior unchanged
only reorganize ownership
Make profiles.py just a registry of feature-block compositions
not special-case logic
Make thresholds.py part of DecisionPolicy
threshold policy and threshold objective should live together
Make runners generic
ablation.py should compare named experiment specs
sweep.py should vary only one axis at a time
Keep settings.py only for defaults
not for experiment identity
experiment identity should come from explicit ExperimentSpec
What I’d Freeze As Reusable Building Blocks

For your next experiment, I’d preserve these as stable reusable components:

normalized data loaders
feature block registry
triple-barrier labeler
walk-forward splitter
event-driven backtest
model adapter interface
decision-policy interface
reporting / metrics layer
That gives you a reusable research framework instead of a SPY-specific project with patched experiments on top.