Richer trend state

1. SMA_200 slope
distance of price from SMA_200
SMA_50 > SMA_200 alignment flag
reason: regime already helps, so trend quality is the most likely next source of lift
Richer volatility state

2. ATR percentile
Bollinger bandwidth percentile
realized vol percentile/z-score beyond the current single vol input
reason: current VIX feature helps, which suggests volatility context matters, but one VIX z-score is probably too thin
Breakout / compression context

3. Donchian channel position
distance to 20-day high / low
recent range compression before breakout
reason: your best label setup is a 10-bar event target, so breakout-style context is likely more aligned than generic oscillators
Regime-conditional thresholding

4. use different decision thresholds in:
bullish / calm regime
bullish / stressed regime
reason: one global threshold per fold is probably too blunt once regime is clearly part of the signal