import numpy as np

def calculate_var(s, s0, confidence=0.95):
    final_prices = s[-1]
    final_returns = (final_prices - s0) / s0
    var = np.percentile(final_returns, (1 - confidence) * 100)
    return var, final_returns

def calculate_historical_var(prices, confidence=0.95):
    #compute 252-day holding period returns from actual price history
    holding_returns = (prices / prices.shift(252) - 1).dropna()

    #need enough samples to reliably estimate the tail quantile
    min_samples = int(20 / (1 - confidence))
    if len(holding_returns) < min_samples:
        return None, None

    var = np.percentile(holding_returns, (1 - confidence) * 100)
    return var, holding_returns