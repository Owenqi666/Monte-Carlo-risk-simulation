import numpy as np

def calculate_var(s, s0, confidence=0.95):
    final_prices = s[-1]
    final_returns = (final_prices - s0) / s0
    var = np.percentile(final_returns, (1 - confidence) * 100)
    return var, final_returns

def calculate_historical_var(prices, confidence=0.95):
    #compute 252-day holding period returns from actual price history
    holding_returns = (prices / prices.shift(252) - 1).dropna()
    
    if len(holding_returns) < 30:
        return None
    
    var = np.percentile(holding_returns, (1 - confidence) * 100)
    return var