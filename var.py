import numpy as np

def calculate_var(S, S0, confidence=0.95):

    final_prices = S[-1]

    final_returns = (final_prices - S0) / S0

    var = np.percentile(final_returns, (1 - confidence) * 100)

    return var, final_returns

def calculate_historical_var(returns, confidence=0.95):

    daily_var = np.percentile(returns, (1 - confidence) * 100)
    
    #annualise using square root of time rule
    annual_var = daily_var * np.sqrt(252)
    
    return daily_var, annual_var