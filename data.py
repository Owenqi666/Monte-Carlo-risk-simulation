import yfinance as yf
import numpy as np
import sys

def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    
    if df.empty:
        sys.exit(f"Error: no data found for '{ticker}'. Check the ticker or date range.")
    
    prices = df['Close'].squeeze().dropna()
    
    if len(prices) < 30:
        sys.exit(f"Error: insufficient data for '{ticker}'. Try a wider date range.")
    
    returns = prices.pct_change().dropna()
    return prices, returns

def estimate_params(returns):
    miu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return miu, sigma