import yfinance as yf
import numpy as np

def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    prices = df['Close'].squeeze().dropna()#convert data frame to series
    returns = prices.pct_change().dropna()
    return prices, returns

def estimate_params(returns):
    miu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return miu, sigma