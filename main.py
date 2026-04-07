from data import get_data, estimate_params
from simulation import simulate
from var import calculate_var, calculate_historical_var
from plot import plot_simulations, plot_var
from date import parse_date
import sys

# input tickers
n = int(input("How many tickers to compare: "))

tickers = []
for i in range(n):
    ticker = input(f"Ticker {i+1}: ").strip().upper()
    if not ticker:
        sys.exit("Error: ticker cannot be empty.")
    tickers.append(ticker)

# input dates
start, start_dt = parse_date("Start date (e.g. 2016-01-01): ")
end, end_dt = parse_date("End date (e.g. 2026-01-01): ")

if end_dt <= start_dt:
    sys.exit(f"Error: start date ({start}) must be before end date ({end}).")

results = {}
for ticker in tickers:
    prices, returns = get_data(ticker, start, end)
    miu, sigma = estimate_params(returns)
    s0 = float(prices.iloc[-1])
    s = simulate(s0, miu, sigma)
    var, final_returns = calculate_var(s, s0)
    historical_var = calculate_historical_var(prices)

    results[ticker] = {'s0': s0, 'miu': miu, 'sigma': sigma,
                    'var': var, 'historical_var': historical_var,
                    'final_returns': final_returns, 's': s}

    print(f'\n{ticker}')
    print(f's0: ${s0:.2f}')
    print(f'miu: {miu:.4f}')
    print(f'sigma: {sigma:.4f}')
    print(f'95% VaR (Monte Carlo): {var:.2%}')

    if historical_var is not None:
        print(f'  95% VaR (Historical)   : {historical_var:.2%}')
    else:
        print(f'  95% VaR (Historical)   : insufficient data')

for ticker, data in results.items():
    plot_simulations(data['s'], ticker)
    plot_var(data['final_returns'], data['var'], ticker)