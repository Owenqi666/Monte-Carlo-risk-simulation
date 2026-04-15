from data import get_data, estimate_params
from simulation import simulate
from garch import fit_garch, simulate_garch
from var import calculate_var, calculate_historical_var
from plot import plot_simulations, plot_var, plot_volatility_paths, plot_distribution_comparison
from checkdate import parse_date
import sys
import numpy as np

#input tickers
n = int(input("How many tickers to compare: "))

tickers = []
for i in range(n):
    ticker = input(f"Ticker {i+1}: ").strip().upper()
    if not ticker:
        sys.exit("Error: ticker cannot be empty.")
    tickers.append(ticker)

#input dates
start, start_dt = parse_date("Start date (e.g. 2016-01-01): ")
end, end_dt = parse_date("End date (e.g. 2026-01-01): ")

if end_dt <= start_dt:
    sys.exit(f"Error: start date ({start}) must be before end date ({end}).")

results = {}
for ticker in tickers:
    prices, returns = get_data(ticker, start, end)
    miu, sigma = estimate_params(returns)
    s0 = float(prices.iloc[-1])

    #GBM simulation (constant volatility)
    s_gbm = simulate(s0, miu, sigma)
    var_gbm, final_returns_gbm = calculate_var(s_gbm, s0)

    #GARCH simulation (time-varying volatility)
    omega, alpha, beta, long_run_var, last_var = fit_garch(returns)
    s_garch, sigmas = simulate_garch(s0, miu, omega, alpha, beta, last_var)
    var_garch, final_returns_garch = calculate_var(s_garch, s0)

    #Historical VaR & returns
    historical_var, historical_returns = calculate_historical_var(prices)

    results[ticker] = {'s0': s0, 'miu': miu, 'sigma': sigma,
                        'omega': omega, 'alpha': alpha, 'beta': beta,
                        'long_run_var': long_run_var, 'last_var': last_var,
                        'var_gbm': var_gbm, 'var_garch': var_garch,
                        'historical_var': historical_var,
                        'final_returns_gbm': final_returns_gbm,
                        'final_returns_garch': final_returns_garch,
                        'historical_returns': historical_returns,
                        's_gbm': s_gbm, 's_garch': s_garch, 'sigmas': sigmas}

    print(f'\n{ticker}')
    print(f's0: ${s0:.2f}')
    print(f'miu: {miu:.4f}')
    print(f'sigma: {sigma:.4f}')
    print(f'GARCH: omega={omega:.2e}, alpha={alpha:.4f}, beta={beta:.4f}, persistence={alpha+beta:.4f}')
    print(f'GARCH vol: current={np.sqrt(last_var*252):.2%}, long-run={np.sqrt(long_run_var*252):.2%}')
    print(f'95% VaR (GBM): {var_gbm:.2%}')
    print(f'95% VaR (GARCH): {var_garch:.2%}')

    if historical_var is not None:
        print(f'95% VaR (Historical): {historical_var:.2%}')
    else:
        print(f'95% VaR (Historical): insufficient data')

for ticker, data in results.items():
    plot_simulations(data['s_gbm'], ticker)
    plot_var(data['final_returns_gbm'], data['var_gbm'], ticker)
    plot_volatility_paths(data['sigmas'], ticker, long_run_var=data['long_run_var'])

    if data['historical_returns'] is not None:
        plot_distribution_comparison(
            data['final_returns_gbm'],
            data['final_returns_garch'],
            data['historical_returns'],
            data['var_gbm'],
            data['var_garch'],
            data['historical_var'],
            ticker
        )