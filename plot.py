import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_simulations(s, ticker):
    plt.figure(figsize=(12, 6))

    #plot all 1000 paths with low opacity
    plt.plot(s, color='steelblue', alpha=0.05, linewidth=0.5)

    #plot 5th and 95th percentile bands
    lower = np.percentile(s, 5, axis=1)
    upper = np.percentile(s, 95, axis=1)
    plt.fill_between(range(s.shape[0]), lower, upper, alpha=0.2, color='steelblue', label='90% confidence band')

    #plot mean path
    plt.plot(s.mean(axis=1), color='red', linewidth=1.5, label='mean path')

    plt.title(f'{ticker} Monte Carlo Simulation (1000 paths, 252 days)')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{ticker}_simulation.png', dpi=150)
    plt.show()


def plot_var(final_returns, var, ticker, confidence=0.95):
    plt.figure(figsize=(10, 5))

    plt.hist(final_returns, bins=50, color='steelblue', edgecolor='white', alpha=0.7)

    plt.axvline(var, color='red', linewidth=2, label=f'VaR ({confidence*100:.0f}%): {var:.2%}')

    plt.title(f'{ticker} Distribution of Final Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{ticker}_var_distribution.png', dpi=150)
    plt.show()

def plot_volatility_paths(sigmas, ticker, long_run_var=None, n_show=100):
    plt.figure(figsize=(12, 6))

    #annualise daily volatility
    sigmas_annual = sigmas * np.sqrt(252)
    sample_idx = np.random.choice(sigmas_annual.shape[1], size=n_show, replace=False)

    #plot sampled paths with low opacity
    plt.plot(sigmas_annual[:, sample_idx], color='steelblue', alpha=0.15, linewidth=0.5)

    #plot mean volatility path
    plt.plot(sigmas_annual.mean(axis=1), color='red', linewidth=1.5, label='mean volatility')

    #plot long-run unconditional volatility
    if long_run_var is not None:
        long_run_vol = np.sqrt(long_run_var * 252)
        plt.axhline(long_run_vol, color='black', linestyle='--', linewidth=1, label='long-run unconditional vol')

    plt.title(f'{ticker} GARCH(1,1) Simulated Volatility Paths (1000 paths, 252 days)')
    plt.xlabel('Trading Days')
    plt.ylabel('Annualised Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{ticker}_garch_volatility.png', dpi=150)
    plt.show()

def plot_distribution_comparison(final_returns_gbm, final_returns_garch, historical_returns,
                                  var_gbm, var_garch, historical_var, ticker):
    
    plt.figure(figsize=(12, 6))

    #build shared x-grid covering all three distributions
    all_returns = np.concatenate([final_returns_gbm, final_returns_garch, historical_returns.values])
    x = np.linspace(all_returns.min(), all_returns.max(), 500)

    #compute KDEs
    kde_gbm = gaussian_kde(final_returns_gbm)(x)
    kde_garch = gaussian_kde(final_returns_garch)(x)
    kde_hist = gaussian_kde(historical_returns)(x)

    #plot KDE curves with filled regions
    plt.fill_between(x, kde_gbm, alpha=0.15, color='steelblue')
    plt.plot(x, kde_gbm, color='steelblue', linewidth=1.5, label='GBM simulated')

    plt.fill_between(x, kde_garch, alpha=0.15, color='seagreen')
    plt.plot(x, kde_garch, color='seagreen', linewidth=1.5, label='GARCH simulated')

    plt.fill_between(x, kde_hist, alpha=0.15, color='black')
    plt.plot(x, kde_hist, color='black', linewidth=2, label='Historical (252-day)')

    #plot 3 VaR lines
    plt.axvline(var_gbm, color='steelblue', linestyle='--', linewidth=1.5,
                label=f'GBM VaR: {var_gbm:.2%}')
    plt.axvline(var_garch, color='seagreen', linestyle='--', linewidth=1.5,
                label=f'GARCH VaR: {var_garch:.2%}')
    plt.axvline(historical_var, color='black', linestyle='--', linewidth=1.5,
                label=f'Historical VaR: {historical_var:.2%}')

    plt.title(f'{ticker} Return Distribution Comparison (252-day horizon)')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{ticker}_distribution_comparison.png', dpi=150)
    plt.show()