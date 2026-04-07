import matplotlib.pyplot as plt
import numpy as np

def plot_simulations(s, ticker):
    plt.figure(figsize=(12, 6))

    # plot all 1000 paths with low opacity
    plt.plot(s, color='steelblue', alpha=0.05, linewidth=0.5)

    # plot 5th and 95th percentile bands
    lower = np.percentile(s, 5, axis=1)
    upper = np.percentile(s, 95, axis=1)
    plt.fill_between(range(s.shape[0]), lower, upper, alpha=0.2, color='steelblue', label='90% confidence band')

    # plot mean path
    plt.plot(s.mean(axis=1), color='red', linewidth=1.5, label='mean path')

    plt.title(f'{ticker} Monte Carlo Simulation (1000 paths, 252 days)')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{ticker}_simulation.png', dpi=150)
    plt.show()


def plot_var(final_returns, var, confidence=0.95):
    plt.figure(figsize=(10, 5))

    plt.hist(final_returns, bins=50, color='steelblue', edgecolor='white', alpha=0.7)

    # mark VaR on the histogram
    plt.axvline(var, color='red', linewidth=2, label=f'VaR ({confidence*100:.0f}%): {var:.2%}')

    plt.title('Distribution of Final Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('var_distribution.png', dpi=150)
    plt.show()