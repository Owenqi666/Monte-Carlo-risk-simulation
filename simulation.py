import numpy as np

def simulate(s0, miu, sigma, days=252, n_simulations=1000):
    np.random.seed(42)  #set seed for reproducibility
    dt = 1 / 252  #1 trading day = 1/252 year

    #generate all random shock
    z = np.random.standard_normal((days, n_simulations))

    #initialise price matrix
    s = np.zeros((days + 1, n_simulations))
    s[0] = s0

    #GBM update
    for t in range(1, days + 1):
        s[t] = s[t-1] * np.exp((miu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[t-1])

    return s