import numpy as np
from arch import arch_model

def fit_garch(returns):
    scaled = returns.dropna() * 100
    model = arch_model(scaled, vol='GARCH', p=1, q=1, mean='Constant')
    res = model.fit(disp='off')

    omega = res.params['omega'] / (100 ** 2)
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    long_run_var = omega / (1 - alpha - beta)

    #extract last day's conditional variance (decimal scale)
    last_var = res.conditional_volatility.iloc[-1] ** 2 / (100 ** 2)

    return omega, alpha, beta, long_run_var, last_var


def simulate_garch(s0, miu, omega, alpha, beta, last_var, days=252, n_simulations=1000):
    np.random.seed(42)  #set seed for reproducibility
    dt = 1 / 252  #1 trading day = 1/252 year

    #generate all random shocks
    z = np.random.standard_normal((days, n_simulations))

    #initialise price and variance matrices
    s = np.zeros((days + 1, n_simulations))
    sigma2 = np.zeros((days + 1, n_simulations))
    s[0] = s0
    sigma2[0] = last_var  #start from TODAY's conditional variance, not long-run average

    r_prev = np.zeros(n_simulations)

    #GARCH-GBM update
    for t in range(1, days + 1):
        sigma2[t] = omega + alpha * r_prev ** 2 + beta * sigma2[t-1]
        sigma_t = np.sqrt(sigma2[t])
        r_t = miu * dt - 0.5 * sigma2[t] + sigma_t * z[t-1]
        s[t] = s[t-1] * np.exp(r_t)
        r_prev = r_t

    sigmas = np.sqrt(sigma2)
    return s, sigmas