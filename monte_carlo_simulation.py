import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load data
msft = pd.read_csv("data/MSFT.csv", parse_dates=["Datetime"], index_col="Datetime")
aapl = pd.read_csv("data/AAPL.csv", parse_dates=["Datetime"], index_col="Datetime")
spy = pd.read_csv("data/SPY.csv", parse_dates=["Datetime"], index_col="Datetime")

# Calculate log returns
msft["LogReturn"] = np.log(msft["Open"] / msft["Open"].shift(1))
aapl["LogReturn"] = np.log(aapl["Open"] / aapl["Open"].shift(1))
spy["LogReturn"] = np.log(spy["Open"] / spy["Open"].shift(1))

# Calculate historical correlations and betas
correlation_matrix = pd.DataFrame({
    "AAPL-MSFT": aapl["LogReturn"].corr(msft["LogReturn"]),
    "AAPL-SPY": aapl["LogReturn"].corr(spy["LogReturn"]),
    "MSFT-SPY": msft["LogReturn"].corr(spy["LogReturn"]),
}, index=["Correlation"])

betas = {
    "AAPL-SPY": np.cov(aapl["LogReturn"].dropna(), spy["LogReturn"].dropna())[0, 1] / np.var(spy["LogReturn"].dropna()),
    "MSFT-SPY": np.cov(msft["LogReturn"].dropna(), spy["LogReturn"].dropna())[0, 1] / np.var(spy["LogReturn"].dropna()),
}

# GBM parameters
def calculate_gbm_params(log_returns):
    drift = log_returns.mean() * 252
    volatility = log_returns.std() * np.sqrt(252)
    return drift, volatility

msft_drift, msft_volatility = calculate_gbm_params(msft["LogReturn"].dropna())
aapl_drift, aapl_volatility = calculate_gbm_params(aapl["LogReturn"].dropna())
spy_drift, spy_volatility = calculate_gbm_params(spy["LogReturn"].dropna())

# Starting prices
msft_start = msft["Open"].iloc[-1]
aapl_start = aapl["Open"].iloc[-1]
spy_start = spy["Open"].iloc[-1]

# Cholesky decomposition for correlated simulations
cov_matrix = np.cov([
    msft["LogReturn"].dropna(),
    aapl["LogReturn"].dropna(),
    spy["LogReturn"].dropna()
])
cholesky_matrix = np.linalg.cholesky(cov_matrix)

# Monte Carlo simulation
simulations = 10000
days = 14

def simulate_gbm(start_price, drift, volatility, shares, days, simulations):
    dt = 1 / 252
    prices = np.zeros((days, simulations))
    valuations = np.zeros((days, simulations))
    prices[0] = start_price
    valuations[0] = start_price * shares
    for t in range(1, days):
        random_shocks = np.random.normal(0, 1, simulations)
        prices[t] = prices[t - 1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks)
        valuations[t] = prices[t] * shares
    return prices, valuations

msft_shares = 7.43  # in billions
aapl_shares = 15.04  # in billions

msft_prices, msft_valuations = simulate_gbm(msft_start, msft_drift, msft_volatility, msft_shares, days, simulations)
aapl_prices, aapl_valuations = simulate_gbm(aapl_start, aapl_drift, aapl_volatility, aapl_shares, days, simulations)
spy_prices, _ = simulate_gbm(spy_start, spy_drift, spy_volatility, 1, days, simulations)

# Analyze conditional probabilities
spy_returns = (spy_prices[-1] - spy_start) / spy_start
valuation_deltas = msft_valuations[-1] - aapl_valuations[-1]

prob_msft_higher_valuation = np.mean(valuation_deltas > 0)
prob_aapl_higher_valuation = np.mean(valuation_deltas < 0)

# Visualizations
plt.figure(figsize=(10, 6))
plt.plot(msft_valuations[:, :100], color="blue", alpha=0.1, label="MSFT Valuations")
plt.plot(aapl_valuations[:, :100], color="green", alpha=0.1, label="AAPL Valuations")
plt.title("Monte Carlo Simulation: 100 Sample Valuation Paths")
plt.xlabel("Days")
plt.ylabel("Valuation (in billions)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(valuation_deltas, kde=True, color="purple", bins=50)
plt.title("Distribution of MSFT - AAPL Valuation Differences")
plt.xlabel("Valuation Difference (in billions)")
plt.ylabel("Frequency")
plt.axvline(0, color="red", linestyle="--", label="Equal Valuation")
plt.legend()
plt.show()

# Print results
print("Correlation Matrix:")
print(correlation_matrix)
print("\nBetas:")
print(betas)
print("\nProbability MSFT valuation > AAPL valuation:", prob_msft_higher_valuation)
print("Probability AAPL valuation > MSFT valuation:", prob_aapl_higher_valuation)