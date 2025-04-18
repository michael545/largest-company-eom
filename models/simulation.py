"""
Module for simulating stock price paths
"""
import numpy as np
from typing import List, Tuple, Optional
from arch import arch_model
import pandas as pd


def simulate_gbm(
    start_price: float,
    mean: float,
    vol: float, 
    days: int,
    simulations: int
) -> np.ndarray:
    """
    Simulate stock prices using Geometric Brownian Motion
    
    Args:
        start_price: Starting price
        mean: Daily mean return
        vol: Daily volatility
        days: Number of days to simulate
        simulations: Number of simulation paths
        
    Returns:
        2D array of simulated prices [days+1, simulations]
    """
    # Create random shocks
    epsilon = np.random.normal(0, 1, size=(days, simulations))
    
    # Daily returns
    daily_returns = mean + vol * epsilon
    
    # Cumulative returns
    cumulative_returns = np.exp(np.cumsum(daily_returns, axis=0))
    
    # Initialize price array (include starting price at index 0)
    prices = np.empty((days + 1, simulations))
    prices[0] = start_price
    prices[1:] = start_price * cumulative_returns
    
    return prices


def simulate_correlated_gbm(
    start_prices: List[float],
    shares: List[float],
    days: int,
    simulations: int,
    cov_matrix: np.ndarray,
    means: Optional[np.ndarray] = None,
    vols: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate correlated stock prices using Geometric Brownian Motion
    
    Args:
        start_prices: Starting prices for each asset
        shares: Shares outstanding (in billions) for each asset
        days: Number of days to simulate
        simulations: Number of simulations to run
        cov_matrix: Covariance matrix of asset returns
        means: Optional mean returns, default 0
        vols: Optional volatilities, default from cov_matrix diagonal
        
    Returns:
        Tuple of (prices, market_caps) arrays
    """
    num_assets = len(start_prices)
    
    # Check if volatilities are provided, otherwise extract from cov_matrix
    if vols is None:
        vols = np.sqrt(np.diag(cov_matrix))
    
    # Check if means are provided, otherwise use zeros
    if means is None:
        means = np.zeros(num_assets)
    
    # Initialize arrays for prices and market caps
    # Shape: (num_assets, days+1, simulations)
    prices = np.zeros((num_assets, days+1, simulations))
    market_caps = np.zeros_like(prices)
    
    # Set initial prices and market caps
    for i in range(num_assets):
        prices[i, 0, :] = start_prices[i]
        market_caps[i, 0, :] = start_prices[i] * shares[i]
    
    # Generate correlated random returns using Cholesky decomposition
    np.random.seed(42)  # For reproducibility
    L = np.linalg.cholesky(cov_matrix)  # Lower triangular matrix
    
    # Generate random paths using GBM
    for t in range(1, days + 1):
        # Generate correlated normal random variables
        rand_norm = np.random.normal(0, 1, (num_assets, simulations))
        correlated_rand = L @ rand_norm  # Matrix multiply to correlate the returns
        
        # Apply the returns to the previous day's prices
        for i in range(num_assets):
            daily_means = means[i] / 252  # Convert annual to daily
            daily_vol = vols[i] / np.sqrt(252)  # Convert annual to daily
            
            # GBM formula: S_t = S_{t-1} * exp((μ - σ²/2)dt + σ√dt * Z)
            drift = daily_means - 0.5 * daily_vol**2
            diffusion = daily_vol * correlated_rand[i]
            
            # Update price
            prices[i, t, :] = prices[i, t-1, :] * np.exp(drift + diffusion)
            
            # Update market cap
            market_caps[i, t, :] = prices[i, t, :] * shares[i]
    
    return prices, market_caps


def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1):
    """
    Fit a GARCH(p,q) model to the return series
    
    Args:
        returns: Series of asset returns
        p: GARCH lag order
        q: ARCH lag order
        
    Returns:
        Fitted GARCH model
    """
    # Create and fit GARCH model
    model = arch_model(
        returns * 100,  # Scale returns for numerical stability
        vol='Garch', 
        p=p, 
        q=q, 
        mean='Constant', 
        dist='normal'
    )
    
    # Fit with robust standard errors
    result = model.fit(disp='off')
    
    return result


def simulate_garch(
    start_price: float,
    returns: pd.Series,
    days: int,
    simulations: int,
    p: int = 1,
    q: int = 1
) -> np.ndarray:
    """
    Simulate stock prices using a GARCH model for volatility
    
    Args:
        start_price: Initial price
        returns: Historical returns series
        days: Number of days to simulate
        simulations: Number of simulation paths
        p: GARCH lag order
        q: ARCH lag order
        
    Returns:
        2D array of simulated prices [days+1, simulations]
    """
    # Fit GARCH model
    garch_result = fit_garch_model(returns, p=p, q=q)
    
    # Get the last conditional variance as starting point
    forecasts = garch_result.forecast(horizon=1, reindex=False)
    last_vol = np.sqrt(forecasts.variance.iloc[-1, 0] / 10000)  # Rescale back
    
    # Parameters for simulation
    omega = garch_result.params['omega'] / 10000
    alpha = garch_result.params['alpha[1]']
    beta = garch_result.params['beta[1]']
    mu = garch_result.params['mu'] / 100  # Rescale mean
    
    # Initialize arrays
    prices = np.empty((days + 1, simulations))
    prices[0] = start_price
    
    # Initialize volatility array (conditional variance)
    sigma2 = np.empty((days + 1, simulations))
    sigma2[0] = last_vol ** 2
    
    # Generate random returns and calculate prices
    for t in range(1, days + 1):
        # Generate random shocks
        z = np.random.normal(0, 1, simulations)
        
        # Update volatility (conditional variance) using GARCH(1,1) formula
        sigma2[t] = omega + alpha * (z[np.newaxis].T**2) * sigma2[t-1] + beta * sigma2[t-1]
        
        # Calculate returns using the updated volatility
        daily_returns = mu + np.sqrt(sigma2[t]) * z
        
        # Update prices
        prices[t] = prices[t-1] * np.exp(daily_returns)
    
    return prices


def simulate_correlated_garch(
    start_prices: List[float],
    shares: List[float],
    returns_df: pd.DataFrame,
    days: int,
    simulations: int,
    p: int = 1,
    q: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate correlated stock prices using GARCH for time-varying volatility
    
    Args:
        start_prices: List of starting prices for each asset
        shares: List of shares outstanding for each asset
        returns_df: DataFrame of historical returns for all assets
        days: Number of days to simulate
        simulations: Number of simulation paths
        p: GARCH lag order
        q: ARCH lag order
        
    Returns:
        Tuple of (prices, market_caps) arrays with shape (num_assets, days+1, simulations)
    """
    num_assets = len(start_prices)
    assets = list(returns_df.columns)
    
    # Initialize arrays for prices and market caps
    prices = np.zeros((num_assets, days+1, simulations))
    market_caps = np.zeros_like(prices)
    
    # Set initial prices and market caps
    for i in range(num_assets):
        prices[i, 0, :] = start_prices[i]
        market_caps[i, 0, :] = start_prices[i] * shares[i]
    
    # Fit GARCH models for each asset
    garch_models = {}
    for i, asset in enumerate(assets):
        returns = returns_df[asset]
        garch_models[asset] = fit_garch_model(returns, p, q)
    
    # Calculate historical residuals
    residuals = pd.DataFrame(index=returns_df.index)
    for asset in assets:
        model = garch_models[asset]
        fitted_mean = model.params['mu'] / 100
        fitted_resid = returns_df[asset] - fitted_mean
        residuals[asset] = fitted_resid / np.sqrt(model.conditional_volatility**2 / 10000)
    
    # Calculate residuals correlation matrix
    residual_corr = residuals.corr().values
    
    # Use Cholesky decomposition for generating correlated random numbers
    try:
        L = np.linalg.cholesky(residual_corr)
    except np.linalg.LinAlgError:
        # If matrix is not positive definite, use nearest PD approximation
        print("Warning: Correlation matrix not positive definite, using nearest approximation")
        # Simple fix - add small value to diagonal
        residual_corr = residual_corr + np.eye(residual_corr.shape[0]) * 1e-6
        L = np.linalg.cholesky(residual_corr)
    
    # Initialize volatility arrays
    sigma = np.zeros((num_assets, days+1, simulations))
    
    # Get last conditional volatilities
    for i, asset in enumerate(assets):
        model = garch_models[asset]
        forecast = model.forecast(horizon=1, reindex=False)
        sigma[i, 0, :] = np.sqrt(forecast.variance.iloc[-1, 0] / 10000)
    
    # Extract GARCH parameters
    params = {}
    for asset in assets:
        model = garch_models[asset]
        params[asset] = {
            'mu': model.params['mu'] / 100,  # Rescale
            'omega': model.params['omega'] / 10000,  # Rescale
            'alpha': model.params['alpha[1]'],
            'beta': model.params['beta[1]']
        }
    
    # Simulation loop
    for t in range(1, days + 1):
        # Generate correlated random numbers
        rand_z = np.random.normal(0, 1, (num_assets, simulations))
        correlated_z = L @ rand_z
        
        # Update volatility and calculate returns for each asset
        for i, asset in enumerate(assets):
            p = params[asset]
            
            # Update volatility using GARCH formula
            sigma[i, t, :] = np.sqrt(
                p['omega'] + 
                p['alpha'] * (sigma[i, t-1, :] * correlated_z[i])**2 +
                p['beta'] * sigma[i, t-1, :]**2
            )
            
            # Calculate returns and update prices
            returns = p['mu'] + sigma[i, t, :] * correlated_z[i]
            prices[i, t, :] = prices[i, t-1, :] * np.exp(returns)
            
            # Update market caps
            market_caps[i, t, :] = prices[i, t, :] * shares[i]
    
    return prices, market_caps