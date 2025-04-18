
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional

try:
    import pandas_datareader as pdr
    HAVE_PDR = True
except ImportError:
    HAVE_PDR = False

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False


def load_data(tickers: Optional[list] = None, start_date: Optional[str] = None, 
              end_date: Optional[str] = None, source: str = 'yahoo', use_cache: bool = True
             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load historical stock data for MSFT, AAPL and SPY
    
    Args:
        tickers: List of tickers to load (default: ['MSFT', 'AAPL', 'SPY'])
        start_date: Start date for data (default: 2 years ago)
        end_date: End date for data (default: today)
        source: Data source ('yahoo', 'yfinance', or 'csv')
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (msft_df, aapl_df, spy_df, price_column)
    """
    # Default parameters
    if tickers is None:
        tickers = ['MSFT', 'AAPL', 'SPY']
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check for cached data
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    
    msft_path = f"{cache_dir}/MSFT.csv"
    aapl_path = f"{cache_dir}/AAPL.csv"
    spy_path = f"{cache_dir}/SPY.csv"
    
    if use_cache and os.path.exists(msft_path) and os.path.exists(aapl_path) and os.path.exists(spy_path):
        try:
            msft = pd.read_csv(msft_path, parse_dates=['Datetime'], index_col='Datetime')
            aapl = pd.read_csv(aapl_path, parse_dates=['Datetime'], index_col='Datetime')
            spy = pd.read_csv(spy_path, parse_dates=['Datetime'], index_col='Datetime')
            price_col = 'Price'  # Default column from saved data
            return msft, aapl, spy, price_col
        except Exception as e:
            print(f"Error reading cached data: {e}")
            # Fall through to downloading data
    
    # Try different data sources
    if source == 'yahoo' and HAVE_PDR:
        # Use pandas_datareader
        msft = pdr.get_data_yahoo('MSFT', start=start_date, end=end_date)
        aapl = pdr.get_data_yahoo('AAPL', start=start_date, end=end_date)
        spy = pdr.get_data_yahoo('SPY', start=start_date, end=end_date)
        price_col = 'Price'
    elif source == 'yfinance' and HAVE_YF:
        # Use yfinance
        msft = yf.download('MSFT', start=start_date, end=end_date)
        aapl = yf.download('AAPL', start=start_date, end=end_date)
        spy = yf.download('SPY', start=start_date, end=end_date)
        price_col = 'Price'
    else:
        # Can't download data, try to use local files as fallback
        raise ValueError(f"Data source {source} not available. Install pandas-datareader or yfinance.")
    
    # Calculate log returns
    for df in [msft, aapl, spy]:
        df['LogReturn'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Save to cache
    msft.to_csv(msft_path)
    aapl.to_csv(aapl_path)
    spy.to_csv(spy_path)
    
    return msft, aapl, spy, price_col


def create_synthetic_data(days: int = 504, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Create synthetic data for demonstration purposes
    
    Args:
        days: Number of trading days to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (msft_df, aapl_df, spy_df, price_column)
    """
    np.random.seed(seed)
    print("Creating synthetic data for demonstration...")
    
    # Set realistic parameters
    mean_returns = {
        'MSFT': 0.0008,  # ~20% annualized
        'AAPL': 0.0008,  # ~20% annualized
        'SPY': 0.0005    # ~12% annualized
    }
    
    volatilities = {
        'MSFT': 0.017,   # ~27% annualized
        'AAPL': 0.020,   # ~32% annualized
        'SPY': 0.013     # ~20% annualized
    }
    
    # Correlation matrix
    corr_matrix = np.array([
        [1.0, 0.7, 0.8],  # MSFT correlations 
        [0.7, 1.0, 0.75], # AAPL correlations
        [0.8, 0.75, 1.0]  # SPY correlations
    ])
    
    # Create covariance matrix
    vols = np.array([volatilities['MSFT'], volatilities['AAPL'], volatilities['SPY']])
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    # Generate correlated returns using Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    
    # Generate random normal samples
    z = np.random.normal(0, 1, size=(days, 3))
    
    # Apply correlation structure
    correlated_returns = z @ L.T
    
    # Apply drift and volatility
    returns = np.zeros_like(correlated_returns)
    returns[:, 0] = correlated_returns[:, 0] + mean_returns['MSFT']
    returns[:, 1] = correlated_returns[:, 1] + mean_returns['AAPL']
    returns[:, 2] = correlated_returns[:, 2] + mean_returns['SPY']
    
    # Starting prices
    start_prices = {
        'MSFT': 400.0,
        'AAPL': 250.0,
        'SPY': 500.0
    }
    
    # Generate prices
    prices = {ticker: [start_prices[ticker]] for ticker in ['MSFT', 'AAPL', 'SPY']}
    
    for i in range(days):
        for j, ticker in enumerate(['MSFT', 'AAPL', 'SPY']):
            prices[ticker].append(prices[ticker][-1] * np.exp(returns[i, j]))
    
    # Create DataFrames
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days+1)
    
    # Create DataFrames
    price_col = 'Close'
    dfs = {}
    
    for j, ticker in enumerate(['MSFT', 'AAPL', 'SPY']):
        df = pd.DataFrame({
            price_col: prices[ticker],
            'Open': prices[ticker],
            'High': [price * (1 + np.random.uniform(0, 0.01)) for price in prices[ticker]],
            'Low': [price * (1 - np.random.uniform(0, 0.01)) for price in prices[ticker]],
            'Volume': np.random.randint(5000000, 50000000, size=days+1)
        }, index=dates)
        
        # Calculate log returns
        df['LogReturn'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Save dataframe
        dfs[ticker] = df
        print(f"Created synthetic data for {ticker}: {len(df)} trading days")
    
    return dfs['MSFT'], dfs['AAPL'], dfs['SPY'], price_col


def prepare_market_data(msft_df: pd.DataFrame, aapl_df: pd.DataFrame, spy_df: pd.DataFrame, price_col: str):
    """
    Prepare aligned market data for analysis
    
    Args:
        msft_df: Microsoft stock data
        aapl_df: Apple stock data
        spy_df: S&P 500 index data
        price_col: Which price column to use
        
    Returns:
        Tuple of aligned data
    """
    # Ensure all dataframes have the same dates
    common_dates = msft_df.index.intersection(aapl_df.index).intersection(spy_df.index)
    
    msft = msft_df.loc[common_dates].copy()
    aapl = aapl_df.loc[common_dates].copy()
    spy = spy_df.loc[common_dates].copy()
    
    # Calculate log returns (if not already present)
    if 'LogReturn' not in msft.columns:
        msft['LogReturn'] = np.log(msft[price_col] / msft[price_col].shift(1))
    if 'LogReturn' not in aapl.columns:
        aapl['LogReturn'] = np.log(aapl[price_col] / aapl[price_col].shift(1))
    if 'LogReturn' not in spy.columns:
        spy['LogReturn'] = np.log(spy[price_col] / spy[price_col].shift(1))
    
    return msft, aapl, spy