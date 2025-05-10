import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Returns:
        Tuple of (msft_df, aapl_df, spy_df, price_column)
    """
    cache_dir = "data"
    
    msft_path = f"{cache_dir}/MSFT.csv"
    aapl_path = f"{cache_dir}/AAPL.csv"
    spy_path = f"{cache_dir}/SPY.csv"
    
    if not os.path.exists(msft_path) or not os.path.exists(aapl_path) or not os.path.exists(spy_path):
        raise FileNotFoundError(
            f"files not found. Something is wrong with the data:\n"
            f"- {msft_path}\n- {aapl_path}\n- {spy_path}"
        )
        
    # Read data from CSV files
    try:
        msft = pd.read_csv(msft_path, parse_dates=['Datetime'], index_col='Datetime')
        aapl = pd.read_csv(aapl_path, parse_dates=['Datetime'], index_col='Datetime')
        spy = pd.read_csv(spy_path, parse_dates=['Datetime'], index_col='Datetime')
        price_col = 'Price'  # Default column from saved data
        
        # Calculate log returns
        for df, name in [(msft, "MSFT"), (aapl, "AAPL"), (spy, "SPY")]:
            if 'LogReturn' not in df.columns:
                df['LogReturn'] = np.log(df[price_col] / df[price_col].shift(1))
                print(f"Added LogReturn to {name} data")
        
        print(f"Successfully loaded data from CSV files")
        return msft, aapl, spy, price_col
    except Exception as e:
        raise RuntimeError(f"Error reading data files: {e}")


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
    # Ensure same dates
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