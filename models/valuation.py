"""
Module for valuation calculations and metrics
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

class CompanyValuation:
    """Class representing a company's valuation"""
    
    def __init__(self, 
                 ticker: str, 
                 price: float, 
                 shares_outstanding: float,
                 price_history: np.ndarray = None,
                 volatility: float = None):
        """
        Initialize company valuation
        
        Args:
            ticker: Company ticker symbol
            price: Current stock price
            shares_outstanding: Shares outstanding in billions
            price_history: Historical price array (optional)
            volatility: Historical volatility (optional)
        """
        self.ticker = ticker
        self.price = price
        self.shares_outstanding = shares_outstanding  # in billions
        self.price_history = price_history
        self.volatility = volatility
        
    @property
    def market_cap(self) -> float:
        """Calculate market capitalization in billions"""
        return self.price * self.shares_outstanding
    
    def __str__(self) -> str:
        return f"{self.ticker}: ${self.price:.2f} × {self.shares_outstanding:.2f}B shares = ${self.market_cap:.2f}B"
    
    def calculate_future_valuations(self, future_prices: np.ndarray) -> np.ndarray:
        """
        Calculate future valuations based on simulated prices
        
        Args:
            future_prices: Array of future prices from simulation
            
        Returns:
            Array of future market caps in billions
        """
        return future_prices * self.shares_outstanding


def calculate_company_valuations(
    prices: np.ndarray,
    shares: List[float],
    tickers: List[str]
) -> List[np.ndarray]:
    """
    Calculate company valuations from simulated prices
    
    Args:
        prices: 3D array of simulated prices [tickers, days, simulations]
        shares: List of shares outstanding per ticker (in billions)
        tickers: List of ticker symbols
        
    Returns:
        List of valuation arrays
    """
    valuations = []
    for i, ticker in enumerate(tickers):
        ticker_prices = prices[i]
        ticker_shares = shares[i]
        ticker_valuations = ticker_prices * ticker_shares
        valuations.append(ticker_valuations)
    
    return valuations


def calculate_valuation_gap(
    company1_valuations: np.ndarray,
    company2_valuations: np.ndarray
) -> np.ndarray:
    """
    Calculate the valuation gap between two companies
    
    Args:
        company1_valuations: Market cap array for company 1 (in billions)
        company2_valuations: Market cap array for company 2 (in billions)
        
    Returns:
        Array of valuation gaps (company1 - company2) in billions
    """
    return company1_valuations - company2_valuations


def calculate_probability_company1_exceeds_company2(valuation_gap: np.ndarray) -> float:
    """
    Calculate probability that company1's valuation exceeds company2's
    
    Args:
        valuation_gap: Array of valuation gaps (company1 - company2)
        
    Returns:
        Probability that company1 > company2
    """
    # If gap > 0, then company1 > company2
    return float(np.mean(valuation_gap > 0))


def calculate_conditional_probabilities(
    spy_returns: np.ndarray,
    valuation_gaps: np.ndarray,
    scenario_return: float,
    tolerance: float = 0.03
) -> Tuple[float, float, int]:
    """
    Calculate conditional probability of valuation gap given a market scenario
    
    Args:
        spy_returns: Array of market returns
        valuation_gaps: Array of valuation gaps (e.g., MSFT - AAPL)
        scenario_return: Target market return to condition on
        tolerance: Return tolerance around the target
        
    Returns:
        Tuple of (probability company1 > company2, average gap, path count)
    """
    # Filter paths where market return is within tolerance
    lower_bound = scenario_return - tolerance
    upper_bound = scenario_return + tolerance
    mask = (spy_returns >= lower_bound) & (spy_returns <= upper_bound)
    
    # Count paths in this scenario
    path_count = np.sum(mask)
    
    if path_count < 20:  # Require at least 20 paths
        return np.nan, np.nan, path_count
    
    # Calculate probability and average gap in this scenario
    valuation_gaps_scenario = valuation_gaps[mask]
    prob_company1_gt_company2 = np.mean(valuation_gaps_scenario > 0)
    avg_gap = np.mean(valuation_gaps_scenario)
    
    return prob_company1_gt_company2, avg_gap, path_count