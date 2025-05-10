"""
Utility functions for financial analysis
"""
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


def ensure_output_dir(output_dir):
    """
    Ensure output directory exists
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def generate_trading_dates(days, start_date=None):
    """
    Generate a list of trading dates (weekdays only)
    
    Args:
        days: Number of trading days to generate
        start_date: Starting date (default: today)
        
    Returns:
        List of datetime objects representing trading days
    """
    if start_date is None:
        start_date = datetime.now()
    
    dates = []
    current_date = start_date
    
    while len(dates) < days + 1:  # +1 to include today
        if current_date.weekday() < 5:  # Mon=0, Fri=4
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    return dates


def calculate_conf_intervals(data, confidence=0.9):
    """
    Calculate confidence intervals for data series
    
    Args:
        data: Data array of shape [time_points, simulations]
        confidence: Confidence level (0-1)
        
    Returns:
        lower_bound, median, upper_bound, mean arrays
    """
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower_bound = np.percentile(data, lower_percentile, axis=1)
    upper_bound = np.percentile(data, upper_percentile, axis=1)
    median = np.median(data, axis=1)
    mean = np.mean(data, axis=1)
    
    return lower_bound, median, upper_bound, mean


def create_return_scenarios(min_return=-0.2, max_return=0.2, num_scenarios=41):
    """
    Create a range of return scenarios
    
    Args:
        min_return: Minimum return (default: -20%)
        max_return: Maximum return (default: +20%)
        num_scenarios: Number of scenarios to create
        
    Returns:
        Array of return scenarios
    """
    return np.linspace(min_return, max_return, num_scenarios)


def filter_scenario_paths(target_return, actual_returns, tolerance=0.03):
    """
    Filter paths based on target return and tolerance
    
    Args:
        target_return: Target return
        actual_returns: Array of actual returns
        tolerance: Tolerance around target return
        
    Returns:
        Boolean mask for paths that match the scenario
    """
    lower_bound = target_return - tolerance
    upper_bound = target_return + tolerance
    return (actual_returns >= lower_bound) & (actual_returns <= upper_bound)


def calculate_scenario_grid(msft_scenarios, aapl_scenarios, msft_price, aapl_price, msft_shares, aapl_shares):
    """
    Calculate scenario grid for all combinations of MSFT and AAPL returns
    
    Args:
        msft_scenarios: Array of MSFT return scenarios
        aapl_scenarios: Array of AAPL return scenarios
        msft_price: Current MSFT price
        aapl_price: Current AAPL price
        msft_shares: MSFT shares outstanding (in billions)
        aapl_shares: AAPL shares outstanding (in billions)
        
    Returns:
        delta_matrix: 2D array of market cap deltas
        outcome_matrix: 2D array of binary outcomes (1=MSFT>AAPL, 0=AAPL>MSFT)
        scenario_results: List of dictionaries with detailed scenario results
    """
    # Set up matrices
    delta_matrix = np.zeros((len(aapl_scenarios), len(msft_scenarios)))
    outcome_matrix = np.zeros((len(aapl_scenarios), len(msft_scenarios)))
    scenario_results = []
    
    # Calculate outcomes for each scenario
    for i, msft_ret in enumerate(msft_scenarios):
        for j, aapl_ret in enumerate(aapl_scenarios):
            # Calculate ending prices based on returns
            msft_end_price = msft_price * (1 + msft_ret)
            aapl_end_price = aapl_price * (1 + aapl_ret)
            
            # Calculate ending market caps
            msft_end_mcap = msft_end_price * msft_shares
            aapl_end_mcap = aapl_end_price * aapl_shares
            
            # Calculate delta
            delta = msft_end_mcap - aapl_end_mcap
            
            # Store delta in matrix
            delta_matrix[j, i] = delta  # Note: j,i for proper orientation
            
            # Store outcome (1 for MSFT > AAPL, 0 otherwise)
            outcome_matrix[j, i] = 1 if delta > 0 else 0
            
            # Determine outcome
            outcome = "MSFT > AAPL" if delta > 0 else "AAPL > MSFT"
            
            # Add to results for select scenarios (to avoid extremely large output)
            if msft_ret % 0.05 == 0 and aapl_ret % 0.05 == 0:  # Only include every 5%
                scenario_results.append({
                    'MSFT Return': f"{msft_ret:.1%}",
                    'AAPL Return': f"{aapl_ret:.1%}",
                    'MSFT Price': f"${msft_end_price:.2f}",
                    'AAPL Price': f"${aapl_end_price:.2f}",
                    'MSFT Market Cap': f"${msft_end_mcap:.2f}B",
                    'AAPL Market Cap': f"${aapl_end_mcap:.2f}B",
                    'Delta': f"${delta:.2f}B",
                    'Outcome': outcome
                })
    
    return delta_matrix, outcome_matrix, scenario_results