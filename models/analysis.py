"""
Module for financial analysis and scenario testing
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False


def analyze_scenarios(
    spy_scenarios: np.ndarray,
    spy_returns: np.ndarray,
    msft_returns: np.ndarray,
    aapl_returns: np.ndarray,
    valuation_gaps: np.ndarray,
    scenario_tolerance: float = 0.03,
    advanced_libraries: bool = False
) -> List[Dict[str, Any]]:
    """
    Analyze scenarios based on SPY returns and calculate corresponding statistics
    
    Args:
        spy_scenarios: Array of SPY return scenarios to analyze
        spy_returns: Array of SPY returns from simulation
        msft_returns: Array of MSFT returns from simulation
        aapl_returns: Array of AAPL returns from simulation
        valuation_gaps: Array of valuation gaps (MSFT-AAPL)
        scenario_tolerance: Tolerance around target SPY return
        advanced_libraries: Whether to use advanced libraries for calculations
        
    Returns:
        List of dictionaries with scenario results
    """
    scenario_results = []
    
    print(f"SPY Return Range: {np.min(spy_returns):.2%} to {np.max(spy_returns):.2%}")
    print(f"MSFT Return Range: {np.min(msft_returns):.2%} to {np.max(msft_returns):.2%}")
    print(f"AAPL Return Range: {np.min(aapl_returns):.2%} to {np.max(aapl_returns):.2%}")
    print(f"Valuation Gap Range: ${np.min(valuation_gaps):.2f}B to ${np.max(valuation_gaps):.2f}B")
    
    for target_return in spy_scenarios:
        # Create a mask for paths where SPY return is within tolerance of target
        lower_bound = target_return - scenario_tolerance
        upper_bound = target_return + scenario_tolerance
        
        mask = (spy_returns >= lower_bound) & (spy_returns <= upper_bound)
        
        # Skip if we don't have enough paths
        if np.sum(mask) < 20:
            print(f"Skipping SPY {target_return:.2%} scenario - only {np.sum(mask)} paths found")
            scenario_results.append({
                'SPY_Target': target_return,
                'Paths': np.sum(mask),
                'P_AAPL_GT_MSFT': np.nan,
                'Avg_ValGap': np.nan,
                'MSFT_Avg_Return': np.nan,
                'AAPL_Avg_Return': np.nan,
                'MSFT_CVaR': np.nan,
                'AAPL_CVaR': np.nan
            })
            continue
        
        # Filter paths
        msft_scenario_returns = msft_returns[mask]
        aapl_scenario_returns = aapl_returns[mask]
        valuation_gaps_scenario = valuation_gaps[mask]
        
        # Calculate probability of AAPL having higher valuation than MSFT
        p_aapl_gt_msft = np.mean(valuation_gaps_scenario < 0)
        
        # Calculate average valuation gap
        avg_val_gap = np.mean(valuation_gaps_scenario)
        
        # Calculate CVaR if riskfolio-lib is available
        if advanced_libraries and RISKFOLIO_AVAILABLE:
            try:
                scenario_returns = pd.DataFrame({
                    'MSFT': msft_scenario_returns,
                    'AAPL': aapl_scenario_returns
                })
                cvar_msft = rp.CVaR_Hist(pd.DataFrame(msft_scenario_returns), alpha=0.05)[0]
                cvar_aapl = rp.CVaR_Hist(pd.DataFrame(aapl_scenario_returns), alpha=0.05)[0]
            except Exception as e:
                print(f"Error calculating CVaR: {e}")
                cvar_msft = cvar_aapl = np.nan
        else:
            cvar_msft = cvar_aapl = np.nan
        
        # Store scenario results
        scenario_results.append({
            'SPY_Target': target_return,
            'Paths': np.sum(mask),
            'P_AAPL_GT_MSFT': p_aapl_gt_msft,
            'Avg_ValGap': avg_val_gap,
            'MSFT_Avg_Return': np.mean(msft_scenario_returns),
            'AAPL_Avg_Return': np.mean(aapl_scenario_returns),
            'MSFT_CVaR': cvar_msft,
            'AAPL_CVaR': cvar_aapl
        })
        
        # Print scenario results
        print(f"\nSPY {target_return*100:.2f}% scenario (± {scenario_tolerance*100:.0f}%): {np.sum(mask)} paths")
        print(f"P(AAPL > MSFT): {p_aapl_gt_msft:.4f}")
        print(f"Avg Valuation Gap: ${avg_val_gap:.2f}B")
        print(f"Avg Returns: MSFT {np.mean(msft_scenario_returns)*100:.2f}%, AAPL {np.mean(aapl_scenario_returns)*100:.2f}%")
    
    # Calculate slope of valuation gap vs SPY returns
    valid_scenario_results = [r for r in scenario_results if not np.isnan(r['Avg_ValGap'])]
    if len(valid_scenario_results) >= 2:
        x = np.array([r['SPY_Target'] for r in valid_scenario_results])
        y = np.array([r['Avg_ValGap'] for r in valid_scenario_results])
        
        # Simple linear regression
        slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
        
        print(f"\nChange in valuation gap per 1% change in SPY: ${slope/0.01:.2f}B")
    
    return scenario_results


def calculate_regime_correlations(
    returns_df: pd.DataFrame,
    market_col: str = 'SPY',
    regime_thresholds: Optional[List[float]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calculate correlations under different market regimes
    
    Args:
        returns_df: DataFrame of return series
        market_col: Column representing market returns
        regime_thresholds: Thresholds for market return regimes
        
    Returns:
        Dictionary of correlation matrices by regime
    """
    if regime_thresholds is None:
        # Default: Bull/Bear/Normal market regimes
        spy_std = returns_df[market_col].std()
        regime_thresholds = [-spy_std, spy_std]
    
    # Determine regimes
    regimes = []
    if returns_df[market_col] <= regime_thresholds[0]:
        regimes.append('Bear')
    elif returns_df[market_col] >= regime_thresholds[1]:
        regimes.append('Bull')
    else:
        regimes.append('Normal')
    
    returns_df['Regime'] = regimes
    
    # Calculate correlations by regime
    regime_corrs = {}
    for regime in ['Bear', 'Normal', 'Bull']:
        regime_data = returns_df[returns_df['Regime'] == regime].drop('Regime', axis=1)
        if len(regime_data) > 10:  # Only calculate if we have enough data
            regime_corrs[regime] = regime_data.corr()
    
    returns_df.drop('Regime', axis=1, inplace=True)
    return regime_corrs