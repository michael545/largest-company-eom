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

from .utils import generate_trading_dates


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
    Analyze conditional probabilities under different market scenarios
    
    Args:
        spy_scenarios: Array of SPY return scenarios to analyze
        spy_returns: Array of simulated SPY returns
        msft_returns: Array of simulated MSFT returns
        aapl_returns: Array of simulated AAPL returns
        valuation_gaps: Array of simulated valuation gaps (MSFT - AAPL)
        scenario_tolerance: Tolerance around target SPY return
        advanced_libraries: Whether to use advanced libraries for risk metrics
        
    Returns:
        List of dictionaries with scenario analysis results
    """
    scenario_results = []
    
    for scenario_return in spy_scenarios:
        # Define scenario bounds
        lower_bound = scenario_return - scenario_tolerance
        upper_bound = scenario_return + scenario_tolerance
        
        # Create mask for paths in this scenario
        mask = (spy_returns >= lower_bound) & (spy_returns <= upper_bound)
        path_count = np.sum(mask)
        
        # Skip if not enough paths in this scenario
        if path_count < 20:
            scenario_results.append({
                'SPY_Target': scenario_return,
                'SPY_Range': f"{lower_bound:.1%} to {upper_bound:.1%}",
                'Paths': path_count,
                'P_AAPL_gt_MSFT': np.nan,
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
            'SPY_Target': scenario_return,
            'SPY_Range': f"{lower_bound:.1%} to {upper_bound:.1%}",
            'Paths': int(path_count),
            'P_AAPL_gt_MSFT': p_aapl_gt_msft,
            'Avg_ValGap': avg_val_gap,
            'MSFT_Avg_Return': np.mean(msft_scenario_returns),
            'AAPL_Avg_Return': np.mean(aapl_scenario_returns),
            'MSFT_CVaR': cvar_msft,
            'AAPL_CVaR': cvar_aapl
        })
        
        # Print detailed results for this scenario
        print(f"\nScenario: SPY Return = {scenario_return:.1%} (±{scenario_tolerance:.1%})")
        print(f"Paths in scenario: {path_count}")
        print(f"P(AAPL > MSFT): {p_aapl_gt_msft:.4f}")
        print(f"Avg Valuation Gap: ${avg_val_gap:.2f}B")
        print(f"Avg Returns: MSFT {np.mean(msft_scenario_returns)*100:.2f}%, AAPL {np.mean(aapl_scenario_returns)*100:.2f}%")
    
    # Calculate slope of valuation gap vs SPY returns
    valid_scenario_results = [r for r in scenario_results if not np.isnan(r['Avg_ValGap'])]
    if len(valid_scenario_results) >= 2:
        x = np.array([r['SPY_Target'] for r in valid_scenario_results])
        y = np.array([r['Avg_ValGap'] for r in valid_scenario_results])
        
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        print(f"\nRelationship between SPY return and valuation gap:")
        print(f"For each 1% change in SPY return, the valuation gap changes by ${slope:.2f}B")
    
    return scenario_results


def calculate_regime_correlations(
    returns_df: pd.DataFrame,
    market_col: str = 'SPY',
    regime_thresholds: Optional[List[float]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calculate correlations under different market regimes
    
    Args:
        returns_df: DataFrame of asset returns
        market_col: Column representing market returns
        regime_thresholds: List of thresholds for regimes, default is [-0.01, 0.01]
        
    Returns:
        Dictionary of correlation matrices for each regime
    """
    if regime_thresholds is None:
        regime_thresholds = [-0.01, 0.01]
    
    # Define regimes
    market_returns = returns_df[market_col]
    bear_mask = market_returns < regime_thresholds[0]
    neutral_mask = (market_returns >= regime_thresholds[0]) & (market_returns <= regime_thresholds[1])
    bull_mask = market_returns > regime_thresholds[1]
    
    # Calculate correlations for each regime
    regime_corrs = {
        'all': returns_df.corr(),
        'bear': returns_df[bear_mask].corr(),
        'neutral': returns_df[neutral_mask].corr(),
        'bull': returns_df[bull_mask].corr()
    }
    
    return regime_corrs


def calculate_scenario_matrices(
    msft_scenarios: List[float],
    aapl_scenarios: List[float],
    start_prices: List[float],
    shares: List[float]
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Calculate scenario matrices for different return combinations
    
    Args:
        msft_scenarios: List of MSFT return scenarios
        aapl_scenarios: List of AAPL return scenarios
        start_prices: List of starting prices [msft_price, aapl_price]
        shares: List of shares outstanding in billions [msft_shares, aapl_shares]
        
    Returns:
        Tuple of (delta_matrix, outcome_matrix, scenario_results)
    """
    # Set up matrices
    delta_matrix = np.zeros((len(aapl_scenarios), len(msft_scenarios)))
    outcome_matrix = np.zeros((len(aapl_scenarios), len(msft_scenarios)))
    
    # Calculate outcomes for each scenario
    scenario_results = []
    
    for i, msft_ret in enumerate(msft_scenarios):
        for j, aapl_ret in enumerate(aapl_scenarios):
            # Calculate ending prices based on returns
            msft_end_price = start_prices[0] * (1 + msft_ret)
            aapl_end_price = start_prices[1] * (1 + aapl_ret)
            
            # Calculate ending market caps
            msft_end_mcap = msft_end_price * shares[0]
            aapl_end_mcap = aapl_end_price * shares[1]
            
            # Calculate delta
            delta = msft_end_mcap - aapl_end_mcap
            
            # Store delta in matrix
            delta_matrix[j, i] = delta  # Note: j,i for proper orientation
            
            # Store outcome (1 for MSFT > AAPL, 0 otherwise)
            outcome_matrix[j, i] = 1 if delta > 0 else 0
            
            # Determine outcome
            outcome = "MSFT > AAPL" if delta > 0 else "AAPL > MSFT"
            
            # Add to results for select scenarios (to avoid extremely large CSV)
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


def run_advanced_analysis(
    days: int = 30,
    simulations: int = 10000,
    display_paths: int = 200,
    confidence_level: float = 0.9,
    random_seed: int = 42,
    output_dir: str = "analysis_output"
) -> Dict[str, Any]:
    """
    Run the full advanced Monte Carlo analysis workflow
    
    Args:
        days: Number of trading days to simulate
        simulations: Number of simulation paths
        display_paths: Number of paths to display in visualizations
        confidence_level: Confidence level for intervals (0-1)
        random_seed: Random seed for reproducibility
        output_dir: Output directory for results
        
    Returns:
        Dictionary with analysis results
    """
    from .data_loader import load_data
    from .simulation import simulate_correlated_gbm, simulate_correlated_garch
    from .visualization import (
        create_delta_visualization,
        create_3d_surface_plot, 
        create_detailed_scenario_grid,
        create_zoomed_scenario_grid,
        create_scenario_surface_3d,
        create_final_summary_table
    )
    from .utils import ensure_output_dir
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Ensure output directory exists
    ensure_output_dir(output_dir)
    
    # Load data
    try:
        msft, aapl, spy, price_col = load_data()
        print(f"Successfully loaded data with {len(msft)} days for MSFT")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise RuntimeError("Unable to load required data files. Please ensure the data files exist in the 'data' folder.")
    
    # Verify we have data and calculate returns if needed
    for df, name in [(msft, "MSFT"), (aapl, "AAPL"), (spy, "SPY")]:
        if "LogReturn" not in df.columns:
            print(f"Adding LogReturn to {name} data")
            df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Get shares outstanding (in billions) - could be updated with latest data
    msft_shares = 7.43
    aapl_shares = 15.04
    
    # Print current market data
    print("\nCurrent market data:")
    print(f"- MSFT: ${msft[price_col].iloc[-1]:.2f} × {msft_shares}B shares = ${msft[price_col].iloc[-1] * msft_shares:.2f}B")
    print(f"- AAPL: ${aapl[price_col].iloc[-1]:.2f} × {aapl_shares}B shares = ${aapl[price_col].iloc[-1] * aapl_shares:.2f}B")
    
    # Calculate gap
    msft_mcap = msft[price_col].iloc[-1] * msft_shares
    aapl_mcap = aapl[price_col].iloc[-1] * aapl_shares
    gap = msft_mcap - aapl_mcap
    
    print(f"\nValuation Gap (MSFT - AAPL): ${gap:.2f}B")
    
    # Prepare aligned returns DataFrame
    msft_returns = msft['LogReturn'].dropna()
    aapl_returns = aapl['LogReturn'].dropna()
    spy_returns = spy['LogReturn'].dropna()
    
    # Get common date index
    common_index = msft_returns.index.intersection(aapl_returns.index).intersection(spy_returns.index)
    
    # Filter returns to common dates
    msft_returns = msft_returns.loc[common_index]
    aapl_returns = aapl_returns.loc[common_index]
    spy_returns = spy_returns.loc[common_index]
    
    # Create aligned DataFrame
    returns_df = pd.DataFrame({
        'MSFT': msft_returns,
        'AAPL': aapl_returns,
        'SPY': spy_returns
    })
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    print(f"\nHistorical Correlation Matrix:")
    print(correlation_matrix)
    
    # Calculate annualized volatilities
    msft_vol = returns_df['MSFT'].std() * np.sqrt(252)
    aapl_vol = returns_df['AAPL'].std() * np.sqrt(252)
    
    print(f"\nAnnualized Volatilities:")
    print(f"- MSFT: {msft_vol:.4f}")
    print(f"- AAPL: {aapl_vol:.4f}")
    
    # Prepare for simulation
    print(f"\nRunning Monte Carlo simulations with {simulations} paths for {days} trading days...")
    
    # Prepare simulation inputs
    ticker_indices = {'MSFT': 0, 'AAPL': 1, 'SPY': 2}
    start_prices = [msft[price_col].iloc[-1], aapl[price_col].iloc[-1], spy[price_col].iloc[-1]]
    shares = [msft_shares, aapl_shares, 1]  # SPY doesn't need shares for market cap
    
    # Create covariance matrix from returns data
    cov_matrix = returns_df.cov().values
    
    # Calculate mean returns (annualized then converted to daily)
    mean_returns = returns_df.mean() * 252
    
    print("\nRunning GBM simulation...")
    # Run GBM simulation
    gbm_prices, gbm_market_caps = simulate_correlated_gbm(
        start_prices=start_prices,
        shares=shares,
        days=days,
        simulations=simulations,
        cov_matrix=cov_matrix,
        means=mean_returns.values
    )
    
    print("Running GARCH simulation...")
    # Run GARCH simulation
    garch_prices, garch_market_caps = simulate_correlated_garch(
        start_prices=start_prices,
        shares=shares,
        returns_df=returns_df,
        days=days,
        simulations=simulations
    )
    
    # Generate trading dates
    trading_dates = generate_trading_dates(days)
    
    # Calculate valuation deltas (MSFT - AAPL)
    gbm_delta_caps = gbm_market_caps[ticker_indices['MSFT'], :, :] - gbm_market_caps[ticker_indices['AAPL'], :, :]
    garch_delta_caps = garch_market_caps[ticker_indices['MSFT'], :, :] - garch_market_caps[ticker_indices['AAPL'], :, :]
    
    # Create visualizations
    print("\nGenerating visualizations...")
    results = create_delta_visualization(
        gbm_delta_caps, 
        garch_delta_caps, 
        trading_dates,
        output_dir,
        display_paths=display_paths,
        confidence=confidence_level
    )
    
    # Create scenario analysis for different return combinations
    print("\nCreating scenario grid analysis...")
    
    # Create grid of possible returns for MSFT and AAPL for probability surface
    msft_returns_range = np.linspace(-0.2, 0.2, 20)  # -20% to +20%
    aapl_returns_range = np.linspace(-0.2, 0.2, 20)  # -20% to +20%
    
    # Create meshgrid
    msft_grid, aapl_grid = np.meshgrid(msft_returns_range, aapl_returns_range)
    prob_grid = np.zeros_like(msft_grid)
    
    # For each combination, calculate probability
    for i in range(len(msft_returns_range)):
        for j in range(len(aapl_returns_range)):
            msft_ret = msft_returns_range[i]
            aapl_ret = aapl_returns_range[j]
            
            # Calculate ending prices based on returns
            msft_end_price = start_prices[0] * (1 + msft_ret)
            aapl_end_price = start_prices[1] * (1 + aapl_ret)
            
            # Calculate ending market caps
            msft_end_mcap = msft_end_price * msft_shares
            aapl_end_mcap = aapl_end_price * aapl_shares
            
            # Calculate if MSFT > AAPL
            prob_grid[j, i] = 1 if msft_end_mcap > aapl_end_mcap else 0
    
    # Create 3D surface and contour plots
    create_3d_surface_plot(
        (msft_returns_range, aapl_returns_range),
        prob_grid,
        output_dir
    )
    
    # Create a more detailed grid visualization for key scenarios
    print("\nGenerating scenario grid visualization...")
    
    # Define specific return scenarios with 1% increments (from -20% to +20%)
    msft_scenarios = [round(i/100, 2) for i in range(-20, 21, 1)]  # 1% increments from -20% to +20%
    aapl_scenarios = [round(i/100, 2) for i in range(-20, 21, 1)]  # 1% increments from -20% to +20%
    
    # Calculate scenario matrices
    delta_matrix, outcome_matrix, scenario_results = calculate_scenario_matrices(
        msft_scenarios,
        aapl_scenarios,
        start_prices[:2],  # Only need MSFT and AAPL prices
        shares[:2]         # Only need MSFT and AAPL shares
    )
    
    # Create detailed scenario grid
    create_detailed_scenario_grid(
        delta_matrix,
        outcome_matrix,
        msft_scenarios,
        aapl_scenarios,
        output_dir
    )
    
    # Create zoomed scenario grid
    create_zoomed_scenario_grid(
        delta_matrix,
        msft_scenarios,
        aapl_scenarios,
        output_dir,
        zoom_min=-0.05,
        zoom_max=0.05
    )
    
    # Create 3D scenario surface
    create_scenario_surface_3d(
        delta_matrix,
        msft_scenarios,
        aapl_scenarios,
        output_dir
    )
    
    # Save scenario results to CSV
    pd.DataFrame(scenario_results).to_csv(f"{output_dir}/scenario_results.csv", index=False)
    
    # Create a summary table
    create_final_summary_table(results, output_dir)
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"GBM Model: Probability MSFT > AAPL = {results['prob_gbm_positive_final']:.2%}")
    print(f"GARCH Model: Probability MSFT > AAPL = {results['prob_garch_positive_final']:.2%}")
    print(f"\nResults saved to {output_dir}/")
    
    return {
        'results': results,
        'scenario_results': scenario_results,
        'delta_matrix': delta_matrix,
        'outcome_matrix': outcome_matrix,
        'gbm_delta_caps': gbm_delta_caps,
        'garch_delta_caps': garch_delta_caps
    }