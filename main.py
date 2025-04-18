import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.data_loader import load_data, create_synthetic_data
from models.simulation import simulate_correlated_gbm
from models.valuation import (
    CompanyValuation, 
    calculate_valuation_gap, 
    calculate_probability_company1_exceeds_company2
)
from models.analysis import analyze_scenarios
from models.visualization import (
    create_3d_probability_plot,
    create_scenario_grid_plot,
    create_summary_table,
    create_valuation_gap_scatter_plot
)

# Configuration parameters
OUTPUT_DIR = "analysis_output"
SIMULATIONS = 10000
DAYS = 14 
SCENARIO_TOLERANCE = 0.03  # ±3% tolerance for scenario analysis


def main():
    """Main analysis function"""
    print("AAPL vs MSFT Valuation Gap Analysis")
    print("===================================")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    try:
        msft, aapl, spy, price_col = load_data()
        print(f"Successfully loaded data with {len(msft)} days for MSFT")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        msft, aapl, spy, price_col = create_synthetic_data()
    
    # Verify we have data and calculate returns if needed
    for df, name in [(msft, "MSFT"), (aapl, "AAPL"), (spy, "SPY")]:
        if "LogReturn" not in df.columns:
            print(f"Adding LogReturn to {name} data")
            df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Fixed shares outstanding (in billions) - constants, not indexed
    msft_shares = 7.43
    aapl_shares = 15.04
    
    # Calculate historical market caps and valuation gap
    msft['MarketCap'] = msft[price_col] * msft_shares
    aapl['MarketCap'] = aapl[price_col] * aapl_shares
    
    # Print current data
    print("\nCurrent market data:")
    print(f"- MSFT: ${msft[price_col].iloc[-1]:.2f} × {msft_shares}B shares = ${msft['MarketCap'].iloc[-1]:.2f}B")
    print(f"- AAPL: ${aapl[price_col].iloc[-1]:.2f} × {aapl_shares}B shares = ${aapl['MarketCap'].iloc[-1]:.2f}B")
    print(f"- SPY Price: ${spy[price_col].iloc[-1]:.2f}")
    
    gap = msft['MarketCap'].iloc[-1] - aapl['MarketCap'].iloc[-1]
    print(f"\nValuation Gap (MSFT - AAPL): ${gap:.2f}B")
    
    # Calculate historical correlations and betas
    # Make sure we align the return series properly
    msft_returns = msft['LogReturn'].dropna()
    aapl_returns = aapl['LogReturn'].dropna()
    spy_returns = spy['LogReturn'].dropna()
    
    # Get common date index
    common_index = msft_returns.index.intersection(aapl_returns.index).intersection(spy_returns.index)
    if len(common_index) < 30:
        print("WARNING: Not enough common trading days for reliable correlation")
    
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
    spy_vol = returns_df['SPY'].std() * np.sqrt(252)
    
    print(f"\nAnnualized Volatilities:")
    print(f"- MSFT: {msft_vol:.4f}")
    print(f"- AAPL: {aapl_vol:.4f}")
    print(f"- SPY: {spy_vol:.4f}")
    
    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation with {SIMULATIONS} paths for {DAYS} trading days...")
    
    # Prepare simulation inputs
    ticker_indices = {'MSFT': 0, 'AAPL': 1, 'SPY': 2}
    start_prices = [msft[price_col].iloc[-1], aapl[price_col].iloc[-1], spy[price_col].iloc[-1]]
    shares = [msft_shares, aapl_shares, 1]  # SPY doesn't need shares for market cap
    
    # Create covariance matrix from returns data
    cov_matrix = returns_df.cov().values
    
    # Run the simulation
    prices, market_caps = simulate_correlated_gbm(
        start_prices=start_prices,
        shares=shares,
        days=DAYS,
        simulations=SIMULATIONS,
        cov_matrix=cov_matrix
    )
    
    # Extract terminal values
    msft_final_prices = prices[ticker_indices['MSFT'], -1, :]
    aapl_final_prices = prices[ticker_indices['AAPL'], -1, :]
    spy_final_prices = prices[ticker_indices['SPY'], -1, :]
    
    # Calculate returns
    msft_returns = (msft_final_prices - msft[price_col].iloc[-1]) / msft[price_col].iloc[-1]
    aapl_returns = (aapl_final_prices - aapl[price_col].iloc[-1]) / aapl[price_col].iloc[-1]
    spy_returns = (spy_final_prices - spy[price_col].iloc[-1]) / spy[price_col].iloc[-1]
    
    # Calculate valuations and valuation gap - using constant share values
    msft_final_valuations = msft_final_prices * msft_shares
    aapl_final_valuations = aapl_final_prices * aapl_shares
    valuation_gaps = calculate_valuation_gap(msft_final_valuations, aapl_final_valuations)
    
    # Calculate the probability that MSFT's valuation exceeds AAPL's
    prob_msft_gt_aapl = calculate_probability_company1_exceeds_company2(valuation_gaps)
    prob_aapl_gt_msft = 1 - prob_msft_gt_aapl
    
    print(f"\nOverall Results:")
    print(f"- Probability MSFT valuation > AAPL valuation: {prob_msft_gt_aapl:.4f}")
    print(f"- Probability AAPL valuation > MSFT valuation: {prob_aapl_gt_msft:.4f}")
    print(f"- Average Final Valuation Gap: ${np.mean(valuation_gaps):.2f}B")
    
    # Define SPY scenarios based on actual simulation range
    min_spy_return = -0.30  # -30%
    max_spy_return = 0.30   # +30%
    print(f"\nSPY return range in simulation: {min_spy_return:.2%} to {max_spy_return:.2%}")
    
    # Create scenarios within this range
    spy_scenarios = np.linspace(min_spy_return, max_spy_return, 9)
    
    # Analyze scenarios
    print("\nAnalyzing conditional probabilities under different market scenarios...")
    scenario_results = analyze_scenarios(
        spy_scenarios=spy_scenarios,
        spy_returns=spy_returns,
        msft_returns=msft_returns,
        aapl_returns=aapl_returns,
        valuation_gaps=valuation_gaps,
        scenario_tolerance=SCENARIO_TOLERANCE
    )
    
    # Save results to CSV
    results_df = pd.DataFrame(scenario_results)
    results_df.to_csv(f"{OUTPUT_DIR}/scenario_results.csv", index=False)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    valid_scenarios = [i for i, result in enumerate(scenario_results) if result['Paths'] >= 20]
    valid_results = [r for r in scenario_results if r['Paths'] >= 20]
    
    # Create scenario grid plot
    if len(valid_scenarios) > 0:
        create_scenario_grid_plot(
            valid_scenarios=valid_scenarios,
            spy_scenarios=spy_scenarios,
            spy_returns=spy_returns,
            scenario_tolerance=SCENARIO_TOLERANCE,
            market_caps=market_caps,
            prices=prices,
            ticker_indices=ticker_indices,
            valuation_gaps=valuation_gaps,
            output_dir=OUTPUT_DIR
        )
    
    # Create 3D probability plot
    if np.sum(~np.isnan(spy_returns)) > 100:
        create_3d_probability_plot(
            spy_returns=spy_returns,
            valuation_gaps=valuation_gaps,
            output_dir=OUTPUT_DIR
        )
    
    # Create summary table
    if valid_results:
        create_summary_table(valid_results=valid_results, output_dir=OUTPUT_DIR)
    
    # Create valuation gap scatter plot
    create_valuation_gap_scatter_plot(
        spy_returns=spy_returns,
        valuation_gaps=valuation_gaps,
        output_dir=OUTPUT_DIR
    )
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}/")
    print("Generated files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(f"- {OUTPUT_DIR}/{filename}")


if __name__ == "__main__":
    main()