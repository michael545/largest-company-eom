#!/usr/bin/env python
"""
Valuation Delta Analysis - Focused on MSFT vs AAPL market cap crossover
This script simulates stock price paths for MSFT, AAPL, and SPY, with a focus on
visualizing when MSFT's market cap exceeds AAPL's market cap.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.data_loader import load_data, create_synthetic_data
from models.simulation import simulate_correlated_gbm

# Configuration parameters
OUTPUT_DIR = "analysis_output"
SIMULATIONS = 10000
DAYS = 14  # Number of trading days to simulate
DISPLAY_PATHS = 300  # Number of paths to display in visualizations
SPY_MIN_RETURN = -0.30  # -30%
SPY_MAX_RETURN = 0.30   # +30%
SCENARIO_TOLERANCE = 0.03  # ±3% tolerance for scenario matching


def ensure_correlated_spy_scenarios(
    prices, market_caps, ticker_indices, spy_targets, scenario_tolerance
):
    """
    Identify paths that match specific SPY return scenarios
    
    Args:
        prices: Simulated price paths
        market_caps: Simulated market cap paths
        ticker_indices: Dictionary mapping tickers to indices
        spy_targets: List of target SPY returns
        scenario_tolerance: Tolerance around target return
        
    Returns:
        Dictionary of scenario paths for each SPY target
    """
    spy_start = prices[ticker_indices['SPY'], 0, 0]
    spy_final = prices[ticker_indices['SPY'], -1, :]
    spy_returns = (spy_final - spy_start) / spy_start
    
    scenario_paths = {}
    for target in spy_targets:
        # Find paths where SPY return is within tolerance of target
        mask = (spy_returns >= target - scenario_tolerance) & (spy_returns <= target + scenario_tolerance)
        scenario_paths[target] = mask
        
    return scenario_paths


def generate_trading_dates(days):
    """Generate future trading dates, skipping weekends"""
    today = datetime.now()
    dates = []
    current_date = today
    
    while len(dates) < days + 1:  # +1 to include today
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    return dates


def create_valuation_delta_paths_plot(
    market_caps, ticker_indices, dates, output_dir, display_paths=100, file_prefix="delta"
):
    """
    Create a visualization showing the MSFT-AAPL valuation delta paths
    
    Args:
        market_caps: Simulated market cap paths
        ticker_indices: Dictionary mapping tickers to indices
        dates: List of trading dates
        output_dir: Directory to save output files
        display_paths: Number of paths to display
        file_prefix: Prefix for output file names
    """
    # Calculate valuation delta paths (MSFT - AAPL)
    msft_idx = ticker_indices['MSFT']
    aapl_idx = ticker_indices['AAPL']
    
    delta_caps = market_caps[msft_idx, :, :] - market_caps[aapl_idx, :, :]
    
    # Create a subset of paths for display
    if market_caps.shape[2] > display_paths:
        # Choose some random paths, making sure to include both positive and negative final outcomes
        final_deltas = delta_caps[-1, :]
        pos_indices = np.where(final_deltas > 0)[0]
        neg_indices = np.where(final_deltas <= 0)[0]
        
        # If we have both positive and negative paths, select a balanced sample
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            pos_sample = np.random.choice(
                pos_indices, 
                min(display_paths // 2, len(pos_indices)), 
                replace=False
            )
            neg_sample = np.random.choice(
                neg_indices, 
                min(display_paths - len(pos_sample), len(neg_indices)), 
                replace=False
            )
            path_indices = np.concatenate([pos_sample, neg_sample])
        else:
            # Otherwise just take a random sample
            path_indices = np.random.choice(
                market_caps.shape[2], display_paths, replace=False
            )
    else:
        path_indices = np.arange(market_caps.shape[2])
    
    # Calculate percentage of paths where MSFT > AAPL at each time point
    prob_msft_gt_aapl_over_time = np.mean(delta_caps > 0, axis=1)
    
    # Format dates for x-axis
    date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Create the first plot - Delta paths
    plt.figure(figsize=(15, 10))
    
    # Plot zero line
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.8, linewidth=2, label="Breakeven Line")
    
    # Plot all paths with transparency
    for i in path_indices:
        end_value = delta_caps[-1, i]
        if end_value > 0:
            plt.plot(date_strings, delta_caps[:, i], color='green', alpha=0.2)
        else:
            plt.plot(date_strings, delta_caps[:, i], color='gray', alpha=0.2)
    
    # Highlight a few distinct paths
    special_paths = np.random.choice(path_indices, 5)
    for i, path_idx in enumerate(special_paths):
        end_value = delta_caps[-1, path_idx]
        color = 'green' if end_value > 0 else 'blue'
        plt.plot(date_strings, delta_caps[:, path_idx], color=color, alpha=0.9, linewidth=2)
    
    # Set plot properties
    plt.title('MSFT-AAPL Valuation Gap Paths', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Valuation Gap ($ Billions)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Add annotation showing probability of MSFT > AAPL
    final_prob = prob_msft_gt_aapl_over_time[-1]
    plt.annotate(
        f'Probability MSFT > AAPL at EOM: {final_prob:.1%}',
        xy=(0.02, 0.95), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
        fontsize=12
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_paths.png", dpi=300)
    
    # Create second plot - Probability of MSFT > AAPL over time
    plt.figure(figsize=(15, 7))
    plt.plot(date_strings, prob_msft_gt_aapl_over_time, 'b-', linewidth=3)
    plt.title('Probability of MSFT Market Cap > AAPL Market Cap Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_probability.png", dpi=300)
    
    # Create third plot - Heatmap of valuation delta paths
    plt.figure(figsize=(15, 10))
    
    # Sort paths by final delta value
    sorted_indices = np.argsort(delta_caps[-1, :])
    sorted_deltas = delta_caps[:, sorted_indices]
    
    # Create a custom colormap: red for negative, white for zero, green for positive
    colors = ['darkred', 'red', 'white', 'green', 'darkgreen']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('delta_cmap', colors, N=n_bins)
    
    # Plot heatmap
    plt.imshow(
        sorted_deltas.T, 
        aspect='auto', 
        cmap=cmap,
        vmin=-max(abs(np.min(sorted_deltas)), abs(np.max(sorted_deltas))),
        vmax=max(abs(np.min(sorted_deltas)), abs(np.max(sorted_deltas)))
    )
    
    # Format x-axis to show dates
    plt.xticks(np.arange(len(dates)), date_strings, rotation=45)
    plt.colorbar(label='Valuation Gap ($ Billions)')
    plt.title('Heatmap of MSFT-AAPL Valuation Gap Paths', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Simulation Path (sorted by final value)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_heatmap.png", dpi=300)
    
    # Create path analysis by final SPY return
    plt.figure(figsize=(12, 8))
    
    # Calculate SPY returns
    spy_idx = ticker_indices['SPY']
    spy_start = market_caps[spy_idx, 0, 0] / (market_caps[spy_idx, 0, 0] / prices[spy_idx, 0, 0])  # To get price, not mkt cap
    spy_final = market_caps[spy_idx, -1, :] / (market_caps[spy_idx, -1, :] / prices[spy_idx, -1, :])
    spy_returns = (spy_final - spy_start) / spy_start
    
    # Plot final valuation gap vs SPY return
    plt.scatter(
        spy_returns, delta_caps[-1, :], 
        alpha=0.5, 
        c=delta_caps[-1, :] > 0,  # Color by positive/negative
        cmap='RdYlGn'
    )
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title('Final MSFT-AAPL Valuation Gap vs SPY Return', fontsize=16)
    plt.xlabel('SPY Return', fontsize=14)
    plt.ylabel('Valuation Gap ($ Billions)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_spy_correlation.png", dpi=300)
    
    # Create an interactive Plotly visualization
    try:
        fig = make_subplots(rows=1, cols=1)
        
        # Add zero line
        fig.add_shape(
            type="line", line=dict(dash="dash", color="red", width=2),
            x0=0, x1=len(dates)-1, y0=0, y1=0
        )
        
        # Add a selection of paths
        for i in path_indices[:min(50, len(path_indices))]:
            end_value = delta_caps[-1, i]
            color = 'green' if end_value > 0 else 'rgba(100,100,100,0.6)'
            fig.add_trace(
                go.Scatter(
                    x=date_strings, 
                    y=delta_caps[:, i], 
                    mode='lines',
                    line=dict(color=color, width=2 if end_value > 0 else 1),
                    opacity=0.7 if end_value > 0 else 0.3,
                    showlegend=False
                )
            )
        
        # Add probability line as a secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=prob_msft_gt_aapl_over_time,
                mode='lines+markers',
                name='P(MSFT>AAPL)',
                line=dict(color='blue', width=3),
                yaxis='y2'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='MSFT-AAPL Valuation Gap Paths and Probability of MSFT > AAPL',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Valuation Gap ($ Billions)'),
            yaxis2=dict(
                title='Probability',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
            height=800,
            width=1200
        )
        
        # Save to HTML
        fig.write_html(f"{output_dir}/{file_prefix}_interactive.html")
        
    except Exception as e:
        print(f"Warning: Could not create interactive visualization: {e}")


def create_spy_scenario_analysis(
    ticker_indices, prices, market_caps, scenario_paths,
    spy_scenarios, output_dir
):
    """
    Create analysis and visualization for different SPY scenarios
    
    Args:
        ticker_indices: Dictionary mapping tickers to indices
        prices: Simulated price paths
        market_caps: Simulated market cap paths
        scenario_paths: Dictionary of paths for each SPY scenario
        spy_scenarios: List of SPY scenario targets
        output_dir: Directory to save output files
    """
    msft_idx = ticker_indices['MSFT']
    aapl_idx = ticker_indices['AAPL']
    spy_idx = ticker_indices['SPY']
    
    # Setup for grid visualization
    scenario_count = len([s for s in spy_scenarios if np.any(scenario_paths[s])])
    if scenario_count == 0:
        print("Warning: No valid scenarios with matching paths found")
        return
        
    rows = int(np.ceil(scenario_count / 3))
    cols = min(scenario_count, 3)
    
    plt.figure(figsize=(15, 5 * rows))
    
    # Create table for scenarios
    scenario_data = []
    
    plot_idx = 1
    for i, target in enumerate(spy_scenarios):
        # Skip scenarios with no matching paths
        if not np.any(scenario_paths[target]):
            print(f"Skipping SPY {target:.2%} scenario - only 0 paths found")
            continue
            
        # Get the matching paths
        paths = np.where(scenario_paths[target])[0]
        
        # Calculate valuation deltas for this scenario
        scenario_deltas = market_caps[msft_idx, :, paths] - market_caps[aapl_idx, :, paths]
        
        # Calculate probability of MSFT > AAPL
        prob_msft_gt_aapl = np.mean(scenario_deltas[-1, :] > 0)
        
        # Calculate average returns
        msft_avg_return = np.mean((prices[msft_idx, -1, paths] - prices[msft_idx, 0, 0]) / prices[msft_idx, 0, 0])
        aapl_avg_return = np.mean((prices[aapl_idx, -1, paths] - prices[aapl_idx, 0, 0]) / prices[aapl_idx, 0, 0])
        
        # Add to scenario data
        scenario_data.append({
            'SPY Scenario': f"{target:.2%}",
            'Paths': len(paths),
            'P(MSFT > AAPL)': prob_msft_gt_aapl,
            'Avg Valuation Gap': np.mean(scenario_deltas[-1, :]),
            'MSFT Avg Return': msft_avg_return,
            'AAPL Avg Return': aapl_avg_return
        })
        
        # Create subplot
        plt.subplot(rows, cols, plot_idx)
        
        # Plot zero line
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        
        # Plot paths
        for p in range(min(50, len(paths))):
            path_idx = paths[p]
            plt.plot(scenario_deltas[:, p], 'gray', alpha=0.3)
            
        # Plot mean
        plt.plot(np.mean(scenario_deltas, axis=1), 'blue', linewidth=3)
        
        plt.title(f"SPY {target:.2%} Scenario\nP(MSFT>AAPL)={prob_msft_gt_aapl:.2f}")
        plt.tight_layout()
        plot_idx += 1
    
    # Save grid plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spy_scenarios_grid.png", dpi=300)
    
    # Save scenario data to CSV
    pd.DataFrame(scenario_data).to_csv(f"{output_dir}/spy_scenario_analysis.csv", index=False)
    
    # Create table visualization of scenario data
    plt.figure(figsize=(12, len(scenario_data) * 0.6 + 1))
    table = plt.table(
        cellText=[[d[k] if not isinstance(d[k], float) else f"{d[k]:.4f}" for k in d.keys()] for d in scenario_data],
        colLabels=scenario_data[0].keys(),
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.title("SPY Scenario Analysis", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spy_scenario_table.png", dpi=300)


def identify_crossover_points(market_caps, ticker_indices):
    """
    Identify when MSFT market cap exceeds AAPL market cap in each path
    
    Args:
        market_caps: Simulated market cap paths
        ticker_indices: Dictionary mapping tickers to indices
        
    Returns:
        List of day indices where crossover happens for each path, -1 if no crossover
    """
    msft_idx = ticker_indices['MSFT']
    aapl_idx = ticker_indices['AAPL']
    num_days = market_caps.shape[1]
    num_paths = market_caps.shape[2]
    
    # Get initial state - is MSFT already > AAPL?
    msft_initial = market_caps[msft_idx, 0, 0]
    aapl_initial = market_caps[aapl_idx, 0, 0]
    msft_initially_higher = msft_initial > aapl_initial
    
    # Initialize array for crossover days
    crossover_days = np.zeros(num_paths, dtype=int) - 1  # -1 means no crossover
    
    # For each path
    for p in range(num_paths):
        msft_caps = market_caps[msft_idx, :, p]
        aapl_caps = market_caps[aapl_idx, :, p]
        
        # If MSFT initially higher, look for when AAPL exceeds MSFT
        if msft_initially_higher:
            for d in range(1, num_days):
                if msft_caps[d] <= aapl_caps[d]:
                    crossover_days[p] = d
                    break
        # Otherwise look for when MSFT exceeds AAPL
        else:
            for d in range(1, num_days):
                if msft_caps[d] > aapl_caps[d]:
                    crossover_days[p] = d
                    break
    
    return crossover_days, msft_initially_higher


def visualize_crossover_analysis(
    crossover_days, msft_initially_higher, dates, output_dir
):
    """
    Visualize analysis of when valuation crossovers occur
    
    Args:
        crossover_days: Array of days where crossover happens for each path
        msft_initially_higher: Boolean if MSFT market cap was initially higher
        dates: List of trading dates
        output_dir: Directory to save output files
    """
    # Count paths with crossover vs no crossover
    crossover_count = np.sum(crossover_days >= 0)
    no_crossover_count = np.sum(crossover_days < 0)
    total_paths = len(crossover_days)
    
    crossover_type = "AAPL exceeding MSFT" if msft_initially_higher else "MSFT exceeding AAPL"
    
    # Create histogram of crossover days
    plt.figure(figsize=(12, 8))
    
    # Filter out -1 values (no crossover)
    valid_crossovers = crossover_days[crossover_days >= 0]
    
    if len(valid_crossovers) > 0:
        # Create histogram
        plt.hist(valid_crossovers, bins=range(len(dates)), color='blue', alpha=0.7)
        plt.xlabel('Trading Day', fontsize=14)
        plt.ylabel('Number of Paths', fontsize=14)
        plt.title(f'Histogram of {crossover_type} Valuation Crossover Days', fontsize=16)
        
        # Set x-ticks to show dates
        plt.xticks(range(len(dates)), [d.strftime('%Y-%m-%d') for d in dates], rotation=45)
        
        # Annotate with probabilities
        plt.annotate(
            f'Paths with crossover: {crossover_count} ({crossover_count/total_paths:.1%})\n'
            f'Paths with no crossover: {no_crossover_count} ({no_crossover_count/total_paths:.1%})',
            xy=(0.02, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
            fontsize=12
        )
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/valuation_crossover_histogram.png", dpi=300)
    else:
        plt.text(0.5, 0.5, f"No {crossover_type} valuation crossovers observed", 
                horizontalalignment='center', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/valuation_crossover_histogram.png", dpi=300)


def main():
    """Main analysis function"""
    print("MSFT vs AAPL Valuation Delta Path Analysis")
    print("==========================================")
    
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
    
    # Generate trading dates for visualization
    trading_dates = generate_trading_dates(DAYS)
    
    # Extract terminal values
    msft_final_prices = prices[ticker_indices['MSFT'], -1, :]
    aapl_final_prices = prices[ticker_indices['AAPL'], -1, :]
    spy_final_prices = prices[ticker_indices['SPY'], -1, :]
    
    # Calculate returns
    msft_returns = (msft_final_prices - msft[price_col].iloc[-1]) / msft[price_col].iloc[-1]
    aapl_returns = (aapl_final_prices - aapl[price_col].iloc[-1]) / aapl[price_col].iloc[-1]
    spy_returns = (spy_final_prices - spy[price_col].iloc[-1]) / spy[price_col].iloc[-1]
    
    # Calculate valuations and valuation gap
    msft_final_valuations = msft_final_prices * msft_shares
    aapl_final_valuations = aapl_final_prices * aapl_shares
    valuation_gaps = msft_final_valuations - aapl_final_valuations
    
    # Calculate probability that MSFT's valuation exceeds AAPL's
    prob_msft_gt_aapl = np.mean(valuation_gaps > 0)
    prob_aapl_gt_msft = 1 - prob_msft_gt_aapl
    
    print(f"\nOverall Results:")
    print(f"- Probability MSFT valuation > AAPL valuation: {prob_msft_gt_aapl:.4f}")
    print(f"- Probability AAPL valuation > MSFT valuation: {prob_aapl_gt_msft:.4f}")
    print(f"- Average Final Valuation Gap: ${np.mean(valuation_gaps):.2f}B")
    
    # Print range of returns observed in the simulation
    print(f"\nSPY return range in simulation: {np.min(spy_returns):.2%} to {np.max(spy_returns):.2%}")
    
    # Define SPY scenarios from -30% to +30%
    spy_scenarios = np.linspace(SPY_MIN_RETURN, SPY_MAX_RETURN, 9)
    scenario_paths = ensure_correlated_spy_scenarios(
        prices, market_caps, ticker_indices, spy_scenarios, SCENARIO_TOLERANCE
    )
    
    # Create valuation delta path visualizations
    print("\nGenerating valuation delta path visualizations...")
    create_valuation_delta_paths_plot(
        market_caps, ticker_indices, trading_dates, OUTPUT_DIR, DISPLAY_PATHS
    )
    
    # Create SPY scenario analysis
    print("\nAnalyzing conditional probabilities under different market scenarios...")
    create_spy_scenario_analysis(
        ticker_indices, prices, market_caps, scenario_paths,
        spy_scenarios, OUTPUT_DIR
    )
    
    # Identify and visualize crossover points
    crossover_days, msft_initially_higher = identify_crossover_points(market_caps, ticker_indices)
    visualize_crossover_analysis(crossover_days, msft_initially_higher, trading_dates, OUTPUT_DIR)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}/")
    print("Generated files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if filename.startswith("delta") or filename.startswith("spy_scenario") or filename.startswith("valuation_crossover"):
            print(f"- {OUTPUT_DIR}/{filename}")


if __name__ == "__main__":
    main()