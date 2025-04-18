#!/usr/bin/env python
"""
Monte Carlo Analysis for MSFT vs AAPL Market Cap Delta
This performs Monte Carlo simulations using GBM and GARCH models
to analyze the market cap delta between Microsoft and Apple over a 30-day period,
with a focus on predicting the probability of a positive delta.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from arch.univariate import GARCH
from models.data_loader import load_data, create_synthetic_data
from models.simulation import simulate_correlated_gbm, simulate_correlated_garch

#Local&params
OUTPUT_DIR = "analysis_output"
SIMULATIONS = 10000
DAYS = 30  # Extend to 30 trading days per requirements
DISPLAY_PATHS = 200
CONFIDENCE_LEVEL = 0.9  # confidence intervals (90%)
RANDOM_SEED = 42  # reproducibility


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def generate_trading_dates(days, start_date=None):
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
    """Calculate confidence intervals for data series"""
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower_bound = np.percentile(data, lower_percentile, axis=1)
    upper_bound = np.percentile(data, upper_percentile, axis=1)
    median = np.median(data, axis=1)
    mean = np.mean(data, axis=1)
    
    return lower_bound, median, upper_bound, mean


def create_delta_visualization(
    gbm_delta_caps, 
    garch_delta_caps, 
    trading_dates,
    output_dir,
    display_paths=100,
    confidence=0.9
):
    """
    Create visualizations comparing GBM and GARCH valuation deltas
    
    Args:
        gbm_delta_caps: Delta market caps from GBM simulations [days+1, simulations]
        garch_delta_caps: Delta market caps from GARCH simulations [days+1, simulations]
        trading_dates: List of trading dates
        output_dir: Directory to save output
        display_paths: Number of individual paths to display
        confidence: Confidence level for intervals (0-1)
    """
    # Format dates for x-axis
    date_strings = [d.strftime('%Y-%m-%d') for d in trading_dates]
    
    # Calculate confidence intervals and statistics
    gbm_lower, gbm_median, gbm_upper, gbm_mean = calculate_conf_intervals(gbm_delta_caps, confidence)
    garch_lower, garch_median, garch_upper, garch_mean = calculate_conf_intervals(garch_delta_caps, confidence)
    
    # Calculate probability of MSFT > AAPL at each time point
    prob_gbm_positive = np.mean(gbm_delta_caps > 0, axis=1)
    prob_garch_positive = np.mean(garch_delta_caps > 0, axis=1)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: GBM Mean with confidence intervals and sample paths
    ax = axes[0, 0]
    ax.fill_between(range(len(trading_dates)), gbm_lower, gbm_upper, alpha=0.3, color='blue', label=f'{confidence*100:.0f}% Confidence Interval')
    ax.plot(gbm_median, color='blue', linestyle='--', label='Median (GBM)', linewidth=2)
    ax.plot(gbm_mean, color='darkblue', label='Mean (GBM)', linewidth=2)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.7, linewidth=1.5, label='Zero Line')
    
    # Add sample paths
    gbm_path_indices = np.random.choice(gbm_delta_caps.shape[1], min(display_paths, gbm_delta_caps.shape[1]), replace=False)
    for i in gbm_path_indices:
        ax.plot(gbm_delta_caps[:, i], color='lightblue', alpha=0.1)
    
    ax.set_title('GBM: MSFT-AAPL Market Cap Delta ($ Billions)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Market Cap Delta ($ Billions)', fontsize=12)
    ax.set_xticks(range(0, len(trading_dates), 5))
    ax.set_xticklabels([date_strings[i] for i in range(0, len(date_strings), 5)], rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 2: GARCH Mean with confidence intervals and sample paths
    ax = axes[0, 1]
    ax.fill_between(range(len(trading_dates)), garch_lower, garch_upper, alpha=0.3, color='green', label=f'{confidence*100:.0f}% Confidence Interval')
    ax.plot(garch_median, color='green', linestyle='--', label='Median (GARCH)', linewidth=2)
    ax.plot(garch_mean, color='darkgreen', label='Mean (GARCH)', linewidth=2)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.7, linewidth=1.5, label='Zero Line')
    
    # Add sample paths
    garch_path_indices = np.random.choice(garch_delta_caps.shape[1], min(display_paths, garch_delta_caps.shape[1]), replace=False)
    for i in garch_path_indices:
        ax.plot(garch_delta_caps[:, i], color='lightgreen', alpha=0.1)
    
    ax.set_title('GARCH: MSFT-AAPL Market Cap Delta ($ Billions)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Market Cap Delta ($ Billions)', fontsize=12)
    ax.set_xticks(range(0, len(trading_dates), 5))
    ax.set_xticklabels([date_strings[i] for i in range(0, len(date_strings), 5)], rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 3: Probability of positive delta over time comparison
    ax = axes[1, 0]
    ax.plot(prob_gbm_positive, color='blue', label='GBM Model', linewidth=2)
    ax.plot(prob_garch_positive, color='green', label='GARCH Model', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Probability')
    
    ax.set_title('Probability of MSFT Market Cap > AAPL Market Cap', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, len(trading_dates), 5))
    ax.set_xticklabels([date_strings[i] for i in range(0, len(date_strings), 5)], rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 4: Histogram of terminal values
    ax = axes[1, 1]
    terminal_gbm = gbm_delta_caps[-1, :]
    terminal_garch = garch_delta_caps[-1, :]
    
    sns.histplot(terminal_gbm, ax=ax, color='blue', alpha=0.5, label='GBM', kde=True)
    sns.histplot(terminal_garch, ax=ax, color='green', alpha=0.5, label='GARCH', kde=True)
    ax.axvline(x=0, color='red', linestyle='-', alpha=0.7, label='Zero Line')
    
    # Add text with probabilities
    prob_gbm_positive_final = np.mean(terminal_gbm > 0)
    prob_garch_positive_final = np.mean(terminal_garch > 0)
    
    ax.text(0.05, 0.95, 
        f"GBM: P(MSFT > AAPL) = {prob_gbm_positive_final:.2%}\n"
        f"GARCH: P(MSFT > AAPL) = {prob_garch_positive_final:.2%}",
        transform=ax.transAxes, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
        verticalalignment='top'
    )
    
    ax.set_title('Distribution of Terminal Values (Day 30)', fontsize=14)
    ax.set_xlabel('Market Cap Delta ($ Billions)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gbm_vs_garch_comparison.png", dpi=300)
    
    # Create interactive Plotly visualization
    try:
        # Create figure for interactive plot
        interactive_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "GBM: Market Cap Delta Projection",
                "GARCH: Market Cap Delta Projection",
                "Probability of MSFT > AAPL Over Time",
                "Terminal Value Distribution (Day 30)"
            )
        )
        
        # Add GBM confidence interval
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=gbm_upper,
                fill=None,
                mode='lines',
                line=dict(color='rgba(0, 0, 255, 0)'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=gbm_lower,
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(0, 0, 255, 0)'),
                fillcolor='rgba(0, 0, 255, 0.2)',
                name='GBM 90% Confidence Interval'
            ),
            row=1, col=1
        )
        
        # Add GBM mean
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=gbm_mean,
                mode='lines',
                line=dict(color='blue', width=2),
                name='GBM Mean'
            ),
            row=1, col=1
        )
        
        # Add GBM zero line
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=np.zeros(len(trading_dates)),
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='Zero Line'
            ),
            row=1, col=1
        )
        
        # Add GARCH confidence interval
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=garch_upper,
                fill=None,
                mode='lines',
                line=dict(color='rgba(0, 128, 0, 0)'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=garch_lower,
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(0, 128, 0, 0)'),
                fillcolor='rgba(0, 128, 0, 0.2)',
                name='GARCH 90% Confidence Interval'
            ),
            row=1, col=2
        )
        
        # Add GARCH mean
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=garch_mean,
                mode='lines',
                line=dict(color='green', width=2),
                name='GARCH Mean'
            ),
            row=1, col=2
        )
        
        # Add GARCH zero line
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=np.zeros(len(trading_dates)),
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='Zero Line',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add probability comparison
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=prob_gbm_positive,
                mode='lines',
                line=dict(color='blue', width=2),
                name='GBM Probability'
            ),
            row=2, col=1
        )
        
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=prob_garch_positive,
                mode='lines',
                line=dict(color='green', width=2),
                name='GARCH Probability'
            ),
            row=2, col=1
        )
        
        # Add 50% probability line
        interactive_fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=[0.5] * len(trading_dates),
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='50% Probability',
            ),
            row=2, col=1
        )
        
        # Add histogram of terminal values
        interactive_fig.add_trace(
            go.Histogram(
                x=terminal_gbm,
                opacity=0.6,
                name='GBM Terminal',
                marker_color='blue',
                nbinsx=50
            ),
            row=2, col=2
        )
        
        interactive_fig.add_trace(
            go.Histogram(
                x=terminal_garch,
                opacity=0.6,
                name='GARCH Terminal',
                marker_color='green',
                nbinsx=50
            ),
            row=2, col=2
        )
        
        # Add zero line to histogram
        interactive_fig.add_vline(
            x=0, 
            line_width=2, 
            line_dash="dash", 
            line_color="red",
            row=2, col=2
        )
        
        # Update layout
        interactive_fig.update_layout(
            title_text="MSFT vs AAPL Market Cap Delta: GBM vs GARCH Comparison",
            height=900,
            width=1200,
            showlegend=True,
        )
        
        # Update axes
        interactive_fig.update_yaxes(title_text="Market Cap Delta ($ Billions)", row=1, col=1)
        interactive_fig.update_yaxes(title_text="Market Cap Delta ($ Billions)", row=1, col=2)
        interactive_fig.update_yaxes(title_text="Probability", row=2, col=1)
        interactive_fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        interactive_fig.update_xaxes(title_text="Date", row=1, col=1)
        interactive_fig.update_xaxes(title_text="Date", row=1, col=2)
        interactive_fig.update_xaxes(title_text="Date", row=2, col=1)
        interactive_fig.update_xaxes(title_text="Market Cap Delta ($ Billions)", row=2, col=2)
        
        # Add annotations with probabilities
        interactive_fig.add_annotation(
            text=f"P(MSFT > AAPL) = {prob_gbm_positive_final:.2%}",
            xref="x4", yref="y4",
            x=0.8 * min(min(terminal_gbm), min(terminal_garch)),
            y=0.9 * interactive_fig.data[-1]['y'].max(),
            showarrow=False,
            font=dict(color="blue", size=14),
            bgcolor="white",
            opacity=0.8,
            align="left"
        )
        
        interactive_fig.add_annotation(
            text=f"P(MSFT > AAPL) = {prob_garch_positive_final:.2%}",
            xref="x4", yref="y4",
            x=0.8 * min(min(terminal_gbm), min(terminal_garch)),
            y=0.8 * interactive_fig.data[-1]['y'].max(),
            showarrow=False,
            font=dict(color="green", size=14),
            bgcolor="white",
            opacity=0.8,
            align="left"
        )
        
        # Save to HTML file
        interactive_fig.write_html(f"{output_dir}/gbm_vs_garch_interactive.html")
        
    except Exception as e:
        print(f"Warning: Could not create interactive visualization: {e}")
    
    # Create additional comparative table
    stats = {
        'Metric': [
            'Probability of Positive Delta (Day 30)',
            'Mean Delta (Day 30, $ Billions)',
            'Median Delta (Day 30, $ Billions)',
            '5th Percentile ($ Billions)',
            '95th Percentile ($ Billions)',
            'Standard Deviation ($ Billions)'
        ],
        'GBM Model': [
            f"{prob_gbm_positive_final:.2%}",
            f"{gbm_mean[-1]:.2f}",
            f"{gbm_median[-1]:.2f}",
            f"{np.percentile(terminal_gbm, 5):.2f}",
            f"{np.percentile(terminal_gbm, 95):.2f}",
            f"{np.std(terminal_gbm):.2f}"
        ],
        'GARCH Model': [
            f"{prob_garch_positive_final:.2%}",
            f"{garch_mean[-1]:.2f}",
            f"{garch_median[-1]:.2f}",
            f"{np.percentile(terminal_garch, 5):.2f}",
            f"{np.percentile(terminal_garch, 95):.2f}",
            f"{np.std(terminal_garch):.2f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f"{output_dir}/model_comparison_statistics.csv", index=False)
    
    # Create a visual table for the statistics
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title("Model Comparison: GBM vs GARCH", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_table.png", dpi=300)
    
    return {
        'prob_gbm_positive_final': prob_gbm_positive_final,
        'prob_garch_positive_final': prob_garch_positive_final
    }


def create_3d_surface_plot(
    factor_grid,
    prob_grid,
    output_dir,
    factor1_name="MSFT Return",
    factor2_name="AAPL Return"
):
    """
    Create a 3D surface plot of probability based on two factors
    
    Args:
        factor_grid: 2D grid of factor values (factor1, factor2)
        prob_grid: 2D grid of probability values
        output_dir: Directory to save output
        factor1_name: Name of the first factor (x-axis)
        factor2_name: Name of the second factor (y-axis)
    """
    try:
        factor1_vals, factor2_vals = factor_grid
        
        fig = go.Figure(data=[
            go.Surface(
                z=prob_grid,
                x=factor1_vals,
                y=factor2_vals,
                colorscale='Viridis',
                colorbar=dict(title="Probability")
            )
        ])
        
        # Add a plane at z=0.5 to visualize the 50% probability level
        x_mesh, y_mesh = np.meshgrid(factor1_vals, factor2_vals)
        z_plane = np.ones_like(x_mesh) * 0.5
        
        fig.add_trace(
            go.Surface(
                z=z_plane,
                x=x_mesh,
                y=y_mesh,
                showscale=False,
                opacity=0.3,
                colorscale=[[0, 'red'], [1, 'red']],
                name="50% Probability"
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Probability of MSFT > AAPL Market Cap',
            scene=dict(
                xaxis_title=factor1_name,
                yaxis_title=factor2_name,
                zaxis_title='Probability',
                xaxis=dict(tickformat='.1%'),
                yaxis=dict(tickformat='.1%'),
                zaxis=dict(range=[0, 1])
            ),
            width=1000,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
        )
        
        # Save as HTML
        fig.write_html(f"{output_dir}/probability_surface.html")
        
        # Also create a 2D contour plot for easier interpretation
        contour_fig = go.Figure(data=
            go.Contour(
                z=prob_grid,
                x=factor1_vals * 100,  # Convert to percentages
                y=factor2_vals * 100,  # Convert to percentages
                colorscale='Viridis',
                contours=dict(
                    start=0,
                    end=1,
                    size=0.05,
                    showlabels=True,
                ),
                colorbar=dict(title="Probability")
            )
        )
        
        # Add contour line for 50% probability
        contour_fig.add_trace(
            go.Contour(
                z=prob_grid,
                x=factor1_vals * 100,  # Convert to percentages
                y=factor2_vals * 100,  # Convert to percentages
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=0.1,
                    showlabels=True,
                    labelfont=dict(color="white"),
                ),
                showscale=False,
                line=dict(color="red", width=2),
                name="50% Probability"
            )
        )
        
        contour_fig.update_layout(
            title='Probability Contour: P(MSFT Market Cap > AAPL Market Cap)',
            xaxis_title=f'{factor1_name} (%)',
            yaxis_title=f'{factor2_name} (%)',
            width=1000,
            height=800
        )
        
        contour_fig.write_html(f"{output_dir}/probability_contour.html")
        
    except Exception as e:
        print(f"Warning: Could not create 3D surface plot: {e}")


def main():
    """Main analysis function"""
    print("Advanced Monte Carlo Analysis: MSFT vs AAPL Market Cap Delta")
    print("==========================================================")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)
    
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
    print(f"\nRunning Monte Carlo simulations with {SIMULATIONS} paths for {DAYS} trading days...")
    
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
        days=DAYS,
        simulations=SIMULATIONS,
        cov_matrix=cov_matrix,
        means=mean_returns.values
    )
    
    print("Running GARCH simulation...")
    # Run GARCH simulation
    garch_prices, garch_market_caps = simulate_correlated_garch(
        start_prices=start_prices,
        shares=shares,
        returns_df=returns_df,
        days=DAYS,
        simulations=SIMULATIONS
    )
    
    # Generate trading dates
    trading_dates = generate_trading_dates(DAYS)
    
    # Calculate valuation deltas (MSFT - AAPL)
    gbm_delta_caps = gbm_market_caps[ticker_indices['MSFT'], :, :] - gbm_market_caps[ticker_indices['AAPL'], :, :]
    garch_delta_caps = garch_market_caps[ticker_indices['MSFT'], :, :] - garch_market_caps[ticker_indices['AAPL'], :, :]
    
    # Create visualizations
    print("\nGenerating visualizations...")
    results = create_delta_visualization(
        gbm_delta_caps, 
        garch_delta_caps, 
        trading_dates,
        OUTPUT_DIR,
        DISPLAY_PATHS,
        CONFIDENCE_LEVEL
    )
    
    # Create scenario analysis for different return combinations
    print("\nCreating scenario grid analysis...")
    
    # Create grid of possible returns for MSFT and AAPL
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
        OUTPUT_DIR
    )
    
    # Create a more detailed grid visualization for key scenarios
    print("\nGenerating scenario grid visualization...")
    
    # Define specific return scenarios with 1% increments (from -20% to +20%)
    msft_scenarios = [round(i/100, 2) for i in range(-20, 21, 1)]  # 1% increments from -20% to +20%
    aapl_scenarios = [round(i/100, 2) for i in range(-20, 21, 1)]  # 1% increments from -20% to +20%
    
    # Create a figure with a bigger size to accommodate the expanded grid
    plt.figure(figsize=(24, 24))
    
    # Calculate number of scenarios and determine grid parameters
    grid_size = len(msft_scenarios)
    
    # Set up a heatmap instead of individual subplots for better visualization
    delta_matrix = np.zeros((len(msft_scenarios), len(aapl_scenarios)))
    outcome_matrix = np.zeros((len(msft_scenarios), len(aapl_scenarios)))
    
    # Calculate outcomes for each scenario
    scenario_results = []
    
    for i, msft_ret in enumerate(msft_scenarios):
        for j, aapl_ret in enumerate(aapl_scenarios):
            # Calculate ending prices based on returns
            msft_end_price = start_prices[0] * (1 + msft_ret)
            aapl_end_price = start_prices[1] * (1 + aapl_ret)
            
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
    
    # Create heatmaps for delta and outcomes
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Delta heatmap
    im0 = axes[0].imshow(
        delta_matrix, 
        cmap='RdYlGn', 
        interpolation='none',
        aspect='auto',
        vmin=-max(abs(np.min(delta_matrix)), abs(np.max(delta_matrix))),
        vmax=max(abs(np.min(delta_matrix)), abs(np.max(delta_matrix)))
    )
    
    # Mark the zero-crossover boundary with a contour line
    CS = axes[0].contour(
        np.arange(len(msft_scenarios)), 
        np.arange(len(aapl_scenarios)), 
        delta_matrix, 
        levels=[0], 
        colors='black', 
        linewidths=2
    )
    
    # Set ticks and labels
    step = 5  # Show every 5th label to avoid crowding
    axes[0].set_xticks(np.arange(0, len(msft_scenarios), step))
    axes[0].set_yticks(np.arange(0, len(aapl_scenarios), step))
    axes[0].set_xticklabels([f"{x:.0%}" for x in msft_scenarios[::step]])
    axes[0].set_yticklabels([f"{y:.0%}" for y in aapl_scenarios[::step]])
    
    # Add labels and title
    axes[0].set_xlabel('MSFT Return', fontsize=14)
    axes[0].set_ylabel('AAPL Return', fontsize=14)
    axes[0].set_title('Market Cap Delta (MSFT - AAPL) in $ Billions', fontsize=16)
    
    # Add colorbar
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('$ Billions', rotation=270, labelpad=20, fontsize=12)
    
    # Outcome heatmap (binary: MSFT > AAPL or not)
    im1 = axes[1].imshow(outcome_matrix, cmap='RdYlGn', interpolation='none', aspect='auto')
    
    # Set ticks and labels
    axes[1].set_xticks(np.arange(0, len(msft_scenarios), step))
    axes[1].set_yticks(np.arange(0, len(aapl_scenarios), step))
    axes[1].set_xticklabels([f"{x:.0%}" for x in msft_scenarios[::step]])
    axes[1].set_yticklabels([f"{y:.0%}" for y in aapl_scenarios[::step]])
    
    # Add labels and title
    axes[1].set_xlabel('MSFT Return', fontsize=14)
    axes[1].set_ylabel('AAPL Return', fontsize=14)
    axes[1].set_title('Outcome: MSFT Market Cap > AAPL Market Cap', fontsize=16)
    
    # Add colorbar with custom labels
    cbar1 = plt.colorbar(im1, ax=axes[1], ticks=[0, 1])
    cbar1.set_ticklabels(['AAPL > MSFT', 'MSFT > AAPL'])
    
    # Add the decision boundary
    CS = axes[1].contour(
        np.arange(len(msft_scenarios)), 
        np.arange(len(aapl_scenarios)), 
        outcome_matrix, 
        levels=[0.5], 
        colors='black', 
        linewidths=2
    )
    axes[1].clabel(CS, inline=1, fontsize=10, fmt='Decision\nBoundary')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/detailed_scenario_grid.png", dpi=300)
    
    # Create a zoomed-in version focused on the -5% to +5% range
    plt.figure(figsize=(15, 15))
    
    # Find indices for the zoom region
    zoom_min = -0.05
    zoom_max = 0.05
    
    msft_min_idx = next(i for i, x in enumerate(msft_scenarios) if x >= zoom_min)
    msft_max_idx = next(i for i, x in enumerate(msft_scenarios) if x > zoom_max) - 1
    aapl_min_idx = next(i for i, x in enumerate(aapl_scenarios) if x >= zoom_min)
    aapl_max_idx = next(i for i, x in enumerate(aapl_scenarios) if x > zoom_max) - 1
    
    # Extract the zoom region
    zoom_delta = delta_matrix[aapl_min_idx:aapl_max_idx+1, msft_min_idx:msft_max_idx+1]
    zoom_msft = msft_scenarios[msft_min_idx:msft_max_idx+1]
    zoom_aapl = aapl_scenarios[aapl_min_idx:aapl_max_idx+1]
    
    # Create the zoomed heatmap
    plt.imshow(
        zoom_delta, 
        cmap='RdYlGn', 
        interpolation='none',
        aspect='auto',
        vmin=-max(abs(np.min(zoom_delta)), abs(np.max(zoom_delta))),
        vmax=max(abs(np.min(zoom_delta)), abs(np.max(zoom_delta))),
        extent=[
            msft_scenarios[msft_min_idx]*100, 
            msft_scenarios[msft_max_idx]*100, 
            aapl_scenarios[aapl_min_idx]*100, 
            aapl_scenarios[aapl_max_idx]*100
        ]
    )
    
    # Add contour line for the decision boundary
    plt.contour(
        np.linspace(zoom_msft[0]*100, zoom_msft[-1]*100, len(zoom_msft)),
        np.linspace(zoom_aapl[0]*100, zoom_aapl[-1]*100, len(zoom_aapl)),
        zoom_delta,
        levels=[0],
        colors='black',
        linewidths=2
    )
    
    # Add grid lines
    plt.grid(alpha=0.3)
    
    # Add labels and title
    plt.xlabel('MSFT Return (%)', fontsize=14)
    plt.ylabel('AAPL Return (%)', fontsize=14)
    plt.title('Zoomed Market Cap Delta (±5% Return Range)', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('$ Billions', rotation=270, labelpad=20, fontsize=12)
    
    # Add annotations for key points
    for i, msft_ret in enumerate(zoom_msft):
        for j, aapl_ret in enumerate(zoom_aapl):
            # Only annotate select points to avoid overcrowding
            if i % 2 == 0 and j % 2 == 0:
                plt.text(
                    msft_ret*100, 
                    aapl_ret*100, 
                    f"{zoom_delta[j, i]:.1f}",
                    ha='center', 
                    va='center', 
                    fontsize=7,
                    color='black' if abs(zoom_delta[j, i]) < max(abs(np.min(zoom_delta)), abs(np.max(zoom_delta)))/2 else 'white'
                )
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/zoomed_scenario_grid.png", dpi=300)
    
    # Save scenario results to CSV
    pd.DataFrame(scenario_results).to_csv(f"{OUTPUT_DIR}/scenario_results.csv", index=False)
    
    # Also create an interactive 3D surface for better visualization
    try:
        x_range = np.array(msft_scenarios) * 100  # Convert to percentages
        y_range = np.array(aapl_scenarios) * 100
        
        surface_fig = go.Figure(data=[
            go.Surface(
                x=x_range,
                y=y_range,
                z=delta_matrix,
                colorscale='RdYlGn',
                colorbar=dict(
                    title="Delta ($ Billions)",
                    titleside="right"
                )
            )
        ])
        
        # Add a plane at z=0 to show the decision boundary
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_plane = np.zeros_like(x_mesh)
        
        surface_fig.add_trace(
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=z_plane,
                showscale=False,
                opacity=0.7,
                colorscale=[[0, 'gray'], [1, 'gray']],
                name="Break-even"
            )
        )
        
        # Update layout
        surface_fig.update_layout(
            title='3D Surface: Market Cap Delta (MSFT - AAPL)',
            scene=dict(
                xaxis_title='MSFT Return (%)',
                yaxis_title='AAPL Return (%)',
                zaxis_title='Market Cap Delta ($ Billions)'
            ),
            autosize=False,
            width=1000,
            height=800
        )
        
        # Save as HTML
        surface_fig.write_html(f"{OUTPUT_DIR}/scenario_surface_3d.html")
    except Exception as e:
        print(f"Warning: Could not create 3D scenario surface: {e}")
    
    # Create a summary table
    summary_stats = {
        'Model': ['GBM', 'GARCH'],
        'Probability MSFT > AAPL': [
            f"{results['prob_gbm_positive_final']:.2%}",
            f"{results['prob_garch_positive_final']:.2%}"
        ],
        'Notes': [
            'Assumes constant volatility',
            'Models time-varying volatility & fat tails'
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create visual table for summary
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title("Summary: Probability of MSFT Market Cap > AAPL Market Cap (30-Day Horizon)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/summary_table.png", dpi=300)
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"GBM Model: Probability MSFT > AAPL = {results['prob_gbm_positive_final']:.2%}")
    print(f"GARCH Model: Probability MSFT > AAPL = {results['prob_garch_positive_final']:.2%}")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()