import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import gaussian_kde
from .utils import calculate_conf_intervals

# Check if plotly is available
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_delta_visualization(
    gbm_delta_caps,
    garch_delta_caps,
    trading_dates,
    output_dir,
    display_paths=100,
    confidence=0.9,
    show_plot: bool = False
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
        show_plot: Whether to display the plot in a popup window
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
    if show_plot:
        plt.show()
    
    # Create interactive Plotly visualization if available
    if PLOTLY_AVAILABLE:
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


def create_3d_probability_plot(
    spy_returns,
    valuation_gaps,
    output_dir: str,
    show_plot: bool = False
):
    """
    Create a 3D scatter plot of valuation gap against SPY returns
    
    Args:
        spy_returns: Array of SPY returns
        valuation_gaps: Array of valuation gaps (MSFT - AAPL)
        output_dir: Directory to save output
        show_plot: Whether to display the plot
    """
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(spy_returns * 100, valuation_gaps, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--', label='Break-even Line')
    plt.xlabel('SPY Return (%)', fontsize=12)
    plt.ylabel('Valuation Gap ($ Billions)', fontsize=12)
    plt.title('Relationship Between SPY Returns and Valuation Gap', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spy_vs_valuation_gap.png", dpi=300)
    if show_plot:
        plt.show()


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
    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available, skipping 3D surface plot")
        return
        
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


def create_scenario_grid_plot(
    valid_scenarios,
    spy_scenarios,
    spy_returns,
    scenario_tolerance,
    market_caps,
    prices,
    ticker_indices,
    valuation_gaps,
    output_dir,
    show_plot: bool = False
):
    """
    Create a grid of scenario subplots showing valuation distributions
    
    Args:
        valid_scenarios: List of valid scenario indices
        spy_scenarios: List of SPY return scenarios
        spy_returns: Array of SPY returns
        scenario_tolerance: Tolerance around target SPY return
        market_caps: Array of market cap simulations
        prices: Array of price simulations
        ticker_indices: Dict mapping tickers to indices
        valuation_gaps: Array of valuation gaps
        output_dir: Directory to save output
        show_plot: Whether to display the plot
    """
    # Create grid of subplots for valid scenarios
    n_scenarios = len(valid_scenarios)
    
    if n_scenarios <= 0:
        print("No valid scenarios to plot")
        return
    
    # Create grid layout
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Ensure axes is iterable for single subplot case
    if n_scenarios == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Plot each scenario
    for i, scenario_idx in enumerate(valid_scenarios):
        if i >= len(axes):
            break
            
        scenario_return = spy_scenarios[scenario_idx]
        lower_bound = scenario_return - scenario_tolerance
        upper_bound = scenario_return + scenario_tolerance
        
        # Create mask for paths in this scenario
        mask = (spy_returns >= lower_bound) & (spy_returns <= upper_bound)
        
        # Filter paths
        valuation_gaps_scenario = valuation_gaps[mask]
        
        # Create histogram for this scenario
        axes[i].hist(valuation_gaps_scenario, bins=30, alpha=0.7)
        axes[i].axvline(x=0, color='red', linestyle='--')
        axes[i].set_title(f"SPY Return: {scenario_return:.1%}")
        axes[i].set_xlabel('Valuation Gap ($ Billions)')
        axes[i].set_ylabel('Frequency')
        
        # Add text with probability
        p_msft_gt_aapl = np.mean(valuation_gaps_scenario > 0)
        axes[i].text(0.05, 0.95, 
            f"P(MSFT > AAPL) = {p_msft_gt_aapl:.2%}\n"
            f"Paths: {np.sum(mask)}",
            transform=axes[i].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
    
    # Hide unused axes
    for i in range(n_scenarios, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario_grid.png", dpi=300)
    if show_plot:
        plt.show()


def create_summary_table(valid_results, output_dir, show_plot: bool = False):
    """
    Create a summary table of scenario results
    
    Args:
        valid_results: List of valid scenario results
        output_dir: Directory to save output
        show_plot: Whether to display the plot
    """
    # Create a visual table for the scenario results
    plt.figure(figsize=(12, len(valid_results) * 0.5 + 2))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create a subset of columns to display
    display_cols = ['SPY_Target', 'SPY_Range', 'Paths', 'P_AAPL_gt_MSFT', 'Avg_ValGap']
    display_data = []
    col_names = ['SPY Target', 'SPY Range', 'Paths', 'P(AAPL > MSFT)', 'Avg Gap ($B)']
    
    for result in valid_results:
        row = [
            f"{result['SPY_Target']:.1%}", 
            result['SPY_Range'], 
            str(result['Paths']),
            f"{result['P_AAPL_gt_MSFT']:.2%}",
            f"{result['Avg_ValGap']:.2f}"
        ]
        display_data.append(row)
    
    table = plt.table(
        cellText=display_data,
        colLabels=col_names,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title("Scenario Analysis Summary", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario_summary_table.png", dpi=300)
    if show_plot:
        plt.show()


def create_valuation_gap_scatter_plot(spy_returns, valuation_gaps, output_dir, show_plot: bool = False):
    """
    Create a scatter plot of valuation gap against SPY returns
    
    Args:
        spy_returns: Array of SPY returns
        valuation_gaps: Array of valuation gaps
        output_dir: Directory to save output
        show_plot: Whether to display the plot
    """
    # Calculate a linear regression fit
    slope, intercept = np.polyfit(spy_returns, valuation_gaps, 1)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(spy_returns * 100, valuation_gaps, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', label='Break-even Line')
    
    # Add regression line
    x_range = np.linspace(min(spy_returns), max(spy_returns), 100)
    plt.plot(x_range * 100, slope * x_range + intercept, 'g-', 
             label=f'Fit: Gap = {slope:.2f}B × SPY% + {intercept:.2f}B')
    
    plt.xlabel('SPY Return (%)', fontsize=12)
    plt.ylabel('Valuation Gap ($ Billions)', fontsize=12)
    plt.title('Relationship Between SPY Returns and Valuation Gap', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spy_vs_valuation_gap.png", dpi=300)
    if show_plot:
        plt.show()
        
    # Print relationship
    print(f"\nRelationship between SPY return and valuation gap:")
    print(f"For each 1% change in SPY return, the valuation gap changes by ${slope:.2f}B")


def create_detailed_scenario_grid(delta_matrix, outcome_matrix, msft_scenarios, aapl_scenarios, output_dir):
    """
    Create detailed scenario grid visualizations
    
    Args:
        delta_matrix: 2D array of valuation deltas for each scenario
        outcome_matrix: 2D array of outcomes (1 for MSFT > AAPL, 0 for AAPL > MSFT)
        msft_scenarios: List of MSFT return scenarios
        aapl_scenarios: List of AAPL return scenarios
        output_dir: Directory to save output
    """
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
    plt.savefig(f"{output_dir}/detailed_scenario_grid.png", dpi=300)


def create_zoomed_scenario_grid(delta_matrix, msft_scenarios, aapl_scenarios, output_dir, 
                                zoom_min=-0.05, zoom_max=0.05):
    """
    Create a zoomed-in scenario grid focused on the most relevant return range
    
    Args:
        delta_matrix: 2D array of valuation deltas for each scenario
        msft_scenarios: List of MSFT return scenarios
        aapl_scenarios: List of AAPL return scenarios
        output_dir: Directory to save output
        zoom_min: Minimum return to include in zoom
        zoom_max: Maximum return to include in zoom
    """
    plt.figure(figsize=(15, 15))
    
    # Find indices for the zoom region
    msft_min_idx = next((i for i, x in enumerate(msft_scenarios) if x >= zoom_min), 0)
    msft_max_idx = next((i for i, x in enumerate(msft_scenarios) if x > zoom_max), len(msft_scenarios)) - 1
    aapl_min_idx = next((i for i, x in enumerate(aapl_scenarios) if x >= zoom_min), 0)
    aapl_max_idx = next((i for i, x in enumerate(aapl_scenarios) if x > zoom_max), len(aapl_scenarios)) - 1
    
    # Extract the zoom region
    zoom_delta = delta_matrix[aapl_min_idx:aapl_max_idx+1, msft_min_idx:msft_max_idx+1]
    zoom_msft = msft_scenarios[msft_min_idx:msft_max_idx+1]
    zoom_aapl = aapl_scenarios[aapl_min_idx:aapl_max_idx+1]
    
    # Handle NaN values
    zoom_delta_clean = np.nan_to_num(zoom_delta, nan=0.0)
    
    # Calculate bounds while avoiding division by zero
    if np.any(zoom_delta_clean != 0):
        vmin = -max(abs(np.nanmin(zoom_delta_clean)), abs(np.nanmax(zoom_delta_clean)))
        vmax = max(abs(np.nanmin(zoom_delta_clean)), abs(np.nanmax(zoom_delta_clean)))
    else:
        vmin = -1
        vmax = 1
    
    # Create the zoomed heatmap
    im = plt.imshow(
        zoom_delta_clean, 
        cmap='RdYlGn', 
        interpolation='none',
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
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
        zoom_delta_clean,
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
    
    # Add colorbar with explicit reference to the image
    cbar = plt.colorbar(im)
    cbar.set_label('$ Billions', rotation=270, labelpad=20, fontsize=12)
    
    # Add annotations for key points
    for i, msft_ret in enumerate(zoom_msft):
        for j, aapl_ret in enumerate(zoom_aapl):
            # Only annotate select points to avoid overcrowding
            if i % 2 == 0 and j % 2 == 0:
                value = zoom_delta_clean[j, i]
                if not np.isnan(value):
                    plt.text(
                        msft_ret*100, 
                        aapl_ret*100, 
                        f"{value:.1f}",
                        ha='center', 
                        va='center', 
                        fontsize=7,
                        color='black' if abs(value) < vmax/2 else 'white'
                    )
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zoomed_scenario_grid.png", dpi=300)


def create_scenario_surface_3d(delta_matrix, msft_scenarios, aapl_scenarios, output_dir):
    """
    Create an interactive 3D surface visualization of the scenario deltas
    
    Args:
        delta_matrix: 2D array of valuation deltas for each scenario
        msft_scenarios: List of MSFT return scenarios
        aapl_scenarios: List of AAPL return scenarios
        output_dir: Directory to save output
    """
    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly is not available, skipping 3D scenario surface")
        return
        
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
        surface_fig.write_html(f"{output_dir}/scenario_surface_3d.html")
    except Exception as e:
        print(f"Warning: Could not create 3D scenario surface: {e}")


def create_final_summary_table(results, output_dir):
    """
    Create a final summary table with key results
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save output
    """
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
    plt.savefig(f"{output_dir}/summary_table.png", dpi=300)