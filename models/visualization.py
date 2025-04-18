
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
PLOTLY_AVAILABLE = True

def create_scenario_grid_plot(
    valid_scenarios: List[int],
    spy_scenarios: np.ndarray,
    spy_returns: np.ndarray,
    scenario_tolerance: float,
    market_caps: np.ndarray,
    prices: np.ndarray,
    ticker_indices: Dict[str, int],
    valuation_gaps: np.ndarray,
    output_dir: str,
    show_plot: bool = False
) -> None:
    """
    Create grid of scenario plots for valid scenarios
    
    Args:
        valid_scenarios: Indices of valid scenarios
        spy_scenarios: Array of SPY return scenarios
        spy_returns: Array of SPY returns from simulation
        scenario_tolerance: Tolerance around target SPY return
        market_caps: Array of market caps from simulation
        prices: Array of prices from simulation
        ticker_indices: Dictionary mapping ticker to index
        valuation_gaps: Array of valuation gaps
        output_dir: Directory to save output files
        show_plot: Whether to display the plot in a popup window
    """
    # Set style
    sns.set(style="whitegrid")
    
    # Determine grid size - use max 4 columns
    num_scenarios = len(valid_scenarios)
    if num_scenarios <= 4:
        rows, cols = 1, num_scenarios
    else:
        cols = 4
        rows = (num_scenarios + cols - 1) // cols
    
    # Select 15 representative scenarios at most to avoid overcrowding
    if num_scenarios > 15:
        # Choose scenarios spaced evenly
        step = num_scenarios // 15
        valid_scenarios = valid_scenarios[::step]
        num_scenarios = len(valid_scenarios)
        # Recalculate grid
        if num_scenarios <= 4:
            rows, cols = 1, num_scenarios
        else:
            cols = 4
            rows = (num_scenarios + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    
    # If we have only one scenario, axes is not a 2D array
    if num_scenarios == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
        
    # ticker indices
    msft_idx = ticker_indices['MSFT']
    aapl_idx = ticker_indices['AAPL']
    
    # Plot each scenario
    for i, scenario_idx in enumerate(valid_scenarios):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        target_return = spy_scenarios[scenario_idx]
        lower_bound = target_return - scenario_tolerance
        upper_bound = target_return + scenario_tolerance
        
        # Create mask for paths where SPY return is within tolerance
        mask = (spy_returns >= lower_bound) & (spy_returns <= upper_bound)
        
        # Calculate probability of AAPL having higher valuation than MSFT
        p_aapl_gt_msft = np.mean(valuation_gaps[mask] < 0)
        
        # Calculate average valuation gap
        avg_val_gap = np.mean(valuation_gaps[mask])
        
        # Extract returns for each company
        msft_rel_perf = prices[msft_idx, -1, mask] / prices[msft_idx, 0, mask] - 1
        aapl_rel_perf = prices[aapl_idx, -1, mask] / prices[aapl_idx, 0, mask] - 1
        
        sns.histplot(msft_rel_perf, color='blue', alpha=0.6, label='MSFT', ax=ax, kde=True)
        sns.histplot(aapl_rel_perf, color='green', alpha=0.6, label='AAPL', ax=ax, kde=True)
        ax.axvline(x=np.mean(msft_rel_perf), color='blue', linestyle='--')
        ax.axvline(x=np.mean(aapl_rel_perf), color='green', linestyle='--')
        
        # Format plot
        ax.set_title(f"SPY {target_return*100:+.1f}%")
        ax.set_xlabel("Return")
        
        # Add stats in bottom left
        stats_text = (f"P(AAPL>MSFT): {p_aapl_gt_msft:.2f}\n"
                    f"Avg Gap: ${avg_val_gap:.1f}B\n"
                    f"MSFT: {np.mean(msft_rel_perf)*100:.1f}%\n"
                    f"AAPL: {np.mean(aapl_rel_perf)*100:.1f}%")
                    
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused axes
    for i in range(len(valid_scenarios), rows*cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    # Add a common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("MSFT vs AAPL Returns Under Different SPY Scenarios", fontsize=16)
    plt.subplots_adjust(bottom=0.1)
    
    # Save the figure
    plt.savefig(f"{output_dir}/scenario_grid.png", dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_3d_probability_plot(
    spy_returns: np.ndarray,
    valuation_gaps: np.ndarray,
    output_dir: str,
    show_plot: bool = False
) -> None:
    """
    Create 3D probability surface plot
    
    Args:
        spy_returns: Array of SPY returns
        valuation_gaps: Array of valuation gaps
        output_dir: Directory to save output files
        show_plot: Whether to display plot in popup window
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping 3D probability plot")
        return
        
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        spy_returns, 
        valuation_gaps,
        bins=[50, 50],
        density=True
    )
    
    # Calculate bin centers
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    
    # Create meshgrid for surface plot
    X, Y = np.meshgrid(x_centers, y_centers)
    Z = H.T  # Transpose H to match meshgrid orientation
    
    # Create plotly figure
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    
    # Calculate probability of AAPL > MSFT based on SPY return
    spy_bins = np.linspace(np.min(spy_returns), np.max(spy_returns), 20)
    prob_aapl_gt_msft = []
    
    for i in range(len(spy_bins)-1):
        lower, upper = spy_bins[i], spy_bins[i+1]
        mask = (spy_returns >= lower) & (spy_returns <= upper)
        if np.sum(mask) > 10:  # Require at least 10 samples
            prob = np.mean(valuation_gaps[mask] < 0)
            prob_aapl_gt_msft.append((np.mean([lower, upper]), prob))
    
    # Update layout WITHOUT the problematic 3D annotations
    fig.update_layout(
        title='Joint Probability Distribution: SPY Returns vs Valuation Gap',
        scene=dict(
            xaxis_title='SPY Return',
            yaxis_title='MSFT-AAPL Valuation Gap ($B)',
            zaxis_title='Probability Density'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1)
        )
    )
    
    # Save as HTML file
    fig.write_html(f"{output_dir}/3d_probability_surface.html")
    
    # Create 2D contour plot
    contour_fig = go.Figure(data=
        go.Contour(
            z=Z,
            x=x_centers, 
            y=y_centers,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        )
    )
    
    # Add scatter for probability by SPY return
    if prob_aapl_gt_msft:
        x_vals, probs = zip(*prob_aapl_gt_msft)
        contour_fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[0 for _ in x_vals],  # Place at y=0
                text=[f"{p:.2f}" for p in probs],
                mode="markers+text",
                name="P(AAPL>MSFT)",
                marker=dict(color="red", size=8),
                textposition="top center"
            )
        )
    
    # Update layout for contour plot
    contour_fig.update_layout(
        title='Probability Contours: SPY Returns vs Valuation Gap',
        xaxis_title='SPY Return',
        yaxis_title='MSFT-AAPL Valuation Gap ($B)',
        yaxis=dict(zeroline=True, zerolinecolor='red', zerolinewidth=2),
        xaxis=dict(zeroline=True, zerolinecolor='red', zerolinewidth=2)
    )
    
    # Save contour plot
    contour_fig.write_html(f"{output_dir}/probability_contour.html")
    
    # Show plots if requested
    if show_plot:
        fig.show()
        contour_fig.show()


def create_regime_correlation_plots(
    spy_returns: np.ndarray,
    msft_returns: np.ndarray,
    aapl_returns: np.ndarray,
    min_spy_return: float,
    max_spy_return: float,
    output_dir: str,
    show_plot: bool = False
) -> None:
    """
    Create correlation plots under different market regimes
    
    Args:
        spy_returns: Array of SPY returns
        msft_returns: Array of MSFT returns
        aapl_returns: Array of AAPL returns
        min_spy_return: Minimum SPY return
        max_spy_return: Maximum SPY return
        output_dir: Directory to save output files
        show_plot: Whether to display plot in popup window
    """
    # Create regime bins
    thresholds = np.linspace(min_spy_return, max_spy_return, 7)  # 6 regimes
    regime_names = [f"{thresholds[i]:.1%} to {thresholds[i+1]:.1%}" 
                   for i in range(len(thresholds)-1)]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(len(regime_names)):
        ax = axes[i]
        mask = (spy_returns >= thresholds[i]) & (spy_returns < thresholds[i+1])
        
        if np.sum(mask) < 50:  # Need enough points for meaningful correlation
            ax.text(0.5, 0.5, f"Insufficient data\n({np.sum(mask)} points)", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(regime_names[i])
            continue
        
        # Calculate correlation
        msft_aapl_corr = np.corrcoef(msft_returns[mask], aapl_returns[mask])[0, 1]
        
        # Create scatter
        ax.scatter(msft_returns[mask], aapl_returns[mask], alpha=0.5)
        ax.axline([0, 0], [1, 1], color='red', linestyle='--')
        
        # Add best fit line
        if np.sum(mask) > 2:  # Need at least 3 points for regression
            m, b = np.polyfit(msft_returns[mask], aapl_returns[mask], 1)
            x_line = np.array([np.min(msft_returns[mask]), np.max(msft_returns[mask])])
            y_line = m * x_line + b
            ax.plot(x_line, y_line, color='green')
        
        ax.set_title(f"{regime_names[i]}\nCorr: {msft_aapl_corr:.2f}")
        ax.set_xlabel('MSFT Return')
        ax.set_ylabel('AAPL Return')
        
        # Add point count
        ax.text(0.05, 0.95, f"n = {np.sum(mask)}", transform=ax.transAxes, 
               verticalalignment='top')
    
    plt.tight_layout()
    plt.suptitle("MSFT-AAPL Return Correlation by SPY Regime", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(f"{output_dir}/regime_correlations.png", dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_summary_table(valid_results, output_dir):
    """Create a summary table of scenario results"""
    plt.figure(figsize=(12, len(valid_results) * 0.5 + 1))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    headers = ["SPY Return", "Probability AAPL > MSFT", "Avg Gap ($B)", "# Paths"]
    table_data = []
    
    for result in valid_results:
        spy_return = f"{result['SPY_Target']:.2%}"
        prob = f"{result['P_AAPL_GT_MSFT']:.4f}"
        gap = f"${result['Avg_ValGap']:.2f}B"
        paths = f"{result['Paths']}"
        table_data.append([spy_return, prob, gap, paths])
    
    # Use named colors instead of rgba strings
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center',
                    colColours=['lightblue'] * 4, 
                    rowColours=['lightgreen'] * len(valid_results))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Summary of Scenario Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_table.png", bbox_inches='tight', dpi=300)
    plt.close()


def create_valuation_gap_scatter_plot(
    spy_returns: np.ndarray,
    valuation_gaps: np.ndarray,
    output_dir: str,
    show_plot: bool = False
) -> None:
    """
    Create scatter plot of SPY returns vs valuation gaps
    
    Args:
        spy_returns: Array of SPY returns
        valuation_gaps: Array of valuation gaps
        output_dir: Directory to save output files
        show_plot: Whether to show plot in popup window
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, creating Matplotlib version of valuation gap scatter")
        # Create with matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(spy_returns, valuation_gaps, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('SPY Return')
        plt.ylabel('MSFT-AAPL Valuation Gap ($B)')
        plt.title('SPY Return vs MSFT-AAPL Valuation Gap')
        
        # Add linear regression
        slope, intercept = np.polyfit(spy_returns, valuation_gaps, 1)
        x_line = np.linspace(np.min(spy_returns), np.max(spy_returns), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='green')
        
        # Add text with slope info
        plt.text(0.05, 0.95, f"Slope: ${slope/0.01:.2f}B per 1% SPY move", 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.savefig(f"{output_dir}/valuation_gap_scatter.png", dpi=300)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return
    
    # Use plotly for interactive version
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=spy_returns,
        y=valuation_gaps,
        mode='markers',
        marker=dict(
            size=6,
            opacity=0.6,
            colorscale='Viridis',
            color=valuation_gaps,
            colorbar=dict(title="Valuation Gap ($B)")
        ),
        name='Simulations'
    ))
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=np.min(spy_returns),
        y0=0,
        x1=np.max(spy_returns),
        y1=0,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add regression line
    slope, intercept = np.polyfit(spy_returns, valuation_gaps, 1)
    x_line = np.linspace(np.min(spy_returns), np.max(spy_returns), 100)
    y_line = slope * x_line + intercept
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color="green", width=3),
        name='Regression Line'
    ))
    
    # Create bins and calculate P(AAPL>MSFT) for each bin
    spy_bins = np.linspace(np.min(spy_returns), np.max(spy_returns), 15)
    bin_centers = []
    probs = []
    
    for i in range(len(spy_bins)-1):
        lower, upper = spy_bins[i], spy_bins[i+1]
        mask = (spy_returns >= lower) & (spy_returns <= upper)
        if np.sum(mask) > 20:  # Require at least 20 samples
            prob = np.mean(valuation_gaps[mask] < 0)
            bin_centers.append((lower + upper) / 2)
            probs.append(prob)
    
    # Add a trace for P(AAPL>MSFT)
    if bin_centers:
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=[-np.max(np.abs(valuation_gaps))*0.9] * len(bin_centers),  # Position at bottom
            text=[f"P(AAPL>MSFT)={p:.2f}" for p in probs],
            mode="markers+text",
            marker=dict(
                size=8,
                color=probs,
                colorscale="RdBu_r",
                cmin=0,
                cmax=1,
                cmid=0.5,
                line=dict(width=2, color="black")
            ),
            textposition="top center",
            name="P(AAPL>MSFT)",
            showlegend=False
        ))
    
    # Update layout
    spy_range = max(abs(np.min(spy_returns)), abs(np.max(spy_returns)))
    fig.update_layout(
        title=f"SPY Returns vs MSFT-AAPL Valuation Gap (${slope/0.01:.2f}B per 1% SPY)",
        xaxis_title="SPY Return",
        yaxis_title="MSFT-AAPL Valuation Gap ($B)",
        hovermode="closest",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            range=[-spy_range, spy_range]  # Symmetrical range
        ),
        yaxis=dict(
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1
        ),
        annotations=[
            dict(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Slope: ${slope/0.01:.2f}B per 1% SPY",
                showarrow=False,
                bgcolor="white",
                borderpad=4
            )
        ]
    )
    
    # Save as HTML
    fig.write_html(f"{output_dir}/valuation_gap_contour.html")
    
    # Show plot if requested
    if show_plot:
        fig.show()