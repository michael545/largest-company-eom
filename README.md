# MSFT vs AAPL Market Cap Analysis

This project performs Monte Carlo simulations to model the 30-day market cap difference trajectory between Microsoft (MSFT) and Apple (AAPL). It uses both Geometric Brownian Motion (GBM) and GARCH models to estimate probabilities and visualize potential outcomes.

## Project Overview

The analysis focuses on predicting the probability that Microsoft's market capitalization will exceed Apple's within a 30-day horizon. It employs:

- **Geometric Brownian Motion (GBM)**: A baseline simulation assuming constant volatility
- **GARCH Model**: More sophisticated modeling of time-varying volatility and fat-tailed distributions
- **Scenario Analysis**: Detailed exploration of market cap delta across different return scenarios
- **Detailed Visualizations**: Confidence intervals, probability projections, and 3D surface plots

## Repository Structure

```
├── advanced_monte_carlo_analysis.py  # Main script for GBM and GARCH comparisons
├── valuation_delta_analysis.py       # Extended valuation gap analysis
├── monte_carlo_simulation.py         # Basic Monte Carlo implementation
├── get_historic_prices.py            # Utility for fetching market data
├── models/                           # Core functionality modules
│   ├── analysis.py                   # Statistical analysis functions
│   ├── data_loader.py                # Data handling and processing
│   ├── simulation.py                 # GBM and GARCH model implementations
│   ├── valuation.py                  # Market cap calculations
│   └── visualization.py              # Chart and plot generation
├── data/                             # Historical price data
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── SPY.csv
└── analysis_output/                  # generated charts and stats
```

## Features

- **Dual Model Approach**: Compares GBM with more advanced GARCH modeling
- **High-Resolution Scenario Grid**: Analyzes market cap delta across 1% return increments
- **Interactive Visualizations**: 3D surfaces and contour plots for scenario exploration
- **Confidence Intervals**: Quantified uncertainty in market cap projections
- **Decision Boundaries**: Identifies critical return combinations for valuation crossover

## Key Visualizations

1. **GBM vs GARCH Comparison**: Visual comparison of both models' projections
2. **Probability Trajectory**: Time series showing probability of MSFT > AAPL
3. **Detailed Scenario Grid**: Heatmap showing market cap delta across return scenarios
4. **Zoomed Decision Boundary**: Close-up of the critical region around current valuation
5. **3D Surface Plots**: Interactive visualization of probabilities across scenarios

## Dependencies

This project requires Python 3.8+ and the following packages:
- numpy
- pandas
- matplotlib
- seaborn
- plotly
- arch
- yfinance

See `requirements.txt` for specific version requirements.

## Results

The analysis produces a comprehensive assessment of the probability that Microsoft's market capitalization will exceed Apple's over a 30-day horizon. Key findings are visualized in the `analysis_output/` directory, including:

- Terminal probability distributions
- Confidence intervals for market cap delta
- Decision boundary visualization
- Detailed scenario grid analysis

## License

MIT License