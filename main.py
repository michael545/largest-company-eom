#!/usr/bin/env python
"""
AAPL vs MSFT Market Cap Analysis
===============================

Entry point for running Monte Carlo simulations to analyze the market cap 
delta between Microsoft and Apple over a 30-day period.
"""
import argparse
import os
from models.analysis import run_advanced_analysis
from models.utils import ensure_output_dir

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Analysis for MSFT vs AAPL Market Cap Delta"
    )
    
    parser.add_argument(
        "-o", "--output-dir", 
        default="analysis_output",
        help="Directory to save output files (default: analysis_output)"
    )
    
    parser.add_argument(
        "-s", "--simulations", 
        type=int, 
        default=10000,
        help="Number of Monte Carlo simulations to run (default: 10000)"
    )
    
    parser.add_argument(
        "-d", "--days", 
        type=int, 
        default=30,
        help="Number of trading days to simulate (default: 30)"
    )
    
    parser.add_argument(
        "-p", "--display-paths", 
        type=int, 
        default=200,
        help="Number of sample paths to display in visualizations (default: 200)"
    )
    
    parser.add_argument(
        "-c", "--confidence", 
        type=float, 
        default=0.9,
        help="Confidence level for intervals (default: 0.9)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    ensure_output_dir(args.output_dir)
    
    print("=" * 80)
    print("MSFT vs AAPL Market Cap Analysis")
    print("=" * 80)
    print(f"Running {args.simulations:,} simulations for {args.days} trading days")
    print(f"Output directory: {args.output_dir}")
    print("-" * 80)

    run_advanced_analysis(
        days=args.days,
        simulations=args.simulations,
        display_paths=args.display_paths,
        confidence_level=args.confidence,
        random_seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()