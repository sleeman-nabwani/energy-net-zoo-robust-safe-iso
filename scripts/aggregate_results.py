#!/usr/bin/env python3
"""
Aggregate evaluation results from baseline runs and generate summary plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def load_all_results(runs_dir="runs/baseline_static"):
    """Load all stress_eval.csv files from baseline runs."""
    results = []
    runs_path = Path(runs_dir)
    
    for csv_file in runs_path.rglob("stress_eval.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Add run path info
            df['run_path'] = str(csv_file.parent.parent)
            results.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not results:
        print(f"No evaluation results found in {runs_dir}")
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)

def analyze_results(df):
    """Generate summary statistics and plots."""
    if df.empty:
        print("No data to analyze")
        return
    
    print("=== BASELINE RESULTS SUMMARY ===")
    print(f"Total runs evaluated: {len(df)}")
    print(f"Algorithms: {df['algo'].unique()}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print()
    
    # Summary by algorithm and mode
    summary = df.groupby(['algo', 'mode']).agg({
        'return_mean': ['mean', 'std', 'count'],
        'avg_step_cost': ['mean', 'std'],
        'pass_cost_limit': 'mean'
    }).round(3)
    
    print("=== PERFORMANCE BY ALGORITHM AND MODE ===")
    print(summary)
    print()
    
    # Create plots
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SafeISO Baseline Results', fontsize=16)
    
    # Return vs Algorithm
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='algo', y='return_mean', hue='mode', ax=ax1)
    ax1.set_title('Return by Algorithm')
    ax1.tick_params(axis='x', rotation=45)
    
    # Cost vs Algorithm  
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='algo', y='avg_step_cost', hue='mode', ax=ax2)
    ax2.set_title('Average Step Cost by Algorithm')
    ax2.tick_params(axis='x', rotation=45)
    
    # Cost Limit Compliance
    ax3 = axes[1, 0]
    compliance = df.groupby(['algo', 'mode'])['pass_cost_limit'].mean().reset_index()
    sns.barplot(data=compliance, x='algo', y='pass_cost_limit', hue='mode', ax=ax3)
    ax3.set_title('Cost Limit Compliance Rate')
    ax3.set_ylabel('Fraction Passing Cost Limit')
    ax3.tick_params(axis='x', rotation=45)
    
    # Violation Types
    ax4 = axes[1, 1]
    violation_cols = ['shortfall_steps', 'freq_oob_steps', 'reserve_violation_steps']
    violation_data = df[violation_cols].mean()
    ax4.bar(range(len(violation_data)), violation_data.values)
    ax4.set_xticks(range(len(violation_data)))
    ax4.set_xticklabels(['Shortfall', 'Freq OOB', 'Reserve'], rotation=45)
    ax4.set_title('Average Violation Steps')
    ax4.set_ylabel('Steps per Episode')
    
    plt.tight_layout()
    plt.savefig('baseline_results.png', dpi=150, bbox_inches='tight')
    print(f"Plots saved to baseline_results.png")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline evaluation results")
    parser.add_argument("--runs_dir", default="runs/baseline_static", 
                       help="Directory containing baseline runs")
    parser.add_argument("--output", default="baseline_summary.csv",
                       help="Output CSV file for aggregated results")
    
    args = parser.parse_args()
    
    # Load and analyze results
    df = load_all_results(args.runs_dir)
    if not df.empty:
        summary = analyze_results(df)
        
        # Save aggregated results
        df.to_csv(args.output, index=False)
        print(f"Full results saved to {args.output}")
        
        # Save summary
        if summary is not None:
            summary.to_csv(args.output.replace('.csv', '_summary.csv'))
            print(f"Summary saved to {args.output.replace('.csv', '_summary.csv')}")

if __name__ == "__main__":
    main()
