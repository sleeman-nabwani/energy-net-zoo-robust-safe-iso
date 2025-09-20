#!/usr/bin/env python3
"""
Unified OmniSafe Analysis Tool for SafeISO

This single script handles all analysis needs:
1. Collects results from training (runs/omnisafe_trained/)
2. Runs evaluation using existing suite_eval.py
3. Generates comprehensive plots and reports
4. Calculates metrics with confidence intervals

Works with existing infrastructure:
- Training: slurm/train_omnisafe_array.sbatch
- Evaluation: safeiso/eval/suite_eval.py
- Metrics: safeiso/eval/metrics.py
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import subprocess
import csv
import math

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class OmniSafeAnalysis:
    """Unified analysis tool for OmniSafe benchmarks."""
    
    def __init__(self, 
                 results_dir: str = "runs",
                 output_dir: str = None,  # Will auto-detect based on batch
                 algorithms: List[str] = None,
                 pcs_modes: List[str] = None,
                 suites: List[str] = None,
                 force_reeval: bool = True,  # Always force fresh evaluations by default
                 enhanced_analysis: bool = False,
                 episodes: int = 10,
                 per_pcs: bool = False):
        
        self.results_dir = Path(results_dir)
        
        # Auto-detect batch and create corresponding output directory
        if output_dir is None:
            batch_id = self._detect_batch_id()
            if batch_id:
                self.output_dir = Path(f"reports/omnisafe_batch_{batch_id}")
            else:
                self.output_dir = Path("reports/omnisafe")
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.algorithms = algorithms or ['PPOLag', 'CPO', 'CUP', 'SautePPO', 'FOCOPS']
        self.pcs_modes = pcs_modes  # Will auto-detect if None
        self.suites = suites or ['standard', 'stress']
        self.force_reeval = force_reeval  # Store force re-evaluation setting
        self.enhanced_analysis = enhanced_analysis
        self.episodes = episodes
        self.per_pcs = per_pcs
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
        
        print(f"Results will be saved to: {self.output_dir}")
    
    def _detect_batch_id(self) -> str:
        """Auto-detect the batch ID from the results directory structure."""
        try:
            # Check for new batch structure: omnisafe_trained_BATCH_ID
            batch_dirs = list(self.results_dir.glob("omnisafe_trained_*"))
            if batch_dirs:
                # Sort by modification time, get the most recent
                batch_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                most_recent = batch_dirs[0]
                batch_id = most_recent.name.replace("omnisafe_trained_", "")
                print(f"Auto-detected batch: {batch_id} from {most_recent.name}")
                return batch_id
            
            # Check for legacy structure: omnisafe_trained (no batch ID)
            legacy_dir = self.results_dir / "omnisafe_trained"
            if legacy_dir.exists():
                print("Using legacy structure (no batch ID)")
                return "legacy"
                
            print("No trained models directory found")
            return None
            
        except Exception as e:
            print(f"Error detecting batch ID: {e}")
            return None
        
    def check_trained_models(self) -> pd.DataFrame:
        """Check which models have been trained."""
        trained = []
        
        # Check if results_dir is already a batch directory
        if self.results_dir.name.startswith("omnisafe_trained_"):
            base_path = self.results_dir
            print(f"Using direct batch directory: {base_path.name}")
        else:
            # Auto-detect the most recent batch directory
            batch_dirs = list(self.results_dir.glob("omnisafe_trained_*"))
            if batch_dirs:
                # Sort by modification time, get the most recent
                batch_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                base_path = batch_dirs[0]
                print(f"Using batch directory: {base_path.name}")
            elif (self.results_dir / "omnisafe_trained").exists():
                # Legacy structure: runs/omnisafe_trained/
                base_path = self.results_dir / "omnisafe_trained"
                print("Using legacy structure")
            else:
                print("❌ No trained models directory found")
                return pd.DataFrame()
        
        # Auto-detect PCS modes if not specified
        if self.pcs_modes is None:
            detected_modes = set()
            for algo in self.algorithms:
                algo_path = base_path / algo
                if algo_path.exists():
                    for pcs_dir in algo_path.iterdir():
                        if pcs_dir.is_dir() and not pcs_dir.name.startswith('.'):
                            # CRITICAL FIX: Exclude ISO models from PCS mode detection
                            # ISO models control the ISO directly and should not be considered as PCS modes
                            if 'iso' in pcs_dir.name.lower():
                                print(f"[skip] Excluding ISO model directory: {pcs_dir.name}")
                                continue
                            detected_modes.add(pcs_dir.name)
            self.pcs_modes = sorted(detected_modes)
            print(f"Auto-detected PCS modes (excluding ISO): {self.pcs_modes}")
        
        for algo in self.algorithms:
            for pcs in self.pcs_modes:
                for seed_dir in (base_path / algo / pcs).glob("seed_*"):
                    if seed_dir.is_dir():
                        # Check for evalpack or SB3 model
                        evalpack = seed_dir / "evalpack" / "actor.ts"
                        sb3_model = seed_dir / "best_model.zip"
                        
                        if evalpack.exists() or sb3_model.exists():
                            seed = int(seed_dir.name.split("_")[-1])
                            trained.append({
                                'algorithm': algo,
                                'pcs_mode': pcs,
                                'seed': seed,
                                'path': str(seed_dir),
                                'has_evalpack': evalpack.exists(),
                                'has_sb3': sb3_model.exists()
                            })
        
        return pd.DataFrame(trained)
    
    def run_evaluation(self, model_path: str, suite: str, episodes: int = None) -> Dict[str, Any]:
        """Run evaluation using the existing suite_eval.py."""
        if episodes is None:
            episodes = self.episodes
        suite_path = f"safeiso/eval/suites/{suite}.yaml"
        
        # Extract algo/pcs from path
        parts = Path(model_path).parts
        algo = parts[-3] if len(parts) > 3 else 'unknown'
        pcs = parts[-2] if len(parts) > 2 else 'unknown'
        seed = parts[-1].split('_')[-1] if len(parts) > 1 else '0'
        
        # Output paths
        eval_dir = Path(model_path) / "eval" / suite
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        scenarios_csv = eval_dir / "scenarios.csv"
        episodes_csv = eval_dir / "episodes.csv"
        
        # Check if already evaluated (unless force re-evaluation is enabled)
        if not self.force_reeval and scenarios_csv.exists() and episodes_csv.exists():
            print(f"  ✓ Already evaluated: {model_path}/{suite}")
            return self._load_eval_results(scenarios_csv, episodes_csv)
        elif self.force_reeval and (scenarios_csv.exists() or episodes_csv.exists()):
            print(f"  🔄 Force re-evaluating: {model_path}/{suite}")
            # Remove existing cached files to ensure fresh evaluation
            scenarios_csv.unlink(missing_ok=True)
            episodes_csv.unlink(missing_ok=True)
        
        print(f"  → Running evaluation: {algo}/{pcs}/seed_{seed} on {suite}")
        
        # Run evaluation using existing suite_eval
        cmd = [
            "python3", "-m", "safeiso.eval.suite_eval",
            "--suite", suite_path,
            "--policy", model_path,
            "--episodes", str(episodes),
            "--horizon", "48",
            "--seed", seed,
            "--algo", algo,
            "--mode", "CMDP",
            "--device", "cuda",
            "--cost_limit", "0.25",
            "--out_scenarios", str(scenarios_csv),
            "--out_episodes", str(episodes_csv)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                return self._load_eval_results(scenarios_csv, episodes_csv)
            else:
                print(f"  ✗ Evaluation failed: {result.stderr}")
                return {}
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {}
    
    def _load_eval_results(self, scenarios_csv: Path, episodes_csv: Path) -> Dict[str, Any]:
        """Load evaluation results from CSV files."""
        try:
            scenarios_df = pd.read_csv(scenarios_csv)
            episodes_df = pd.read_csv(episodes_csv)
            
            return {
                'scenarios': len(scenarios_df),
                'episodes': len(episodes_df),
                'pass_rate': scenarios_df['pass_cost_limit'].mean() if 'pass_cost_limit' in scenarios_df else 0,
                'avg_cost': scenarios_df['avg_step_cost'].mean() if 'avg_step_cost' in scenarios_df else np.nan,
                'avg_reward': scenarios_df['return_mean'].mean() if 'return_mean' in scenarios_df else np.nan,
                'violations': {
                    'shortfall': scenarios_df['shortfall_steps'].sum() if 'shortfall_steps' in scenarios_df else 0,
                    'freq': scenarios_df['freq_oob_steps'].sum() if 'freq_oob_steps' in scenarios_df else 0,
                }
            }
        except Exception:
            return {}
    
    def collect_all_results(self) -> pd.DataFrame:
        """Collect all evaluation results at episode level."""
        print("\n📊 Collecting Results")
        print("=" * 60)
        
        # Check trained models
        trained_df = self.check_trained_models()
        print(f"Found {len(trained_df)} trained models")
        
        if trained_df.empty:
            print("❌ No trained models found!")
            return pd.DataFrame()
        
        # Run evaluations and collect episode-level data
        results = []
        for _, model in trained_df.iterrows():
            for suite in self.suites:
                # Run evaluation to ensure CSV files exist
                self.run_evaluation(model['path'], suite)
                
                # Load episode-level data directly from CSV
                eval_dir = Path(model['path']) / "eval" / suite
                episodes_csv = eval_dir / "episodes.csv"
                
                if episodes_csv.exists():
                    try:
                        episodes_df = pd.read_csv(episodes_csv)
                        
                        # Add metadata columns
                        episodes_df['algorithm'] = model['algorithm']
                        episodes_df['pcs_mode'] = model['pcs_mode']
                        episodes_df['model_seed'] = model['seed']
                        episodes_df['suite'] = suite
                        
                        # Rename columns to match expected names
                        column_mapping = {
                            'ep_cost_mean': 'avg_cost',
                            'ep_return': 'avg_reward',
                            'ep_cost_sum': 'total_cost'
                        }
                        episodes_df = episodes_df.rename(columns=column_mapping)
                        
                        # Add missing columns with defaults
                        if 'soc_oob_steps' not in episodes_df.columns:
                            episodes_df['soc_oob_steps'] = 0
                        if 'ep_length' not in episodes_df.columns:
                            episodes_df['ep_length'] = episodes_df.get('horizon', 48)
                        if 'total_steps' not in episodes_df.columns:
                            episodes_df['total_steps'] = episodes_df['ep_length']
                            
                        results.append(episodes_df)
                        
                    except Exception as e:
                        print(f"  ✗ Error loading {episodes_csv}: {e}")
        
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            print(f"Collected {len(combined_df)} episodes from {len(results)} evaluations")
            return combined_df
        else:
            return pd.DataFrame()
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated metrics with confidence intervals."""
        if df.empty:
            return pd.DataFrame()
        
        metrics = []
        
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            
            # Bootstrap confidence intervals
            pass_rates = algo_data['pass_rate'].values
            costs = algo_data['avg_cost'].dropna().values
            rewards = algo_data['avg_reward'].dropna().values
            
            metrics.append({
                'algorithm': algo,
                'n_runs': len(algo_data),
                'pass_rate_mean': pass_rates.mean(),
                'pass_rate_std': pass_rates.std(),
                'pass_rate_ci': self._bootstrap_ci(pass_rates),
                'cost_mean': costs.mean() if len(costs) > 0 else np.nan,
                'cost_std': costs.std() if len(costs) > 0 else np.nan,
                'cost_ci': self._bootstrap_ci(costs) if len(costs) > 0 else (np.nan, np.nan),
                'reward_mean': rewards.mean() if len(rewards) > 0 else np.nan,
                'reward_std': rewards.std() if len(rewards) > 0 else np.nan,
                'reward_ci': self._bootstrap_ci(rewards) if len(rewards) > 0 else (np.nan, np.nan),
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Calculate rankings
        if not metrics_df.empty:
            metrics_df['safety_score'] = (1 - metrics_df['cost_mean'] / metrics_df['cost_mean'].max())
            metrics_df['rank_safety'] = metrics_df['safety_score'].rank(ascending=False)
            metrics_df['rank_reward'] = metrics_df['reward_mean'].rank(ascending=False)
            metrics_df['rank_overall'] = (metrics_df['rank_safety'] + metrics_df['rank_reward']) / 2
            metrics_df = metrics_df.sort_values('rank_overall')
        
        return metrics_df
    
    def _bootstrap_ci(self, data: np.ndarray, n_bootstrap: int = 5000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if len(data) == 0:
            return (np.nan, np.nan)
        
        rng = np.random.default_rng(42)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_means.append(sample.mean())
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def _ci95_norm(self, x: pd.Series) -> pd.Series:
        """Calculate 95% confidence interval using normal approximation."""
        m = x.mean()
        s = x.std(ddof=1)
        n = x.count()
        ci = 1.96 * s / max(np.sqrt(n), 1.0)
        return pd.Series({'mean': m, 'ci95': ci, 'n': n})
    
    def compute_violation_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate violation metrics grouped by algorithm × PCS mode."""
        if df.empty:
            return pd.DataFrame()

        # Ensure total_steps column exists
        if 'total_steps' not in df.columns:
            df = df.assign(total_steps=df.get('ep_length', df.get('steps', 480)))
        
        # Group by algorithm and PCS mode and calculate metrics
        results = []
        for (algo, pcs), group in df.groupby(['algorithm', 'pcs_mode']):
            cost_stats = self._ci95_norm(group['avg_cost'])
            reward_stats = self._ci95_norm(group['avg_reward'])
            
            result = {
                'algorithm': algo,
                'pcs_mode': pcs,
                'avg_cost_mean': cost_stats['mean'],
                'avg_cost_ci95': cost_stats['ci95'],
                'avg_cost_n': cost_stats['n'],
                'avg_reward_mean': reward_stats['mean'],
                'avg_reward_ci95': reward_stats['ci95'],
                'avg_reward_n': reward_stats['n'],
                'shortfall_steps': group['shortfall_steps'].sum(),
                'freq_oob_steps': group['freq_oob_steps'].sum(),
                'soc_oob_steps': group['soc_oob_steps'].sum(),
                'total_steps': group['total_steps'].sum(),
            }
            results.append(result)
        
        grouped = pd.DataFrame(results)
        
        # Calculate violation rates
        grouped['shortfall_rate'] = grouped['shortfall_steps'] / grouped['total_steps']
        grouped['freq_rate'] = grouped['freq_oob_steps'] / grouped['total_steps']
        grouped['soc_rate'] = grouped['soc_oob_steps'] / grouped['total_steps']
        
        # Rank by cost (lower is better)
        grouped['rank_by_cost'] = grouped['avg_cost_mean'].rank(method='dense', ascending=True).astype(int)
        
        return grouped
    
    def compute_algo_comparison_table(self, metrics_df: pd.DataFrame, per_pcs: bool = False) -> pd.DataFrame:
        """Generate algorithm comparison table, optionally aggregated across PCS modes."""
        if metrics_df.empty:
            return pd.DataFrame()
            
        if per_pcs:
            table = metrics_df.copy()
        else:
            # Aggregate across PCS modes for headline table
            table = (metrics_df
                     .groupby('algorithm', as_index=False)
                     .agg({
                         'avg_cost_mean': 'mean',
                         'avg_cost_ci95': 'mean',
                         'avg_reward_mean': 'mean',
                         'avg_reward_ci95': 'mean',
                         'shortfall_rate': 'mean',
                         'freq_rate': 'mean',
                         'soc_rate': 'mean'
                     }))
            table['rank_by_cost'] = table['avg_cost_mean'].rank(method='dense', ascending=True).astype(int)
            
        return table.sort_values(['rank_by_cost', 'algorithm'])
    
    def plot_algo_pcs_heatmap(self, metrics_df: pd.DataFrame, outpath: Path):
        """Generate heatmap of average cost by algorithm × PCS mode."""
        if metrics_df.empty:
            return
            
        pivot = metrics_df.pivot(index='algorithm', columns='pcs_mode', values='avg_cost_mean')
        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                        linewidths=.5, cbar_kws={'label': 'Avg Cost'})
        ax.set_title('Average Cost by Algorithm × PCS Mode')
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    
    def plot_cost_reward_scatter(self, metrics_df: pd.DataFrame, outpath: Path):
        """Generate cost vs reward scatter plot with PCS mode annotations."""
        if metrics_df.empty:
            return
            
        plt.figure(figsize=(8, 6))
        for algo, g in metrics_df.groupby('algorithm'):
            plt.scatter(g['avg_cost_mean'], g['avg_reward_mean'], label=algo, s=60)
            for _, r in g.iterrows():
                plt.annotate(r['pcs_mode'], (r['avg_cost_mean'], r['avg_reward_mean']), 
                           fontsize=8, alpha=0.8)
        
        plt.xlabel('Cost (lower better)')
        plt.ylabel('Reward (higher better)')
        plt.title('Cost–Reward by Algorithm × PCS Mode')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    
    def plot_algo_pcs_line(self, metrics_df: pd.DataFrame, outpath: Path):
        """Generate line plot of cost across PCS modes for each algorithm."""
        if metrics_df.empty:
            return
            
        plt.figure(figsize=(11, 5))
        # Stable ordering for PCS modes
        order = sorted(metrics_df['pcs_mode'].unique())
        
        for algo, g in metrics_df.groupby('algorithm'):
            g = g.set_index('pcs_mode').loc[order].reset_index()
            plt.plot(g['pcs_mode'], g['avg_cost_mean'], marker='o', label=algo)
            plt.fill_between(g['pcs_mode'], 
                           g['avg_cost_mean'] - g['avg_cost_ci95'], 
                           g['avg_cost_mean'] + g['avg_cost_ci95'], 
                           alpha=0.15)
        
        plt.xticks(rotation=45)
        plt.ylabel('Avg Cost')
        plt.title('Cost across PCS Modes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    
    def plot_per_algo_pcs(self, metrics_df: pd.DataFrame, algo: str, outdir: Path):
        """Generate per-algorithm PCS analysis with 3-panel figure."""
        sub = metrics_df[metrics_df['algorithm'] == algo].copy()
        if sub.empty:
            return
            
        order = sorted(sub['pcs_mode'].unique())
        fig, axs = plt.subplots(1, 3, figsize=(17, 5))
        
        # Panel 1: Cost
        costs = [sub[sub.pcs_mode == p]['avg_cost_mean'].values[0] for p in order]
        cost_cis = [sub[sub.pcs_mode == p]['avg_cost_ci95'].values[0] for p in order]
        axs[0].bar(order, costs, yerr=cost_cis, capsize=3)
        axs[0].set_title(f'{algo} — Avg Cost by PCS')
        axs[0].set_ylabel('Avg Cost')
        axs[0].tick_params(axis='x', rotation=45)
        
        # Panel 2: Reward
        rewards = [sub[sub.pcs_mode == p]['avg_reward_mean'].values[0] for p in order]
        reward_cis = [sub[sub.pcs_mode == p]['avg_reward_ci95'].values[0] for p in order]
        axs[1].bar(order, rewards, yerr=reward_cis, capsize=3)
        axs[1].set_title(f'{algo} — Avg Reward by PCS')
        axs[1].set_ylabel('Avg Reward')
        axs[1].tick_params(axis='x', rotation=45)
        
        # Panel 3: Violations stacked (%)
        s = [sub[sub.pcs_mode == p]['shortfall_rate'].values[0] for p in order]
        f = [sub[sub.pcs_mode == p]['freq_rate'].values[0] for p in order]
        o = [sub[sub.pcs_mode == p]['soc_rate'].values[0] for p in order]
        width = 0.6
        axs[2].bar(order, s, width, label='shortfall')
        axs[2].bar(order, f, width, bottom=s, label='freq')
        axs[2].bar(order, o, width, bottom=np.array(s) + np.array(f), label='soc')
        axs[2].set_title(f'{algo} — Violation Breakdown by PCS')
        axs[2].set_ylabel('Rate')
        axs[2].legend()
        axs[2].tick_params(axis='x', rotation=45)
        
        fig.suptitle(f'{algo} — PCS Analysis', y=1.03)
        plt.tight_layout()
        
        # Save both PNG and PDF
        for ext in ('png', 'pdf'):
            plt.savefig(outdir / f'{algo}_analysis.{ext}', dpi=200)
        plt.close()
    
    def generate_enhanced_composite(self, df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Generate enhanced composite plot with cost/reward/PCS focus."""
        print("📊 Generating Enhanced Composite Plot...")
        
        # Create a figure with multiple subplots (3x3 grid)
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Average Cost per Algorithm (replaces pass rate)
        ax1 = fig.add_subplot(gs[0, 0])
        if not metrics_df.empty:
            algo_costs = metrics_df.groupby('algorithm')['avg_cost_mean'].mean()
            algo_cost_cis = metrics_df.groupby('algorithm')['avg_cost_ci95'].mean()
            
            bars = ax1.bar(range(len(algo_costs)), algo_costs.values, 
                          yerr=algo_cost_cis.values, capsize=3)
            ax1.set_xticks(range(len(algo_costs)))
            ax1.set_xticklabels(algo_costs.index, rotation=45)
            ax1.set_ylabel('Average Cost')
            ax1.set_title('Average Cost by Algorithm')
            ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Target Limit')
            ax1.legend()
            
            # Add value labels
            for i, (v, ci) in enumerate(zip(algo_costs.values, algo_cost_cis.values)):
                ax1.text(i, v + ci + 0.01, f'{v:.3f}', ha='center', fontsize=9)
        
        # 2. Average Reward per Algorithm (replaces pass rate)
        ax2 = fig.add_subplot(gs[0, 1])
        if not metrics_df.empty:
            algo_rewards = metrics_df.groupby('algorithm')['avg_reward_mean'].mean()
            algo_reward_cis = metrics_df.groupby('algorithm')['avg_reward_ci95'].mean()
            
            bars = ax2.bar(range(len(algo_rewards)), algo_rewards.values,
                          yerr=algo_reward_cis.values, capsize=3)
            ax2.set_xticks(range(len(algo_rewards)))
            ax2.set_xticklabels(algo_rewards.index, rotation=45)
            ax2.set_ylabel('Average Reward')
            ax2.set_title('Average Reward by Algorithm')
            
            # Add value labels
            for i, (v, ci) in enumerate(zip(algo_rewards.values, algo_reward_cis.values)):
                ax2.text(i, v + ci + 0.5, f'{v:.1f}', ha='center', fontsize=9)
        
        # 3. Cost vs Reward Scatter with PCS annotations (replaces empty panel)
        ax3 = fig.add_subplot(gs[0, 2])
        if not metrics_df.empty:
            for algo, g in metrics_df.groupby('algorithm'):
                ax3.scatter(g['avg_cost_mean'], g['avg_reward_mean'], 
                           label=algo, s=80, alpha=0.7)
                for _, row in g.iterrows():
                    ax3.annotate(row['pcs_mode'], 
                               (row['avg_cost_mean'], row['avg_reward_mean']),
                               xytext=(3, 3), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
            
            ax3.set_xlabel('Average Cost')
            ax3.set_ylabel('Average Reward')
            ax3.set_title('Cost-Reward Trade-off by PCS')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # 4. Violations Analysis (keep existing)
        ax4 = fig.add_subplot(gs[1, 0])
        if not metrics_df.empty:
            vio_summary = metrics_df.groupby('algorithm')[['shortfall_rate', 'freq_rate', 'soc_rate']].mean()
            vio_summary.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_ylabel('Violation Rate')
            ax4.set_title('Violation Rates by Algorithm')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            ax4.legend(title='Violation Type')
        
        # 5. Performance by PCS Mode Heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        if not metrics_df.empty:
            pivot = metrics_df.pivot_table(values='avg_cost_mean', index='algorithm', 
                                          columns='pcs_mode', aggfunc='mean')
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                          ax=ax5, cbar_kws={'label': 'Avg Cost'})
                ax5.set_title('Cost by Algorithm × PCS Mode')
        
        # 6. Cost Distribution by Algorithm
        ax6 = fig.add_subplot(gs[1, 2])
        if not df.empty and 'avg_cost' in df.columns:
            algo_groups = [group['avg_cost'].values for name, group in df.groupby('algorithm')]
            algo_names = [name for name, _ in df.groupby('algorithm')]
            
            ax6.boxplot(algo_groups, labels=algo_names)
            ax6.set_ylabel('Episode Cost')
            ax6.set_title('Cost Distribution by Algorithm')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. PCS Mode Comparison (Line plot)
        ax7 = fig.add_subplot(gs[2, 0])
        if not metrics_df.empty:
            pcs_order = sorted(metrics_df['pcs_mode'].unique())
            for algo, g in metrics_df.groupby('algorithm'):
                g_ordered = g.set_index('pcs_mode').reindex(pcs_order).reset_index()
                ax7.plot(g_ordered['pcs_mode'], g_ordered['avg_cost_mean'], 
                        marker='o', label=algo)
            ax7.set_ylabel('Average Cost')
            ax7.set_title('Cost Across PCS Modes')
            ax7.tick_params(axis='x', rotation=45)
            ax7.legend()
        
        # 8. Ranking Summary
        ax8 = fig.add_subplot(gs[2, 1])
        if not metrics_df.empty:
            algo_ranks = metrics_df.groupby('algorithm')['rank_by_cost'].mean().sort_values()
            bars = ax8.barh(range(len(algo_ranks)), algo_ranks.values)
            ax8.set_yticks(range(len(algo_ranks)))
            ax8.set_yticklabels(algo_ranks.index)
            ax8.set_xlabel('Average Cost Rank')
            ax8.set_title('Algorithm Cost Rankings')
            
            # Color bars by rank
            for i, (bar, rank) in enumerate(zip(bars, algo_ranks.values)):
                if rank <= 2:
                    bar.set_color('green')
                elif rank <= 4:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # 9. Violation Breakdown by PCS
        ax9 = fig.add_subplot(gs[2, 2])
        if not metrics_df.empty:
            pcs_vio = metrics_df.groupby('pcs_mode')[['shortfall_rate', 'freq_rate', 'soc_rate']].mean()
            pcs_vio.plot(kind='bar', ax=ax9)
            ax9.set_ylabel('Violation Rate')
            ax9.set_title('Violations by PCS Mode')
            ax9.tick_params(axis='x', rotation=45)
            ax9.legend(title='Violation Type')
        
        plt.suptitle('OmniSafe Enhanced Analysis - Cost & PCS Focus', fontsize=18, y=1.02)
        
        # Save composite plot
        plt.savefig(self.plots_dir / 'composite_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'composite_overview.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved enhanced composite plot to {self.plots_dir}/")
    
    def generate_plots(self, df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Generate comprehensive plots."""
        print("\n📈 Generating Plots")
        print("=" * 60)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Pass Rate by Algorithm
        ax1 = fig.add_subplot(gs[0, 0])
        if 'pass_rate_mean' in metrics_df.columns:
            bars = ax1.bar(range(len(metrics_df)), metrics_df['pass_rate_mean'])
            ax1.set_xticks(range(len(metrics_df)))
            ax1.set_xticklabels(metrics_df['algorithm'], rotation=45)
            ax1.set_ylabel('Pass Rate')
            ax1.set_ylim(0, 1.1)
            ax1.set_title('Safety Constraint Satisfaction')
            ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% target')
            ax1.legend()
            
            # Add value labels
            for i, v in enumerate(metrics_df['pass_rate_mean']):
                ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=9)
        
        # 2. Cost vs Reward Scatter
        ax2 = fig.add_subplot(gs[0, 1])
        if 'cost_mean' in metrics_df.columns and 'reward_mean' in metrics_df.columns:
            scatter = ax2.scatter(metrics_df['cost_mean'], metrics_df['reward_mean'], 
                                 s=200, alpha=0.7, edgecolors='black', linewidth=2)
            
            # Add algorithm labels
            for _, row in metrics_df.iterrows():
                ax2.annotate(row['algorithm'], 
                           (row['cost_mean'], row['reward_mean']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Cost Limit')
            ax2.set_xlabel('Average Cost')
            ax2.set_ylabel('Average Reward')
            ax2.set_title('Cost-Reward Trade-off')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance by PCS Mode
        ax3 = fig.add_subplot(gs[0, 2])
        if not df.empty:
            pivot = df.pivot_table(values='pass_rate', index='algorithm', 
                                  columns='pcs_mode', aggfunc='mean')
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                          vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Pass Rate'})
                ax3.set_title('Performance by PCS Mode')
        
        # 4. Violations Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        if not df.empty and 'violations' in df.columns:
            violations_data = []
            for _, row in df.iterrows():
                if isinstance(row['violations'], dict):
                    violations_data.append({
                        'algorithm': row['algorithm'],
                        'shortfall': row['violations'].get('shortfall', 0),
                        'freq': row['violations'].get('freq', 0)
                    })
            
            if violations_data:
                vio_df = pd.DataFrame(violations_data)
                vio_summary = vio_df.groupby('algorithm')[['shortfall', 'freq']].sum()
                vio_summary.plot(kind='bar', stacked=True, ax=ax4)
                ax4.set_ylabel('Total Violations')
                ax4.set_title('Violation Types by Algorithm')
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
                ax4.legend(title='Violation Type')
        
        # 5. Performance by Suite
        ax5 = fig.add_subplot(gs[1, 1])
        if not df.empty:
            suite_perf = df.pivot_table(values='pass_rate', index='algorithm', 
                                       columns='suite', aggfunc='mean')
            if not suite_perf.empty:
                suite_perf.plot(kind='bar', ax=ax5)
                ax5.set_ylabel('Pass Rate')
                ax5.set_title('Performance Across Test Suites')
                ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
                ax5.legend(title='Suite')
                ax5.set_ylim(0, 1.1)
        
        # 6. Safety Score Ranking
        ax6 = fig.add_subplot(gs[1, 2])
        if 'safety_score' in metrics_df.columns:
            sorted_df = metrics_df.sort_values('safety_score', ascending=True)
            bars = ax6.barh(range(len(sorted_df)), sorted_df['safety_score'])
            ax6.set_yticks(range(len(sorted_df)))
            ax6.set_yticklabels(sorted_df['algorithm'])
            ax6.set_xlabel('Safety Score')
            ax6.set_title('Algorithm Safety Ranking')
            ax6.set_xlim(0, 1.1)
            
            # Color bars by score
            for bar, score in zip(bars, sorted_df['safety_score']):
                if score > 0.8:
                    bar.set_color('green')
                elif score > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # 7. Confidence Intervals
        ax7 = fig.add_subplot(gs[2, 0])
        if 'cost_ci' in metrics_df.columns:
            x = range(len(metrics_df))
            means = metrics_df['cost_mean'].values
            
            # Extract CI bounds
            ci_lower = [ci[0] for ci in metrics_df['cost_ci']]
            ci_upper = [ci[1] for ci in metrics_df['cost_ci']]
            
            # Calculate error bars
            yerr = [means - ci_lower, ci_upper - means]
            
            ax7.bar(x, means)
            ax7.errorbar(x, means, yerr=yerr, fmt='none', color='black', capsize=5)
            ax7.set_xticks(x)
            ax7.set_xticklabels(metrics_df['algorithm'], rotation=45)
            ax7.set_ylabel('Average Cost')
            ax7.set_title('Cost with 95% Confidence Intervals')
            ax7.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Cost Limit')
            ax7.legend()
        
        # 8. Pareto Frontier
        ax8 = fig.add_subplot(gs[2, 1])
        if 'cost_mean' in metrics_df.columns and 'reward_mean' in metrics_df.columns:
            # Plot all points
            ax8.scatter(metrics_df['cost_mean'], metrics_df['reward_mean'], 
                       s=100, alpha=0.5, label='Algorithms')
            
            # Find Pareto frontier
            points = metrics_df[['cost_mean', 'reward_mean']].values
            pareto_indices = []
            
            for i, point in enumerate(points):
                if np.isnan(point).any():
                    continue
                dominated = False
                for j, other in enumerate(points):
                    if i != j and not np.isnan(other).any():
                        # Check if other dominates point (lower cost AND higher reward)
                        if other[0] <= point[0] and other[1] >= point[1]:
                            if other[0] < point[0] or other[1] > point[1]:
                                dominated = True
                                break
                if not dominated:
                    pareto_indices.append(i)
            
            # Highlight Pareto optimal points
            if pareto_indices:
                pareto_data = metrics_df.iloc[pareto_indices]
                ax8.scatter(pareto_data['cost_mean'], pareto_data['reward_mean'],
                          s=200, color='red', marker='*', label='Pareto Optimal', zorder=5)
                
                # Add labels for Pareto points
                for _, row in pareto_data.iterrows():
                    ax8.annotate(row['algorithm'], 
                               (row['cost_mean'], row['reward_mean']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, fontweight='bold', color='red')
            
            ax8.set_xlabel('Cost (lower is better)')
            ax8.set_ylabel('Reward (higher is better)')
            ax8.set_title('Pareto Frontier Analysis')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Overall Rankings
        ax9 = fig.add_subplot(gs[2, 2])
        if 'rank_overall' in metrics_df.columns:
            sorted_df = metrics_df.sort_values('rank_overall')
            y_pos = np.arange(len(sorted_df))
            bars = ax9.barh(y_pos, sorted_df['rank_overall'])
            
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels(sorted_df['algorithm'])
            ax9.set_xlabel('Overall Rank (lower is better)')
            ax9.set_title('Final Algorithm Rankings')
            ax9.invert_xaxis()  # Lower rank on the right
            
            # Color by rank
            for i, (bar, rank) in enumerate(zip(bars, sorted_df['rank_overall'])):
                if i == 0:
                    bar.set_color('gold')
                elif i == 1:
                    bar.set_color('silver')
                elif i == 2:
                    bar.set_color('#CD7F32')  # Bronze
                else:
                    bar.set_color('steelblue')
        
        plt.suptitle('OmniSafe Algorithm Benchmark Analysis', fontsize=18, y=1.02)
        plt.savefig(self.plots_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'comprehensive_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved plots to {self.plots_dir}/")
    
    def generate_report(self, df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Generate comprehensive report."""
        print("\n📝 Generating Report")
        print("=" * 60)
        
        # Save raw data
        df.to_csv(self.tables_dir / 'raw_results.csv', index=False)
        metrics_df.to_csv(self.tables_dir / 'metrics_summary.csv', index=False)
        
        # Generate LaTeX table
        if not metrics_df.empty:
            latex_columns = ['algorithm', 'pass_rate_mean', 'cost_mean', 'reward_mean', 'safety_score', 'rank_overall']
            latex_df = metrics_df[latex_columns].round(3)
            latex_df.columns = ['Algorithm', 'Pass Rate', 'Avg Cost', 'Avg Reward', 'Safety Score', 'Rank']
            
            with open(self.tables_dir / 'results_table.tex', 'w') as f:
                f.write(latex_df.to_latex(index=False,
                                        caption="OmniSafe Algorithm Performance",
                                        label="tab:omnisafe_performance"))
        
        # Generate Markdown report
        report = []
        report.append("# OmniSafe Algorithm Benchmark Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        
        if not metrics_df.empty:
            best_overall = metrics_df.iloc[0]
            report.append(f"- **Best Overall Algorithm**: {best_overall['algorithm']}\n")
            report.append(f"- **Best Pass Rate**: {metrics_df.loc[metrics_df['pass_rate_mean'].idxmax(), 'algorithm']} ")
            report.append(f"({metrics_df['pass_rate_mean'].max():.1%})\n")
            report.append(f"- **Lowest Cost**: {metrics_df.loc[metrics_df['cost_mean'].idxmin(), 'algorithm']} ")
            report.append(f"({metrics_df['cost_mean'].min():.3f})\n")
            report.append(f"- **Highest Reward**: {metrics_df.loc[metrics_df['reward_mean'].idxmax(), 'algorithm']} ")
            report.append(f"({metrics_df['reward_mean'].max():.2f})\n\n")
        
        # Detailed Results
        report.append("## Algorithm Performance\n\n")
        
        if not metrics_df.empty:
            for _, row in metrics_df.iterrows():
                report.append(f"### {row['algorithm']}\n")
                report.append(f"- **Overall Rank**: {row['rank_overall']:.0f}\n")
                report.append(f"- **Pass Rate**: {row['pass_rate_mean']:.1%} ± {row['pass_rate_std']:.1%}\n")
                report.append(f"- **Average Cost**: {row['cost_mean']:.3f} ± {row['cost_std']:.3f}\n")
                report.append(f"- **Average Reward**: {row['reward_mean']:.2f} ± {row['reward_std']:.2f}\n")
                report.append(f"- **Safety Score**: {row['safety_score']:.3f}\n\n")
        
        # Key Findings
        report.append("## Key Findings\n\n")
        
        if not metrics_df.empty:
            # Safety analysis
            safe_algos = metrics_df[metrics_df['pass_rate_mean'] > 0.9]
            if not safe_algos.empty:
                report.append("### ✅ Algorithms Meeting Safety Target (>90% pass rate)\n")
                for _, row in safe_algos.iterrows():
                    report.append(f"- {row['algorithm']}: {row['pass_rate_mean']:.1%}\n")
            else:
                report.append("### ⚠️ No algorithms achieved >90% pass rate\n")
            
            report.append("\n")
            
            # Pareto optimal algorithms
            report.append("### 🏆 Pareto Optimal Algorithms\n")
            report.append("(Best trade-off between cost and reward)\n")
            
            # Find Pareto optimal
            points = metrics_df[['cost_mean', 'reward_mean']].values
            pareto_indices = []
            for i, point in enumerate(points):
                if np.isnan(point).any():
                    continue
                dominated = False
                for j, other in enumerate(points):
                    if i != j and not np.isnan(other).any():
                        if other[0] <= point[0] and other[1] >= point[1]:
                            if other[0] < point[0] or other[1] > point[1]:
                                dominated = True
                                break
                if not dominated:
                    pareto_indices.append(i)
            
            if pareto_indices:
                for idx in pareto_indices:
                    algo = metrics_df.iloc[idx]['algorithm']
                    report.append(f"- {algo}\n")
        
        # Save report
        with open(self.output_dir / 'benchmark_report.md', 'w') as f:
            f.writelines(report)
        
        print(f"✅ Saved report to {self.output_dir}/benchmark_report.md")
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "=" * 60)
        print("OmniSafe Benchmark Analysis")
        if self.enhanced_analysis:
            print("(Enhanced PCS & Cost-Centric Mode)")
        print("=" * 60)
        
        # Collect results
        df = self.collect_all_results()
        
        if df.empty:
            print("\n❌ No results to analyze!")
            print("Please train models first using:")
            print("  sbatch --array=0-29 slurm/train_omnisafe_array.sbatch")
            return
        
        if self.enhanced_analysis:
            # Enhanced analysis pipeline
            print("\n🔬 Running Enhanced Analysis")
            print("=" * 60)
            
            # Compute violation metrics by algorithm × PCS
            metrics_df = self.compute_violation_metrics(df)
            
            if metrics_df.empty:
                print("❌ No metrics computed!")
                return
            
            # Generate comparison tables
            algo_table = self.compute_algo_comparison_table(metrics_df, per_pcs=False)
            pcs_table = self.compute_algo_comparison_table(metrics_df, per_pcs=True)
            
            # Save CSV outputs
            metrics_df.to_csv(self.tables_dir / 'violations_by_algo_pcs.csv', index=False)
            algo_table.to_csv(self.tables_dir / 'algo_comparison_table.csv', index=False)
            if self.per_pcs:
                pcs_table.to_csv(self.tables_dir / 'algo_pcs_comparison_table.csv', index=False)
            
            # Generate enhanced plots
            print("📈 Generating Enhanced Plots...")
            
            # PCS analysis plots
            self.plot_algo_pcs_heatmap(metrics_df, self.plots_dir / 'pcs_heatmap.png')
            self.plot_cost_reward_scatter(metrics_df, self.plots_dir / 'cost_reward_scatter.png')
            self.plot_algo_pcs_line(metrics_df, self.plots_dir / 'pcs_line_plot.png')
            
            # Per-algorithm PCS reports
            for algo in self.algorithms:
                self.plot_per_algo_pcs(metrics_df, algo, self.plots_dir)
            
            # Generate modified composite plot (replace pass-rate panels)
            self.generate_enhanced_composite(df, metrics_df)
            
            # Console summary
            print("\n" + "=" * 60)
            print("=== Algorithm × PCS (cost-centric) ===")
            print("=" * 60)
            print(f"{'Algorithm':<12} {'PCS':<20} {'AvgCost':<8} {'CI95':<6} {'Shortfall%':<10} {'Freq%':<6} {'SOC%':<6} {'Rank':<4}")
            print("-" * 80)
            
            for _, row in metrics_df.iterrows():
                print(f"{row['algorithm']:<12} {row['pcs_mode']:<20} "
                      f"{row['avg_cost_mean']:<8.3f} {row['avg_cost_ci95']:<6.3f} "
                      f"{row['shortfall_rate']*100:<10.2f} {row['freq_rate']*100:<6.2f} "
                      f"{row['soc_rate']*100:<6.2f} #{row['rank_by_cost']:<3}")
            
        else:
            # Legacy analysis pipeline
            metrics_df = self.calculate_metrics(df)
            self.generate_plots(df, metrics_df)
            self.generate_report(df, metrics_df)
            
            # Print legacy summary
            if not metrics_df.empty:
                print("\nTop 3 Algorithms:")
                for i, (_, row) in enumerate(metrics_df.head(3).iterrows(), 1):
                    cost_mean = row.get('cost_mean', row.get('avg_cost_mean', np.nan))
                    reward_mean = row.get('reward_mean', row.get('avg_reward_mean', np.nan))
                    pass_rate = row.get('pass_rate_mean', 0)
                    print(f"{i}. {row['algorithm']}: Pass Rate={pass_rate:.1%}, "
                          f"Cost={cost_mean:.3f}, Reward={reward_mean:.2f}")
        
        print(f"\n📁 All results saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Unified OmniSafe Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with defaults
  python omnisafe_analysis.py
  
  # Analyze specific algorithms
  python omnisafe_analysis.py --algorithms PPOLag CPO --pcs-modes static responsive
  
  # Use specific suites
  python omnisafe_analysis.py --suites standard stress
  
  # Custom directories
  python omnisafe_analysis.py --results-dir runs --output-dir reports/analysis
        """
    )
    
    parser.add_argument('--results-dir', default='runs',
                       help='Directory containing training results')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for analysis (auto-detects batch if not specified)')
    parser.add_argument('--algorithms', nargs='+',
                       default=['PPOLag', 'CPO', 'CUP', 'SautePPO', 'FOCOPS'],
                       help='Algorithms to analyze')
    parser.add_argument('--pcs-modes', nargs='+',
                       default=['static', 'responsive'],
                       help='PCS modes to analyze')
    parser.add_argument('--suites', nargs='+',
                       default=['standard', 'stress'],
                       help='Evaluation suites to use')
    parser.add_argument('--force-reeval', action='store_true', default=True,
                       help='Force fresh evaluation (ignore cached results) - DEFAULT: True')
    parser.add_argument('--use-cache', action='store_true', default=False,
                       help='Use cached evaluation results if available (opposite of --force-reeval)')
    parser.add_argument('--enhanced-analysis', action='store_true', default=False,
                       help='Enable enhanced PCS and cost-centric analysis')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--per-pcs', action='store_true', default=False,
                       help='Generate per-PCS algorithm comparison table')
    
    args = parser.parse_args()
    
    # Determine force_reeval setting (--use-cache overrides --force-reeval)
    force_reeval = args.force_reeval and not args.use_cache
    
    # Run analysis
    analyzer = OmniSafeAnalysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        algorithms=args.algorithms,
        pcs_modes=args.pcs_modes,
        suites=args.suites,
        force_reeval=force_reeval,
        enhanced_analysis=args.enhanced_analysis,
        episodes=args.episodes,
        per_pcs=args.per_pcs
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()