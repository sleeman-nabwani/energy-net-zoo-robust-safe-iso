from __future__ import annotations
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_cost_reward_frontier(df: pd.DataFrame, out_dir: str, title: str = "Reward vs. Avg Step-Cost"):
    """
    Scatter with error bars:
      x = avg_step_cost_mean  (with CI)
      y = return_mean         (with CI)
    Grouped by (algo, mode); pcs is shown as text label near the point.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    groups = df.groupby(["algo","mode"], sort=False)

    for (algo, mode), g in groups:
        x = g["avg_step_cost_mean"].values
        y = g["return_mean"].values
        xerr = np.vstack([g["avg_step_cost_mean"] - g["avg_step_cost_CI_low"],
                          g["avg_step_cost_CI_high"] - g["avg_step_cost_mean"]])
        yerr = np.vstack([g["return_mean"] - g["return_CI_low"],
                          g["return_CI_high"] - g["return_mean"]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", label=f"{algo} / {mode}", capsize=3)
        # annotate pcs near each point
        for xi, yi, pcs in zip(x, y, g["pcs"].values):
            ax.annotate(str(pcs), (xi, yi), xytext=(5, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Average Step Cost")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frontier_cost_vs_reward.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "frontier_cost_vs_reward.pdf"))
    plt.close(fig)

def plot_pass_rates(df: pd.DataFrame, out_dir: str, title: str = "Pass Rate by Algo/Mode"):
    """
    Bar chart of scenario-level pass rate; if episode_pass_rate present, overlay markers.
    """
    fig, ax = plt.subplots(figsize=(9,5))
    # Aggregate over pcs for a simple view
    g = df.groupby(["algo","mode"], as_index=False).agg(
        pass_rate_scenario=("pass_rate_scenario","mean"),
        scenarios=("scenarios","sum")
    )
    x = np.arange(len(g))
    ax.bar(x, g["pass_rate_scenario"].values)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{m}" for a,m in zip(g["algo"], g["mode"])], rotation=0)
    ax.set_ylim(0,1.0)
    ax.set_ylabel("Pass Rate (avg over PCS)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Optional overlay of episode-level pass rate if available (mean over pcs)
    if "episode_pass_rate" in df.columns:
        e = df.groupby(["algo","mode"], as_index=False)["episode_pass_rate"].mean()
        # align order with g
        e = g[["algo","mode"]].merge(e, on=["algo","mode"], how="left")
        ax.plot(x, e["episode_pass_rate"].values, marker="o", linestyle="None")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pass_rates.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "pass_rates.pdf"))
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot benchmark figures from bench_summary.csv")
    ap.add_argument("--summary", required=True, help="Path to bench_summary.csv")
    ap.add_argument("--out_dir", required=True, help="Directory to save figures")
    ap.add_argument("--title_tag", default="", help="Optional suffix for plot titles, e.g., 'Baseline Static'")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    df = pd.read_csv(args.summary)

    # Basic validation
    required = {"algo","mode","pcs","avg_step_cost_mean","avg_step_cost_CI_low","avg_step_cost_CI_high",
                "return_mean","return_CI_low","return_CI_high","pass_rate_scenario","scenarios"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bench_summary.csv missing columns: {missing}")

    tag = f" ({args.title_tag})" if args.title_tag else ""
    plot_cost_reward_frontier(df, args.out_dir, title="Reward vs. Avg Step-Cost"+tag)
    plot_pass_rates(df, args.out_dir, title="Pass Rate by Algo/Mode"+tag)
    print(f"Wrote plots to {args.out_dir}")

if __name__ == "__main__":
    main()
