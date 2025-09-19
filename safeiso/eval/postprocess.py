from __future__ import annotations
from typing import List
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    # Treat empty path as current directory
    if not path:
        path = "."
    os.makedirs(path, exist_ok=True)


def compute_bootstrap_ci(df: pd.DataFrame, cols: List[str], boot: int = 5000, alpha: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(0)
    for col in cols:
        x = out[col].dropna().values
        if len(x) == 0:
            out[f"{col}_CI_low"] = np.nan
            out[f"{col}_CI_high"] = np.nan
            continue
        boots = rng.choice(x, size=(boot, len(x)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
        out[f"{col}_CI_low"] = lo
        out[f"{col}_CI_high"] = hi
    return out


def aggregate_episodes_to_scenarios(episodes_csv: str) -> str:
    """Aggregate per-episode CSV to per-scenario CSV with mean/std and pass rate.

    Returns the path to the scenarios CSV written next to the episodes CSV.
    """
    df = pd.read_csv(episodes_csv)
    req = ["algo","mode","seed","name","preset","pcs","ep_cost_mean","ep_return"]
    # Backward-compatible columns
    if "ep_return" not in df.columns and "ep_return" in df.columns:
        pass

    g = df.groupby(["algo","mode","pcs"], as_index=False)
    scen = g.agg(
        return_mean=("ep_return","mean"),
        return_std=("ep_return","std"),
        avg_step_cost=("ep_cost_mean","mean"),
        pass_cost_limit=("passed_cost_limit","mean"),
        scenarios=("name","nunique")
    )
    scen = compute_bootstrap_ci(scen, cols=["return_mean","avg_step_cost"], boot=5000, alpha=0.05)

    out_dir = os.path.dirname(episodes_csv)
    out_csv = os.path.join(out_dir, "scenarios.csv")
    scen.to_csv(out_csv, index=False)
    return out_csv


def make_standard_plots(scenarios_csv: str) -> None:
    df = pd.read_csv(scenarios_csv)
    out_dir = os.path.dirname(scenarios_csv)
    _ensure_dir(out_dir)

    # pass rate by scenario (per pcs)
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(df))
    ax.bar(x, df["pass_cost_limit"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pcs"].astype(str).values, rotation=30, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel("Pass rate")
    ax.set_title("Pass Rate by Scenario")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pass_rate_by_scenario.png"), dpi=200); plt.close(fig)

    # avg step cost box
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(df["avg_step_cost"].dropna().values, vert=True)
    ax.set_ylabel("Avg Step Cost")
    ax.set_title("Average Step Cost (scenarios)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "avg_step_cost_box.png"), dpi=200); plt.close(fig)

    # return box
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(df["return_mean"].dropna().values, vert=True)
    ax.set_ylabel("Return")
    ax.set_title("Return (scenarios)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "return_box.png"), dpi=200); plt.close(fig)

    # violation burst histogram if available later from merged episodes
    # Placeholder: if a merged episodes.csv is colocated, try to read and plot
    ep_csv = os.path.join(os.path.dirname(out_dir), "stress_eval_episodes.csv")
    if os.path.isfile(ep_csv):
        try:
            ep = pd.read_csv(ep_csv)
            if "violation_burst_max" in ep.columns:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(ep["violation_burst_max"].dropna().values, bins=20)
                ax.set_xlabel("Violation Burst Max (steps)")
                ax.set_ylabel("Count")
                ax.set_title("Violation Burst Histogram")
                fig.tight_layout(); fig.savefig(os.path.join(out_dir, "violation_burst_hist.png"), dpi=200); plt.close(fig)
        except Exception:
            pass


