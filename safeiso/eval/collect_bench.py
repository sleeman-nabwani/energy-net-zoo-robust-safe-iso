from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd

def ci_mean(x, boot=5000, alpha=0.05, rng=None):
    x = np.array(x, dtype=np.float64)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(rng)
    boots = rng.choice(x, size=(boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return (x.mean(), lo, hi)

def main():
    p = argparse.ArgumentParser(description="Aggregate suite evaluation CSVs into an organised table.")
    p.add_argument("--root", default="runs/baseline_static")
    p.add_argument("--out", default="bench_summary.csv")
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    scen_files = glob.glob(os.path.join(args.root, "**", "eval", "stress_eval.csv"), recursive=True)
    ep_files   = glob.glob(os.path.join(args.root, "**", "eval", "stress_eval_episodes.csv"), recursive=True)
    if not scen_files:
        print("No scenario CSVs found under", args.root); return

    df_s = pd.concat([pd.read_csv(f) for f in scen_files], ignore_index=True)
    df_e = pd.concat([pd.read_csv(f) for f in ep_files], ignore_index=True) if ep_files else None

    if df_e is not None and "passed_cost_limit" in df_e.columns:
        ep_pass = df_e.groupby(["algo","mode","pcs"], as_index=False)["passed_cost_limit"].mean() \
                      .rename(columns={"passed_cost_limit":"episode_pass_rate"})
    else:
        ep_pass = None

    rows = []
    for (algo, mode, pcs), g in df_s.groupby(["algo","mode","pcs"]):
        cost = g["avg_step_cost"].dropna().values
        ret  = g["return_mean"].dropna().values
        m_cost, lo_cost, hi_cost = ci_mean(cost, boot=args.boot, alpha=args.alpha)
        m_ret,  lo_ret,  hi_ret  = ci_mean(ret,  boot=args.boot, alpha=args.alpha)
        pass_rate = float(np.mean(g["pass_cost_limit"].astype(bool))) if "pass_cost_limit" in g else np.nan
        rows.append({
            "algo": algo, "mode": mode, "pcs": pcs,
            "scenarios": g["name"].nunique() if "name" in g else len(g),
            "pass_rate_scenario": pass_rate,
            "avg_step_cost_mean": m_cost, "avg_step_cost_CI_low": lo_cost, "avg_step_cost_CI_high": hi_cost,
            "return_mean": m_ret, "return_CI_low": lo_ret, "return_CI_high": hi_ret,
        })

    summary = pd.DataFrame(rows).sort_values(["algo","mode","pcs"]).reset_index(drop=True)
    if ep_pass is not None:
        summary = summary.merge(ep_pass, on=["algo","mode","pcs"], how="left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
