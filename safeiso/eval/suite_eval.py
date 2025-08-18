#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, dataclasses, glob, math, os, re, json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import omnisafe

# Enforce OmniSafe version pin for reproducible evaluation
assert getattr(omnisafe, "__version__", "0").startswith("0.5."), \
    f"Evaluation pinned to OmniSafe 0.5.x; found {getattr(omnisafe,'__version__','unknown')}"
import yaml

# Register env IDs (no-op if already imported)
try:
    import safeiso.utils.register_envs  # noqa: F401
except Exception:
    pass

# Build ISO-only CMDP env and inject PCS
from safeiso.utils.make_env import make_iso_cmdp_basic
import numpy as np
from safeiso.eval.evalpack_loader import load_evalpack


# OmniSafe Evaluator loader
def _load_evalpack_policy(run_dir: str) -> Tuple[Callable, Dict[str, Any]]:
    act, meta = load_evalpack(run_dir)
    return act, meta


# PCS env builder (use unified PCS spec)


def _build_env(seed: int, pcs_spec: str, preset: Optional[str], horizon: int):
    # make_iso_cmdp_basic internally resolves the PCS policy from the spec string
    env = make_iso_cmdp_basic(seed=seed, pcs=pcs_spec, preset=preset)
    return env


# rows
@dataclass
class ScenarioRow:
    algo: str; mode: str; seed: int; name: str; preset: str; pcs: str
    episodes: int; horizon: int
    return_mean: float; return_std: float; avg_step_cost: float
    shortfall_steps: int; freq_oob_steps: int; reserve_violation_steps: int
    pass_cost_limit: bool

@dataclass
class EpisodeRow:
    algo: str; mode: str; seed: int; name: str; preset: str; pcs: str; ep_idx: int
    horizon: int; ep_return: float; ep_cost_mean: float; ep_cost_sum: float
    shortfall_steps: int; freq_oob_steps: int; reserve_violation_steps: int
    passed_cost_limit: bool


def _count_flags(info: Dict[str, Any]) -> Tuple[int,int,int]:
    dbg = info.get("cost_debug", {}) or {}
    shortfall = 1 if float(info.get("shortfall", 0.0)) > 0.0 else 0
    vf = dbg.get("vf_states", {}) or {}
    freq_hz = float(vf.get("freq_hz", 50.0))
    freq = 1 if (freq_hz < 49.0 or freq_hz > 51.0 or "freq" in str(dbg.get("hard_trigger","")).lower()) else 0
    reserve = 1 if ("reserve" in str(dbg.get("hard_trigger","")).lower()) else 0
    return shortfall, freq, reserve


# core API (importable)
def run_suite(
    env_id: str,
    suite_path: str,
    policy: str,
    episodes: int,
    horizon: int,
    base_seed: int,
    algo: str,
    mode: str,
    device: str,
    cost_limit: float,
    out_scenarios: str,
    out_episodes: str,
    *,
    verbose: bool = True,
    print_prompt: bool = True,
    dry_run: bool = False,
):
    with open(suite_path, "r") as f:
        suite = yaml.safe_load(f)
    scenarios = suite.get("scenarios", [])

    if dry_run:
        if verbose:
            print(f"[dry_run] Would evaluate policy='{policy}' on {len(scenarios)} scenarios from '{suite_path}'.")
        return

    scen_rows: List[ScenarioRow] = []
    ep_rows_all: List[EpisodeRow] = []

    for i, sc in enumerate(scenarios):
        name = str(sc.get("name", f"scenario_{i}"))
        preset = str(sc.get("preset", "default"))
        pcs_spec = str(sc.get("pcs", "static:0.0"))
        eps = int(sc.get("episodes", episodes))

        # Build env for this scenario with the requested PCS policy
        env = _build_env(seed=base_seed, pcs_spec=pcs_spec, preset=preset, horizon=horizon)

        # ISO policy tied to this env's action space
        if policy == "mean":
            iso_act = lambda obs, deterministic=True: ((env.action_space.low + env.action_space.high)/2.0).astype(np.float32)  # noqa: E731
            meta = {}
        elif policy == "random":
            iso_act = lambda obs, deterministic=True: env.action_space.sample().astype(np.float32)  # noqa: E731
            meta = {}
        else:
            iso_act, meta = _load_evalpack_policy(policy)

        # Resolve Sauté flags/params
        is_saute = ("saute" in str(algo).lower()) or (isinstance(meta, dict) and "saute" in meta)
        saute_cfg = (meta.get("saute", {}) if isinstance(meta, dict) else {}) if is_saute else {}
        saute_gamma = float(saute_cfg.get("gamma", 1.0))
        safety_budget = saute_cfg.get("budget", None)  # if None, fallback to cost_limit*horizon later

        # episode loop
        ret_list, costs_all = [], []
        sh_total = fq_total = rv_total = 0
        for e in range(eps):
            obs, info = env.reset(seed=base_seed + i + e)
            ep_ret = 0.0; ep_costs = []; sh=fq=rv=0
            
            # init sauté safety state once per episode
            if is_saute:
                s = float(safety_budget if safety_budget is not None else (cost_limit * horizon))
            
            for t in range(horizon):
                ob = obs
                # to numpy on CPU
                if not isinstance(ob, np.ndarray):
                    try:
                        ob = ob.detach().cpu().numpy()
                    except Exception:
                        ob = np.asarray(ob, dtype=np.float32)
                if ob.ndim == 2 and ob.shape[0] == 1:
                    ob = ob[0]

                # augment if sauté
                if is_saute:
                    ob_aug = np.concatenate([ob.astype(np.float32, copy=False),
                                           np.array([s], dtype=np.float32)], axis=0)
                else:
                    ob_aug = ob.astype(np.float32, copy=False)

                a = iso_act(ob_aug, True)
                
                # step
                a_t = a
                try:
                    import torch
                    if isinstance(obs, torch.Tensor) and not isinstance(a, torch.Tensor):
                        a_t = torch.as_tensor(a, dtype=torch.float32, device=obs.device)
                except Exception:
                    pass
                
                step = env.step(a_t)
                if len(step) == 6:
                    obs, rew, cost, terminated, truncated, info = step
                else:
                    obs, rew, terminated, truncated, info = step
                    cost = float(info.get("cost", 0.0))
                
                # Debug prints for first steps
                if args.debug and t < 5:
                    # robust access for both list/tuple or dict info
                    ii = info[0] if isinstance(info, (list, tuple)) else info
                    pcs_a = ii.get('pcs_action', 'NA')
                    vio = ii.get('violations', {})
                    rbc = ii.get('raw_cost_before_clip', 'NA')
                    cst = float(cost if not hasattr(cost, 'item') else cost.item())
                    print(f"[debug ep{e} t{t}] pcs_action={pcs_a} shortfall={vio.get('shortfall')} freq_oob={vio.get('freq_oob')} raw_before_clip={rbc} cost={cst}")
                
                # sauté update: s_{t+1} = (s_t - cost_t) / gamma
                if is_saute:
                    try:
                        c_val = float(cost) if not hasattr(cost, "item") else float(cost.item())
                    except Exception:
                        c_val = float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
                    s = (s - c_val) / max(1e-12, saute_gamma)
                
                ep_ret += float(rew)
                c = float(cost); ep_costs.append(c)
                s_count,f_count,r_count = _count_flags(info); sh+=s_count; fq+=f_count; rv+=r_count
                if terminated or truncated:
                    break
            ret_list.append(ep_ret); costs_all.extend(ep_costs)
            ep_mean = float(np.mean(ep_costs)) if ep_costs else math.nan
            ep_sum  = float(np.sum(ep_costs))  if ep_costs else math.nan
            ep_rows_all.append(EpisodeRow(
                algo=algo, mode=mode, seed=base_seed, name=name, preset=preset, pcs=pcs_spec, ep_idx=e,
                horizon=horizon, ep_return=ep_ret, ep_cost_mean=ep_mean, ep_cost_sum=ep_sum,
                shortfall_steps=sh, freq_oob_steps=fq, reserve_violation_steps=rv,
                passed_cost_limit=(ep_mean <= cost_limit) if not math.isnan(ep_mean) else False
            ))
            sh_total+=sh; fq_total+=fq; rv_total+=rv

        scen_rows.append(ScenarioRow(
            algo=algo, mode=mode, seed=base_seed, name=name, preset=preset, pcs=pcs_spec,
            episodes=eps, horizon=horizon,
            return_mean=float(np.mean(ret_list)) if ret_list else math.nan,
            return_std=float(np.std(ret_list)) if ret_list else math.nan,
            avg_step_cost=float(np.mean(costs_all)) if costs_all else math.nan,
            shortfall_steps=sh_total, freq_oob_steps=fq_total, reserve_violation_steps=rv_total,
            pass_cost_limit=(float(np.mean(costs_all)) <= cost_limit) if costs_all else False
        ))

        # Close env for this scenario
        try:
            env.close()
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_scenarios) or ".", exist_ok=True)
    with open(out_scenarios, "w", newline="") as f:
        w = csv.writer(f); first=True
        for r in scen_rows:
            d = dataclasses.asdict(r)
            if first: w.writerow(list(d.keys())); first=False
            w.writerow(list(d.values()))
    with open(out_episodes, "w", newline="") as f:
        w = csv.writer(f); first=True
        for r in ep_rows_all:
            d = dataclasses.asdict(r)
            if first: w.writerow(list(d.keys())); first=False
            w.writerow(list(d.values()))
    # All per-scenario envs were closed above


# CLI
def main():
    ap = argparse.ArgumentParser(description="Evaluate ONE policy against a scenario suite (deterministic).")
    ap.add_argument("--env_id", default="SafeISO-CMDP-omni-v0")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--policy", required=True)  # 'mean'|/path/to/run_dir
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algo", default="(unknown)")
    ap.add_argument("--mode", default="CMDP")  # CMDP|ISOOnly
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cost_limit", type=float, default=0.10)
    ap.add_argument("--out_scenarios", default=None)
    ap.add_argument("--out_episodes", default=None)
    ap.add_argument("--debug", action="store_true", help="Print first steps diagnostics per episode")
    args = ap.parse_args()

    # Default outputs under run_dir/eval/...
    out_scenarios = args.out_scenarios
    out_episodes = args.out_episodes
    if out_scenarios is None and isinstance(args.policy, str) and os.path.isdir(os.path.join(args.policy, "evalpack")):
        os.makedirs(os.path.join(args.policy, "eval"), exist_ok=True)
        out_scenarios = os.path.join(args.policy, "eval", "stress_eval.csv")
    if out_episodes is None and isinstance(args.policy, str) and os.path.isdir(os.path.join(args.policy, "evalpack")):
        out_episodes = os.path.join(args.policy, "eval", "stress_eval_episodes.csv")
    
    # Fallback defaults
    if out_scenarios is None:
        out_scenarios = "stress_eval.csv"
    if out_episodes is None:
        out_episodes = "stress_eval_episodes.csv"

    run_suite(
        env_id=args.env_id, suite_path=args.suite, policy=args.policy,
        episodes=args.episodes, horizon=args.horizon, base_seed=args.seed,
        algo=args.algo, mode=args.mode, device=args.device, cost_limit=args.cost_limit,
        out_scenarios=out_scenarios, out_episodes=out_episodes,
    )

if __name__ == "__main__":
    main()
