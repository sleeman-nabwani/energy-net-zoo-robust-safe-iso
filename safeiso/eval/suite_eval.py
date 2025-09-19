#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, dataclasses, glob, math, os, re, json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
  
from safeiso.utils.logging_config import get_logger

logger = get_logger("eval.suite_eval")
try:
    import omnisafe  # type: ignore
    # Enforce OmniSafe version pin for reproducible evaluation when available
    assert getattr(omnisafe, "__version__", "0").startswith("0.5."), \
        f"Evaluation pinned to OmniSafe 0.5.x; found {getattr(omnisafe,'__version__','unknown')}"
except Exception:
    omnisafe = None  # optional: only needed for --init-algo path
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
from safeiso.eval.evalpack_loader import load_evalpack_with_denormalization
import os
from pathlib import Path
import time
import socket
try:
    from stable_baselines3 import PPO  # SB3 fallback loader for ISO policies
    _HAS_SB3 = True
except Exception:
    _HAS_SB3 = False


# Simple schema validator (tiny, to avoid new deps)
def _validate_suite_schema(suite: Dict[str, Any], schema_path: str) -> None:
    required_top = ["name", "mode", "algo", "horizon", "episodes", "cost_limit", "device", "scenarios"]
    missing = [k for k in required_top if k not in suite]
    if missing:
        raise ValueError(f"suite yaml missing required fields: {missing}")
    if not isinstance(suite.get("scenarios"), list) or len(suite.get("scenarios")) == 0:
        raise ValueError("suite yaml: 'scenarios' must be a non-empty list")
    for i, sc in enumerate(suite["scenarios"]):
        for k in ("name", "preset", "pcs"):
            if k not in sc:
                raise ValueError(f"suite yaml: scenario[{i}] missing '{k}'")
    # Optional: check mode enum
    if str(suite.get("mode")).upper() not in ("CMDP", "ISOONLY"):
        raise ValueError("suite yaml: 'mode' must be one of ['CMDP','ISOOnly']")


def _pkg_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pkg_name in ("numpy", "gymnasium", "stable_baselines3", "torch"):
        try:
            mod = __import__(pkg_name)
            out[pkg_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg_name] = "not_installed"
    return out


def _write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# OmniSafe Evaluator loader (with SB3 fallback)
def _load_evalpack_policy(run_dir: str, *, force_no_normalize: bool = False, env_action_space=None) -> Tuple[Callable, Dict[str, Any]]:
    # Preferred: EvalPack (TorchScript actor) with denormalization fix
    try:
        if env_action_space is not None:
            # Use fixed loader with denormalization
            act, meta = load_evalpack_with_denormalization(run_dir, env_action_space)
            m = dict(meta or {})
            m["loader"] = "evalpack_fixed"
            return act, m
        else:
            # Fallback to original (with warning)
            act, meta = load_evalpack(run_dir)
            m = dict(meta or {})
            m["loader"] = "evalpack_original"
            return act, m
    except Exception:
        pass

    # Fallback: SB3 policy directory produced by RL-Zoo3
    # Expect a .zip model file and optional vecnormalize.pkl
    if _HAS_SB3:
        rd = Path(run_dir)
        # heuristics to find a model zip
        candidates = [
            rd / "best_model.zip",
            rd / "ISO-RLZoo-v0.zip",
        ] + list(rd.glob("*.zip"))
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is not None:
            model = PPO.load(str(model_path), device="cpu")
            # try to load vecnormalize stats to normalize observations similarly to training
            vec_stats = None
            try:
                import cloudpickle as cp
                stats_path = rd / "ISO-RLZoo-v0" / "vecnormalize.pkl"
                if stats_path.exists():
                    with open(stats_path, "rb") as f:
                        vn = cp.load(f)
                    mean = getattr(getattr(vn, "obs_rms", None), "mean", None)
                    var = getattr(getattr(vn, "obs_rms", None), "var", None)
                    clip = getattr(vn, "clip_obs", None)
                    if mean is not None and var is not None:
                        import numpy as np
                        vec_stats = {
                            "mean": np.asarray(mean, dtype=np.float32),
                            "var": np.asarray(var, dtype=np.float32),
                            "clip": clip,
                        }
            except Exception:
                vec_stats = None

            if vec_stats is None and not force_no_normalize:
                raise RuntimeError(
                    f"SB3 fallback requires vecnormalize.pkl under {rd/'ISO-RLZoo-v0'}; "
                    "pass --force-no-normalize to bypass (not recommended)."
                )

            import numpy as np
            def _norm_obs(obs: np.ndarray) -> np.ndarray:
                if vec_stats is None:
                    return np.asarray(obs, dtype=np.float32)
                x = np.asarray(obs, dtype=np.float32)
                m, v = vec_stats["mean"], vec_stats["var"]
                # Note: we proceed even if shape differs subtly; better to error
                if x.shape != m.shape and not force_no_normalize:
                    raise RuntimeError(
                        f"vecnormalize mean shape {m.shape} != obs shape {x.shape}; "
                        "use --force-no-normalize to bypass."
                    )
                eps = 1e-8
                y = (x - m) / np.sqrt(v + eps)
                c = vec_stats.get("clip", None)
                return np.clip(y, -c, c) if c is not None else y

            def iso_act(ob, deterministic: bool = True):
                x = _norm_obs(ob)
                a, _ = model.predict(x, deterministic=deterministic)
                return np.asarray(a, dtype=np.float32)

            return iso_act, {"loader": "sb3", "vecnormalize": bool(vec_stats is not None)}

    # If neither EvalPack nor SB3 model found, re-raise to signal caller
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
    soc_low_steps: int; soc_high_steps: int  # Added SOC violation tracking
    pass_cost_limit: bool

@dataclass
class EpisodeRow:
    algo: str; mode: str; seed: int; name: str; preset: str; pcs: str; ep_idx: int
    horizon: int; ep_return: float; ep_cost_mean: float; ep_cost_sum: float
    shortfall_steps: int; freq_oob_steps: int; reserve_violation_steps: int
    passed_cost_limit: bool


def _count_flags(info: Dict[str, Any]) -> Tuple[int,int,int,int,int]:
    dbg = info.get("cost_debug", {}) or {}
    shortfall = 1 if float(info.get("shortfall", 0.0)) > 0.0 else 0
    vf = dbg.get("vf_states", {}) or {}
    freq_hz = float(vf.get("freq_hz", 50.0))
    freq = 1 if (freq_hz < 49.0 or freq_hz > 51.0 or "freq" in str(dbg.get("hard_trigger","")).lower()) else 0
    reserve = 1 if ("reserve" in str(dbg.get("hard_trigger","")).lower()) else 0
    
    # Add SOC violation tracking
    soc_frac = float(info.get("soc_fraction", 0.5))
    soc_low = 1 if soc_frac < 0.20 else 0   # Undercharge violation (< 20%)
    soc_high = 1 if soc_frac > 0.80 else 0  # Overcharge violation (> 80%)
    
    return shortfall, freq, reserve, soc_low, soc_high


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
    debug: bool = False,
    post_process: bool = False,
    force_no_normalize: bool = False,
    manifest_out: Optional[str] = None,
):
    with open(suite_path, "r") as f:
        suite = yaml.safe_load(f)
    # Validate suite schema (tiny validator)
    try:
        _validate_suite_schema(suite, os.path.join(os.path.dirname(suite_path), "schema.json"))
    except Exception as e:
        raise RuntimeError(f"Suite schema validation failed: {e}")

    scenarios = suite.get("scenarios", [])

    if dry_run:
        if verbose:
            logger.info(f"[dry_run] Would evaluate policy='{policy}' on {len(scenarios)} scenarios from '{suite_path}'.")
        return

    t0 = time.time()
    host = socket.gethostname()
    manifest: Dict[str, Any] = {
        "suite": os.path.abspath(suite_path),
        "policy": os.path.abspath(policy) if isinstance(policy, str) else str(policy),
        "args": {
            "episodes": episodes,
            "horizon": horizon,
            "algo": algo,
            "mode": mode,
            "device": device,
            "cost_limit": cost_limit,
            "force_no_normalize": force_no_normalize,
            "post_process": post_process,
        },
        "env": {"host": host},
        "pkg_versions": _pkg_versions(),
        "loader": None,
    }

    scen_rows: List[ScenarioRow] = []
    ep_rows_all: List[EpisodeRow] = []

    # For manifest snapshot of vf/cost configs, capture first env configs
    vf_snapshot = None
    cost_snapshot = None

    for i, sc in enumerate(scenarios):
        name = str(sc.get("name", f"scenario_{i}"))
        preset = str(sc.get("preset", "default"))
        pcs_spec = str(sc.get("pcs", "static:0.0"))
        eps = int(sc.get("episodes", episodes))

        # Build env for this scenario with the requested PCS policy
        env = _build_env(seed=base_seed, pcs_spec=pcs_spec, preset=preset, horizon=horizon)
        # Snapshot configs once
        if vf_snapshot is None:
            try:
                from safeiso.wrappers.cmdp_adapter import CMDPAdapter
                cur = env
                while hasattr(cur, 'env') and not isinstance(cur, CMDPAdapter):
                    cur = getattr(cur, 'env')
                if isinstance(cur, CMDPAdapter):
                    vf_snapshot = dataclasses.asdict(cur.vf)
                    cost_snapshot = dataclasses.asdict(cur.cfg)
            except Exception:
                pass

        # ISO policy tied to this env's action space
        if policy == "mean":
            iso_act = lambda obs, deterministic=True: ((env.action_space.low + env.action_space.high)/2.0).astype(np.float32)  # noqa: E731
            meta = {}
        elif policy == "random":
            iso_act = lambda obs, deterministic=True: env.action_space.sample().astype(np.float32)  # noqa: E731
            meta = {}
        elif policy == "load_follow":
            from safeiso.eval.policies.baselines import iso_load_follow_builder
            act_fn, meta = iso_load_follow_builder(env)
            iso_act = act_fn
        else:
            # CRITICAL FIX: Use denormalized evalpack loader to fix action space mismatch
            # Trained policies output actions in [-1,1] but evaluation environment expects [1-10, 1-10, 0-300]
            iso_act, meta = _load_evalpack_policy(policy, force_no_normalize=force_no_normalize, env_action_space=env.action_space)
        if manifest.get("loader") is None:
            manifest["loader"] = meta.get("loader", "unknown") if isinstance(meta, dict) else "unknown"

        # Resolve Sauté flags/params
        is_saute = ("saute" in str(algo).lower()) or (isinstance(meta, dict) and "saute" in meta)
        saute_cfg = (meta.get("saute", {}) if isinstance(meta, dict) else {}) if is_saute else {}
        saute_gamma = float(saute_cfg.get("gamma", 1.0))
        safety_budget = saute_cfg.get("budget", None)  # if None, fallback to cost_limit*horizon later

        # episode loop
        ret_list, costs_all = [], []
        sh_total = fq_total = rv_total = soc_low_total = soc_high_total = 0
        for e in range(eps):
            obs, info = env.reset(seed=base_seed + i + e)
            ep_ret = 0.0; ep_costs = []; sh=fq=rv=soc_low=soc_high=0
            # analytics accumulators
            vio_burst = 0
            vio_burst_max = 0
            ttf_violation = None
            comp_acc: Dict[str, List[float]] = {}
            
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
                
                # Debug prints for first few steps
                if debug and t < 5:
                    # robust access for both list/tuple or dict info
                    ii = info[0] if isinstance(info, (list, tuple)) else info
                    pcs_a = ii.get('pcs_action', 'NA')
                    vio = ii.get('violations', {})
                    rbc = ii.get('raw_cost_before_clip', 'NA')
                    cst = float(cost if not hasattr(cost, 'item') else cost.item())
                    logger.debug(f"[ep{e} t{t}] pcs_action={pcs_a} shortfall={vio.get('shortfall')} freq_oob={vio.get('freq_oob')} raw_before_clip={rbc} cost={cst}")
                
                # sauté update: s_{t+1} = (s_t - cost_t) / gamma
                if is_saute:
                    try:
                        c_val = float(cost) if not hasattr(cost, "item") else float(cost.item())
                    except Exception:
                        c_val = float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
                    s = (s - c_val) / max(1e-12, saute_gamma)
                
                ep_ret += float(rew)
                c = float(cost); ep_costs.append(c)
                s_count,f_count,r_count,soc_low_count,soc_high_count = _count_flags(info); sh+=s_count; fq+=f_count; rv+=r_count; soc_low+=soc_low_count; soc_high+=soc_high_count
                # violation analytics
                any_vio = int(bool(s_count or f_count or r_count or soc_low_count or soc_high_count))
                if any_vio:
                    vio_burst += 1
                    if ttf_violation is None:
                        ttf_violation = t
                else:
                    vio_burst = 0
                if vio_burst > vio_burst_max:
                    vio_burst_max = vio_burst
                # cost component stats if available
                try:
                    ii = info[0] if isinstance(info, (list, tuple)) else info
                    comps = (ii.get("cost_debug") or {}).get("components") or {}
                    for k, v in comps.items():
                        comp_acc.setdefault(k, []).append(float(v))
                except Exception:
                    pass
                if terminated or truncated:
                    break
            ret_list.append(ep_ret); costs_all.extend(ep_costs)
            ep_mean = float(np.mean(ep_costs)) if ep_costs else math.nan
            ep_sum  = float(np.sum(ep_costs))  if ep_costs else math.nan
            # summarize component stats
            def _p95(x: List[float]) -> float:
                if not x: return math.nan
                xx = sorted(x); idx = int(0.95*len(xx))
                idx = min(max(idx, 0), len(xx)-1)
                return float(xx[idx])
            comp_mean = {k: (float(np.mean(v)) if v else math.nan) for k, v in comp_acc.items()}
            comp_p95  = {k: _p95(v) for k, v in comp_acc.items()}
            ep_rows_all.append(EpisodeRow(
                algo=algo, mode=mode, seed=base_seed, name=name, preset=preset, pcs=pcs_spec, ep_idx=e,
                horizon=horizon, ep_return=ep_ret, ep_cost_mean=ep_mean, ep_cost_sum=ep_sum,
                shortfall_steps=sh, freq_oob_steps=fq, reserve_violation_steps=rv,
                passed_cost_limit=(ep_mean <= cost_limit) if not math.isnan(ep_mean) else False
            ))
            # attach extra analytics by appending to last row via dataclass replace is overkill; write directly in CSV write phase below
            ep_rows_all[-1].__dict__["violation_burst_max"] = int(vio_burst_max)
            ep_rows_all[-1].__dict__["time_to_first_violation"] = int(ttf_violation) if ttf_violation is not None else -1
            ep_rows_all[-1].__dict__["violation_counts_by_type"] = json.dumps({"shortfall": sh, "freq": fq, "reserve": rv, "soc_low": soc_low, "soc_high": soc_high})
            ep_rows_all[-1].__dict__["cost_components_mean"] = json.dumps(comp_mean)
            ep_rows_all[-1].__dict__["cost_components_p95"]  = json.dumps(comp_p95)
            sh_total+=sh; fq_total+=fq; rv_total+=rv; soc_low_total+=soc_low; soc_high_total+=soc_high

        scen_rows.append(ScenarioRow(
            algo=algo, mode=mode, seed=base_seed, name=name, preset=preset, pcs=pcs_spec,
            episodes=eps, horizon=horizon,
            return_mean=float(np.mean(ret_list)) if ret_list else math.nan,
            return_std=float(np.std(ret_list)) if ret_list else math.nan,
            avg_step_cost=float(np.mean(costs_all)) if costs_all else math.nan,
            shortfall_steps=sh_total, freq_oob_steps=fq_total, reserve_violation_steps=rv_total,
            soc_low_steps=soc_low_total, soc_high_steps=soc_high_total,
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
    # Episodes CSV with extended analytics: ensure stable header
    with open(out_episodes, "w", newline="") as f:
        w = csv.writer(f)
        base_keys = list(dataclasses.asdict(EpisodeRow("","",0,"","","",0,0,0.0,0.0,0.0,0,0,0,False)).keys())
        extra_keys = ["violation_burst_max","time_to_first_violation","violation_counts_by_type","cost_components_mean","cost_components_p95"]
        header = base_keys + extra_keys
        w.writerow(header)
        for r in ep_rows_all:
            d = dataclasses.asdict(r)
            row = [d.get(k, "") for k in base_keys]
            row += [r.__dict__.get(k, "") for k in extra_keys]
            w.writerow(row)
    # All per-scenario envs were closed above

    # Write manifest
    if manifest_out is None:
        manifest_out = os.path.join(os.path.dirname(out_scenarios) or ".", "eval_manifest.json")
    manifest["vf_config"] = vf_snapshot
    manifest["cost_config"] = cost_snapshot
    manifest["walltime_s"] = float(time.time() - t0)
    _write_manifest(manifest_out, manifest)

    # Optional post-process (aggregate + plots)
    if post_process:
        try:
            from safeiso.eval.postprocess import aggregate_episodes_to_scenarios, make_standard_plots
            scen_csv = aggregate_episodes_to_scenarios(out_episodes)
            make_standard_plots(scen_csv)
        except Exception as e:
            logger.error(f"[post-process] failed: {e}")


# CLI
def main():
    ap = argparse.ArgumentParser(description="Evaluate ONE policy against a scenario suite (deterministic).")
    ap.add_argument("--env_id", default="SafeISO-CMDP-omni-v0", 
                   help="(Deprecated) training env id hint; not used for scoring.")
    ap.add_argument("--metric_env_id", default="SafeISO-CMDP-omni-v0",
                   help="Env used to compute reward/cost metrics for ALL policies.")
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
    ap.add_argument("--post-process", action="store_true", help="Aggregate and plot after run")
    ap.add_argument("--force-no-normalize", action="store_true", help="Allow SB3 fallback without vecnormalize stats")
    ap.add_argument("--manifest-out", default=None, help="Path to write eval_manifest.json")
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
        env_id=args.metric_env_id, suite_path=args.suite, policy=args.policy,
        episodes=args.episodes, horizon=args.horizon, base_seed=args.seed,
        algo=args.algo, mode=args.mode, device=args.device, cost_limit=args.cost_limit,
        out_scenarios=out_scenarios, out_episodes=out_episodes,
        debug=args.debug, post_process=args.post_process,
        force_no_normalize=args.force_no_normalize, manifest_out=args.manifest_out,
    )

if __name__ == "__main__":
    main()
