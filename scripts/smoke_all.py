#!/usr/bin/env python3
"""
Fail-fast config smoke for all OmniSafe algos.

- Builds a resolved config (dict) for each algo with:
  * shared BASE defaults,
  * trainer-standard fields (seed/device/steps/env_cfgs),
  * algo-specific overrides,
  * validation of required keys (per algo).
- Exits non-zero on the first missing key (so your SLURM sweep won’t waste time).

Optional: with --init-algo, it will import OmniSafe, register SafeISO envs,
and try constructing the Algo(env_id, cfgs) once (no learning). This is slower
but catches adapter/constructor regressions early.

Usage:
  python scripts/smoke_all.py --mode cmdp --horizon 48 --steps 96 \
      --pcs "static:0.0" --cost-limit 0.10 --preset default
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any

# Make package importable when running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from safeiso.train.algo_defaults import BASE, deep_update, algo_overrides, validate_required

ALGOS = ("PPOLag", "CUP", "CPO", "FOCOPS", "SautePPO")


def build_cfg_dict(
    *,
    algo: str,
    steps: int,
    seed: int,
    device: str,
    cmdp: bool,
    pcs: str,
    preset: str,
    cost_limit: float,
    horizon: int,
    save_dir: str,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    deep_update(cfg, BASE)

    pcs_tag = pcs.split("?", 1)[0].replace(":", "-").replace("/", "_")
    exp = f"{algo}_{'cmdp' if cmdp else 'mdp'}_{preset}_{pcs_tag}"

    # Standard trainer/env fields (mirror the trainer)
    deep_update(cfg, {
        "seed": int(seed),
        "device": device,
        "num_envs": 1,
        "exp_name": exp,
        "train_cfgs": {
            "device": device,
            "total_steps": int(steps),
            "seed": int(seed),
            "log_dir": os.path.join(save_dir, f"{algo}_omni"),
        },
        "algo_cfgs": {
            "cost_limit": float(cost_limit),
            "steps_per_epoch": int(steps),
            "use_cost": bool(cmdp),
            "max_ep_len": int(horizon),
        },
        "lagrange_cfgs": {
            "cost_limit": float(cost_limit),
            "lagrangian_multiplier_init": 0.0,
            "lambda_lr": 0.005,
            "lambda_optimizer": "Adam",
            "lagrangian_upper_bound": None,
        },
        "env_cfgs": {
            "preset": preset,
            "pcs": pcs,
            "spread_limit": 5.0,
            "return_cost_in_info": True,
            "max_episode_steps": int(horizon),
        },
        "logger_cfgs": {
            "log_dir": os.path.join(save_dir, f"{algo}_omni"),
        },
    })

    # Layer algo-specific overrides & validate
    deep_update(cfg, algo_overrides(algo, horizon=horizon, cost_limit=cost_limit))
    validate_required(cfg, algo)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cmdp", "mdp"], default="cmdp")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--pcs", type=str, default="static:0.0")
    ap.add_argument("--preset", type=str, default="default")
    ap.add_argument("--cost-limit", type=float, default=0.10)
    ap.add_argument("--save-root", type=str, default="runs/smoke_configs")
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--init-algo", action="store_true",
                    help="Also try constructing Algo(env_id, cfgs) once (slower).")
    args = ap.parse_args()

    cmdp = (args.mode == "cmdp")
    os.makedirs(args.save_root, exist_ok=True)

    print(f"Config pre-flight (mode={args.mode}, horizon={args.horizon}, pcs={args.pcs})")
    for algo in ALGOS:
        save_dir = os.path.join(args.save_root, f"{algo}_{args.mode}")
        cfg = build_cfg_dict(
            algo=algo, steps=args.steps, seed=args.seed, device=args.device,
            cmdp=cmdp, pcs=args.pcs, preset=args.preset, cost_limit=args.cost_limit,
            horizon=args.horizon, save_dir=save_dir,
        )
        if args.print_config:
            path = os.path.join(save_dir, "config_resolved.json")
            os.makedirs(save_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
        print(f"  ✓ {algo}: keys OK")

        if args.init_algo:
            # Optional heavier check: construct Algo once to ensure adapter compatibility.
            import safeiso.utils.register_envs  # noqa: F401
            env_id = "SafeISO-CMDP-omni-v0" if cmdp else "SafeISO-ISOOnly-omni-v0"
            from omnisafe.utils.config import Config
            cfg_obj = Config.dict2config(cfg)
            # Import the right class
            if   algo == "PPOLag":
                from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag as AlgoCls
            elif algo == "CUP":
                from omnisafe.algorithms.on_policy.first_order.cup import CUP as AlgoCls
            elif algo == "CPO":
                from omnisafe.algorithms.on_policy.second_order.cpo import CPO as AlgoCls
            elif algo == "FOCOPS":
                from omnisafe.algorithms.on_policy.first_order.focops import FOCOPS as AlgoCls
            elif algo in ("SautePPO", "PPOSaute"):
                from omnisafe.algorithms.on_policy.saute.ppo_saute import PPOSaute as AlgoCls
            else:
                raise ValueError(algo)

            _ = AlgoCls(env_id=env_id, cfgs=cfg_obj)  # no learn(), just init
            print(f"    • {algo}: Algo(env_id, cfgs) constructed")

    print("All algos passed config validation")


if __name__ == "__main__":
    main() 