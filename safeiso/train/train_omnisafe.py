# safeiso/train/train_omnisafe.py
"""
OmniSafe trainer for SafeISO.

Assumptions:
- Env IDs are registered by importing safeiso.utils.register_envs.
- OmniSafe always returns a 6-tuple: (obs, reward, cost, terminated, truncated, info).
- PCS policy is passed as a single string spec via --pcs (e.g. 'static:0.0',
  'sb3:/abs/model.zip', 'responsive:spread_prop?gain=0.6&clip=5').

Example:
  python -m safeiso.train.train_omnisafe \
    --algo PPOLag --steps 2000 --seed 0 --device cpu \
    --cmdp --pcs "responsive:spread_prop?gain=0.6&clip=5" \
    --preset default --eval_every 500 --eval_episodes 3 --save_dir runs/dev
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, Tuple
import numpy as np

# Ensure SafeISO env IDs are registered
import safeiso.utils.register_envs  # noqa: F401

from omnisafe.envs import make as omni_make
from omnisafe.utils.config import Config

from safeiso.train.algo_defaults import BASE, deep_update, algo_overrides, validate_required
from safeiso.eval.metrics import rollout_metrics
from safeiso.eval.policy_api import make_policy_act

ALGOS = ("PPOLag", "CUP", "CPO", "SautePPO", "FOCOPS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SafeISO OmniSafe Trainer")
    p.add_argument("--algo", choices=ALGOS, default="PPOLag")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--cmdp", action="store_true")
    p.add_argument("--no-cmdp", dest="cmdp", action="store_false")
    p.set_defaults(cmdp=True)

    # Env/PCS config
    p.add_argument(
        "--pcs",
        type=str,
        default="static:0.0",
        help=('PCS spec: "static:0.0", "sb3:/abs/model.zip", or '
              '"responsive:spread_prop?gain=0.6&clip=5"'),
    )
    p.add_argument("--preset", type=str, default="default")
    p.add_argument("--spread_limit", type=float, default=5.0)
    p.add_argument("--return_cost_in_info", action="store_true", default=True)
    p.add_argument("--max_episode_steps", type=int, default=48, help="Episode horizon.")

    # Safety & eval
    p.add_argument("--cost_limit", type=float, default=0.10)
    p.add_argument("--eval_every", type=int, default=2_000)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_mode", choices=("random", "policy", "none"), default="random")

    # IO
    p.add_argument("--save_dir", type=str, default="runs/dev")
    return p.parse_args()


def build_env_id(args) -> str:
    return "SafeISO-CMDP-omni-v0" if args.cmdp else "SafeISO-ISOOnly-omni-v0"


def get_algo_class(name: str):
    alias = {"SautePPO": "PPOSaute"}
    key = alias.get(name, name)
    if key == "PPOLag":
        from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag as A
        return A
    if key == "CUP":
        from omnisafe.algorithms.on_policy.first_order.cup import CUP as A
        return A
    if key == "CPO":
        from omnisafe.algorithms.on_policy.second_order.cpo import CPO as A
        return A
    if key == "FOCOPS":
        from omnisafe.algorithms.on_policy.first_order.focops import FOCOPS as A
        return A
    if key == "PPOSaute":
        from omnisafe.algorithms.on_policy.saute.ppo_saute import PPOSaute as A
        return A
    raise ValueError(f"Unsupported algo '{name}'")


def make_cfgs(args: argparse.Namespace, env_id: str) -> Config:
    pcs_tag = args.pcs.split('?', 1)[0].replace(':', '-').replace('/', '_')

    cfgs_dict: Dict[str, Any] = {}
    # Start from shared defaults
    deep_update(cfgs_dict, BASE)

    # Standard top-level & env
    deep_update(cfgs_dict, {
        "seed": int(args.seed),
        "device": args.device,
        "num_envs": 1,
        "exp_name": f"{args.algo}_{'cmdp' if args.cmdp else 'mdp'}_{args.preset}_{pcs_tag}",
        "train_cfgs": {
            "device": args.device,
            "total_steps": int(args.steps),
            "seed": int(args.seed),
            "log_dir": os.path.join(args.save_dir, f"{args.algo}_omni"),
        },
        "algo_cfgs": {
            "cost_limit": float(args.cost_limit),
            "steps_per_epoch": int(args.steps),
            "use_cost": bool(args.cmdp),
            "max_ep_len": int(args.max_episode_steps),
        },
        "lagrange_cfgs": {
            "cost_limit": float(args.cost_limit),
            "lagrangian_multiplier_init": 0.0,
            "lambda_lr": 0.005,
            "lambda_optimizer": "Adam",
            "lagrangian_upper_bound": None,
        },
        "env_cfgs": {
            "preset": args.preset,
            "pcs": args.pcs,
            "spread_limit": float(args.spread_limit),
            "return_cost_in_info": bool(args.return_cost_in_info),
            "max_episode_steps": int(args.max_episode_steps),
        },
        "logger_cfgs": {
            "log_dir": os.path.join(args.save_dir, f"{args.algo}_omni"),
        },
    })

    # Apply algo-specific overrides
    deep_update(cfgs_dict, algo_overrides(args.algo, horizon=args.max_episode_steps, cost_limit=args.cost_limit))

    # Validate required keys for the chosen algo
    validate_required(cfgs_dict, args.algo)

    return Config.dict2config(cfgs_dict)


def evaluate_random(env, episodes: int = 3) -> Tuple[float, float]:
    import torch
    import numpy as np
    ep_rew, ep_avg_cost = [], []
    for _ in range(max(1, episodes)):
        obs, _ = env.reset()
        done = torch.zeros((1,), dtype=torch.bool, device=obs.device)
        r = 0.0
        c = 0.0
        n = 0
        while not done.item():
            a = env.action_space.sample()
            obs, rew, cost, term, trunc, info = env.step(a)
            r += float(rew.item()) if hasattr(rew, "item") else float(rew)
            c += float(cost.item()) if hasattr(cost, "item") else float(cost)
            n += 1
            done = term | trunc
        ep_rew.append(r)
        ep_avg_cost.append(c / max(1, n))
    return float(np.mean(ep_rew)), float(np.mean(ep_avg_cost))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(json.dumps({
        "algo": args.algo, "cmdp": args.cmdp, "steps": args.steps,
        "seed": args.seed, "device": args.device, "pcs": args.pcs,
        "preset": args.preset, "cost_limit": args.cost_limit,
        "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
        "save_dir": args.save_dir
    }, indent=2), flush=True)

    env_id = build_env_id(args)
    Algo = get_algo_class(args.algo)
    cfgs = make_cfgs(args, env_id)

    # Persist resolved config for reproducibility
    try:
        with open(os.path.join(args.save_dir, "config_resolved.json"), "w") as f:
            json.dump(Config.config2dict(cfgs), f, indent=2)
    except Exception:
        pass

    # Construct algorithm; OmniSafe will build envs internally
    agent = Algo(env_id=env_id, cfgs=cfgs)
    agent.learn()

    # Write manifest
    with open(os.path.join(args.save_dir, "run_manifest.json"), "w") as f:
        json.dump({
            "algo": args.algo,
            "env_id": env_id,
            "cmdp": args.cmdp,
            "seed": args.seed,
            "pcs": args.pcs,
            "preset": args.preset,
            "cost_limit": args.cost_limit,
            "total_steps": args.steps,
        }, f, indent=2)

    # Eval
    if args.eval_mode != "none":
        # For SautePPO, the policy expects safety-augmented observations provided by the Saute adapter.
        # Use the agent's wrapped env to avoid observation-dimension mismatch.
        if args.algo in ("SautePPO", "PPOSaute"):
            eval_env = getattr(agent, "_env", None)
        else:
            eval_env = omni_make(
                env_id,
                num_envs=1,
                device=args.device,
                seed=args.seed,
                pcs=args.pcs,
                preset=args.preset,
                spread_limit=args.spread_limit,
                return_cost_in_info=args.return_cost_in_info,
                max_episode_steps=args.max_episode_steps,
            )
        policy_act = None
        if args.eval_mode == "policy":
            try:
                policy_act = make_policy_act(agent)
            except Exception as e:
                print(f"[eval] policy mode unavailable, falling back to random: {e!r}")
                policy_act = None
        metrics = rollout_metrics(eval_env, episodes=args.eval_episodes, policy_act=policy_act, deterministic=True)
        print(json.dumps({
            "eval_reward_mean": metrics["reward_mean"],
            "eval_cost_avg_mean": metrics["avg_step_cost_mean"],
            "eval_ep_len_mean": metrics["ep_len_mean"],
            "eval_violation_counts": metrics["violation_counts"],
            "eval_violation_rate_by_type": metrics["violation_rate_by_type"],
            "eval_steps_total": metrics["steps_total"],
            "eval_mode": args.eval_mode,
        }, indent=2))


if __name__ == "__main__":
    main()
