from __future__ import annotations
from typing import Dict, Any


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


BASE: Dict[str, Any] = {
    "train_cfgs": {
        "epochs": 1,
        "vector_env_nums": 1,
    },
    "algo_cfgs": {
        "gamma": 0.99,
        "lam": 0.95,
        "lam_c": 0.95,
        "adv_estimation_method": "gae",
        "standardized_rew_adv": False,
        "standardized_cost_adv": False,
        "penalty_coef": 0.0,
        "update_epochs": 10,
        "update_iters": 10,
        "batch_size": 2048,
        "clip": 0.2,
        "entropy_coef": 0.0,
        "target_kl": 0.01,
        "kl_early_stop": False,
        "use_max_grad_norm": False,
        "max_grad_norm": 0.5,
        "use_critic_norm": False,
        "critic_norm_coef": 0.0,
        # Enable observation normalization by default for stability
        "obs_normalize": True,
        "reward_normalize": False,
        "cost_normalize": False,
    },
    "model_cfgs": {
        "exploration_noise_anneal": False,
        "weight_initialization_mode": "xavier_uniform",
        "actor_type": "gaussian_learning",
        "critic_type": "value",
        "linear_lr_decay": False,
        "actor": {"hidden_sizes": [64, 64], "activation": "tanh", "lr": 3e-4},
        "critic": {"hidden_sizes": [64, 64], "activation": "tanh", "lr": 3e-4},
    },
    "logger_cfgs": {
        "use_tensorboard": False,
        "use_wandb": False,
        "save_model_freq": 1,
    },
}


REQ_BY_ALGO = {
    "PPOLag":   [],
    "CUP":      [],
    "FOCOPS":   ["algo_cfgs.focops_eta", "algo_cfgs.focops_lam"],
    "CPO":      ["algo_cfgs.cg_iters", "algo_cfgs.cg_damping", "algo_cfgs.fvp_sample_freq"],
    "SautePPO": ["algo_cfgs.safety_budget", "algo_cfgs.saute_gamma", "algo_cfgs.max_ep_len", "algo_cfgs.unsafe_reward"],
    "PPOSaute": ["algo_cfgs.safety_budget", "algo_cfgs.saute_gamma", "algo_cfgs.max_ep_len", "algo_cfgs.unsafe_reward"],
}


def algo_overrides(name: str, *, horizon: int, cost_limit: float) -> Dict[str, Any]:
    # PPOLag: Add robust configuration for Lagrangian methods to prevent NaN errors
    if name == "PPOLag":
        # Special handling for very short training runs (< 100 steps)
        if horizon < 100:
            return {
                "algo_cfgs": {
                    # Ultra-conservative settings for short runs
                    "lagrange_init": 0.0,        # Start with zero multiplier
                    "lagrange_lr": 0.0,          # Disable learning entirely for very short runs
                    "lagrange_max": 0.1,         # Very low maximum value
                    "cost_limit": 1.0,           # Very relaxed limit
                    "use_cost": False,           # Disable cost constraint entirely
                    
                    # Minimal updates to prevent NaN
                    "update_epochs": 1,          # Single update epoch
                    "batch_size": max(32, horizon),  # Small batch size
                    "clip": 0.5,                 # Large clipping for stability
                    "target_kl": 0.1,           # High KL tolerance
                    
                    # Disable problematic features
                    "cost_normalize": False,     # Skip normalization for short runs
                    "obs_normalize": False,      # Skip observation normalization
                    "reward_normalize": False,   # Skip reward normalization
                },
                "lagrange_cfgs": {
                    "cost_limit": 1.0,
                    "lagrangian_multiplier_init": 0.0,
                    "lambda_lr": 0.0,            # Disable lambda learning
                    "lagrangian_upper_bound": 0.1,
                }
            }
        else:
            return {
                "algo_cfgs": {
                    # Standard robust configuration for normal runs
                    "lagrange_init": 0.001,      # Start with very small multiplier
                    "lagrange_lr": 0.0001,       # Conservative learning rate
                    "lagrange_max": 10.0,        # Cap maximum value
                    
                    # Cost normalization for numerical stability
                    "cost_normalize": True,       # Critical for preventing NaN
                    "cost_gamma": 0.99,          # Discount for cost advantages
                    
                    # Training stability improvements
                    "update_epochs": min(8, max(3, horizon // 2000)),  # Adaptive update epochs
                    "batch_size": min(1024, max(64, horizon // 4)),    # Adaptive batch size
                    "clip": 0.2,                 # Standard clipping
                    "target_kl": 0.02,          # Slightly higher for robustness
                    
                    # Normalization settings
                    "obs_normalize": True,       # Essential for stability
                    "reward_normalize": False,   # Keep rewards unnormalized
                },
                "lagrange_cfgs": {
                    "cost_limit": float(cost_limit),
                    "lagrangian_multiplier_init": 0.001,
                    "lambda_lr": 0.0001,
                    "lagrangian_upper_bound": 10.0,
                }
            }
    
    if name == "FOCOPS":
        return {
            "algo_cfgs": {
                "focops_eta": 0.02, 
                "focops_lam": 1.0,
                # Add stability improvements
                "cost_normalize": True,
                "obs_normalize": True,
            },
            "lagrange_cfgs": {
                "cost_limit": float(cost_limit),
                "lagrangian_multiplier_init": 0.0,
                "lambda_lr": 0.005,
            }
        }
    
    if name == "CPO":
        return {
            "algo_cfgs": {
                "cg_iters": 10, 
                "cg_damping": 0.1, 
                "fvp_sample_freq": 1,
                # Add CPO-specific stability settings
                "target_kl": 0.02,          # Slightly higher for robustness
                "cost_normalize": True,     # Normalize cost values
                "obs_normalize": True,      # Observation normalization
            }
        }
    
    if name == "CUP":
        return {
            "algo_cfgs": {
                # CUP-specific settings for stability
                "cost_normalize": True,      # Normalize cost statistics
                "obs_normalize": True,       # Observation normalization
                "target_kl": 0.02,          # Conservative KL target
                "clip": 0.2,                # Standard clipping
            }
        }
    
    if name in ("SautePPO", "PPOSaute"):
        budget = float(cost_limit) * int(horizon)
        return {
            "algo_cfgs": {
                "safety_budget": budget,
                "saute_gamma": 0.99,
                "max_ep_len": int(horizon),
                "unsafe_reward": 0.0,
                # Explicitly keep obs normalization on for Sauté
                "obs_normalize": True,
                "reward_normalize": False,
                "cost_normalize": False,
            },
            # optional mirror for adapter variants; harmless if unused
            "saute_cfgs": {"safety_budget": budget},
        }
    
    return {}


def validate_required(cfg: Dict[str, Any], name: str) -> None:
    import functools, operator

    def getp(d, path):
        keys = path.split(".")
        try:
            return functools.reduce(operator.getitem, keys, d)
        except Exception:
            raise KeyError(path)

    missing = []
    for key in REQ_BY_ALGO.get(name, []):
        try:
            getp(cfg, key)
        except KeyError:
            missing.append(key)
    if missing:
        raise KeyError(f"[{name}] missing required config keys: {', '.join(missing)}")
