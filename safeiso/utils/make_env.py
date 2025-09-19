# safeiso/make_env.py
from __future__ import annotations
from typing import Optional
import gymnasium as gym
import importlib
import warnings

# Suppress Gym deprecation warnings from EnergyNet dependency
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Gym has been unmaintained since 2022")
    warnings.filterwarnings("ignore", message=".*Gym.*unmaintained.*")
    import energy_net.env.register_envs

from ..wrappers.iso_env_wrapper import IsoEnvWrapper
from ..wrappers.cmdp_adapter import CMDPAdapter, CostConfig, VFConfig
from ..pcs.pcs_spec import build_pcs_policy, parse_pcs_spec
from .config_validate import load_and_validate_configs, debug_print_summary


class RewardScalingWrapper(gym.Wrapper):
    """Wrapper to scale rewards for evaluation consistency with training.
    
    This ensures that models trained with reward scaling (1e6 factor)
    are evaluated in the same reward scale, preventing performance degradation.
    """
    
    def __init__(self, env: gym.Env, scale_factor: float = 1e6):
        super().__init__(env)
        self.scale_factor = scale_factor
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Scale reward to match training environment
        scaled_reward = float(reward) / self.scale_factor
        # Store original reward for debugging
        info["unscaled_reward"] = float(reward)
        info["reward_scale_factor"] = self.scale_factor
        return obs, scaled_reward, terminated, truncated, info


def _resolve_factory(env_id: str):
    spec = gym.spec(env_id)
    entry = spec.entry_point
    if isinstance(entry, str):
        module_name, obj_name = entry.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    return entry


def _resolve_pcs_from_spec(pcs: str):
    kind, kw = parse_pcs_spec(pcs)
    # use unified builder to avoid drift
    return build_pcs_policy(pcs)


def make_iso_single_agent_env(*, seed: int = 0, pcs: str = "static:0.0") -> gym.Env:
    # loader_shim: normalizes legacy YAML fields and validates configs before env init
    try:
        env_cfg = load_and_validate_configs("configs/environment_config.yaml")
        # Optional: log a concise summary
        from safeiso.utils.logging_config import get_logger
        logger = get_logger("utils.make_env")
        logger.debug("[safeiso] config_ok " + debug_print_summary(env_cfg))
    except Exception as e:
        raise ValueError(f"Config validation failed before env init: {e}")

    factory = _resolve_factory("EnergyNetEnv-v0")
    base = factory()
    base.reset(seed=seed)
    pcs_policy = _resolve_pcs_from_spec(pcs)
    wrapped = IsoEnvWrapper(base, pcs_policy)
    return gym.wrappers.PassiveEnvChecker(wrapped)


def make_iso_cmdp_basic(
    *,
    seed: int = 0,
    pcs: str = "static:0.0",
    preset: Optional[str] = None,
    spread_limit: Optional[float] = None,
    return_cost_in_info: bool = True,
    use_reward_scaling: bool = True,
) -> gym.Env:
    env = make_iso_single_agent_env(seed=seed, pcs=pcs)
    cfg = CostConfig.presets(preset or "default") if hasattr(CostConfig, "presets") else CostConfig()
    if spread_limit is not None:
        cfg.spread_soft_start = float(spread_limit)
    # Build VF config (use adapter presets where available)
    key = (preset or "").strip().lower()
    try:
        vf = VFConfig.presets(key) if hasattr(VFConfig, 'presets') else VFConfig()
    except Exception:
        vf = VFConfig()
    # Relax negative spread hard head for calibration runs via CostConfig preset
    env = CMDPAdapter(env, cost_cfg=cfg, vf_cfg=vf, return_cost_in_info=return_cost_in_info)
    
    # Apply reward scaling for evaluation consistency with training
    # This ensures models trained with scaled rewards work correctly during evaluation
    if use_reward_scaling:
        env = RewardScalingWrapper(env, scale_factor=1e6)
    
    return env
