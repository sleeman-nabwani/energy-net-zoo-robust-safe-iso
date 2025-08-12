# safeiso/make_env.py
from __future__ import annotations
from typing import Optional
import gymnasium as gym
import importlib
import energy_net.env.register_envs

from ..wrappers.iso_env_wrapper import IsoEnvWrapper
from ..wrappers.cmdp_adapter import CMDPAdapter, CostConfig, VFConfig
from ..pcs.pcs_spec import build_pcs_policy, parse_pcs_spec


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
) -> gym.Env:
    env = make_iso_single_agent_env(seed=seed, pcs=pcs)
    cfg = CostConfig.presets(preset or "default") if hasattr(CostConfig, "presets") else CostConfig()
    if spread_limit is not None:
        cfg.spread_soft_start = float(spread_limit)
    env = CMDPAdapter(env, cost_cfg=cfg, vf_cfg=VFConfig(), return_cost_in_info=return_cost_in_info)
    return env
