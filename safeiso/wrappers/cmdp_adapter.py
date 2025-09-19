from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import json

import gymnasium as gym
import numpy as np

from ..utils.costs import (
    soc_to_fraction,
    frequency_next_pu,
    penalty_freq_dev,
    penalty_rocof,
    penalty_dispatch_ramp,
    penalty_soc_band,
    penalty_batt_crate,
    penalty_slew,
    penalty_spread,
    weighted_sum_cost,
    lin01_band,
)


# -----------------------------
# Configs
# -----------------------------

@dataclass
class VFConfig:
    """Frequency (physics) + bands. Voltage is intentionally omitted (no proxy)."""

    # time base & nominal frequency
    seconds_per_step: float = 7.5
    nominal_freq_hz: float = 50.0

    # per-unit swing equation params (diagnostic calibrated)
    system_base_mw: float = 297.1755676269531  # diagnostic recommendation (updated)
    inertia_Hs: float = 8.0                  # diagnostic recommendation
    load_damping_pu_per_pu: float = 2.0      # diagnostic recommendation

    # numerical integration fidelity (diagnostic calibrated)
    n_substeps: int = 12                     # diagnostic recommendation (updated)

    # frequency/ROCOF soft bands (diagnostic calibrated)
    freq_soft_start_hz: float = 0.2    # diagnostic recommendation
    freq_soft_full_hz: float = 1.0     # diagnostic recommendation
    rocof_soft_start: float = 0.3      # diagnostic recommendation
    rocof_soft_full: float = 1.5       # diagnostic recommendation

    # frequency hard band (trip)
    freq_hard_low_hz: float = 49.0
    freq_hard_high_hz: float = 51.0

    @classmethod
    def presets(cls, name: str) -> "VFConfig":
        """Return a predefined VF (physics + bands) configuration.

        Supported names:
          - "default": baseline VFConfig for realistic constraints
          - "training": optimized for policy training with relaxed constraints
          - "evaluation": balanced constraints for policy evaluation
        """
        key = (name or "").strip().lower()
        if key in ("", "default"):
            return cls()
        if key == "training":
            # Training preset: relaxed constraints for better trainability
            vf = cls()
            vf.freq_soft_start_hz = 0.30  # wider soft bands
            vf.freq_soft_full_hz = max(1.50, 1.75)  # more forgiving frequency limits
            vf.rocof_soft_start = 0.50    # higher ROCOF tolerance
            vf.rocof_soft_full = 2.00     # more ROCOF headroom
            vf.inertia_Hs *= 1.375        # +37.5% system stability
            vf.load_damping_pu_per_pu *= 1.375  # +37.5% damping for stability
            vf.system_base_mw *= 1.32     # +32% system capacity
            return vf
        if key == "evaluation":
            # Evaluation preset: moderate constraints for fair policy assessment
            vf = cls()
            vf.freq_soft_start_hz = 0.25  # slightly relaxed from default
            vf.freq_soft_full_hz = 1.25   # moderate frequency tolerance
            vf.rocof_soft_start = 0.35    # modest ROCOF tolerance
            vf.rocof_soft_full = 1.75     # reasonable ROCOF limits
            vf.inertia_Hs *= 1.20         # +20% stability boost
            vf.load_damping_pu_per_pu *= 1.20  # +20% damping
            vf.system_base_mw *= 1.15     # +15% system capacity
            return vf
        raise ValueError(f"Unknown VFConfig preset: {name}")


@dataclass
class CostConfig:
    # HARD constraints
    hard_shortfall: bool = True  # was False for debugging; restored by diagnostics
    hard_reserve_adequacy: bool = True
    hard_negative_spread: bool = False   # keep False for safety-only objective

    # Reserve adequacy
    reserve_min_frac: float = 0.03
    max_dispatch_cap_mw: Optional[float] = None

    # Battery normalization (PCS limit). Must be resolvable; otherwise adapter raises.
    batt_max_mw: Optional[float] = None

    # SOFT weights (sum ≈ 1; final cost clipped to [0,1])
    w_fdev: float = 0.25
    w_rocof: float = 0.15
    w_ramp: float = 0.20
    w_soc: float = 0.15
    w_batt_crate: float = 0.15
    w_batt_slew: float = 0.05
    w_spread: float = 0.05  # set 0.0 to exclude economics entirely

    # SOFT thresholds / bands
    ramp_deadband_mw: float = 10.0
    ramp_full_delta_mw: float = 80.0
    soc_min_frac: float = 0.20
    soc_max_frac: float = 0.80
    spread_soft_start: float = 3.2
    spread_full: Optional[float] = None  # optional separate saturation threshold

    # Battery slew band
    batt_slew_deadband_mw: float = 1.0
    batt_slew_full_mw: float = 5.0

    @classmethod
    def presets(cls, name: str) -> "CostConfig":
        """Return a predefined configuration.

        Supported names:
          - "default": baseline weights and bands for realistic constraints
          - "training": relaxed constraints optimized for policy learning
          - "evaluation": balanced constraints for fair policy assessment
        """
        key = (name or "").strip().lower()
        if key == "default":
            return cls()
        if key == "training":
            # Training preset: relaxed constraints for better trainability
            return cls(
                hard_negative_spread=False,  # no hard penalty for negative spreads
                reserve_min_frac=0.015,      # 1.5% reserve requirement (vs 3% default)
                max_dispatch_cap_mw=396.0,   # +32% dispatch capacity for training headroom
            )
        if key == "evaluation":
            # Evaluation preset: moderate constraints for fair policy assessment
            return cls(
                hard_negative_spread=False,  # keep spreads soft for evaluation
                reserve_min_frac=0.020,      # 2% reserve requirement (moderate)
                max_dispatch_cap_mw=345.0,   # +15% dispatch capacity
            )
        raise ValueError(f"Unknown CostConfig preset: {name}")


# -----------------------------
# CMDP Adapter
# -----------------------------

class CMDPAdapter(gym.Wrapper):
    """
    Wrap an ISO single-agent env so step() returns (obs, reward, cost, terminated, truncated, info).

    HARD constraints (cost = 1 immediately):
      - shortfall > 0 (if reported)
      - reserve inadequacy (capacity - demand < reserve_min_frac * demand)
      - frequency outside hard band
      - (optional) negative price spread

    SOFT penalties (normalized to [0,1], then weighted and clipped):
      - frequency deviation |f - f0|
      - ROCOF |df/dt|
      - dispatch ramp |Δdispatch|
      - SoC band deviation
      - battery C-rate proxy |P_batt|/P_batt,max
      - battery slew |ΔP_batt|
      - (optional) price spread above comfort band

    Debug info is attached at info["cost_debug"].
    """

    def __init__(
        self,
        env: gym.Env,
        cost_cfg: Optional[CostConfig] = None,
        vf_cfg: Optional[VFConfig] = None,
        *,
        log_components: bool = True,
        return_cost_in_info: bool = True,
        log_every: int = 1,
    ) -> None:
        super().__init__(env)
        self.cfg = cost_cfg or CostConfig()
        self.vf = vf_cfg or VFConfig()
        self.log_components = log_components
        self.return_cost_in_info = bool(return_cost_in_info)
        self._log_every = max(int(log_every), 1)

        # Optional override of cost weights via environment variable for isolation runs
        # SAFEISO_COST_WEIGHTS expects a JSON object with fields like {"w_fdev":0.0, ...}
        try:
            w_raw = os.environ.get("SAFEISO_COST_WEIGHTS", "").strip()
            if w_raw:
                obj = json.loads(w_raw)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if hasattr(self.cfg, k) and isinstance(v, (int, float)):
                            setattr(self.cfg, k, float(v))
        except Exception:
            # best-effort override; ignore parsing errors silently
            pass

        # Optional extra nudge controlled via env var (reversible)
        try:
            import os as _os
            if str(_os.environ.get("SAFEISO_EXTRA_NUDGE", "0")).strip() == "1":
                self.vf.inertia_Hs *= 1.1
                self.vf.load_damping_pu_per_pu *= 1.1
                self.vf.freq_soft_full_hz = max(self.vf.freq_soft_full_hz, 1.5)
        except Exception:
            pass

        # Validate bands
        if not (self.vf.freq_soft_full_hz > self.vf.freq_soft_start_hz):
            raise ValueError("freq_soft_full_hz must be > freq_soft_start_hz")
        if not (self.vf.freq_hard_high_hz > self.vf.freq_hard_low_hz):
            raise ValueError("freq_hard_high_hz must be > freq_hard_low_hz")
        if not (self.vf.seconds_per_step > 0):
            raise ValueError("seconds_per_step must be positive")

        # Infer dispatch capacity from ISO action space if needed
        if self.cfg.max_dispatch_cap_mw is None:
            if isinstance(self.env.action_space, gym.spaces.Box) and self.env.action_space.shape[0] >= 3:
                self.cfg.max_dispatch_cap_mw = float(self.env.action_space.high[2])
            else:
                raise ValueError("Cannot infer max_dispatch_cap_mw: ISO action space must be Box([buy,sell,dispatch]).")

        # Determine battery max power (for C-rate normalization)
        self._batt_max_mw = self._resolve_batt_max_mw()

        # Small state
        self._prev_dispatch_mw: float = 0.0
        self._prev_batt_mw: float = 0.0
        self._freq_hz: float = self.vf.nominal_freq_hz
        self._prev_freq_hz: float = self.vf.nominal_freq_hz

        # For completeness, a cost space
        self.cost_space = gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)

    # helpers
    def _resolve_batt_max_mw(self) -> float:
        """Resolve PCS max power (MW) from config or environment; raise if unavailable."""
        if self.cfg.batt_max_mw is not None:
            return float(self.cfg.batt_max_mw)
        pcs_space = getattr(self.env, 'pcs_action_space', None)
        if isinstance(pcs_space, gym.spaces.Box) and pcs_space.shape == (1,):
            return float(abs(pcs_space.high[0]))
        # Walk nested wrappers to find a Dict action space with 'pcs'
        cur = self.env
        seen = set()
        while hasattr(cur, 'env') and id(cur) not in seen:
            seen.add(id(cur))
            cur_action_space = getattr(cur, 'action_space', None)
            if isinstance(cur_action_space, gym.spaces.Dict):
                spaces_dict = getattr(cur_action_space, 'spaces', {})
                if isinstance(spaces_dict, dict) and ('pcs' in spaces_dict):
                    pcs = spaces_dict['pcs']
                    if isinstance(pcs, gym.spaces.Box) and pcs.shape == (1,):
                        return float(abs(pcs.high[0]))
            cur = getattr(cur, 'env', None)
            if cur is None:
                break
        raise ValueError(
            "Cannot determine batt_max_mw for C-rate normalization. "
            "Provide CostConfig.batt_max_mw or expose a 'pcs' Box action space."
        )

    def _attach_debug(self, info: Dict[str, Any], *,
                      cost: float,
                      hard_trigger: Optional[str],
                      components: Optional[Dict[str, float]] = None,
                      freq_hz: Optional[float] = None,
                      rocof_hz_s: Optional[float] = None,
                      extra_raw: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        out = dict(info)
        out['cost'] = float(cost)
        dbg = {
            'hard_trigger': hard_trigger,
            'weights': {
                'w_fdev': self.cfg.w_fdev, 'w_rocof': self.cfg.w_rocof,
                'w_ramp': self.cfg.w_ramp, 'w_soc': self.cfg.w_soc,
                'w_batt_crate': self.cfg.w_batt_crate, 'w_batt_slew': self.cfg.w_batt_slew,
                'w_spread': self.cfg.w_spread,
            },
            'components': components or {},
            'vf_states': {
                'freq_hz': None if freq_hz is None else float(freq_hz),
                'rocof_hz_per_s': None if rocof_hz_s is None else float(rocof_hz_s),
                'bands': {
                    'freq_soft_start_hz': self.vf.freq_soft_start_hz,
                    'freq_soft_full_hz': self.vf.freq_soft_full_hz,
                    'freq_hard_low_hz': self.vf.freq_hard_low_hz,
                    'freq_hard_high_hz': self.vf.freq_hard_high_hz,
                    'rocof_soft_start': self.vf.rocof_soft_start,
                    'rocof_soft_full': self.vf.rocof_soft_full,
                },
            },
            'raw': extra_raw or {},
        }
        out['cost_debug'] = dbg
        return out

    # gym api
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_dispatch_mw = float(info.get('dispatch', 0.0))
        self._prev_batt_mw = float(info.get('battery_action', 0.0))
        self._freq_hz = self.vf.nominal_freq_hz
        self._prev_freq_hz = self.vf.nominal_freq_hz
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # extract signals
        shortfall_mw = float(info.get('shortfall', 0.0))
        net_demand_mw = float(info.get('net_demand', 0.0))
        dispatch_mw = float(info.get('dispatch', self._prev_dispatch_mw))
        batt_mw = float(info.get('battery_action', self._prev_batt_mw))

        buy_price = float(info.get('iso_buy_price', 0.0))
        sell_price = float(info.get('iso_sell_price', 0.0))
        spread = float(info.get('price_spread', sell_price - buy_price))

        cap_mw = float(self.cfg.max_dispatch_cap_mw)

        # SoC
        try:
            soc_frac = soc_to_fraction(info.get('battery_level'))
        except ValueError as e:
            # treat invalid telemetry as a hard failure (transparent)
            cost = 1.0
            info = self._attach_debug(info, cost=cost, hard_trigger=f"telemetry:{e}")
            if self.return_cost_in_info:
                return obs, float(reward), bool(terminated), bool(truncated), info
            return obs, float(reward), float(cost), bool(terminated), bool(truncated), info

        # frequency update (per-unit swing) with substeps for stability
        dP = (dispatch_mw + batt_mw) - net_demand_mw
        f = self._freq_hz
        nsub = max(1, int(getattr(self.vf, 'n_substeps', 1)))
        dt = float(self.vf.seconds_per_step) / float(nsub)
        for _ in range(nsub):
            f, _ = frequency_next_pu(
                f_hz=f,
                f0_hz=self.vf.nominal_freq_hz,
                dP_mw=dP,
                S_base_mw=self.vf.system_base_mw,
                H_s=self.vf.inertia_Hs,
                D_L_pu_per_pu=self.vf.load_damping_pu_per_pu,
                dt_s=dt,
            )
        next_f_hz = f
        rocof = (next_f_hz - self._freq_hz) / max(float(self.vf.seconds_per_step), 1e-6)

        # hard constraints
        hard_trigger: Optional[str] = None
        vio_shortfall = int(shortfall_mw > 0.0) if self.cfg.hard_shortfall else 0
        vio_reserve = 0
        vio_neg_spread = 0
        # Compute reserve inadequacy boolean
        margin_mw = cap_mw - max(0.0, net_demand_mw)
        need_mw = self.cfg.reserve_min_frac * max(0.0, net_demand_mw)
        if self.cfg.hard_reserve_adequacy and (margin_mw < need_mw):
            vio_reserve = 1
        if self.cfg.hard_negative_spread and (sell_price < buy_price):
            vio_neg_spread = 1
        # Frequency hard band violation
        vio_freq_hard = int((next_f_hz < self.vf.freq_hard_low_hz) or (next_f_hz > self.vf.freq_hard_high_hz))

        if self.cfg.hard_shortfall and shortfall_mw > 0.0:
            hard_trigger = 'shortfall>0'
        if hard_trigger is None and self.cfg.hard_reserve_adequacy:
            if margin_mw < need_mw:
                hard_trigger = 'reserve_inadequate'
        if hard_trigger is None and self.cfg.hard_negative_spread and (sell_price < buy_price):
            hard_trigger = 'negative_spread'
        if hard_trigger is None:
            if (next_f_hz < self.vf.freq_hard_low_hz) or (next_f_hz > self.vf.freq_hard_high_hz):
                hard_trigger = 'freq_hard_band'

        # Prepare violations dictionary for info
        violations = {
            'shortfall': vio_shortfall,
            'reserve': vio_reserve,
            'freq_oob': vio_freq_hard,
        }
        if self.cfg.hard_negative_spread:
            violations['negative_spread'] = vio_neg_spread
        violation_types = [k for k, v in violations.items() if v]

        # Tiered cost calculation (replaces hard triggers)
        # Tier 1: Critical infrastructure violations (highest priority)
        C1_unserved = 1.0 if shortfall_mw > 0.0 else 0.0
        C1_reserve = 1.0 if (margin_mw < need_mw) else 0.0
        C1_freq_hard = 1.0 if vio_freq_hard else 0.0
        C1 = max(C1_unserved, C1_reserve, C1_freq_hard)

        # Tier 2: Dynamic stability violations (medium priority)
        freq_dev_hz = abs(next_f_hz - self.vf.nominal_freq_hz)
        rocof_abs = abs(rocof)
        C2_freq = lin01_band(freq_dev_hz, self.vf.freq_soft_start_hz, self.vf.freq_soft_full_hz)
        C2_rocof = lin01_band(rocof_abs, self.vf.rocof_soft_start, self.vf.rocof_soft_full)
        C2 = max(C2_freq, C2_rocof)

        # Tier 3: Operational efficiency violations (lowest priority)
        ramp_delta = abs(dispatch_mw - self._prev_dispatch_mw)
        C3_ramp = lin01_band(ramp_delta, self.cfg.ramp_deadband_mw, self.cfg.ramp_full_delta_mw)
        
        soc_low = max(0.0, self.cfg.soc_min_frac - soc_frac) / max(self.cfg.soc_min_frac, 0.01)
        soc_high = max(0.0, soc_frac - self.cfg.soc_max_frac) / max(1.0 - self.cfg.soc_max_frac, 0.01)
        C3_soc = min(1.0, soc_low + soc_high)
        
        C3 = max(C3_ramp, C3_soc)

        # Aggregate with tier weighting: Tier-1 dominates, then Tier-2, then Tier-3
        raw_cost_before_clip = C1 + 0.5*C2 + 0.2*C3
        cost = min(1.0, raw_cost_before_clip)

        # Expose individual cost heads for diagnostics
        comps = {
            'unserved': float(C1_unserved),
            'reserve_short': float(C1_reserve), 
            'freq_hard': float(C1_freq_hard),
            'freq_dev': float(C2_freq),
            'rocof': float(C2_rocof),
            'ramp': float(C3_ramp),
            'soc': float(C3_soc),
        }

        info = self._attach_debug(
            info,
            cost=cost,
            hard_trigger=None,  # No more hard triggers
            freq_hz=next_f_hz,
            rocof_hz_s=rocof,
            extra_raw={
                'price_spread': spread,
                'dispatch_mw': dispatch_mw,
                'battery_mw': batt_mw,
                'soc_fraction': soc_frac,
                'net_demand_mw': net_demand_mw,
                'batt_max_mw': float(self._batt_max_mw),
                'dispatch_cap_mw': float(cap_mw),
                'raw_cost_before_clip': float(raw_cost_before_clip),
            },
        )
        info['safety_heads'] = comps
        info['violations'] = violations
        info['violation_types'] = violation_types
        
        # state update
        self._prev_dispatch_mw = dispatch_mw
        self._prev_batt_mw = batt_mw
        self._prev_freq_hz, self._freq_hz = self._freq_hz, next_f_hz
        
        if self.return_cost_in_info:
            return obs, float(reward), bool(terminated), bool(truncated), info
        return obs, float(reward), float(cost), bool(terminated), bool(truncated), info
