from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
)


# -----------------------------
# Configs
# -----------------------------

@dataclass
class VFConfig:
    """Frequency (physics) + bands. Voltage is intentionally omitted (no proxy)."""

    # time base & nominal frequency
    seconds_per_step: float = 60.0
    nominal_freq_hz: float = 50.0

    # per-unit swing equation params
    system_base_mw: float = 300.0      # set to dispatch cap or observed peak demand
    inertia_Hs: float = 5.0            # aggregate inertia seconds
    load_damping_pu_per_pu: float = 1.0

    # frequency/ROCOF soft bands (defaults for 50 Hz)
    freq_soft_start_hz: float = 0.2    # start penalizing at ±0.2 Hz (49.8–50.2)
    freq_soft_full_hz: float = 0.5     # full penalty by ±0.5 Hz (49.5–50.5)
    rocof_soft_start: float = 0.5      # Hz/s
    rocof_soft_full: float = 1.0       # Hz/s

    # frequency hard band (trip)
    freq_hard_low_hz: float = 49.0
    freq_hard_high_hz: float = 51.0


@dataclass
class CostConfig:
    # HARD constraints
    hard_shortfall: bool = True
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
        """Return a predefined configuration. Minimal implementation to support tests.

        Supported names:
          - "default": baseline weights and bands
          - "no_spread": exclude economic spread component (safety-only)
          - "strict": slightly stricter reserve requirement
        """
        key = (name or "").strip().lower()
        if key == "default":
            return cls()
        if key == "no_spread":
            return cls(w_spread=0.0)
        if key == "strict":
            return cls(reserve_min_frac=0.05)
        raise ValueError(f"Unknown CostConfig preset: {name}")


# -----------------------------
# CMDP Adapter
# -----------------------------

class CMDPAdapter(gym.Wrapper):
    """
    Wrap an ISO single-agent env so step() returns (obs, reward, cost, terminated, truncated, info).

    HARD constraints (cost = 1 immediately):
      - shortfall > 0 (if reported)
      - reserve inadequacy (capacity − demand < reserve_min_frac * demand)
      - frequency outside hard band
      - (optional) negative price spread

    SOFT penalties (normalized to [0,1], then weighted and clipped):
      - frequency deviation |f − f0|
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

        # frequency update (per-unit swing)
        dP = (dispatch_mw + batt_mw) - net_demand_mw
        next_f_hz, rocof = frequency_next_pu(
            f_hz=self._freq_hz,
            f0_hz=self.vf.nominal_freq_hz,
            dP_mw=dP,
            S_base_mw=self.vf.system_base_mw,
            H_s=self.vf.inertia_Hs,
            D_L_pu_per_pu=self.vf.load_damping_pu_per_pu,
            dt_s=self.vf.seconds_per_step,
        )

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

        if hard_trigger is not None:
            cost = 1.0
            info = self._attach_debug(
                info,
                cost=cost,
                hard_trigger=hard_trigger,
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
                },
            )
            info['violations'] = violations
            info['violation_types'] = violation_types
            
            # Diagnostic fields for hard trigger case too
            info['raw_cost_before_clip'] = 1.0  # Hard triggers set cost to 1.0 directly
            if 'pcs_action' not in info:
                try:
                    info['pcs_action'] = float(getattr(self, 'last_pcs_action', 0.0))
                except Exception:
                    pass
            # state update
            self._prev_dispatch_mw = dispatch_mw
            self._prev_batt_mw = batt_mw
            self._prev_freq_hz, self._freq_hz = self._freq_hz, next_f_hz
            if self.return_cost_in_info:
                return obs, float(reward), bool(terminated), bool(truncated), info
            return obs, float(reward), float(cost), bool(terminated), bool(truncated), info

        # soft components
        comps: Dict[str, float] = {}
        comps['freq_dev'] = penalty_freq_dev(next_f_hz - self.vf.nominal_freq_hz,
                                             self.vf.freq_soft_start_hz,
                                             self.vf.freq_soft_full_hz)
        comps['rocof'] = penalty_rocof(rocof, self.vf.rocof_soft_start, self.vf.rocof_soft_full)
        comps['dispatch_ramp'] = penalty_dispatch_ramp(abs(dispatch_mw - self._prev_dispatch_mw),
                                                       self.cfg.ramp_deadband_mw,
                                                       self.cfg.ramp_full_delta_mw)
        comps['soc'] = penalty_soc_band(soc_frac, self.cfg.soc_min_frac, self.cfg.soc_max_frac)
        comps['batt_crate'] = penalty_batt_crate(batt_mw, float(self._batt_max_mw))
        comps['batt_slew'] = penalty_slew(abs(batt_mw - self._prev_batt_mw),
                                           self.cfg.batt_slew_deadband_mw,
                                           self.cfg.batt_slew_full_mw)
        comps['spread'] = penalty_spread(spread, self.cfg.spread_soft_start, self.cfg.spread_full)

        weights = {
            'freq_dev': self.cfg.w_fdev,
            'rocof': self.cfg.w_rocof,
            'dispatch_ramp': self.cfg.w_ramp,
            'soc': self.cfg.w_soc,
            'batt_crate': self.cfg.w_batt_crate,
            'batt_slew': self.cfg.w_batt_slew,
            'spread': self.cfg.w_spread,
        }

        raw_cost = weighted_sum_cost(comps, weights)
        
        # Keep a copy of the pre-clip value for diagnostics
        raw_cost_before_clip = float(raw_cost)
        cost = np.clip(raw_cost, 0.0, 1.0).astype(np.float32)

        # attach debug
        info = self._attach_debug(
            info,
            cost=cost,
            hard_trigger=None,
            components=comps,
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
            },
        )
        
        # Ensure violation flags exist (booleans) - already computed above
        vio = info.get('violations', {})
        vio.setdefault('shortfall', bool(vio_shortfall))
        vio.setdefault('freq_oob', bool(vio_freq_hard))  
        vio.setdefault('reserve_violation', bool(vio_reserve))
        info['violations'] = vio
        info['violation_types'] = violation_types
        
        # Diagnostic fields
        info['raw_cost_before_clip'] = raw_cost_before_clip
        # pcs_action: the scalar actually used by the env this step (if available)
        if 'pcs_action' not in info:
            try:
                info['pcs_action'] = float(getattr(self, 'last_pcs_action', 0.0))
            except Exception:
                pass

        # state update
        self._prev_dispatch_mw = dispatch_mw
        self._prev_batt_mw = batt_mw
        self._prev_freq_hz, self._freq_hz = self._freq_hz, next_f_hz

        # Return 5-tuple when cost is in info, otherwise 6-tuple with positional cost
        if self.return_cost_in_info:
            return obs, float(reward), bool(terminated), bool(truncated), info
        # SB3-style (positional cost available)
        return obs, float(reward), float(cost), bool(terminated), bool(truncated), info