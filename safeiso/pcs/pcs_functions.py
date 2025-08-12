# safeiso/pcs/pcs_functions.py
from __future__ import annotations
from typing import Callable, Any, Optional
import numpy as np

from ..utils.costs import soc_to_fraction

# Helpers

def _get_soc_fraction(info: dict, default: float = 0.5) -> float:
    """Read SoC from info; fall back to `default` if not present/invalid."""
    try:
        return float(soc_to_fraction(info.get("battery_level")))
    except Exception:
        return float(default)

def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# Factories (each returns a policy core: (obs_pcs, info, action_space) -> float)

def spread_prop_factory(gain: float = 0.5, clip: Optional[float] = None,
                        invert_sign: bool = False) -> Callable:
    """
    Proportional to price spread (sell - buy):
      power_raw = gain * spread
    `clip` (if set) limits |power_raw| before the wrapper clips to Box.
    Set `invert_sign=True` if your env uses the opposite sign convention.
    """
    clip = float(abs(clip)) if clip is not None else None

    def f(_obs_pcs, info, _space):
        spread = _coerce_float(
            info.get("price_spread", info.get("iso_sell_price", 0.0) - info.get("iso_buy_price", 0.0)),
            0.0,
        )
        raw = gain * spread
        if invert_sign:
            raw = -raw
        if clip is not None:
            raw = np.clip(raw, -clip, clip)
        return float(raw)
    return f


def soc_hold_factory(low: float = 0.25, high: float = 0.80,
                     gain: float = 6.0, invert_sign: bool = False) -> Callable:
    """
    Bang-bang with deadband:
      if SoC < low  -> charge  (+gain)
      if SoC > high -> discharge(-gain)
      else          -> 0.0
    """
    low = float(low); high = float(high); gain = float(gain)

    def f(_obs_pcs, info, _space):
        soc = _get_soc_fraction(info, default=0.5)
        if soc < low:
            raw = +gain
        elif soc > high:
            raw = -gain
        else:
            raw = 0.0
        if invert_sign:
            raw = -raw
        return float(raw)
    return f


def soc_track_factory(target: float = 0.50, k: float = 12.0,
                      invert_sign: bool = False, saturate: Optional[float] = None) -> Callable:
    """
    Proportional SoC tracker:
      error = target - SoC; power_raw = k * error
    `saturate` caps |power_raw| before Box clipping (optional).
    """
    target = float(target); k = float(k)
    sat = float(abs(saturate)) if saturate is not None else None

    def f(_obs_pcs, info, _space):
        soc = _get_soc_fraction(info, default=target)
        raw = k * (target - soc)   
        if invert_sign:
            raw = -raw
        if sat is not None:
            raw = np.clip(raw, -sat, sat)
        return float(raw)
    return f


def net_follow_factory(k: float = 0.2, use_dispatch: bool = True,
                       invert_sign: bool = False, saturate: Optional[float] = None) -> Callable:
    """
    Counteract net load with storage (simple grid support):
      mismatch = net_demand - (dispatch if use_dispatch else 0)
      power_raw = -k * mismatch   # oppose mismatch
    """
    k = float(k)
    sat = float(abs(saturate)) if saturate is not None else None

    def f(_obs_pcs, info, _space):
        net = _coerce_float(info.get("net_demand"), 0.0)
        disp = _coerce_float(info.get("dispatch"), 0.0) if use_dispatch else 0.0
        mismatch = net - disp
        raw = -k * mismatch
        if invert_sign:
            raw = -raw
        if sat is not None:
            raw = np.clip(raw, -sat, sat)
        return float(raw)
    return f


def ramp_smoother_factory(alpha: float = 0.5, max_step: Optional[float] = None,
                          invert_sign: bool = False) -> Callable:
    """
    Gentle first-order smoother on ISO dispatch ramps.
    Uses internal state to oppose fast changes:
      raw_t = -alpha * (dispatch_t - dispatch_{t-1})
    `max_step` caps |raw_t| per call (optional).
    """
    alpha = float(alpha)
    cap = float(abs(max_step)) if max_step is not None else None
    prev_dispatch = {"val": None}  # tiny closure state

    def f(_obs_pcs, info, _space):
        cur = _coerce_float(info.get("dispatch"), 0.0)
        prev = prev_dispatch["val"]
        prev_dispatch["val"] = cur
        if prev is None:
            # first call: no ramp info yet
            return 0.0
        ramp = cur - prev
        raw = -alpha * ramp
        if invert_sign:
            raw = -raw
        if cap is not None:
            raw = np.clip(raw, -cap, cap)
        return float(raw)
    return f


# Optional: a simple combiner for quick experiments

def blend_factory(main: Callable, aux: Callable, w_main: float = 0.7, w_aux: float = 0.3,
                  saturate: Optional[float] = None) -> Callable:
    """
    Linearly combine two cores already constructed:
      power_raw = w_main * main(...) + w_aux * aux(...)
    """
    w_main = float(w_main); w_aux = float(w_aux)
    sat = float(abs(saturate)) if saturate is not None else None

    def f(obs_pcs, info, action_space):
        raw = w_main * float(main(obs_pcs, info, action_space)) + \
              w_aux  * float(aux(obs_pcs, info, action_space))
        if sat is not None:
            raw = np.clip(raw, -sat, sat)
        return float(raw)
    return f
