"""
Research-grade cost primitives for Safe ISO in EnergyNet.

This module provides:
  - Helpers (clipping, band mappers, SoC parsing)
  - A per-unit aggregate swing-equation step for frequency & ROCOF
  - Normalized [0,1] penalties for each soft safety component

Design choices:
  * Frequency is computed from a standard aggregate swing model in per-unit
    using ONLY signals already available in EnergyNet: dispatch, battery_action,
    and net_demand.
  * Each penalty returns a number in [0,1]; combine them with weights and clip.
"""

from __future__ import annotations
from typing import Tuple, Dict

# utilities
def clip01(x: float) -> float:
    """Clamp to [0,1]."""
    return float(min(max(x, 0.0), 1.0))


def lin01_band(x_abs: float, start: float, full: float) -> float:
    """
    Map |x| to [0,1]. 0 when |x|<=start, linear to 1.0 by |x|>=full.

    Args:
        x_abs:    a value whose absolute magnitude determines penalty.
        start:    begin penalizing once |x| > start (no penalty at/below start).
        full:     saturate (penalty=1) once |x| >= full.

    Returns:
        float in [0,1].
    """
    d = abs(float(x_abs))
    if d <= start:
        return 0.0
    if full <= start:
        raise ValueError(f"lin01_band requires full>start (got start={start}, full={full}).")
    if d >= full:
        return 1.0
    return (d - start) / (full - start)


def soc_to_fraction(raw_soc: float | int | None, assume_percent_if_gt: float = 1.0) -> float:
    """
    Convert a battery SoC reading to a fraction in [0,1].

    Policy:
      - If value > assume_percent_if_gt (default 1.0), interpret as percent (0..100).
      - Invalid/missing/out-of-range inputs raise ValueError.

    Args:
        raw_soc:                input value (fraction or percent).
        assume_percent_if_gt:   threshold above which units are treated as percent.

    Returns:
        soc in [0,1]
    """
    if raw_soc is None:
        raise ValueError("soc_to_fraction: missing battery_level.")
    try:
        s = float(raw_soc)
    except Exception:
        raise ValueError(f"soc_to_fraction: cannot parse battery_level={raw_soc!r}.")

    # Unit interpretation
    if s > assume_percent_if_gt:
        if s <= 100.0:
            s = s / 100.0
        else:
            raise ValueError(f"soc_to_fraction: percent out of range: {s}")

    # Final range check / clamp
    if not (0.0 <= s <= 1.0):
        raise ValueError(f"soc_to_fraction: fraction out of range: {s}")
    return clip01(s)


# Frequency model (per-unit aggregate swing equation)
def frequency_next_pu(
    f_hz: float,
    f0_hz: float,
    dP_mw: float,            # P_dispatch + P_batt - P_demand (MW)
    S_base_mw: float,        # system power base (MW) e.g., dispatch cap or peak demand
    H_s: float,              # total inertia seconds (area aggregate)
    D_L_pu_per_pu: float,    # load damping: pu power per pu frequency deviation
    dt_s: float,
) -> Tuple[float, float]:
    """
    Aggregate swing with frequency-dependent load:
        ΔP_pu = dP_mw / S_base_mw
        df/dt = (f0 / (2 H_s)) * (ΔP_pu - D_L * ((f - f0)/f0))
        f_next = f + dt * df/dt
        ROCOF  = (f_next - f) / dt

    All units are physical (Hz, Hz/s). Choose S_base_mw, H_s, D_L, and dt_s
    so that ROCOF and |f-f0| fall in realistic ranges for your scenario.

    Returns:
        (f_next_hz, rocof_hz_per_s)
    """
    DeltaP_pu = float(dP_mw) / max(S_base_mw, 1e-6)
    Delta_f   = float(f_hz) - float(f0_hz)
    dfdt = (f0_hz / max(2.0 * H_s, 1e-6)) * (DeltaP_pu - D_L_pu_per_pu * (Delta_f / max(f0_hz, 1e-6)))
    f_next = float(f_hz) + float(dt_s) * dfdt
    rocof  = (f_next - float(f_hz)) / max(float(dt_s), 1e-6)
    return float(f_next), float(rocof)


# Component penalties (all return [0,1])
def penalty_spread(spread: float, soft_start: float) -> float:
    """
    Price spread penalty (optional, economic). Penalize only above soft_start.

    Args:
        spread:      price_spread (sell - buy).
        soft_start:  threshold where penalty begins.
    """
    if spread <= soft_start:
        return 0.0
    return clip01((spread - soft_start) / max(soft_start, 1e-6))


def penalty_dispatch_ramp(delta_mw: float, deadband_mw: float, full_delta_mw: float) -> float:
    """
    Dispatch ramp penalty: 0 inside deadband, 1 by full_delta_mw.

    Args:
        delta_mw:       |dispatch_t - dispatch_{t-1}| (MW).
        deadband_mw:    no penalty up to this change.
        full_delta_mw:  penalty saturates at this change or above.
    """
    if delta_mw <= deadband_mw:
        return 0.0
    return clip01((delta_mw - deadband_mw) / max(full_delta_mw - deadband_mw, 1e-6))


def penalty_soc_band(soc_frac: float, s_min: float, s_max: float) -> float:
    """
    SoC band penalty: distance outside [s_min, s_max] normalized by band width.

    Args:
        soc_frac:  SoC fraction in [0,1].
        s_min:     lower bound.
        s_max:     upper bound.
    """
    if s_min <= soc_frac <= s_max:
        return 0.0
    dist = max(s_min - soc_frac, 0.0, soc_frac - s_max)
    return clip01(dist / max(s_max - s_min, 1e-6))


def penalty_freq_dev(f_dev_hz: float, start: float, full: float) -> float:
    """
    Frequency deviation penalty: |f - f0| mapped with a two-point band.
    """
    return lin01_band(f_dev_hz, start, full)


def penalty_rocof(rocof_hz_s: float, start: float, full: float) -> float:
    """
    ROCOF penalty: |df/dt| mapped with a two-point band.
    """
    return lin01_band(rocof_hz_s, start, full)


def penalty_batt_crate(batt_mw: float, batt_max_mw: float) -> float:
    """
    Battery C-rate proxy: normalize absolute battery power by PCS limit.

    Args:
        batt_mw:      battery_action (MW), discharge positive (by convention).
        batt_max_mw:  PCS power capability (MW) for normalization.

    Returns:
        clip01(|batt_mw| / batt_max_mw)
    """
    return clip01(abs(batt_mw) / max(batt_max_mw, 1e-6))


def penalty_slew(delta_batt_mw: float, deadband_mw: float, full_mw: float) -> float:
    """
    Battery power slew penalty: 0 inside deadband, 1 by full_mw.

    Args:
        delta_batt_mw:  |battery_action_t - battery_action_{t-1}| (MW).
        deadband_mw:    no penalty up to this change.
        full_mw:        penalty saturates at this change or above.
    """
    if delta_batt_mw <= deadband_mw:
        return 0.0
    return clip01((delta_batt_mw - deadband_mw) / max(full_mw - deadband_mw, 1e-6))


# aggregation helper
def weighted_sum_cost(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Combine normalized component penalties with weights and clip to [0,1].

    Args:
        components:  dict of component_name -> penalty in [0,1]
        weights:     dict of component_name -> nonnegative weight

    Returns:
        final cost in [0,1]
    """
    total = 0.0
    for name, pen in components.items():
        w = float(weights.get(name, 0.0))
        # be defensive if caller forgot to clamp a component
        p = clip01(float(pen))
        total += w * p
    return clip01(total)


__all__ = [
    "clip01", "lin01_band", "soc_to_fraction",
    "frequency_next_pu",
    "penalty_spread", "penalty_dispatch_ramp", "penalty_soc_band",
    "penalty_freq_dev", "penalty_rocof",
    "penalty_batt_crate", "penalty_slew",
    "weighted_sum_cost",
]
