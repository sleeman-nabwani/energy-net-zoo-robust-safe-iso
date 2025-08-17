# SAFETY COSTS (Research‑Grade) for Safe ISO in EnergyNet

This document explains **exactly** how our safety cost is computed from the signals the current EnergyNet environment already provides. It describes the physics‑backed frequency surrogate, the set of hard constraints, the normalized soft penalties, and how they aggregate into the final cost used by OmniSafe algorithms (e.g., PPO‑Lagrangian, CPO).

---

## Signals used (from `info` at each step)

* `dispatch` (MW): ISO dispatch (action #3 of the ISO Box).
* `battery_action` (MW): PCS charge (−) / discharge (+) power.
* `net_demand` (MW): aggregate demand the ISO must meet.
* `battery_level` (fraction 0–1 or percent 0–100): state of charge (SoC).
* `iso_buy_price`, `iso_sell_price`, `price_spread` (optional economic term).
* We also read `action_space.high[2]` to infer the ISO **dispatch capacity** (MW). If available, we read the PCS Box high to infer the **battery power limit**.

These are sufficient to construct a frequency surrogate and practical asset‑safety penalties.

---
## Frequency model

Real power systems obey the swing equation. We use its **aggregate, per‑unit** form with frequency‑dependent load:

* Power mismatch (per‑unit):

  $\Delta P_{\text{pu}} = \frac{P_\text{dispatch} + P_\text{batt} - P_\text{demand}}{S_{\text{base}}}$

* Frequency dynamics:

  $\frac{df}{dt} = \frac{f_0}{2H}\Big(\Delta P_{\text{pu}} - D_L\,\frac{f - f_0}{f_0}\Big)$

* Discrete update per env step of size $\Delta t$:

  $f_{t+1} = f_t + \Delta t \cdot \frac{df}{dt}, \qquad \text{ROCOF} = \frac{f_{t+1}-f_t}{\Delta t}.$

**Parameters (set in config):**

* `S_base_mw` — System power base (MW). Choose dispatch cap or peak demand.
* `inertia_Hs` (seconds) — Aggregate inertia; larger → slower frequency motion.
* `load_damping_pu_per_pu` — Load damping; larger → faster recentring toward nominal.
* `seconds_per_step` — Your environment’s step time (s).

This produces **physical units**: `freq_hz` (Hz) and `rocof_hz_per_s` (Hz/s), suitable for constraints.

---

## Hard vs soft constraints

### Hard constraints (any trigger ⇒ `cost = 1.0`)

1. **Shortfall**: `shortfall > 0` (if the env reports it). When present, this indicates unmet demand and is treated as catastrophic.
2. **Reserve adequacy**: `capacity − demand ≥ r_min * demand`, where `capacity` is the ISO dispatch cap and `r_min` (e.g., 3%) is a configurable reserve margin.
3. **Frequency hard band**: resulting `freq_hz` outside a wide safety band (default: `[49.0, 51.0]` Hz for 50 Hz systems).


### Soft penalties (normalized **\[0,1]**, then weighted)

All soft components are normalized to \[0,1] so they can be combined transparently:

1. **Frequency deviation**: penalty on $|f - f_0|$ using a two‑point band.

   * Start penalizing at `freq_soft_start_hz` (e.g., 0.2 Hz → 49.8–50.2 band).
   * Saturate by `freq_soft_full_hz` (e.g., 0.5 Hz → 49.5–50.5).

2. **ROCOF**: penalty on $|df/dt|$ using a two‑point band.

   * Start at `rocof_soft_start` (e.g., 0.5 Hz/s); full by `rocof_soft_full` (e.g., 1.0 Hz/s).

3. **Dispatch ramp**: penalty on step‑to‑step dispatch change $|\Delta\text{dispatch}|$.

   * Deadband (no penalty) up to `ramp_deadband_mw` (e.g., 10 MW); saturate by `ramp_full_delta_mw` (e.g., 80 MW).

4. **SoC band**: penalty for SoC outside a longevity‑friendly band (default `[0.2, 0.8]`).

   * Normalized distance outside the band, i.e., `(distance) / (band_width)`.

5. **Battery C‑rate proxy**: $|P_\text{batt}| / P_\text{batt,max}|$.

   * Penalizes sustained operation near PCS limits (thermal/electrical stress proxy).

6. **Battery slew**: penalty on $|\Delta P_\text{batt}|$.

   * Deadband up to `batt_slew_deadband_mw` (e.g., 1 MW); saturate by `batt_slew_full_mw` (e.g., 5 MW).

7. **(Optional) Price spread**: Disabled by default. In a CMDP, the agent maximizes reward subject to an expected cost budget, and this cost is intended to reflect safety only. Set `w_spread = 0.0` for a pure safety objective. Enable this term only if you intentionally want an additional constraint that discourages operation during extreme price stress; otherwise, express economic preferences in the reward.

---

## Normalization & aggregation

Each component returns a **bounded** penalty `p_i ∈ [0,1]`. The soft cost is

$J_\text{soft} = \mathrm{clip}\!\left( \sum_i w_i\,p_i,\; 0,\; 1 \right).$

Weights `w_i` reflect priorities (physics > assets > economics by default). If any **hard** constraint triggers, the final cost is **overridden** to `1.0` for that step.

---

## Recommended starting configuration

* **Frequency bands (50 Hz)**: soft start = 0.2 Hz, soft full = 0.5 Hz, hard = \[49.0, 51.0] Hz.
* **ROCOF**: soft 0.5 Hz/s, full 1.0 Hz/s.
* **Dispatch ramp**: deadband 10 MW, full 80 MW.
* **SoC band**: 0.20–0.80 (lithium‑ion longevity proxy).
* **Battery C‑rate**: normalized by PCS power limit.
* **Battery slew**: deadband 1 MW, full 5 MW.
* **Spread (optional)**: start 3.2; small weight.

**Suggested weights (sum ≈ 1):**

* `w_fdev=0.25`, `w_rocof=0.15`, `w_ramp=0.20`,
* `w_soc=0.15`, `w_batt_crate=0.15`, `w_batt_slew=0.05`,
* `w_spread=0.05` (set to `0.0` to exclude economics entirely).

---

## Calibration checklist (do this after a smoke run)

1. Set `S_base_mw` to dispatch cap or observed peak demand.
2. Start with `inertia_Hs = 5.0 s`, `load_damping_pu_per_pu = 1.0`, `seconds_per_step = 60`.
3. Inspect prints:

   * Typical `|f − f0|` ≤ \~0.2 Hz, typical `|ROCOF|` ≤ \~0.5 Hz/s.
   * No spurious hard trips in normal operation.
   * Battery penalties (C‑rate/slew/SoC) light up only when stressed.
4. If frequency is too jumpy → **increase** `inertia_Hs`.
5. If it recenters too slowly → **increase** `load_damping_pu_per_pu`.
6. Adjust ramp/slew/SoC thresholds & weights so the **average cost** under a reasonable hand policy lands near your target budget (e.g., 0.05–0.10).

---

## How the trainer uses this

Your CMDP adapter:

1. Reads signals from `info`.
2. Computes power mismatch $\Delta P$ and calls the swing step to get `freq_hz` and `rocof_hz_per_s`.
3. Checks **hard** constraints (shortfall, reserve, frequency hard band). If any, set `cost=1.0`.
4. Computes each **soft** penalty (frequency deviation, ROCOF, dispatch ramp, SoC, battery C‑rate/slew, and optionally spread).
5. Combines with weights and clips to \[0,1].
6. Exposes `info["cost_debug"]` with raw values, bands, penalty components, weights, and the final `cost`.

OmniSafe (e.g., PPO‑Lagrangian, CPO) then optimizes reward subject to an **expected cost constraint** (e.g., `cost_limit=0.10`).

---

## Example usage (inside your adapter)

```python
from safeiso.costs import (
    soc_to_fraction, frequency_next_pu,
    penalty_freq_dev, penalty_rocof,
    penalty_dispatch_ramp, penalty_soc_band,
    penalty_batt_crate, penalty_slew, penalty_spread,
    weighted_sum_cost, clip01,
)

# power mismatch (MW)
dP = (dispatch_mw + batt_mw) - net_demand_mw

# frequency step (Hz and Hz/s)
f_next, rocof = frequency_next_pu(
    f_hz=f, f0_hz=50.0, dP_mw=dP,
    S_base_mw=base_mw, H_s=H, D_L_pu_per_pu=D_L,
    dt_s=60.0,
)

# normalized component penalties (all ∈ [0,1])
comps = {
    "freq_dev":      penalty_freq_dev(f_next - 50.0, 0.2, 0.5),
    "rocof":         penalty_rocof(rocof, 0.5, 1.0),
    "dispatch_ramp": penalty_dispatch_ramp(abs(dispatch_mw - prev_dispatch), 10.0, 80.0),
    "soc":           penalty_soc_band(soc_to_fraction(battery_level, strict=True), 0.2, 0.8),
    "batt_crate":    penalty_batt_crate(batt_mw, batt_max_mw),
    "batt_slew":     penalty_slew(abs(batt_mw - prev_batt_mw), 1.0, 5.0),
    "spread":        penalty_spread(price_spread, 3.2),  # optional
}
weights = {
    "freq_dev":0.25, "rocof":0.15, "dispatch_ramp":0.20,
    "soc":0.15, "batt_crate":0.15, "batt_slew":0.05,
    "spread":0.05,
}

cost_soft = weighted_sum_cost(comps, weights)  # -> [0,1]
# if any hard rule trips earlier, override cost to 1.0 for that step
```

---

