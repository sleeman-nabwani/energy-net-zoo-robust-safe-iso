
---

# EnergyNet Environment Notes

## What we’re using

* **Base class:** `EnergyNetV0` (module: `energy_net.env`)
* **Multi-agent:** yes — two agents with **lowercase** keys:

  * Agent IDs: `'iso'`, `'pcs'`
* **Gym IDs registered by the package (FYI):**

  * `PCSUnitEnv-v0`, `ISOEnv-v0`, `EnergyNetEnv-v0`, `ISO-RLZoo-v0`, `PCS-RLZoo-v0`
    *(We construct `EnergyNetV0` directly rather than relying on these IDs.)*

## Typical constructor (working on this fork)

```python
from energy_net.env import EnergyNetV0

env = EnergyNetV0(
    pricing_policy="ONLINE",
    demand_pattern="SINUSOIDAL",
    cost_type="CONSTANT",
    num_pcs_agents=1,
)
```

## Spaces (from the probe)

* **Action space** (dict):

  * `iso`: `Box(low=[1., 1., 0.], high=[10., 10., 300.], shape=(3,), dtype=float32)`
  * `pcs`: `Box(low=-10.0, high=10.0, shape=(1,), dtype=float32)`
* **Observation space:** dict with keys `['iso', 'pcs']`
  *(Tensor shapes depend on config; not enumerated here.)*

## Episode interface (Gymnasium v0.29 style)

* `reset(seed=...) -> (obs, info)`

  * `obs`: dict `{'iso': ..., 'pcs': ...}`
  * `info@reset` keys (observed):
    `['iso', 'pcs', 'shared', 'iso_total_reward']`
* `step(action_dict) -> (obs, reward, terminated, truncated, info)`

  * `reward`: **dict** keyed by `'iso'`, `'pcs'`
  * `terminated`: **dict** keyed by `'iso'`, `'pcs'`
  * `truncated`: **dict** keyed by `'iso'`, `'pcs'`
  * `info` keys (observed during steps):

    ```
    ['iso', 'pcs', 'shared', 'iso_total_reward',
     'iso_buy_price', 'iso_sell_price', 'predicted_demand', 'realized_demand',
     'net_demand', 'dispatch', 'shortfall', 'dispatch_cost', 'reserve_cost',
     'price_spread', 'iso_action', 'battery_level', 'battery_action',
     'net_exchange', 'pcs_exchange_cost', 'pcs_action', 'time', 'step',
     'terminated', 'truncated', 'episode_iso_reward', 'episode_pcs_reward']
    ```

### Especially useful signals (for analysis/safety design later)

* `shortfall` — unserved demand (>0 means a violation)
* `price_spread` — `iso_sell_price - iso_buy_price` (proxy for price instability)
* `iso_buy_price`, `iso_sell_price` — raw prices
* Operational context: `dispatch`, `net_demand`, `battery_level`, `battery_action`

## Known dependency gotcha (fixed)

* The `energy_net` package imports `scipy.misc.derivative` (removed in newer SciPy).
* **Pin SciPy to `1.10.1`** in your environment. Example:

  ```
  mamba install -y scipy=1.10.1
  ```

  After that, `from scipy.misc import derivative` works and the env imports cleanly.

## Quick probe (to re-confirm later if needed)

```python
from energy_net.env import EnergyNetV0
env = EnergyNetV0(pricing_policy="ONLINE", demand_pattern="SINUSOIDAL", cost_type="CONSTANT", num_pcs_agents=1)

print("action_space:", env.action_space)
obs, info = env.reset(seed=0)
print("obs keys:", list(obs.keys()))
print("info@reset:", list(info.keys()))

action = {k: v.sample() for k, v in env.action_space.items()}  # dict action
obs, reward, terminated, truncated, info = env.step(action)
print("reward keys:", list(reward.keys()))
print("terminated keys:", list(terminated.keys()))
print("truncated keys:", list(truncated.keys()))
print("info keys:", list(info.keys()))
```

## Notes

* Actions must be provided as a **dict**: `{'iso': <...>, 'pcs': <...>}` matching the spaces above.
* Rewards/termination/truncation are **per-agent** dicts; if you need a single-agent view for experiments, you’ll need a wrapper (not included here).

---
