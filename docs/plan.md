

# The picture (one ISO step with a fixed PCS)

1. **Base env** expects **joint action**: `{"iso": ..., "pcs": ...}` and returns
   `obs = {"iso": ..., "pcs": ...}`, `reward = {"iso": r_iso, "pcs": r_pcs}`, `terminated`, `truncated`, `info`.

2. **Your wrapper** exposes **only ISO** to the trainer. It **caches** the last full `(obs, info)`.

3. Trainer gives wrapper an **ISO action** → `iso_action`.

4. Wrapper asks your **PCS policy** for the other half:
   `pcs_action = pcs_policy(last_obs["pcs"], last_info, env.action_space["pcs"])`.

5. Wrapper calls base env:
   `env.step({"iso": iso_action, "pcs": pcs_action})`.

6. Wrapper returns to trainer: `(obs["iso"], reward["iso"], terminated["iso"], truncated["iso"], info)`.

7. (Optional) **CMDP adapter** also computes `cost` from `info` for OmniSafe.

That’s it. ISO learns; PCS is a fixed “autopilot.”

---

# The 3 arguments to every PCS policy (and why)

* `obs_pcs: np.ndarray` → **PCS observation vector** (from `last_obs["pcs"]`).
* `info: dict` → extra signals (e.g., `price_spread`, `battery_level`) your rule/models can use.
* `action_space: gym.spaces.Box` → so the policy can **clip** and return the right **shape/dtype**.

Return must be **`np.float32` with shape `(1,)`**, clipped to that Box.

---

# What your `SB3_Policy` actually does

* **Load** `pcs_td3_*.zip` (with “pickle shims” and `custom_objects` so old RNG/noise don’t break).
* **Eval mode**: freezes nets; `deterministic=True` → no exploration noise.
* **VecNormalize (if present)**: loads `vec_normalize_pcs.pkl` and applies
  `z = (obs - mean) / sqrt(var + 1e-8)` (+optional clip) **before** predict.
* **Predict** with shape `(1, obs_dim)`, then **clip** to PCS Box, return `(1,)`.

So: `obs_pcs → (normalize) → predict → clip → pcs_action`.

---

# Minimal responsibilities (who does what)

* **PCS policy**: produce a valid, clipped `pcs_action`.
* **ISO wrapper**: combine ISO action from trainer + PCS action from policy → call env.
* **CMDP adapter (optional)**: turn `info` into `cost`.
* **Trainer (OmniSafe)**: sees only ISO’s MDP or CMDP.

---

# “Do I have the right shapes?” — 3 assertions to keep

Inside your `SB3_Policy.__call__`:

```python
obs = self._norm_obs(obs_pcs).reshape(1, -1)
assert obs.dtype == np.float32
action, _ = self.model.predict(obs, deterministic=self.det)
act = _clip_to_space(action, action_space)
assert act.shape == action_space.shape
assert (act >= action_space.low).all() and (act <= action_space.high).all()
return act
```

---

# Tiny checklist (use this order every time)

1. **PCS smoke**: Can `SB3_Policy(zip)` produce a valid `(1,)` action for one reset?
2. **Wrapper smoke**: ISOEnvWrapper + that PCS: one `step()` works?
3. **CMDP smoke** (if using OmniSafe): CMDPAdapter returns `(obs, reward, cost, terminated, truncated, info)`?
4. **Training**: point PPOLag at the ISO wrapper (or CMDP adapter) env.

---

# One-minute “why” for the tricky bits

* **VecNormalize**: the PCS was trained on **normalized** obs; using the same stats at inference makes behavior match training.
* **deterministic=True**: we’re not training PCS; noise would just confuse ISO.
* **`custom_objects` + numpy shim**: lets you load zips saved under different NumPy/SB3 versions by bypassing training-only junk (noise, RNG, buffers).

---
