# imports (top of safeiso/pcs_policies.py)
import json, zipfile
import numpy as np
try:
    from gymnasium.spaces import Box
except Exception:
    from gym.spaces import Box
import sys
from typing import Dict, Any
from stable_baselines3 import TD3, SAC, PPO, DDPG, A2C, DQN
# optional contrib algos
try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

def patch_numpy_for_pickle():
    """Map NumPy 2.x private paths to 1.x so cloudpickle can resolve them."""
    try:
        import numpy.core as _np_core
    except Exception:
        return
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)

def _as_dtype(s):
    try:
        return np.dtype(str(s).replace("dtype(", "").replace(")", "").strip("'\""))
    except Exception:
        return np.float32

def _to_ndarray(x, dtype, shape):
    if isinstance(x, str):
        arr = np.fromstring(x.strip("[]"), sep=" ", dtype=dtype)
    else:
        arr = np.asarray(x, dtype=dtype)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    if arr.size == 1 and np.prod(shape) > 1:
        arr = np.full(shape, arr.item(), dtype=dtype)
    else:
        arr = arr.reshape(shape)
    return arr

def _extract_box(d):
    if not isinstance(d, dict):
        return None
    src = d.get("kwargs", d)
    shp = src.get("_shape") or src.get("shape") or []
    if isinstance(shp, int):
        shp = (shp,)
    shp = tuple(shp)
    if not shp:
        return None
    dt = _as_dtype(src.get("dtype", "float32"))
    low = src.get("low"); high = src.get("high")
    if low is None or high is None:
        return None
    low = _to_ndarray(low, dt, shp)
    high = _to_ndarray(high, dt, shp)
    return Box(low=low, high=high, dtype=dt)

def _spaces_from_model_zip(model_path: str):
    out = {}
    try:
        with zipfile.ZipFile(model_path) as zf:
            data = json.loads(zf.read("data").decode("utf-8"))
        obs = _extract_box(data.get("observation_space", {}))
        act = _extract_box(data.get("action_space", {}))
        if obs is not None: out["observation_space"] = obs
        if act is not None: out["action_space"] = act
    except Exception:
        pass
    return out

def _read_data(model_path: str) -> Dict[str, Any]:
    with zipfile.ZipFile(model_path) as zf:
        return json.loads(zf.read("data").decode("utf-8"))

_ALGO_MAP = {
    "td3": TD3, "sac": SAC, "ppo": PPO, "ddpg": DDPG, "a2c": A2C, "dqn": DQN,
    # best-effort recurrent PPO (sb3_contrib)
    "recurrentppo": RecurrentPPO, "recurrent_ppo": RecurrentPPO,
}

def resolve_algo_class(model_path: str):
    data = _read_data(model_path)
    mod = (data.get("policy_class") or {}).get("__module__", "")
    low = mod.lower()
    for key, cls in _ALGO_MAP.items():
        if cls is None:
            continue
        if key in low or f".{key}." in low:
            return cls
    # special-case contrib recurrent policies
    if ("sb3_contrib" in low or "recurrent" in low) and RecurrentPPO is not None:
        return RecurrentPPO
    raise RuntimeError(f"Could not infer algorithm from policy_class: {mod}")

def learning_rate_from_zip(model_path: str, default: float = 3e-4) -> float:
    data = _read_data(model_path)
    try:
        return float(data.get("learning_rate", default))
    except Exception:
        return default
