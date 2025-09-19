import numpy as np
from typing import Callable, Dict, Any, Protocol, Union
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize
from pathlib import Path
from .version_helper import _spaces_from_model_zip, patch_numpy_for_pickle, resolve_algo_class, learning_rate_from_zip


def _clip_to_space(x: np.ndarray, action_space) -> np.ndarray:
    """Ensure right dtype/shape and keep the action within Box bounds."""
    x = np.asarray(x, dtype=np.float32)
    
    # Handle shape mismatch - ensure x can be reshaped to action_space.shape
    if x.size != np.prod(action_space.shape):
        if action_space.shape == (1,) and x.size > 1:
            # For PCS action space (1,), take the last element regardless of input size
            # This handles size 2, 3, 7, or any other size from SB3 models
            x = x.flatten()[-1:]
        elif x.size < np.prod(action_space.shape):
            padding_needed = np.prod(action_space.shape) - x.size
            x = np.concatenate([x.flatten(), np.zeros(padding_needed, dtype=np.float32)])
        else:
            x = x.flatten()[:np.prod(action_space.shape)]
    
    x = x.reshape(action_space.shape)
    return np.clip(x, action_space.low, action_space.high)


class PCS_Policy(Protocol):
    def __call__(self, obs_pcs:np.ndarray, info:Dict[str, Any], action_space) -> np.ndarray:
        ...


class static_Policy(PCS_Policy):
    '''used for establishing a baseline'''
    def __init__(self, value = 0.0):
        self.value = value
        
    def __call__(self, obs_pcs:np.ndarray, info:Dict[str, Any], action_space) -> np.ndarray:
        res = np.full(shape=action_space.shape, fill_value=self.value)
        return _clip_to_space(res, action_space)


class responsive_Policy(PCS_Policy):
    def __init__(self, fn: Callable[[np.ndarray, Dict[str, Any], Any], Union[float, np.ndarray]]):
        self.fn = fn

    def __call__(self, obs_pcs:np.ndarray, info:Dict[str, Any], action_space) -> np.ndarray:
        raw = self.fn(obs_pcs, info, action_space)
        return _clip_to_space(raw, action_space)


class SB3_Policy(PCS_Policy):
    def __init__(self, model_path: str, deterministic: bool = True, device: str = "auto"):
        patch_numpy_for_pickle()
        
        # Add NumPy compatibility fix for numpy._core issue
        import sys
        if 'numpy._core' not in sys.modules:
            try:
                import numpy
                sys.modules['numpy._core'] = numpy
                sys.modules['numpy._core.multiarray'] = numpy
                sys.modules['numpy._core.umath'] = numpy
            except Exception:
                pass

        lr = learning_rate_from_zip(model_path)
        custom_objects = {
            "action_noise": None,
            "replay_buffer": None,
            "rng": np.random.default_rng(0),
            "np_random": np.random.default_rng(0),
            "random_generator": np.random.default_rng(0),
            "generator": np.random.default_rng(0),
            "lr_schedule": lambda _: float(lr),   # avoids unpickling code
            "learning_rate": float(lr),
        }
        custom_objects.update(_spaces_from_model_zip(model_path))
        if "action_space" not in custom_objects or "observation_space" not in custom_objects:
            raise RuntimeError("Failed to reconstruct spaces from model 'data'. Use a compatible ZIP or pin versions.")

        algo_cls = resolve_algo_class(model_path)
        self.model = algo_cls.load(model_path, device=device, custom_objects=custom_objects)
        self.det = deterministic

        # Ensure evaluation-only
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "set_training_mode"):
            self.model.policy.set_training_mode(False)
        try:
            import torch as th
            for p in self.model.policy.parameters():
                p.requires_grad = False
            self._torch = th
        except Exception:
            self._torch = None
        
        # locate normalization stats file (only use if shape matches observation space)
        self._vecstats = None
        P = Path(model_path).resolve()
        expected_shape = None
        try:
            # custom_objects already includes spaces
            from .version_helper import _spaces_from_model_zip as _spaces_fn
            sp = _spaces_fn(model_path)
            obs_space = sp.get("observation_space")
            if obs_space is not None:
                expected_shape = tuple(obs_space.shape)
        except Exception:
            expected_shape = None

        candidates = ["vec_normalize_pcs.pkl", "vec_normalize_iso.pkl", "vec_normalize.pkl"]
        for fname in candidates:
            norm_path = P.with_name(fname)
            if not norm_path.exists():
                continue
            try:
                import cloudpickle as cp
                with open(norm_path, "rb") as f:
                    vn = cp.load(f)
                mean = getattr(getattr(vn, "obs_rms", None), "mean", None)
                var = getattr(getattr(vn, "obs_rms", None), "var", None)
                clip = getattr(vn, "clip_obs", None)
                if mean is None or var is None:
                    continue
                mean = np.asarray(mean, dtype=np.float32)
                var = np.asarray(var, dtype=np.float32)
                if expected_shape is not None and tuple(mean.shape) != expected_shape:
                    continue
                self._vecstats = {"mean": mean, "var": var, "clip": clip}
                break
            except Exception:
                continue
    
    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        s = self._vecstats
        if not s:
            return obs
        mean, var = s["mean"], s["var"]
        if mean.shape != obs.shape:
            return obs
        eps = 1e-8
        out = (obs - mean) / np.sqrt(var + eps)
        clip = s.get("clip", None)
        return np.clip(out, -clip, clip) if clip is not None else out

    def __call__(self, obs_pcs:np.ndarray, info:Dict[str, Any], action_space) -> np.ndarray:
        obs_normalized = self._norm_obs(obs_pcs)
        
        # Handle observation shape mismatch - SB3 models expect (3,) observations
        if obs_normalized.shape == (1, 3):
            obs_normalized = obs_normalized.flatten()  # (1,3) -> (3,)
        elif obs_normalized.ndim == 1 and obs_normalized.shape[0] == 4:
            # Environment observation is (4,) but SB3 model expects (3,) - truncate last element
            obs_normalized = obs_normalized[:3]
        elif obs_normalized.ndim == 1 and obs_normalized.shape[0] > 3:
            # Truncate to 3 elements if too large
            obs_normalized = obs_normalized[:3]
        elif obs_normalized.ndim == 1 and obs_normalized.shape[0] < 3:
            # Pad to 3 elements if too small
            padding_needed = 3 - obs_normalized.shape[0]
            obs_normalized = np.concatenate([obs_normalized, np.zeros(padding_needed, dtype=np.float32)])
            
        # SB3 predict expects (3,) shape, not (1, 3)
        if obs_normalized.shape != (3,):
            # Ensure we have exactly 3 elements
            if obs_normalized.size > 3:
                obs_normalized = obs_normalized.flatten()[:3]
            elif obs_normalized.size < 3:
                padding = np.zeros(3 - obs_normalized.size, dtype=np.float32)
                obs_normalized = np.concatenate([obs_normalized.flatten(), padding])
            else:
                obs_normalized = obs_normalized.reshape((3,))
        
        obs = obs_normalized  # Don't add batch dimension - SB3 predict handles this
        
        if self._torch is not None:
            with self._torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=self.det)
        else:
            action, _ = self.model.predict(obs, deterministic=self.det)
        
        # SB3 models output the full joint action space (3D), but PCS only needs the last component
        # Extract PCS component (assuming it's the last dimension based on EnergyNet structure)
        if isinstance(action, np.ndarray):
            if action.ndim > 1:
                action = action.flatten()  # Flatten multi-dimensional arrays
            
            if len(action) > 1 and action_space.shape == (1,):
                # PCS action space expects (1,) - take the last component
                action = np.array([action[-1]], dtype=np.float32)
            elif len(action) == 1 and action_space.shape == (1,):
                # Already correct shape, ensure proper type
                action = np.array([action[0]], dtype=np.float32)
            elif len(action) > 1:
                # Multiple elements but action_space doesn't specify (1,) - take last
                action = np.array([action[-1]], dtype=np.float32)
        else:
            # Scalar action - convert to proper array format
            action = np.array([float(action)], dtype=np.float32)
        
        return _clip_to_space(action, action_space)
