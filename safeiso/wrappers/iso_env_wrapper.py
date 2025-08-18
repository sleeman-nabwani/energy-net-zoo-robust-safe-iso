import gymnasium as gym
import numpy as np
from ..pcs.pcs_policies import PCS_Policy
from typing import Dict, Any, Optional
from collections.abc import Mapping

class IsoEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, pcs_policy: PCS_Policy):
        super().__init__(env)
        obs_space = env.observation_space
        act_space = env.action_space

        if isinstance(obs_space, Mapping):
            if not all(isinstance(v, gym.spaces.Space) for v in obs_space.values()):
                raise TypeError("observation_space mapping must contain gym.spaces.Space values")
            obs_space = gym.spaces.Dict(dict(obs_space))
        if isinstance(act_space, Mapping):
            if not all(isinstance(v, gym.spaces.Space) for v in act_space.values()):
                raise TypeError("action_space mapping must contain gym.spaces.Space values")
            act_space = gym.spaces.Dict(dict(act_space))

        if not isinstance(obs_space, gym.spaces.Dict) or not isinstance(act_space, gym.spaces.Dict):
            raise TypeError("Base env must expose ISO/PCS spaces via Dict (mapping-of-spaces or gym.spaces.Dict).")
        for k in ("iso", "pcs"):
            if k not in obs_space.spaces or k not in act_space.spaces:
                raise KeyError(f"Missing key {k!r} in observation/action Dict spaces")

        self._joint_observation_space = obs_space
        self._joint_action_space = act_space
        self.observation_space = obs_space["iso"]
        self.action_space = act_space["iso"]

        self.pcs_policy = pcs_policy
        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_info: Optional[Dict[str, Any]] = None

    @property
    def pcs_action_space(self):
        """Expose the PCS Box space for downstream wrappers."""
        return self._joint_action_space["pcs"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = info
        return obs["iso"], info

    def step(self, iso_action: np.ndarray):
        if self._last_obs is None or self._last_info is None:
            raise RuntimeError("must call reset before step")

        pcs_action = self.pcs_policy(self._last_obs["pcs"], self._last_info, self._joint_action_space["pcs"])
        joint_action = {
            "iso": np.asarray(iso_action, dtype=np.float32).reshape(self.action_space.shape),
            "pcs": np.asarray(pcs_action, dtype=np.float32).reshape(self._joint_action_space["pcs"].shape),
        }
        obs, reward, terminated, truncated, info = self.env.step(joint_action)
        self._last_obs, self._last_info = obs, info

        # Add PCS action to info for diagnostics
        if isinstance(info, dict):
            info['pcs_action'] = float(pcs_action[0]) if len(pcs_action) > 0 else float(pcs_action)
        
        r_iso = float(reward.get("iso", reward) if isinstance(reward, dict) else reward)
        te = bool(terminated.get("iso", terminated) if isinstance(terminated, dict) else terminated)
        tr = bool(truncated.get("iso", truncated) if isinstance(truncated, dict) else truncated)
        return obs["iso"], r_iso, te, tr, info