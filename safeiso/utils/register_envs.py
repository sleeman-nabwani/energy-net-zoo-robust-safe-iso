"""SafeISO environment registration.

- Provides builder functions for Gymnasium env construction
- Registers OmniSafe IDs only (no Gym ID registration)
"""

from typing import Any, ClassVar, Tuple

import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register

from .make_env import make_iso_single_agent_env, make_iso_cmdp_basic

# -----------------------------
# Builder functions (entry points)
# -----------------------------

def _make_iso_only(**kwargs: Any):
    """Build ISO-only single-agent Gym env (no cost)."""
    return make_iso_single_agent_env(
        seed=kwargs.get("seed", 0),
        pcs=kwargs.get("pcs", "static:0.0"),
    )


def _make_iso_cmdp(**kwargs: Any):
    """Build ISO CMDP Gym env with cost in info['cost'] by default (5-tuple)."""
    return make_iso_cmdp_basic(
        seed=kwargs.get("seed", 0),
        pcs=kwargs.get("pcs", "static:0.0"),
        preset=kwargs.get("preset", None),
        spread_limit=kwargs.get("spread_limit", None),
        return_cost_in_info=kwargs.get("return_cost_in_info", True),
    )


# -----------------------------
# OmniSafe registration
# -----------------------------

def _sanitize_tensor(x: torch.Tensor) -> torch.Tensor:
    # Replace NaN/±Inf and clamp to a large but finite range to keep nets stable
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e6, 1e6)


@env_register
class SafeISOOmniEnv(CMDP):
    """Thin adapter to expose SafeISO Gym envs to OmniSafe."""

    _support_envs: ClassVar[list[str]] = [
        "SafeISO-ISOOnly-omni-v0",
        "SafeISO-CMDP-omni-v0",
    ]

    # Follow OmniSafe template: allow their wrappers to attach
    need_time_limit_wrapper = True
    need_auto_reset_wrapper = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id=env_id, device=kwargs.get("device", "cpu"))
        self._device = torch.device(kwargs.get("device", "cpu"))

        seed = kwargs.get("seed", None)
        if env_id == "SafeISO-ISOOnly-omni-v0":
            self._env = _make_iso_only(**kwargs)
        elif env_id == "SafeISO-CMDP-omni-v0":
            self._env = _make_iso_cmdp(**kwargs)
        else:
            raise ValueError(f"Unknown env_id for SafeISOOmniEnv: {env_id}")

        # define episode horizon for OmniSafe time-limit wrapper (read-only property below)
        self._max_episode_steps = int(kwargs.get("max_episode_steps", 48))
        # single-env adapter
        self._num_envs = 1

        self._env.reset(seed=seed)
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def single_observation_space(self):
        return self._observation_space

    @property
    def max_episode_steps(self) -> int | None:
        return self._max_episode_steps

    def set_seed(self, seed: int) -> None:
        self._env.reset(seed=seed)

    def render(self):
        return getattr(self._env, "render", lambda: None)()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[torch.Tensor, dict]:
        try:
            obs, info = self._env.reset(seed=seed, options=options)
        except TypeError:
            obs, info = self._env.reset(seed=seed)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        obs_t = _sanitize_tensor(obs_t)
        return obs_t, info

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(action, torch.Tensor):
            act = action.detach().to("cpu").contiguous().numpy().astype(np.float32, copy=False)
        else:
            act = np.asarray(action, dtype=np.float32, order="C")

        ret = self._env.step(act)
        if len(ret) == 6:
            obs, reward, cost, terminated, truncated, info = ret
        else:
            obs, reward, terminated, truncated, info = ret
            cost = float(info.get("cost", 0.0))

        # Ensure info['cost'] is consistent and finite
        if not isinstance(info, dict):
            info = dict(info)
        info = dict(info)
        c = float(cost)
        if not np.isfinite(c):
            c = 0.0
        # Optional clamp to plausible range [0, 1]
        if c < 0.0:
            c = 0.0
        elif c > 1.0:
            c = 1.0
        info["cost"] = c

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        obs_t = _sanitize_tensor(obs_t)
        rew_t = torch.tensor(reward, dtype=torch.float32, device=self._device)
        cost_t = torch.tensor(c, dtype=torch.float32, device=self._device)
        term_t = torch.tensor(bool(terminated), dtype=torch.bool, device=self._device)
        trunc_t = torch.tensor(bool(truncated), dtype=torch.bool, device=self._device)
        return obs_t, rew_t, cost_t, term_t, trunc_t, info

    @property
    def single_action_space(self):
        return self._action_space

    def close(self) -> None:
        self._env.close()
