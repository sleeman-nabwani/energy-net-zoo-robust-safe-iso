from __future__ import annotations
from typing import Any, Callable
import numpy as np
import torch


def make_policy_act(agent: Any) -> Callable:
    """
    Return a function policy_act(obs, deterministic=True) -> np.ndarray.
    Tries common OmniSafe actor access; raise if not available.
    """
    ac = getattr(agent, "_actor_critic", None)
    actor = getattr(ac, "actor", None) if ac is not None else None

    @torch.no_grad()
    def _act(obs, deterministic: bool = True):
        if actor is None:
            raise RuntimeError("Policy actor not found on agent; cannot do policy eval.")
        out = actor(obs)
        # If Distribution-like
        if hasattr(out, "mean") and hasattr(out, "rsample"):
            action = out.mean if deterministic else out.rsample()
        else:
            if isinstance(out, (tuple, list)):
                action = out[0]
            else:
                action = out
        if isinstance(action, torch.Tensor):
            action = action.detach().to("cpu").numpy().astype(np.float32, copy=False)
        return action

    return _act
