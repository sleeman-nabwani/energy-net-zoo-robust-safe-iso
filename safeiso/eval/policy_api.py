from __future__ import annotations
from typing import Any, Callable, Optional
import numpy as np
import torch


def make_policy_act(
    agent: Any,
    action_space=None,
    device: Optional[str] = None,
    keep_batch_dim: bool = True,
) -> Callable[[np.ndarray | torch.Tensor, bool], np.ndarray]:
    """
    Build a deterministic policy callable from an OmniSafe agent.

    - Keeps batch dim if input is batched, unless keep_batch_dim=False.
    - Auto-selects device from obs or actor if device is None.
    - Clips to action_space if provided.
    """
    # Tolerate OmniSafe version drift
    ac = getattr(agent, "_actor_critic", None) or getattr(agent, "actor_critic", None)
    actor = getattr(ac, "actor", None) or getattr(agent, "actor", None) or getattr(agent, "pi", None)
    if actor is None:
        raise RuntimeError("Could not locate actor on the agent (no .actor).")

    def _actor_device() -> torch.device:
        try:
            return next(actor.parameters()).device  # type: ignore[arg-type]
        except Exception:
            return torch.device("cpu")

    @torch.no_grad()
    def policy_act(obs: np.ndarray | torch.Tensor, deterministic: bool = True) -> np.ndarray:
        # Convert obs to torch on the desired device and ensure batch dim
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        dev = obs_t.device if device is None else torch.device(device)
        if device is None and not isinstance(obs, torch.Tensor):
            # If obs was numpy, prefer actor's device
            dev = _actor_device()
        obs_t = obs_t.to(dev, non_blocking=True)

        was_batched = obs_t.ndim > 1
        if not was_batched:
            obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]

        # Forward through actor
        out = actor(obs_t)
        if hasattr(out, "mean") and (hasattr(out, "rsample") or hasattr(out, "sample")):
            act_t = out.mean if deterministic else (out.rsample() if hasattr(out, "rsample") else out.sample())
        elif isinstance(out, (tuple, list)):
            act_t = out[0]
        else:
            act_t = out

        if not was_batched and not keep_batch_dim:
            act_t = act_t.squeeze(0)

        act = act_t.detach().to("cpu").numpy().astype(np.float32, copy=False)

        if action_space is not None and hasattr(action_space, "low"):
            act = np.clip(act, action_space.low, action_space.high, dtype=np.float32)
        return act

    return policy_act
