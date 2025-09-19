from __future__ import annotations
from pathlib import Path
from typing import Tuple, Any, Optional
import json
import numpy as np


def load_evalpack(run_dir: str | Path) -> Tuple[Any, dict]:
    """
    Load evalpack policy - returns NORMALIZED actions [-1,1].
    
    This is the original loader that returns actions as trained (normalized).
    For evaluation environments expecting denormalized actions, the caller
    should use load_evalpack_with_denormalization instead.
    """
    import torch
    p = Path(run_dir) / "evalpack"
    actor_p = p / "actor.ts"
    meta_p = p / "meta.json"
    if not actor_p.exists() or not meta_p.exists():
        raise FileNotFoundError(f"EvalPack missing under {p}")
    actor = torch.jit.load(str(actor_p), map_location="cpu")
    actor.eval()
    meta = json.loads(meta_p.read_text())

    @torch.no_grad()
    def policy_act(obs, deterministic: bool = True) -> np.ndarray:
        x = obs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        a = actor(x)
        if a.ndim > 1:
            a = a.squeeze(0)
        return a.cpu().numpy().astype(np.float32, copy=False)

    return policy_act, meta


def load_evalpack_with_denormalization(
    run_dir: str | Path,
    env_action_space: Optional[Any] = None
) -> Tuple[Any, dict]:
    """
    Load evalpack policy and return a wrapper that denormalizes actions.
    
    This is the FIXED version that properly transforms policy outputs from
    the normalized [-1, 1] range to the evaluation environment's expected range.
    
    Args:
        run_dir: Path to the run directory containing evalpack
        env_action_space: The actual environment's action space (for denormalization)
        
    Returns:
        Tuple of (policy_function, metadata_dict)
    """
    import torch
    
    p = Path(run_dir) / "evalpack"
    actor_p = p / "actor.ts"
    meta_p = p / "meta.json"
    
    if not actor_p.exists() or not meta_p.exists():
        raise FileNotFoundError(f"EvalPack missing under {p}")
    
    # Load the TorchScript actor
    actor = torch.jit.load(str(actor_p), map_location="cpu")
    actor.eval()
    
    # Load metadata
    meta = json.loads(meta_p.read_text())
    
    @torch.no_grad()
    def policy_act_denormalized(obs, deterministic: bool = True) -> np.ndarray:
        """Policy that outputs denormalized actions for the evaluation environment."""
        x = obs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        # Get normalized action from the trained policy
        a = actor(x)
        if a.ndim > 1:
            a = a.squeeze(0)
        a_normalized = a.cpu().numpy().astype(np.float32, copy=False)
        
        # Denormalize if we have the environment action space
        if env_action_space is not None:
            # Get bounds from the actual environment
            low = env_action_space.low.astype(np.float32)
            high = env_action_space.high.astype(np.float32)
            
            # The trained policy outputs in [-1, 1], denormalize to [low, high]
            a_denormalized = low + (a_normalized + 1.0) * 0.5 * (high - low)
            a_denormalized = np.clip(a_denormalized, low, high)
            
            return a_denormalized
        else:
            # Without proper denormalization info, return as-is (will likely fail)
            return a_normalized

    return policy_act_denormalized, meta


