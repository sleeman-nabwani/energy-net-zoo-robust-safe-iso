from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import json
import numpy as np


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    try:
        import numpy as _np
        if x is None:
            return None
        if isinstance(x, _np.ndarray):
            return x.astype(_np.float32, copy=False)
        if hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy().astype(_np.float32, copy=False)
        return _np.asarray(x, dtype=_np.float32)
    except Exception:
        return None


class _EvalWrapper:
    """
    Minimal TorchScript-friendly callable wrapper around an actor.
    Performs optional observation normalization and action clipping.
    """
    def __init__(self, actor, low: np.ndarray, high: np.ndarray,
                 obs_mean: Optional[np.ndarray] = None,
                 obs_var: Optional[np.ndarray] = None,
                 eps: float = 1e-8):
        import torch
        self.actor = actor
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.has_norm = (obs_mean is not None) and (obs_var is not None)
        if self.has_norm:
            self.mean = torch.as_tensor(obs_mean, dtype=torch.float32)
            self.var = torch.as_tensor(obs_var, dtype=torch.float32)
            self.eps = float(eps)

    def __call__(self, obs):
        import torch
        x = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if getattr(self, "has_norm", False):
            x = (x - self.mean) / torch.sqrt(self.var + self.eps)  # type: ignore[attr-defined]
        out = self.actor(x)
        if hasattr(out, "mean") and (hasattr(out, "rsample") or hasattr(out, "sample")):
            act = out.mean
        elif isinstance(out, (tuple, list)):
            act = out[0]
        else:
            act = out
        act = torch.clamp(act, self.low, self.high)
        return act


def export_evalpack(*,
    actor,
    action_space,
    run_dir: str | Path,
    algo: str,
    env_id: str,
    seed: int,
    obs_rms: Any | None = None,
) -> Path:
    """Export a minimal, version-agnostic EvalPack for evaluation.

    Writes:
    - evalpack/actor.ts   (TorchScript-saved callable)
    - evalpack/meta.json  (algo/env/seed/obs_norm/action_space)
    """
    import torch
    run_dir = Path(run_dir)
    out_dir = run_dir / "evalpack"
    out_dir.mkdir(parents=True, exist_ok=True)

    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    obs_mean = None
    obs_var = None
    eps = 1e-8
    if obs_rms is not None:
        obs_mean = _to_numpy(getattr(obs_rms, "mean", None))
        obs_var = _to_numpy(getattr(obs_rms, "var", None))
        eps = float(getattr(obs_rms, "eps", 1e-8))

    wrapper = _EvalWrapper(actor=actor.cpu(), low=low, high=high,
                           obs_mean=obs_mean, obs_var=obs_var, eps=eps)
    scripted = torch.jit.script(wrapper)  # type: ignore[arg-type]
    scripted.save(str(out_dir / "actor.ts"))

    meta = {
        "format_version": 1,
        "algo": str(algo),
        "env_id": str(env_id),
        "seed": int(seed),
        "obs_norm": None if obs_mean is None or obs_var is None else {
            "mean": obs_mean.tolist(),
            "var": obs_var.tolist(),
            "eps": float(eps),
        },
        "action_space": {"low": low.tolist(), "high": high.tolist()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


