# safeiso/train/evalpack_export.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch


def _to_pylist(x):
    """Return a plain Python list (or None) from numpy/torch/list/tuple/scalar."""
    if x is None:
        return None
    # torch tensor
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    # numpy array / scalar
    if hasattr(x, "tolist"):
        return x.tolist()
    # python list/tuple
    if isinstance(x, (list, tuple)):
        return list(x)
    # scalar fallback
    try:
        return [float(x)]
    except Exception:
        raise TypeError(f"Cannot convert type {type(x)} to list for EvalPack meta.")


class _EvalWrapper(torch.nn.Module):
    """
    TorchScript-able wrapper:
      - optional obs normalization
      - actor forward
      - deterministic mean if distribution-like
      - clip to action bounds
    """
    def __init__(
        self,
        actor: torch.nn.Module,
        low: np.ndarray,
        high: np.ndarray,
        obs_mean: list[float] | None = None,
        obs_var: list[float] | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.actor = actor
        self.register_buffer("low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(high, dtype=torch.float32))
        if obs_mean is not None and obs_var is not None:
            self.register_buffer("mean", torch.tensor(np.asarray(obs_mean, dtype=np.float32)))
            self.register_buffer("var", torch.tensor(np.asarray(obs_var, dtype=np.float32)))
            self.eps = float(eps)
        else:
            self.mean = None  # type: ignore[assignment]
            self.var = None   # type: ignore[assignment]
            self.eps = 0.0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs if obs.ndim > 1 else obs.unsqueeze(0)
        if self.mean is not None and self.var is not None:
            x = (x - self.mean) / torch.sqrt(self.var + self.eps)
        out = self.actor(x)
        if hasattr(out, "mean") and hasattr(out, "rsample"):
            act = out.mean
        elif isinstance(out, (tuple, list)):
            act = out[0]
        else:
            act = out
        return torch.clamp(act, self.low, self.high)


def export_evalpack(
    *,
    actor: torch.nn.Module,
    action_space,  # gymnasium.spaces.Box
    run_dir: str | Path,
    algo: str,
    env_id: str,
    seed: int,
    obs_rms: object | None = None,  # expects optional .mean/.var/.eps
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / "evalpack"
    out_dir.mkdir(parents=True, exist_ok=True)

    low = np.array(action_space.low, dtype=np.float32)
    high = np.array(action_space.high, dtype=np.float32)

    # Extract & sanitize normalization (once!)
    obs_mean = getattr(obs_rms, "mean", None)
    obs_var = getattr(obs_rms, "var", None)
    eps = getattr(obs_rms, "eps", 1e-8)

    obs_mean_list = _to_pylist(obs_mean)
    obs_var_list = _to_pylist(obs_var)

    # Script the wrapper
    wrapper = _EvalWrapper(
        actor=actor.cpu(),
        low=low,
        high=high,
        obs_mean=obs_mean_list,
        obs_var=obs_var_list,
        eps=float(eps),
    )
    scripted = torch.jit.script(wrapper)
    scripted.save(str(out_dir / "actor.ts"))

    # Build JSON-serializable meta; DO NOT .tolist() lists again
    meta = {
        "format_version": 1,
        "algo": str(algo),
        "env_id": str(env_id),
        "seed": int(seed),
        "obs_norm": None
        if (obs_mean_list is None or obs_var_list is None)
        else {"mean": obs_mean_list, "var": obs_var_list, "eps": float(eps)},
        "action_space": {"low": low.tolist(), "high": high.tolist()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir