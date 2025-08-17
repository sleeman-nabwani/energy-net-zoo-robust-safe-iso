from __future__ import annotations
from pathlib import Path
from typing import Tuple, Any
import json
import numpy as np


def load_evalpack(run_dir: str | Path) -> Tuple[Any, dict]:
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


