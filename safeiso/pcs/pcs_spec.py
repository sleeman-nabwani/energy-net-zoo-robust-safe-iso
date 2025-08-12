"""PCS spec parser and policy builder.

This module provides a simple spec parser and policy builder for PCS policies.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs
import os
import importlib

from .pcs_functions import *
from .pcs_policies import *


def _coerce_value(val: str) -> bool | float | int | str:
    s = val.strip()
    if s == "":
        return ""
    low = s.lower()
    if low in {"true", "t", "yes", "y", "1"}:
        return True
    if low in {"false", "f", "no", "n", "0"}:
        return False
    try:
        # float if decimal or scientific notation appears
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except Exception:
        return s


def parse_pcs_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a PCS spec string into (kind, kwargs) for the policy builder.

    Forms:
      static:<value>
      sb3:/abs/path/to/model.zip
      responsive:<recipe_or_dotted>[?k=v&k2=v2...]

    Returns:
      kind in {"static","sb3","responsive"}, plus kwargs.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"Bad PCS spec: {spec!r}")

    kind, rest = spec.split(":", 1)
    kind = kind.strip().lower()
    rest = rest.strip()

    if kind == "static":
        if rest == "":
            raise ValueError("static spec requires a value, e.g. static:0.0")
        return "static", {"value": float(rest)}

    if kind == "sb3":
        if rest == "":
            raise ValueError("sb3 spec requires a model path, e.g. sb3:/path/model.zip")
        path = os.path.abspath(os.path.expanduser(rest))
        if not os.path.exists(path):
            raise FileNotFoundError(f"SB3 model not found: {path}")
        return "sb3", {"model_path": path}

    if kind == "responsive":
        if rest == "":
            raise ValueError("responsive spec requires a target, e.g. responsive:spread_prop?gain=0.5")
        if "?" in rest:
            target, q = rest.split("?", 1)
            params = {k: _coerce_value(v[-1]) for k, v in parse_qs(q, keep_blank_values=True).items()}
        else:
            target, params = rest, {}
        target = target.strip()
        return "responsive", {"factory": target, "params": params}

    raise ValueError(f"Unknown PCS kind: {kind!r}")


def _resolve_factory(target: str):
    """Resolve a recipe name or dotted path to a callable factory."""
    key = target.strip().replace("-", "_")

    # Try built-ins: exact and *_factory
    import safeiso.pcs.pcs_functions as recipes
    for cand in (key, f"{key}_factory"):
        fn = getattr(recipes, cand, None)
        if callable(fn):
            return fn

    # Dotted path fallback
    if "." in key:
        module_name, attr_name = key.rsplit(".", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, attr_name)
        if callable(fn):
            return fn

    raise ValueError(f"Unknown responsive PCS recipe: {target!r}")


def build_pcs_policy(spec: str):
    """Construct a PCS policy callable from a user spec.

    Returns a policy with signature:
      policy(obs_pcs, info, action_space) -> np.ndarray of shape (1,), float32, Box-clipped
    """
    kind, cfg = parse_pcs_spec(spec)

    if kind == "static":
        return static_Policy(cfg["value"])
    if kind == "sb3":
        return SB3_Policy(cfg["model_path"])
    if kind == "responsive":
        target = cfg["factory"]
        params = cfg.get("params", {})
        factory = _resolve_factory(target)
        core = factory(**params)
        if not callable(core):
            raise TypeError(f"Responsive factory {factory} must return a callable, got: {type(core)}")
        return responsive_Policy(core)

    raise RuntimeError(f"Unhandled PCS kind: {kind}")
