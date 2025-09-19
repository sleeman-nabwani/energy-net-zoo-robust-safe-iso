from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, Tuple

import yaml


_NUMERIC_RE = re.compile(r"^[\s]*([+-]?(?:\d+\.\d*|\d*\.\d+|\d+))(?:[\s]*([a-zA-Z%]+))?[\s]*$")


def _to_number_maybe(x: Any) -> Any:
    """Convert strings like "123", "3.14", "300MW", "inf", "-inf" to numbers.

    - If unit suffix exists and is alphabetic (e.g., MW), it is ignored and value is parsed as float.
    - Returns original if conversion is not possible.
    """
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("inf", "+inf", "infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        if s in ("nan",):
            return float("nan")
        if s in ("true", "yes", "on"):
            return True
        if s in ("false", "no", "off"):
            return False
        m = _NUMERIC_RE.match(x)
        if m:
            num = float(m.group(1))
            # ignore unit suffix (group 2) by design for legacy YAMLs
            return num
        # fallback: int-like
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
    return x


def _normalize(obj: Any) -> Any:
    """Recursively normalize legacy YAML values.

    - numeric strings and unit-suffixed strings → floats/ints
    - "true"/"false" → bool
    - lists/dicts processed recursively
    """
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return _to_number_maybe(obj)


def _require_keys(d: Dict[str, Any], keys: Tuple[str, ...], *, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{ctx}: missing required keys {missing}")


def _require_type(v: Any, typ, *, ctx: str, key: str) -> None:
    if not isinstance(v, typ):
        raise ValueError(f"{ctx}: field '{key}' must be {typ.__name__}, got {type(v).__name__}")


def validate_energy_net_cfg(obj: Dict[str, Any]) -> None:
    """Validate normalized environment_config style dict.

    Required sections and common pitfalls are checked. Raises ValueError with
    explicit field names on error.
    """
    ctx = "environment_config"
    _require_keys(obj, ("time", "pricing", "iso_config_path", "pcs_unit_config_path"), ctx=ctx)
    # time
    t = obj["time"]
    _require_type(t, dict, ctx=ctx, key="time")
    for k in ("step_duration", "max_steps_per_episode"):
        if k not in t:
            raise ValueError(f"{ctx}.time: missing '{k}'")
        if not isinstance(t[k], (int, float)):
            raise ValueError(f"{ctx}.time.{k}: must be number, got {type(t[k]).__name__}")
    # pricing
    _require_type(obj["pricing"], dict, ctx=ctx, key="pricing")
    # referenced files
    for k in ("iso_config_path", "pcs_unit_config_path"):
        p = obj.get(k)
        if not isinstance(p, str) or not p:
            raise ValueError(f"{ctx}.{k}: must be non-empty string path")
        if not os.path.exists(p):
            raise ValueError(f"{ctx}.{k}: file not found: {p}")

    # ISO config
    with open(obj["iso_config_path"], "r") as f:
        iso_cfg = _normalize(yaml.safe_load(f) or {})
    iso_ctx = "iso_config"
    _require_keys(iso_cfg, ("pricing", "dispatch", "action_spaces", "observation_space", "type"), ctx=iso_ctx)
    _require_type(iso_cfg["pricing"], dict, ctx=iso_ctx, key="pricing")
    _require_type(iso_cfg["dispatch"], dict, ctx=iso_ctx, key="dispatch")
    # action_spaces.online fields
    aspc = iso_cfg.get("action_spaces", {})
    if not isinstance(aspc, dict) or not aspc:
        raise ValueError("iso_config.action_spaces: must be non-empty mapping")
    online = (aspc.get("online") or aspc.get("ONLINE"))
    if isinstance(online, dict):
        for k in ("buy_price", "sell_price", "dispatch"):
            if k in online:
                lo = online[k].get("min", None); hi = online[k].get("max", None)
                if lo is None or hi is None:
                    raise ValueError(f"iso_config.action_spaces.online.{k}: requires 'min' and 'max'")
                if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                    raise ValueError(f"iso_config.action_spaces.online.{k}.min/max must be numbers")

    # PCS config
    with open(obj["pcs_unit_config_path"], "r") as f:
        pcs_cfg = _normalize(yaml.safe_load(f) or {})
    pcs_ctx = "pcs_unit_config"
    _require_keys(pcs_cfg, ("action", "battery"), ctx=pcs_ctx)
    # Battery limits must be numeric
    batt = pcs_cfg["battery"]
    if not isinstance(batt, dict) or not isinstance(batt.get("model_parameters", {}), dict):
        raise ValueError("pcs_unit_config.battery.model_parameters must be a mapping")
    mp = batt["model_parameters"]
    for k in ("charge_rate_max", "discharge_rate_max", "max", "min"):
        if k in mp and not isinstance(mp[k], (int, float)):
            raise ValueError(f"pcs_unit_config.battery.model_parameters.{k} must be number")


def load_and_validate_configs(env_cfg_path: str = "configs/environment_config.yaml") -> Dict[str, Any]:
    """Load, normalize, and validate environment + referenced ISO/PCS configs.

    Returns the top-level environment config (normalized). Writes normalized snapshots
    under artifacts/configs_normalized for debugging.
    """
    with open(env_cfg_path, "r") as f:
        env_cfg_raw = yaml.safe_load(f) or {}
    env_cfg = _normalize(env_cfg_raw)
    # Resolve relative paths: prefer as-written if it exists from CWD; else relative to env_cfg directory
    base_dir = os.path.dirname(env_cfg_path) or "."
    for k in ("iso_config_path", "pcs_unit_config_path"):
        p = env_cfg.get(k)
        if isinstance(p, str) and not os.path.isabs(p):
            if os.path.exists(p):
                # use as provided (relative to CWD)
                env_cfg[k] = p
            else:
                env_cfg[k] = os.path.normpath(os.path.join(base_dir, p))

    validate_energy_net_cfg(env_cfg)

    # Write normalized snapshots (best-effort)
    out_dir = os.path.join("artifacts", "configs_normalized")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "environment_config.normalized.yaml"), "w") as f:
            yaml.safe_dump(env_cfg, f, sort_keys=False)
        # Also copy referenced configs in normalized form
        with open(env_cfg["iso_config_path"], "r") as f:
            iso = _normalize(yaml.safe_load(f) or {})
        with open(os.path.join(out_dir, "iso_config.normalized.yaml"), "w") as f:
            yaml.safe_dump(iso, f, sort_keys=False)
        with open(env_cfg["pcs_unit_config_path"], "r") as f:
            pcs = _normalize(yaml.safe_load(f) or {})
        with open(os.path.join(out_dir, "pcs_unit_config.normalized.yaml"), "w") as f:
            yaml.safe_dump(pcs, f, sort_keys=False)
    except Exception:
        # Best-effort only
        pass

    return env_cfg


def debug_print_summary(env_cfg: Dict[str, Any]) -> str:
    """Return a small JSON string summarizing key fields for logs."""
    t = env_cfg.get("time", {})
    iso_p = env_cfg.get("iso_config_path")
    pcs_p = env_cfg.get("pcs_unit_config_path")
    summary = {
        "time": {k: t.get(k) for k in ("step_duration", "max_steps_per_episode")},
        "iso_config_path": iso_p,
        "pcs_unit_config_path": pcs_p,
    }
    return json.dumps(summary, indent=2)


