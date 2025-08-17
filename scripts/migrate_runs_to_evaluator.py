#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path


def infer_algo_mode_seed(run_dir: str):
    algo = "(unknown)"; mode = "CMDP"; seed = None
    m = re.search(r"(PPOLag|CPO|CUP|FOCOPS|SautePPO)_(cmdp|mdp)_", run_dir, re.I)
    if m:
        algo = m.group(1)
        mode = "CMDP" if m.group(2).lower() == "cmdp" else "ISOOnly"
    m = re.search(r"seed(\d+)", run_dir)
    if m:
        seed = int(m.group(1))
    return algo, mode, seed


def ensure_env_seed(cfg: dict, mode: str, seed: int | None):
    if not cfg.get("env_id"):
        cfg["env_id"] = "SafeISO-CMDP-omni-v0" if mode == "CMDP" else "SafeISO-ISOOnly-omni-v0"
    if seed is not None and "seed" not in cfg:
        cfg["seed"] = seed
    cfg.setdefault("dict_cfgs", {})


def migrate_dir(run_dir: str, dry: bool = False) -> tuple[bool, str]:
    run = Path(run_dir)
    cfg_p = run / "config.json"
    ts_dir = run / "torch_save"
    if not ts_dir.exists() or not list(ts_dir.glob("*.pt")):
        return False, "no checkpoints (*.pt)"
    cfg = {}
    if cfg_p.exists():
        try:
            cfg = json.loads(cfg_p.read_text())
        except Exception:
            pass
    algo, mode, seed = infer_algo_mode_seed(run_dir)
    ensure_env_seed(cfg, mode, seed)
    if not dry:
        cfg_p.write_text(json.dumps(cfg))
    return True, "patched config.json"


def main():
    ap = argparse.ArgumentParser(description="Patch legacy runs for OmniSafe Evaluator compatibility.")
    ap.add_argument("--roots", nargs="+", default=["runs"], help="Directories to scan recursively")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    targets = []
    for root in args.roots:
        for d in glob.glob(os.path.join(root, "**", "seed*", ""), recursive=True):
            if os.path.isdir(os.path.join(d, "torch_save")) and glob.glob(os.path.join(d, "torch_save", "*.pt")):
                targets.append(os.path.abspath(d))
    targets = sorted(set(targets))
    if not targets:
        print("No run dirs found.")
        return
    ok = err = 0
    for t in targets:
        o, msg = migrate_dir(t, dry=args.dry)
        print(("[OK] " if o else "[ERR]") + t + " :: " + msg)
        ok += int(o); err += int(not o)
    print(f"Done. OK={ok}, ERR={err}")


if __name__ == "__main__":
    main()


