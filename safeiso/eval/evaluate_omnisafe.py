from __future__ import annotations
import argparse, glob, os, re
from typing import Iterable, List, Tuple

# Runner: evaluate ONE policy/run against a suite (deterministic)
from safeiso.eval.suite_eval import run_suite


def has_evalpack(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir, "evalpack", "actor.ts"))

def find_seed_dirs(root: str) -> List[str]:
    candidates = glob.glob(os.path.join(root, "**", "seed*", ""), recursive=True)
    return sorted([os.path.abspath(d) for d in candidates if has_evalpack(d)])

def normalize_targets(inputs: Iterable[str]) -> List[str]:
    targets: List[str] = []
    for path in inputs:
        path = os.path.abspath(path)
        if has_evalpack(path):
            targets.append(path)  # already a seed dir
        else:
            targets.extend(find_seed_dirs(path))
    return sorted(list(dict.fromkeys(targets)))

def infer_algo_mode(run_dir: str) -> Tuple[str, str]:
    """Infer (algo, mode) from a path segment like PPOLag_cmdp_default_static-0.0/seed0."""
    algo = "(unknown)"; mode = "CMDP"
    pat = re.compile(r"(PPOLag|CPO|CUP|FOCOPS|SautePPO)_(cmdp|mdp)_", re.IGNORECASE)
    for part in run_dir.split(os.sep):
        m = pat.search(part)
        if m:
            algo = m.group(1)
            mode = "CMDP" if m.group(2).lower() == "cmdp" else "ISOOnly"
            break
    return algo, mode

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate OmniSafe SafeISO runs against a scenario suite (discovers run dirs, calls suite runner)."
    )
    ap.add_argument("--baseline-only", action="store_true", help="Scan runs/baseline_static (default if nothing else set)")
    ap.add_argument("--all", action="store_true", help="Scan runs/ recursively")
    ap.add_argument("--dirs", nargs="*", default=[], help="Specific directories (algo dir or seed dir)")
    ap.add_argument("--suite", required=True, help="Path to suite YAML (e.g., safeiso/eval/suites/baseline_suite.yaml)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cost_limit", type=float, default=0.10)
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if eval CSVs exist")
    # Logging controls
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--dry_run", action="store_true", help="Discover targets only; do not run episodes")
    args = ap.parse_args()

    search_roots: List[str] = []
    if args.all:
        search_roots.append("runs")
    if args.baseline_only or (not args.all and not args.dirs):
        search_roots.append("runs/baseline_static")
    search_roots.extend(args.dirs)

    targets = normalize_targets(search_roots)
    if not targets:
        print("No runs found to evaluate.")
        return

    print(f"Discovered {len(targets)} runs.")
    failed = []
    for run_dir in targets:
        algo, mode = infer_algo_mode(run_dir)
        env_id = "SafeISO-CMDP-omni-v0" if mode == "CMDP" else "SafeISO-ISOOnly-omni-v0"

        out_dir = os.path.join(run_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        out_scen = os.path.join(out_dir, "stress_eval.csv")            # per-scenario
        out_eps  = os.path.join(out_dir, "stress_eval_episodes.csv")   # per-episode

        if (not args.overwrite) and os.path.isfile(out_scen) and os.path.isfile(out_eps):
            print(f"[skip] {run_dir} (already evaluated)")
            continue

        # Require EvalPack
        if not has_evalpack(run_dir):
            print(f"[skip] {run_dir} (no EvalPack found; retrain required)")
            continue

        print(f"[eval] {run_dir} -> {out_scen}")

        # Ensure env IDs registered in this process (harmless if already imported)
        try:
            import safeiso.utils.register_envs  # noqa: F401
        except Exception:
            pass

        # Run the suite for this run dir
        try:
            run_suite(
                env_id=env_id,
                suite_path=args.suite,
                policy=run_dir,
                episodes=args.episodes,
                horizon=args.horizon,
                base_seed=args.seed,
                algo=algo,
                mode=mode,
                device=args.device,
                cost_limit=args.cost_limit,
                out_scenarios=out_scen,
                out_episodes=out_eps,
                verbose=args.verbose,
                print_prompt=True,
                dry_run=args.dry_run,
            )
        except Exception as e:
            failed.append((run_dir, str(e)))
            print(f"[fail] {run_dir} :: {e}")

    if failed:
        print("Some runs failed EvalPack evaluation:")
        for p, err in failed:
            print(" -", p, "::", err)

    print("Done.")

if __name__ == "__main__":
    main()
