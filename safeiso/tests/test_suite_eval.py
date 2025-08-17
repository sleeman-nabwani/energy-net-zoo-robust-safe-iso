import os, csv
from safeiso.eval.suite_eval import run_suite


def test_suite_smoke(tmp_path):
    out_scen = tmp_path / "scen.csv"; out_ep = tmp_path / "ep.csv"
    root = "runs/baseline_static"
    run_dir = None
    for dirpath, _, files in os.walk(root):
        if dirpath.endswith("seed0") and os.path.isfile(os.path.join(dirpath, "evalpack", "actor.ts")):
            run_dir = dirpath; break
    if run_dir is None:
        return  # skip if no runs
    run_suite(env_id="SafeISO-CMDP-omni-v0", suite_path="safeiso/eval/suites/baseline_suite.yaml",
              policy=run_dir, episodes=2, horizon=48, base_seed=0,
              algo="(test)", mode="CMDP", device="cpu", cost_limit=0.10,
              out_scenarios=str(out_scen), out_episodes=str(out_ep),
              verbose=False, print_prompt=False, dry_run=False)
    assert out_scen.exists() and out_ep.exists()
    with open(out_scen) as f:
        headers = next(csv.reader(f))
    for must in ["algo","mode","avg_step_cost","return_mean","pcs","name","pass_cost_limit"]:
        assert must in headers


