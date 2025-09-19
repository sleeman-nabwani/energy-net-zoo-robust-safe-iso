import os, numpy as np, pytest
from safeiso.eval.evalpack_loader import load_evalpack


def test_evalpack_determinism_and_bounds():
    root = "runs/baseline_static"
    target = None
    for dirpath, _, files in os.walk(root):
        if dirpath.endswith("seed0") and os.path.isfile(os.path.join(dirpath, "evalpack", "actor.ts")):
            target = dirpath; break
    if target is None:
        pytest.skip("No EvalPack run found (train baseline first)")
    act, meta = load_evalpack(target)
    low = np.array(meta["action_space"]["low"], dtype=np.float32)
    high = np.array(meta["action_space"]["high"], dtype=np.float32)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((1,), dtype=np.float32)
    a1 = act(obs, True); a2 = act(obs, True)
    assert np.allclose(a1, a2)
    assert a1.shape == (1,)
    assert np.all(a1 >= low) and np.all(a1 <= high)


