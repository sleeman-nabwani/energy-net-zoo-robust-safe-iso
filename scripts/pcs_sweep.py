import argparse, numpy as np, pandas as pd
from safeiso.utils.register_envs import SafeISOOmniEnv

def eval_static(k: float, steps=48, seed=0):
    env = SafeISOOmniEnv('SafeISO-ISOOnly-omni-v0', 
                         preset='default', pcs_spec=f"static:{k}", 
                         max_episode_steps=steps, device='cpu')
    obs,_ = env.reset(seed=seed)
    import torch
    # ISO action space is 3D (frequency, voltage, reserves), use zeros
    a_iso = torch.zeros(env.action_space.shape, dtype=torch.float32, device=getattr(obs,'device','cpu'))
    tot_c=0.0; n=0; short=0; freq=0; res=0
    while True:
        obs, r, c, term, trunc, info = env.step(a_iso)
        ii = info[0] if isinstance(info,(list,tuple)) else info
        vio = ii.get('violations', {})
        short += int(bool(vio.get('shortfall', False)))
        freq  += int(bool(vio.get('freq_oob', False)))
        res   += int(bool(vio.get('reserve_violation', False)))
        tot_c += float(c if not hasattr(c,'item') else c.item()); n+=1
        if (term|trunc).item(): break
    env.close()
    return tot_c/n, short, freq, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmin', type=float, default=-20)
    ap.add_argument('--kmax', type=float, default=20)
    ap.add_argument('--kstep', type=float, default=1)
    ap.add_argument('--steps', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--csv', type=str, default='pcs_sweep.csv')
    args = ap.parse_args()

    ks = np.arange(args.kmin, args.kmax + 1e-9, args.kstep)
    rows=[]
    for k in ks:
        c, sh, fq, rv = eval_static(float(k), args.steps, args.seed)
        rows.append(dict(k=float(k), avg_step_cost=c, shortfall_steps=sh, freq_oob_steps=fq, reserve_violation_steps=rv))
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print(df.head().to_string(index=False))
    # Show first relief point
    relief = df[df.shortfall_steps < args.steps]
    if not relief.empty:
        print("First k with partial relief (shortfall < steps):", float(relief.iloc[0].k))
    else:
        print("No relief found in range.")

if __name__ == "__main__":
    main()
