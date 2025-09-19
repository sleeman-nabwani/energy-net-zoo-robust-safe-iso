## Slurm job usage and monitoring guide

This guide shows how to submit, monitor, and debug jobs for this repository on your Slurm cluster.

### Submit jobs

- Submit the baseline array job (uses throttling and GPU by default):
```bash
sbatch -p all /home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch
```
- Override parameters at submit time:
```bash
sbatch -p all --export=ALL,STEPS=96000,COST_LIMIT=0.2,PRESET=default,DEVICE=cuda \
  /home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch
```
- Run a single task (useful for dry runs):
```bash
sbatch -p all --array=0 /home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch
```

### Job arrays and throttling

- The script uses an array of 30 tasks (grid over algorithms, modes, seeds):
  - `#SBATCH --array=0-29%4` → at most 4 tasks run concurrently.
- Change concurrency at submit time (overrides script):
```bash
sbatch -p all --array=0-29%2 /home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch
```

### GPU requests

- The script requests one GPU per task: `#SBATCH --gres=gpu:1`.
- Verify the cluster supports this (your cluster does).
- Request a specific GPU type (optional):
```bash
sbatch -p all --gres=gpu:2080ti:1 /home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch
```

### Check job status

- All your jobs:
```bash
squeue -u "$USER" -o "%.18i %.9P %.8j %.2t %.10M %R"
```
- A specific job/array with reason:
```bash
squeue -j <JOBID> -o "%.18i %.2t %R"
```
- Common states:
  - PD: Pending (waiting); R: Running; CG: Completing; F/FAILED: Failed; CA: Cancelled; CD/COMPLETED: Finished.
- Common pending reasons you may see:
  - QOSMaxCpuPerUserLimit: over per-user CPU quota → reduce `--cpus-per-task` or throttle `%N`.
  - QOSMaxGRESPerUser: over per-user GPU quota → throttle `%N`.

### Inspect allocations (CPU/GPU, node)

- Accounting summary (requested vs allocated):
```bash
sacct -j <JOBID> -X --format=JobID,State,ReqTRES,AllocTRES%60,NodeList
```
- Detailed view of one task (check GPUs):
```bash
scontrol show job <JOBID_TASK> | egrep -i "TRES|Gres|TresPer|NodeList|State"
```
- GPU allocated if `AllocTRES` or `TRES` shows `gres/gpu=1`.

### Logs

- Logs are written by Slurm using this pattern from the script:
  - Stdout: `logs/%x_%A_%a.out` → e.g., `logs/safeiso_baseline_<JOBID>_<TASKID>.out`
  - Stderr: `logs/%x_%A_%a.err`
- Tail and follow:
```bash
tail -n 200 logs/safeiso_baseline_<JOBID>_<TASKID>.out
tail -f logs/safeiso_baseline_<JOBID>_<TASKID>.out
```
- Search for GPU and device diagnostics included by the script:
```bash
grep -E "CUDA_VISIBLE_DEVICES|torch.cuda|gpu_name|\"device\"" logs/safeiso_baseline_<JOBID>_*.out
```
- Evaluation summary printed at the end:
```bash
grep -H "eval_reward_mean\|eval_cost_avg_mean" logs/safeiso_baseline_<JOBID>_*.out
```

### Cancel jobs

- Cancel a whole array:
```bash
scancel <JOBID>
```
- Cancel only running tasks of an array:
```bash
scancel $(squeue -h -j <JOBID> -t R -o %i)
```
- Cancel everything for your user:
```bash
scancel -u "$USER"
```

### Resource tuning

- Concurrency: control with array throttle `%N` (e.g., `--array=0-29%4`).
- CPUs: `#SBATCH --cpus-per-task=4` in the script; lowering to 2 may ease QOS pressure.
- Memory/time: `#SBATCH --mem=24G`, `#SBATCH --time=06:00:00` — adjust as needed.
- Partition: your cluster’s default is `all`. Use `-p all` (or a GPU partition if different).

### Repository-specific notes

- Script path: `/home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch`.
- The script handles environment activation using absolute Miniconda paths and prints CUDA diagnostics on start.
- Algorithms grid includes: PPOLag, CUP, CPO, FOCOPS, SautePPO; modes: `cmdp` and `mdp`.
- Known OmniSafe config expectations handled in code:
  - SautePPO: `algo_cfgs.safety_budget` is set from `cost_limit * max_episode_steps`.
  - CPO: default `algo_cfgs.fvp_sample_freq=1`.
  - FOCOPS: default `algo_cfgs.focops_lam=1.0`.

### Quick reference

```bash
# Submit (GPU, throttled by script)
sbatch -p all /home/sleemann/energy-net-zoo-robust-safe-iso/safeiso/slurm/train_baseline_static.sbatch

# Live status and reasons
squeue -u "$USER" -o "%.18i %.9P %.8j %.2t %.10M %R"

# Allocations (look for gres/gpu=1)
sacct -j <JOBID> -X --format=JobID,State,ReqTRES,AllocTRES%60,NodeList

# Inspect a running task’s resources
aTask=<JOBID>_0; scontrol show job "$aTask" | egrep -i "TRES|Gres|NodeList|State"

# Logs for a task
tail -n 200 logs/safeiso_baseline_<JOBID>_<TASKID>.out

# Cancel an array
scancel <JOBID>
``` 