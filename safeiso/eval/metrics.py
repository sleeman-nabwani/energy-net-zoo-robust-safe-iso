from __future__ import annotations
from typing import Callable, Dict, Any, List
import numpy as np
import torch


def _to_torch_like(action, ref_obs):
	"""Convert numpy action to torch on the same device as obs if obs is torch; otherwise return as-is."""
	try:
		import torch  # imported lazily
	except Exception:
		torch = None

	if torch is not None and isinstance(ref_obs, torch.Tensor):
		return torch.as_tensor(action, dtype=torch.float32, device=ref_obs.device)
	return action


def _as_bool(x) -> bool:
	"""Torch/numpy/python bool → python bool."""
	return bool(x.item()) if hasattr(x, "item") else bool(x)


def _extract_info_dict(info) -> Dict[str, Any]:
	"""Support gymnasium's info (dict) and vec env style [info]."""
	if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
		return info[0]
	return info if isinstance(info, dict) else {}


def _derive_violations(info: Dict[str, Any]) -> Dict[str, int]:
	"""
	Prefer an explicit 'violations' dict if present; otherwise derive
	common types from SafeISO cost_debug/fields:
	  - 'shortfall': shortfall > 0
	  - 'freq': frequency out-of-band or hard_trigger mentions 'freq'
	  - 'reserve': hard_trigger mentions 'reserve'
	"""
	vio = {}
	maybe = info.get("violations")
	if isinstance(maybe, dict):
		for k, v in maybe.items():
			try:
				vio[k] = int(bool(v))
			except Exception:
				pass
		return vio

	dbg = info.get("cost_debug", {}) or {}
	vf = dbg.get("vf_states", {}) or {}
	freq_hz = float(vf.get("freq_hz", 50.0))
	shortfall = float(info.get("shortfall", 0.0)) > 0.0
	hard = str(dbg.get("hard_trigger", "")).lower()
	freq = (freq_hz < 49.0) or (freq_hz > 51.0) or ("freq" in hard)
	reserve = ("reserve" in hard)
	vio["shortfall"] = int(shortfall)
	vio["freq"] = int(freq)
	vio["reserve"] = int(reserve)
	return vio


def rollout_metrics(
	env,
	episodes: int = 5,
	policy_act: Callable | None = None,
	deterministic: bool = True,
) -> Dict[str, Any]:
	"""
	Run `episodes` episodes and return aggregate metrics.

	Returns
	-------
	dict with:
	  - reward_mean
	  - avg_step_cost_mean
	  - ep_len_mean
	  - violation_counts: {type: total across all episodes}
	  - violation_rate_by_type: {type: total_steps_with_violation / total_steps}
	  - steps_total

	Notes
	-----
	- Works with both 6-tuple (obs, reward, cost, terminated, truncated, info)
	  and 5-tuple (obs, reward, terminated, truncated, info) envs.
	- Robust to torch/numpy obs; converts numpy actions to torch if needed.
	- Counts violations via `info['violations']` if present; otherwise derives
	  shortfall/frequency/reserve heuristics from common SafeISO info fields.
	"""
	rew_list: List[float] = []
	avg_cost_list: List[float] = []
	len_list: List[int] = []
	vio_counts: Dict[str, int] = {}
	steps_total = 0

	for _ in range(max(1, episodes)):
		obs, info0 = env.reset()
		ep_r = 0.0
		ep_cost_sum = 0.0
		n_steps = 0
		done = False

		while not done:
			# Choose action
			if policy_act is None:
				action = env.action_space.sample()
			else:
				action = policy_act(obs, deterministic=deterministic)

			# Convert action type to match env expectations (torch if obs is torch)
			action_t = _to_torch_like(action, obs)

			step = env.step(action_t)
			# Support 6-tuple (CMDP) vs 5-tuple
			if len(step) == 6:
				obs, rew, cost, terminated, truncated, info = step
			else:
				obs, rew, terminated, truncated, info = step
				# Fallback cost from info if adapter writes it
				info_d = _extract_info_dict(info)
				cost = float(info_d.get("cost", 0.0))

			# Numbers to python floats
			rew_f = float(rew.item()) if hasattr(rew, "item") else float(rew)
			cost_f = float(cost.item()) if hasattr(cost, "item") else float(cost)

			ep_r += rew_f
			ep_cost_sum += cost_f
			n_steps += 1
			steps_total += 1

			# Violations
			inf = _extract_info_dict(info)
			vio = _derive_violations(inf)
			for k, v in vio.items():
				vio_counts[k] = vio_counts.get(k, 0) + (1 if v else 0)

			done = _as_bool(terminated) or _as_bool(truncated)

		rew_list.append(ep_r)
		avg_cost_list.append(ep_cost_sum / max(1, n_steps))
		len_list.append(n_steps)

	violation_rate = {k: (v / max(1, steps_total)) for k, v in vio_counts.items()}
	return {
		"reward_mean": float(np.mean(rew_list)) if rew_list else 0.0,
		"avg_step_cost_mean": float(np.mean(avg_cost_list)) if avg_cost_list else 0.0,
		"ep_len_mean": float(np.mean(len_list)) if len_list else 0.0,
		"violation_counts": vio_counts,
		"violation_rate_by_type": violation_rate,
		"steps_total": int(steps_total),
	}