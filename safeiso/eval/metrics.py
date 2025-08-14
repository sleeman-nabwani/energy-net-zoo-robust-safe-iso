from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np
import torch


def rollout_metrics(env, episodes: int = 5, policy_act: Callable | None = None, deterministic: bool = True) -> Dict[str, Any]:
	"""
	Run episodes and return aggregated metrics:
	  - reward_mean
	  - avg_step_cost_mean
	  - ep_len_mean
	  - violation_counts: {type: total across all episodes}
	  - violation_rate_by_type: {type: total_steps_with_violation/total_steps}
	  - steps_total
	If policy_act is None, take random actions from env.action_space.
	"""
	rew_list, avg_cost_list, len_list = [], [], []
	vio_counts: Dict[str, int] = {}
	steps_total = 0
	for _ in range(max(1, episodes)):
		obs, _info = env.reset()
		done = torch.zeros((1,), dtype=torch.bool, device=obs.device)
		ep_r = 0.0
		ep_c = 0.0
		n = 0
		while not done.item():
			if policy_act is None:
				action = env.action_space.sample()
				# If env is wrapped to expect batched actions (e.g., Saute Unsqueeze), add batch dim
				if isinstance(action, np.ndarray) and action.ndim == 1 and hasattr(obs, 'shape') and len(obs.shape) == 2 and obs.shape[0] == 1:
					action = action[None, :]
			else:
				action = policy_act(obs, deterministic=deterministic)
			# Convert numpy action to torch on the same device as obs if needed (OmniSafe wrappers expect torch)
			if isinstance(action, np.ndarray) and isinstance(obs, torch.Tensor):
				action_t = torch.as_tensor(action, dtype=torch.float32, device=obs.device)
			else:
				action_t = action
			obs, rew, cost, term, trunc, info = env.step(action_t)
			ep_r += float(rew) if not hasattr(rew, "item") else float(rew.item())
			ep_c += float(cost) if not hasattr(cost, "item") else float(cost.item())
			n += 1
			steps_total += 1
			vio = info[0].get("violations", {}) if isinstance(info, (list, tuple)) else info.get("violations", {})
			for k, v in vio.items():
				vio_counts[k] = vio_counts.get(k, 0) + (1 if v else 0)
			done = term | trunc
		rew_list.append(ep_r)
		avg_cost_list.append(ep_c / max(1, n))
		len_list.append(n)
	violation_rate = {k: (v / max(1, steps_total)) for k, v in vio_counts.items()}
	return {
		"reward_mean": float(np.mean(rew_list)) if rew_list else 0.0,
		"avg_step_cost_mean": float(np.mean(avg_cost_list)) if avg_cost_list else 0.0,
		"ep_len_mean": float(np.mean(len_list)) if len_list else 0.0,
		"violation_counts": vio_counts,
		"violation_rate_by_type": violation_rate,
		"steps_total": steps_total,
	} 