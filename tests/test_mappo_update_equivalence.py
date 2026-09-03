from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from agents.mappo import MAPPOAgent, MAPPORolloutBuffer


def _make_agent(device: str = "cpu") -> MAPPOAgent:
    return MAPPOAgent(
        obs_dim=5,
        global_state_dim=4,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        agent_ids=["car_0", "car_1", "car_2", "car_3"],
        params={
            "device": device,
            "hidden_dims": [8],
            "n_steps": 8,
            "n_epochs": 2,
            "batch_size": 8,
            "critic_mode": "agent_conditioned",
        },
    )


def _fill_rollout(agent: MAPPOAgent) -> np.ndarray:
    rng = np.random.default_rng(321)
    rollout = rng.normal(size=agent._rollout_storage.shape).astype(np.float32)
    action_start = agent.obs_dim + agent.global_state_dim
    action_end = action_start + agent.action_dim
    rollout[:, :, action_start:action_end] = np.tanh(
        rollout[:, :, action_start:action_end]
    ) * 0.8
    rollout[:, :, action_end + 3 :] = 0.0
    rollout[0, 3, action_end + 3] = 1.0
    rollout[1, 5, action_end + 4] = 1.0
    agent._rollout_storage.copy_(torch.from_numpy(rollout).to(agent.device))
    for buffer in agent.buffers.values():
        buffer.ptr = agent.n_steps
    return rng.normal(size=agent.global_state_dim).astype(np.float32)


def _legacy_update(
    agent: MAPPOAgent, next_global_state: np.ndarray
) -> dict[str, float]:
    all_obs = []
    all_gs = []
    all_acts = []
    all_old_lp = []
    all_adv = []
    all_ret = []
    rollout_agent_ids = [
        aid for aid in agent.agent_ids if agent.buffers[aid].size() > 0
    ]
    next_values = agent.evaluate_states(next_global_state, rollout_agent_ids)
    for aid in rollout_agent_ids:
        buffer = agent.buffers[aid]
        n = buffer.size()
        adv, ret = buffer.compute_gae(
            next_values[aid], agent.gamma, agent.gae_lambda
        )
        all_obs.append(buffer.obs[:n])
        all_gs.append(agent._critic_batch(buffer.global_states[:n], aid))
        all_acts.append(buffer.actions[:n])
        all_old_lp.append(buffer.log_probs[:n])
        all_adv.append(adv)
        all_ret.append(ret)

    obs_pool = torch.cat(all_obs, dim=0)
    gs_pool = torch.cat(all_gs, dim=0)
    acts_pool = torch.cat(all_acts, dim=0)
    old_lp_pool = torch.cat(all_old_lp, dim=0)
    adv_pool = torch.cat(all_adv, dim=0)
    ret_pool = torch.cat(all_ret, dim=0)
    adv_pool = (adv_pool - adv_pool.mean()) / (
        adv_pool.std(correction=0) + 1e-8
    )

    totals = [0.0, 0.0, 0.0, 0.0]
    updates = 0
    for _ in range(agent.n_epochs):
        idx_all = torch.randperm(obs_pool.shape[0], device=agent.device)
        for start in range(0, obs_pool.shape[0], agent.batch_size):
            idx = idx_all[start : start + agent.batch_size]
            obs_b = obs_pool[idx]
            gs_b = gs_pool[idx]
            acts_b = acts_pool[idx]
            old_lp_b = old_lp_pool[idx]
            adv_b = adv_pool[idx]
            ret_b = ret_pool[idx]
            new_lp_b, entropy_b = agent.actor.evaluate_actions(obs_b, acts_b)
            ratio = (new_lp_b - old_lp_b).exp()
            pi_loss = torch.max(
                -adv_b * ratio,
                -adv_b
                * ratio.clamp(1 - agent.clip_range, 1 + agent.clip_range),
            ).mean()
            vf_loss = nn.functional.mse_loss(agent.critic(gs_b), ret_b)
            entropy = entropy_b.mean()
            loss = (
                pi_loss
                + agent.vf_coef * vf_loss
                - agent.ent_coef * entropy
            )
            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent._optim_parameters, agent.max_grad_norm)
            agent.optimizer.step()
            with torch.no_grad():
                approx_kl = ((old_lp_b - new_lp_b).mean()).abs().item()
            totals[0] += pi_loss.item()
            totals[1] += vf_loss.item()
            totals[2] += entropy.item()
            totals[3] += approx_kl
            updates += 1
    values = [value / updates for value in totals]
    return {
        "train/policy_loss": values[0],
        "train/value_loss": values[1],
        "train/entropy": values[2],
        "train/approx_kl": values[3],
    }


def test_packed_update_matches_legacy_losses_gradients_and_parameters() -> None:
    torch.manual_seed(123)
    legacy = _make_agent()
    optimized = _make_agent()
    optimized.actor.load_state_dict(copy.deepcopy(legacy.actor.state_dict()))
    optimized.critic.load_state_dict(copy.deepcopy(legacy.critic.state_dict()))
    optimized.optimizer.load_state_dict(copy.deepcopy(legacy.optimizer.state_dict()))
    next_state = _fill_rollout(legacy)
    optimized._rollout_storage.copy_(legacy._rollout_storage)
    for buffer in optimized.buffers.values():
        buffer.ptr = optimized.n_steps

    torch.manual_seed(999)
    expected_metrics = _legacy_update(legacy, next_state)
    torch.manual_seed(999)
    actual_metrics = optimized.update(next_state)

    for name in expected_metrics:
        assert actual_metrics[name] == pytest.approx(
            expected_metrics[name], rel=1e-6, abs=1e-7
        )
    for expected, actual in zip(
        legacy._optim_parameters, optimized._optim_parameters
    ):
        assert torch.equal(actual, expected)
        assert expected.grad is not None
        assert actual.grad is not None
        assert torch.equal(actual.grad, expected.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bulk_cuda_gae_matches_scalar_cpu_recurrence_exactly() -> None:
    packed = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [2.0, 0.4, 0.0, 0.0],
            [3.0, 0.6, 1.0, 0.0],
            [4.0, 0.8, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    def make_buffer(device: torch.device) -> MAPPORolloutBuffer:
        buffer = MAPPORolloutBuffer(4, 0, 0, 0, device)
        buffer.rewards.copy_(packed[:, 0].to(device))
        buffer.values.copy_(packed[:, 1].to(device))
        buffer.terminated.copy_(packed[:, 2].to(device))
        buffer.truncated.copy_(packed[:, 3].to(device))
        buffer.ptr = 4
        return buffer

    expected_adv, expected_ret = make_buffer(torch.device("cpu")).compute_gae(
        1.25, 0.99, 0.95
    )
    actual_adv, actual_ret = make_buffer(torch.device("cuda")).compute_gae(
        1.25, 0.99, 0.95
    )

    assert torch.equal(actual_adv.cpu(), expected_adv)
    assert torch.equal(actual_ret.cpu(), expected_ret)
