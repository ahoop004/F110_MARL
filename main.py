import yaml
import numpy as np
from f110x.envs import F110ParallelEnv

def random_policy(action_space, obs):
    return action_space.sample()

def simple_heuristic(action_space,obs):
    # Example: small forward velocity, no steer
    return np.array([0.0, 1.0], dtype=np.float32)

POLICIES = {
    "random": random_policy,
    "heuristic": simple_heuristic,
}


with open("/home/aaron/F110_MARL/configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
with open("/home/aaron/F110_MARL/scenarios/test.yaml") as f:
    scenario_cfg = yaml.safe_load(f)
scenario = scenario_cfg["agents"]

env = F110ParallelEnv(**cfg["env"])

def run_eval(env, policy_fn, episodes=5):
    results = []
    for ep in range(episodes):
        obs, infos = env.reset()
        done = {aid: False for aid in env.agents}
        totals = {aid: 0.0 for aid in env.agents}

        while not all(done.values()):
            actions = {}
            for aid in env.agents:
                if not done[aid]:
                    actions[aid] = policy_fn(env.action_space(aid), obs[aid])
            obs, rewards, terms, truncs, infos = env.step(actions)
            for aid, r in rewards.items():
                totals[aid] += r
            done = {aid: terms[aid] or truncs[aid] for aid in env.possible_agents}
        results.append(totals)
    return results


for name, fn in POLICIES.items():
    scores = run_eval(env, fn, episodes=3)
    print(name, scores)