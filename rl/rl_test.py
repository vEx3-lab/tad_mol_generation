import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# ----------------------------
# 1. Policy 网络
# ----------------------------
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return Categorical(logits=self.net(x))


# ----------------------------
# 2. 采样一整条 trajectory
# ----------------------------
@torch.no_grad()
def sample_episode(env, policy):
    obs, _ = env.reset()
    done = False

    states, actions, logps = [], [], []
    total_reward = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        dist = policy(obs_t)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(obs_t)
        actions.append(action)
        logps.append(dist.log_prob(action))

        total_reward += reward
        obs = next_obs

    return {
        "states": states,
        "actions": actions,
        "logps": logps,
        "reward": total_reward,
        "length": len(actions)
    }


# ----------------------------
# 3. GSPO loss（序列级）
# ----------------------------
def gspo_loss(
    policy,
    ref_policy,
    trajectories,
    advantages,
    clip_eps=0.2,
    kl_beta=0.01
):
    losses = []
    kl_terms = []

    for traj, A in zip(trajectories, advantages):
        states = torch.stack(traj["states"])
        actions = torch.stack(traj["actions"])
        logp_old = torch.stack(traj["logps"]).detach()

        dist_new = policy(states)
        logp_new = dist_new.log_prob(actions)

        # -------- GSPO 核心：序列级 ratio --------
        log_ratio = (logp_new - logp_old).mean()
        ratio = torch.exp(log_ratio)

        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surrogate = torch.min(ratio * A, clipped * A)

        # -------- KL(πθ || πref) --------
        with torch.no_grad():
            ref_dist = ref_policy(states)
        kl = torch.distributions.kl_divergence(dist_new, ref_dist).mean()

        losses.append(-(surrogate - kl_beta * kl))
        kl_terms.append(kl.item())

    return torch.stack(losses).mean(), np.mean(kl_terms)


# ----------------------------
# 4. 训练主循环
# ----------------------------
def train():
    env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim)
    old_policy = Policy(obs_dim, act_dim)
    ref_policy = Policy(obs_dim, act_dim)

    old_policy.load_state_dict(policy.state_dict())
    ref_policy.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=5e-4)

    reward_history = []
    for iteration in range(300):
        trajectories = []
        rewards = []

        # -------- on-policy sampling --------
        for _ in range(16):
            traj = sample_episode(env, old_policy)
            trajectories.append(traj)
            rewards.append(traj["reward"])

        rewards = np.array(rewards)
        reward_history.append(rewards.mean())
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        loss, kl = gspo_loss(
            policy, ref_policy, trajectories, advantages
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        old_policy.load_state_dict(policy.state_dict())

        if iteration % 10 == 0:
            print(
                f"Iter {iteration:03d} | "
                f"Reward: {rewards.mean():.1f} | "
                f"Loss: {loss.item():.3f} | "
                f"KL: {kl:.4f}"
            )

    plt.figure()
    plt.plot(reward_history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.title("GSPO Reward Curve")
    plt.show()

    env.close()


if __name__ == "__main__":
    train()
