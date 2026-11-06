"""
train.py
Training pipeline for Comm-MAPPO and IPPO in the Industrial Intersection Environment.
Uses Generalized Advantage Estimation (GAE) and Centralized Value Critic.
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
from src.env.intersection_env import IndustrialIntersectionEnv, IntersectionConfig
from src.marl.mappo import CommMAPPO
from src.baselines.ippo_agent import IPPOAgent


class RolloutBuffer:
    """Stores multi-agent trajectory rollouts and computes GAE advantages."""
    
    def __init__(self, buffer_size: int, max_vehicles: int, obs_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.max_vehicles = max_vehicles
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.buffer_size, self.max_vehicles, self.obs_dim), dtype=np.float32)
        self.adj = np.zeros((self.buffer_size, self.max_vehicles, self.max_vehicles), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.max_vehicles, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.max_vehicles), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.max_vehicles), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.max_vehicles), dtype=bool)
        self.values = np.zeros((self.buffer_size, self.max_vehicles), dtype=np.float32)
        self.ptr = 0

    def insert(self, obs, adj, actions, log_probs, rewards, dones, values):
        self.obs[self.ptr] = obs
        self.adj[self.ptr] = adj
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray) -> dict:
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = np.zeros(self.max_vehicles, dtype=np.float32)

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
                next_values = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + self.values

        # Normalize advantages
        adv_mean = np.mean(advantages[:self.ptr])
        adv_std = np.std(advantages[:self.ptr]) + 1e-8
        norm_advantages = (advantages[:self.ptr] - adv_mean) / adv_std

        return {
            "obs": torch.as_tensor(self.obs[:self.ptr]),
            "adj": torch.as_tensor(self.adj[:self.ptr]),
            "actions": torch.as_tensor(self.actions[:self.ptr]),
            "log_probs": torch.as_tensor(self.log_probs[:self.ptr]),
            "advantages": torch.as_tensor(norm_advantages),
            "returns": torch.as_tensor(returns[:self.ptr])
        }


def train(
    algo: str = "comm_mappo",
    total_episodes: int = 60,
    steps_per_episode: int = 400,
    save_dir: str = "models",
    log_dir: str = "results"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"==================================================")
    print(f"Starting Training: {algo.upper()}")
    print(f"Episodes: {total_episodes} | Steps/Ep: {steps_per_episode}")
    print(f"==================================================")

    config = IntersectionConfig()
    config.max_steps = steps_per_episode
    env = IndustrialIntersectionEnv(config=config)

    # Initialize agent
    if algo == "comm_mappo":
        agent = CommMAPPO(obs_dim=env.obs_dim, action_dim=env.action_dim, max_vehicles=config.max_vehicles)
    else:
        agent = IPPOAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, max_vehicles=config.max_vehicles)

    buffer = RolloutBuffer(steps_per_episode, config.max_vehicles, env.obs_dim)
    training_logs = []
    best_throughput = -1.0
    start_time = time.time()

    for ep in range(1, total_episodes + 1):
        obs, adj, info = env.reset(seed=ep * 10)
        buffer.reset()
        ep_rewards = []

        for step in range(steps_per_episode):
            actions, log_probs, values = agent.select_action(obs, adj)
            next_obs, next_adj, rewards, dones, step_info = env.step(actions)

            buffer.insert(obs, adj, actions, log_probs, rewards, dones, values)
            ep_rewards.append(np.sum(rewards))

            obs = next_obs
            adj = next_adj

        # Final step value estimate for GAE
        _, _, last_values = agent.select_action(obs, adj)
        trajectory_data = buffer.compute_returns_and_advantages(last_values)
        
        # PPO Update
        loss_dict = agent.train_step(trajectory_data, ppo_epochs=4, batch_size=64)
        
        # Telemetry
        final_info = env._get_info()
        throughput = final_info["throughput_veh_per_step"] * 1000.0
        avg_wait = final_info["average_wait_steps"]
        collisions = final_info["total_collisions"]
        total_ep_reward = float(np.sum(ep_rewards))

        log_entry = {
            "episode": ep,
            "total_reward": total_ep_reward,
            "completed_vehicles": final_info["completed_vehicles"],
            "throughput_per_1000": throughput,
            "collisions": collisions,
            "avg_wait_steps": avg_wait,
            "actor_loss": loss_dict["actor_loss"],
            "critic_loss": loss_dict["critic_loss"],
            "time_elapsed_sec": time.time() - start_time
        }
        training_logs.append(log_entry)

        if ep % 5 == 0 or ep == total_episodes:
            print(f"Ep {ep:02d}/{total_episodes} | Rew: {total_ep_reward:7.1f} | Completed: {final_info['completed_vehicles']:2d} | "
                  f"Throughput/1k: {throughput:5.1f} | Collisions: {collisions} | Wait: {avg_wait:4.1f} | "
                  f"ActLoss: {loss_dict['actor_loss']:6.3f} | CritLoss: {loss_dict['critic_loss']:6.3f}")

        # Checkpoint best policy
        if throughput > best_throughput:
            best_throughput = throughput
            save_path = os.path.join(save_dir, f"best_{algo}.pt")
            agent.save(save_path)

    # Save training metrics CSV
    log_file = os.path.join(log_dir, f"training_log_{algo}.csv")
    pd.DataFrame(training_logs).to_csv(log_file, index=False)
    print(f"✓ Training finished. Saved model to {save_path} and logs to {log_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="comm_mappo", choices=["comm_mappo", "ippo", "all"])
    parser.add_argument("--episodes", type=int, default=60)
    args = parser.parse_args()

    if args.algo == "all":
        train("comm_mappo", total_episodes=args.episodes)
        train("ippo", total_episodes=args.episodes)
    else:
        train(args.algo, total_episodes=args.episodes)
