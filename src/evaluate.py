"""
evaluate.py
Standardized evaluation benchmark comparing FIFO, IDM-TTC, IPPO, and Comm-MAPPO.
Includes cyber-physical network stress testing (packet loss sweep).
"""

import os
import argparse
import numpy as np
import pandas as pd
from src.env.intersection_env import IndustrialIntersectionEnv, IntersectionConfig
from src.baselines.fifo_controller import FIFOController
from src.baselines.idm_ttc_agent import IDMTTCAgent
from src.baselines.ippo_agent import IPPOAgent
from src.marl.mappo import CommMAPPO
from src.analytics.metrics import FleetMetricsTracker


def run_benchmark(num_episodes: int = 20, steps_per_ep: int = 400, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    tracker = FleetMetricsTracker()
    
    config = IntersectionConfig()
    config.max_steps = steps_per_ep
    env = IndustrialIntersectionEnv(config=config)

    # Instantiate agents
    fifo = FIFOController()
    idm = IDMTTCAgent()
    
    ippo = IPPOAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, max_vehicles=config.max_vehicles)
    if os.path.exists("models/best_ippo.pt"):
        ippo.load("models/best_ippo.pt")
        print("Loaded trained IPPO checkpoint.")

    comm_mappo = CommMAPPO(obs_dim=env.obs_dim, action_dim=env.action_dim, max_vehicles=config.max_vehicles)
    if os.path.exists("models/best_comm_mappo.pt"):
        comm_mappo.load("models/best_comm_mappo.pt")
        print("Loaded trained Comm-MAPPO checkpoint.")

    strategies = [
        ("FIFO (Centralized)", fifo),
        ("IDM-TTC (Decentralized)", idm),
        ("IPPO (No Comms)", ippo),
        ("Comm-MAPPO (Proposed)", comm_mappo)
    ]

    print(f"\n=======================================================")
    print(f"Executing Operational Benchmark across {num_episodes} test episodes")
    print(f"=======================================================")

    for name, agent in strategies:
        print(f"Evaluating strategy: {name}...")
        for ep in range(num_episodes):
            seed = 1000 + ep
            obs, adj, info = env.reset(seed=seed)

            for step in range(steps_per_ep):
                if isinstance(agent, FIFOController) or isinstance(agent, IDMTTCAgent):
                    actions = agent.get_actions(env.vehicles)
                else:
                    actions, _, _ = agent.select_action(obs, adj, deterministic=True)

                obs, adj, rewards, dones, step_info = env.step(actions)

            final_info = env._get_info()
            tracker.record_episode(
                strategy_name=name,
                episode_idx=ep,
                steps=steps_per_ep,
                completed_vehicles=final_info["completed_vehicles"],
                collisions=final_info["total_collisions"],
                avg_speed=final_info["average_speed"],
                avg_wait_steps=final_info["average_wait_steps"],
                avg_jerk=final_info["average_jerk"]
            )

    df_results = tracker.to_dataframe()
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparative benchmark results to: {csv_path}")

    # Summary table
    summary_df = tracker.summary()
    print("\n--- Benchmark Performance Summary ---")
    print(summary_df)

    # 2. Network Stress Test: Packet Drop Rate Sweep for Comm-MAPPO
    print(f"\n=======================================================")
    print(f"Executing Cyber-Physical Network Stress Test (Packet Loss Sweep)")
    print(f"=======================================================")
    
    packet_loss_records = []
    loss_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for p_loss in loss_rates:
        print(f"Testing Comm-MAPPO under packet drop rate: {p_loss*100:.0f}%...")
        env_loss = IndustrialIntersectionEnv(config=config, packet_drop_rate=p_loss)
        
        ep_completed = []
        ep_wait = []
        ep_collisions = []

        for ep in range(10):
            obs, adj, info = env_loss.reset(seed=2000 + ep)
            for step in range(steps_per_ep):
                actions, _, _ = comm_mappo.select_action(obs, adj, deterministic=True)
                obs, adj, rewards, dones, step_info = env_loss.step(actions)

            f_info = env_loss._get_info()
            ep_completed.append(f_info["completed_vehicles"])
            ep_wait.append(f_info["average_wait_steps"])
            ep_collisions.append(f_info["total_collisions"])

        packet_loss_records.append({
            "packet_drop_rate": p_loss,
            "mean_completed": np.mean(ep_completed),
            "std_completed": np.std(ep_completed),
            "throughput_per_1000": (np.mean(ep_completed) / steps_per_ep) * 1000.0,
            "mean_wait_steps": np.mean(ep_wait),
            "total_collisions": np.sum(ep_collisions)
        })

    df_loss = pd.DataFrame(packet_loss_records)
    loss_csv = os.path.join(output_dir, "packet_loss_sweep.csv")
    df_loss.to_csv(loss_csv, index=False)
    print(f"✓ Saved packet loss sweep results to: {loss_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    run_benchmark(num_episodes=args.episodes)
