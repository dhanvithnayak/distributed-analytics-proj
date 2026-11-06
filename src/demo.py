"""
demo.py
Generates high-definition animated GIF demonstrations of Comm-MAPPO and FIFO
for presentation slides and project showcases.
"""

import os
import imageio
import numpy as np
from src.env.intersection_env import IndustrialIntersectionEnv, IntersectionConfig
from src.env.renderer import IntersectionRenderer
from src.baselines.fifo_controller import FIFOController
from src.marl.mappo import CommMAPPO


def record_demonstration(
    strategy_name: str,
    output_gif: str,
    steps: int = 200,
    seed: int = 42
):
    print(f"Recording demonstration for: {strategy_name} -> {output_gif}")
    config = IntersectionConfig()
    config.max_steps = steps
    env = IndustrialIntersectionEnv(config=config)
    obs, adj, info = env.reset(seed=seed)

    renderer = IntersectionRenderer(width=600, height=600)
    frames = []

    if strategy_name == "fifo":
        controller = FIFOController()
    else:
        controller = CommMAPPO(obs_dim=env.obs_dim, action_dim=env.action_dim, max_vehicles=config.max_vehicles)
        if os.path.exists("models/best_comm_mappo.pt"):
            controller.load("models/best_comm_mappo.pt")
            print("Loaded Comm-MAPPO weights for demonstration.")

    for step in range(steps):
        if strategy_name == "fifo":
            actions = controller.get_actions(env.vehicles)
        else:
            actions, _, _ = controller.select_action(obs, adj, deterministic=True)

        obs, adj, rewards, dones, step_info = env.step(actions)

        # Record every frame (or every 2nd frame for compact GIF size)
        frame = renderer.render_frame(env, render_mode="rgb_array")
        if frame is not None:
            frames.append(frame)

    renderer.close()
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    imageio.mimsave(output_gif, frames[::2], fps=15)
    print(f"✓ Saved {len(frames[::2])} frames to: {output_gif} ({os.path.getsize(output_gif)/1024:.1f} KB)\n")


if __name__ == "__main__":
    record_demonstration("comm_mappo", "assets/demo_comm_mappo.gif", steps=150)
    record_demonstration("fifo", "assets/demo_baseline_fifo.gif", steps=150)
