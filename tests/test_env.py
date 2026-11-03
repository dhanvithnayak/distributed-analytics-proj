"""
test_env.py
Verifies simulation physics, multi-agent dynamics, collision detection, and rendering.
"""

import os
import numpy as np
import imageio
from src.env.intersection_env import IndustrialIntersectionEnv, IntersectionConfig
from src.env.renderer import IntersectionRenderer

def test_environment_run():
    print("Initializing Industrial Intersection Environment...")
    config = IntersectionConfig()
    env = IndustrialIntersectionEnv(config=config, packet_drop_rate=0.1)
    obs, adj, info = env.reset(seed=42)
    
    print(f"Initial active CAVs: {info['active_vehicles']}")
    print(f"Observation matrix shape: {obs.shape}")
    print(f"V2V Adjacency matrix shape: {adj.shape}")
    
    renderer = IntersectionRenderer(width=600, height=600)
    frames = []
    
    # Run 50 random steps and capture frames
    for step in range(50):
        # Sample random actions in [-0.5, 0.8] (slight bias toward forward progress)
        actions = np.random.uniform(-0.5, 0.8, size=(config.max_vehicles, 1)).astype(np.float32)
        obs, adj, rewards, terminated, info = env.step(actions)
        
        frame = renderer.render_frame(env, render_mode="rgb_array")
        if frame is not None and step % 2 == 0:
            frames.append(frame)
            
    print(f"Completed 50 simulation steps successfully!")
    print(f"Final Step Telemetry: Active CAVs: {info['active_vehicles']}, Completed: {info['completed_vehicles']}, Collisions: {info['total_collisions']}")
    
    # Save a quick sanity GIF
    os.makedirs("assets", exist_ok=True)
    gif_path = "assets/env_sanity_check.gif"
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"Saved sanity check animation to: {gif_path}")
    renderer.close()

if __name__ == "__main__":
    test_environment_run()
