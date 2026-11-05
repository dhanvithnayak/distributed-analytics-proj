"""
test_baselines_and_marl.py
Unit tests verifying FIFO, IDM-TTC, IPPO, and Comm-MAPPO implementations.
"""

import torch
import numpy as np
from src.env.intersection_env import IndustrialIntersectionEnv
from src.baselines.fifo_controller import FIFOController
from src.baselines.idm_ttc_agent import IDMTTCAgent
from src.baselines.ippo_agent import IPPOAgent
from src.marl.mappo import CommMAPPO


def test_baselines_and_marl():
    print("Testing Baselines and MARL Components...")
    env = IndustrialIntersectionEnv()
    obs, adj, info = env.reset(seed=123)

    # 1. Test FIFO Baseline
    fifo = FIFOController()
    fifo_actions = fifo.get_actions(env.vehicles)
    assert fifo_actions.shape == (len(env.vehicles), 1)
    assert np.all(fifo_actions >= -1.0) and np.all(fifo_actions <= 1.0)
    print("✓ FIFO Controller test passed.")

    # 2. Test IDM-TTC Baseline
    idm = IDMTTCAgent()
    idm_actions = idm.get_actions(env.vehicles)
    assert idm_actions.shape == (len(env.vehicles), 1)
    assert np.all(idm_actions >= -1.0) and np.all(idm_actions <= 1.0)
    print("✓ IDM+TTC Agent test passed.")

    # 3. Test Comm-MAPPO
    mappo = CommMAPPO(obs_dim=24, action_dim=1, max_vehicles=8)
    actions, log_probs, values = mappo.select_action(obs, adj)
    assert actions.shape == (8, 1)
    assert log_probs.shape == (8,)
    assert values.shape == (8,)
    print(f"✓ Comm-MAPPO forward pass test passed on device: {mappo.device}")

    # 4. Test IPPO
    ippo = IPPOAgent(obs_dim=24, action_dim=1, max_vehicles=8)
    ippo_actions, _, _ = ippo.select_action(obs, adj)
    assert ippo_actions.shape == (8, 1)
    print("✓ IPPO forward pass test passed.")

    print("\nAll baseline and MARL modules passed testing successfully!")

if __name__ == "__main__":
    test_baselines_and_marl()
