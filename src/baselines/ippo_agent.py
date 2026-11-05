"""
ippo_agent.py
Independent PPO (IPPO) Baseline.
Non-cooperative Multi-Agent Reinforcement Learning without inter-vehicle communication.
Ablation baseline demonstrating why dynamic V2V communication is critical for bottleneck coordination.
"""

from src.marl.mappo import CommMAPPO


class IPPOAgent(CommMAPPO):
    """Independent PPO Agent without V2V communication."""
    
    def __init__(self, obs_dim: int = 24, action_dim: int = 1, max_vehicles: int = 8, **kwargs):
        # Force enable_comm=False to disable inter-agent message passing
        super(IPPOAgent, self).__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_vehicles=max_vehicles,
            enable_comm=False,
            **kwargs
        )
