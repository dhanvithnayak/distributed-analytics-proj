"""
metrics.py
Industrial and Systems Engineering (ISE) Performance Metrics Evaluator.
Calculates throughput, queue delay, safety, and energy metrics for automated fleet systems.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class FleetMetricsTracker:
    """Tracks and aggregates operational logistics KPIs across simulation episodes."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_records: List[Dict[str, Any]] = []

    def record_episode(
        self,
        strategy_name: str,
        episode_idx: int,
        steps: int,
        completed_vehicles: int,
        collisions: int,
        avg_speed: float,
        avg_wait_steps: float,
        avg_jerk: float,
        packet_drop_rate: float = 0.0
    ):
        throughput_per_1k = (completed_vehicles / max(1, steps)) * 1000.0
        collision_rate = collisions / max(1, completed_vehicles + collisions)
        
        record = {
            "strategy": strategy_name,
            "episode": episode_idx,
            "steps": steps,
            "completed_vehicles": completed_vehicles,
            "throughput_per_1000_steps": throughput_per_1k,
            "collisions": collisions,
            "collision_rate": collision_rate,
            "avg_speed_mps": avg_speed,
            "avg_wait_steps": avg_wait_steps,
            "avg_jerk": avg_jerk,
            "packet_drop_rate": packet_drop_rate
        }
        self.episode_records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_records)

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return df.groupby("strategy").agg({
            "throughput_per_1000_steps": ["mean", "std"],
            "avg_wait_steps": ["mean", "std"],
            "collisions": ["mean", "sum"],
            "avg_speed_mps": ["mean", "std"],
            "avg_jerk": ["mean", "std"]
        })
