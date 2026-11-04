"""
idm_ttc_agent.py
Intelligent Driver Model (IDM) + Time-To-Collision (TTC) Reactive Heuristic.
Decentralized rule-based baseline: each vehicle independently computes
longitudinal car-following and brakes reactively if cross-traffic TTC violates safety margins.
"""

from typing import List
import numpy as np
from src.env.kinematics import Vehicle


class IDMTTCAgent:
    """Decentralized IDM + Time-To-Collision cross-traffic yielding controller."""
    
    def __init__(
        self,
        s0: float = 4.0,           # Minimum jam distance (meters)
        time_headway: float = 1.2, # Desired time headway T (seconds)
        delta_exp: float = 4.0,    # Acceleration exponent
        ttc_threshold: float = 2.5 # Critical Time-To-Collision threshold (seconds)
    ):
        self.s0 = s0
        self.T = time_headway
        self.delta = delta_exp
        self.ttc_threshold = ttc_threshold

    def get_actions(self, vehicles: List[Vehicle]) -> np.ndarray:
        """
        Compute control action for all active vehicles.
        Returns: array of shape (len(vehicles), 1) with normalized acceleration in [-1, 1].
        """
        active_vehicles = [v for v in vehicles if v.active]
        actions = np.zeros((len(vehicles), 1), dtype=np.float32)

        for idx, vi in enumerate(vehicles):
            if not vi.active or vi.completed:
                continue

            a_max = vi.config.max_accel
            b_comfort = abs(vi.config.max_decel) * 0.6
            v0 = vi.config.target_speed
            v_curr = vi.speed

            # 1. Standard IDM Car-Following along same lane
            leader_dist = float("inf")
            delta_v = 0.0
            for vj in active_vehicles:
                if vj.id != vi.id and vj.route_id == vi.route_id and vj.s > vi.s:
                    d = vj.s - vi.s
                    if d < leader_dist:
                        leader_dist = d
                        delta_v = v_curr - vj.speed

            # Free-road acceleration term
            free_term = 1.0 - (v_curr / max(v0, 1e-4)) ** self.delta

            if leader_dist < 100.0:
                s_star = self.s0 + max(0.0, v_curr * self.T + (v_curr * delta_v) / (2.0 * np.sqrt(a_max * b_comfort)))
                interaction_term = (s_star / max(leader_dist, 0.5)) ** 2
            else:
                interaction_term = 0.0

            accel_idm = a_max * (free_term - interaction_term)

            # 2. Time-To-Collision (TTC) reactive cross-traffic check
            dist_to_center = np.hypot(vi.x, vi.y)
            has_passed = vi.s > (vi.total_path_length / 2.0)
            
            cross_yield = False
            if not has_passed and dist_to_center < 30.0:
                # Estimate time to reach intersection center
                ttc_self = dist_to_center / max(v_curr, 0.5)

                for vj in active_vehicles:
                    if vj.id != vi.id and vj.route_id != vi.route_id:
                        dist_j = np.hypot(vj.x, vj.y)
                        has_j_passed = vj.s > (vj.total_path_length / 2.0)
                        
                        if not has_j_passed and dist_j < 30.0:
                            ttc_other = dist_j / max(vj.speed, 0.5)
                            # If vehicles are predicted to arrive at conflict point simultaneously
                            if abs(ttc_self - ttc_other) < self.ttc_threshold:
                                # Tie-breaker: vehicle closer to center has right of way; farther yields
                                if dist_to_center > dist_j or (abs(dist_to_center - dist_j) < 1.0 and vi.route_id > vj.route_id):
                                    cross_yield = True
                                    break

            if cross_yield:
                # Apply defensive braking to avoid collision
                final_accel = vi.config.max_decel * 0.8
            else:
                final_accel = accel_idm

            # Normalize to [-1, 1]
            if final_accel >= 0:
                norm_action = np.clip(final_accel / vi.config.max_accel, 0.0, 1.0)
            else:
                norm_action = np.clip(final_accel / abs(vi.config.max_decel), -1.0, 0.0)

            actions[idx, 0] = float(norm_action)

        return actions
