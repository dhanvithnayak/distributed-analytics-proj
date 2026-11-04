"""
fifo_controller.py
First-In, First-Out (FIFO) Virtual Traffic Light / Reservation Controller.
Standard industrial baseline: grants exclusive conflict-zone reservation
to the earliest arriving vehicle, forcing others to queue.
"""

from typing import List, Optional
import numpy as np
from src.env.kinematics import Vehicle


class FIFOController:
    """Centralized FIFO reservation baseline for industrial bottleneck intersections."""
    
    def __init__(self, conflict_radius: float = 8.0, stop_line_offset: float = 12.0):
        self.conflict_radius = conflict_radius
        self.stop_line_offset = stop_line_offset
        self.reserved_agent_id: Optional[int] = None

    def get_actions(self, vehicles: List[Vehicle]) -> np.ndarray:
        """
        Compute control action for all active vehicles based on FIFO reservation.
        Returns: array of shape (len(vehicles), 1) with normalized acceleration in [-1, 1].
        """
        active_vehicles = [v for v in vehicles if v.active]
        actions = np.zeros((len(vehicles), 1), dtype=np.float32)
        
        # 1. Update reservation status
        # If current occupant cleared the conflict zone, release reservation
        if self.reserved_agent_id is not None:
            occupant = next((v for v in active_vehicles if v.id == self.reserved_agent_id), None)
            if occupant is None or np.hypot(occupant.x, occupant.y) > self.conflict_radius:
                self.reserved_agent_id = None

        # 2. If no vehicle currently holds reservation, grant to vehicle closest to conflict center
        if self.reserved_agent_id is None:
            approaching_candidates = [
                v for v in active_vehicles
                if np.hypot(v.x, v.y) < (self.conflict_radius + self.stop_line_offset + 10.0)
                and (v.s < v.total_path_length / 2.0 + 5.0)  # Has not yet passed through
            ]
            if approaching_candidates:
                # Earliest / closest gets the reservation
                approaching_candidates.sort(key=lambda v: np.hypot(v.x, v.y))
                self.reserved_agent_id = approaching_candidates[0].id

        # 3. Compute acceleration for each vehicle
        for idx, vi in enumerate(vehicles):
            if not vi.active or vi.completed:
                continue

            dist_to_center = np.hypot(vi.x, vi.y)
            has_passed_center = vi.s > (vi.total_path_length / 2.0)

            # Check for leading vehicle in same lane
            leader_dist = float("inf")
            leader_speed = vi.config.target_speed
            for vj in active_vehicles:
                if vj.id != vi.id and vj.route_id == vi.route_id and vj.s > vi.s:
                    d = vj.s - vi.s
                    if d < leader_dist:
                        leader_dist = d
                        leader_speed = vj.speed

            # Car-following brake if too close to lane leader
            if leader_dist < vi.config.safe_distance * 1.5:
                # Hard brake to prevent rear-end collision
                action = -1.0
            elif leader_dist < vi.config.safe_distance * 2.5:
                # Match leader speed smoothly
                target_a = (leader_speed - vi.speed) / 2.0
                action = np.clip(target_a / vi.config.max_accel, -1.0, 1.0)
            elif not has_passed_center and dist_to_center <= (self.stop_line_offset + 2.0) and vi.id != self.reserved_agent_id:
                # Yield / Stop at the stop line before conflict zone
                if vi.speed > 1.0:
                    action = -0.8  # Decelerate to stop
                else:
                    action = -0.1  # Hold stationary in queue
            else:
                # Cruise towards target speed
                target_a = (vi.config.target_speed - vi.speed) / 2.0
                action = np.clip(target_a / vi.config.max_accel, -1.0, 1.0)

            actions[idx, 0] = float(action)

        return actions
