"""
kinematics.py
Continuous 2D vehicle kinematic modeling, collision detection, and V2V topology.
Designed for high-throughput vectorized simulation in industrial logistics networks.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VehicleConfig:
    length: float = 4.5        # meters
    width: float = 2.0         # meters
    max_speed: float = 12.0    # m/s (~43 km/h industrial logistics speed limit)
    target_speed: float = 8.0  # m/s (~29 km/h nominal flow speed)
    max_accel: float = 2.5     # m/s^2 (comfortable acceleration for industrial cargo)
    max_decel: float = -4.5    # m/s^2 (emergency braking capability)
    dt: float = 0.1            # seconds per control step (10 Hz control loop)
    comm_range: float = 40.0   # meters (V2V DSRC/C-V2X communication radius)
    sensor_range: float = 25.0 # meters (onboard LiDAR/radar sensing radius)
    safe_distance: float = 6.0 # meters minimum safe headway buffer


class Vehicle:
    """Represents an individual Connected and Autonomous Vehicle (CAV)."""
    
    def __init__(
        self,
        agent_id: int,
        route_id: int,
        path_waypoints: np.ndarray,
        initial_speed: float = 6.0,
        config: Optional[VehicleConfig] = None
    ):
        self.id = agent_id
        self.route_id = route_id
        self.config = config or VehicleConfig()
        
        # Path: Array of shape (M, 2) defining (x, y) coordinates of the route centerline
        self.path = path_waypoints
        self.path_index = 0
        self.s = 0.0  # Distance traveled along path (arc length)
        
        # Initial position & heading
        self.x = self.path[0, 0]
        self.y = self.path[0, 1]
        self.speed = initial_speed
        self.accel = 0.0
        self.prev_accel = 0.0
        self.heading = self._compute_heading(0)
        
        # Status
        self.active = True
        self.completed = False
        self.collided = False
        self.time_in_system = 0
        self.total_jerk = 0.0
        self.wait_steps = 0  # Steps with speed < 0.5 m/s
        
        # Precompute path total length
        diffs = np.diff(self.path, axis=0)
        self.segment_lengths = np.linalg.norm(diffs, axis=1)
        self.total_path_length = np.sum(self.segment_lengths)

    def _compute_heading(self, idx: int) -> float:
        """Compute heading angle (radians) at waypoint index."""
        if idx < len(self.path) - 1:
            dx = self.path[idx + 1, 0] - self.path[idx, 0]
            dy = self.path[idx + 1, 1] - self.path[idx, 1]
            return np.arctan2(dy, dx)
        elif idx > 0:
            dx = self.path[idx, 0] - self.path[idx - 1, 0]
            dy = self.path[idx, 1] - self.path[idx - 1, 1]
            return np.arctan2(dy, dx)
        return 0.0

    def step(self, action_accel: float):
        """
        Advance vehicle state by one time-step using 2D kinematics.
        action_accel: normalized action in [-1, 1] mapped to [max_decel, max_accel].
        """
        if not self.active or self.completed:
            return

        self.time_in_system += 1

        # Denormalize action to acceleration (m/s^2)
        if action_accel >= 0:
            accel = action_accel * self.config.max_accel
        else:
            accel = abs(action_accel) * self.config.max_decel
            
        # Physical rate of change of acceleration (jerk)
        jerk = (accel - self.prev_accel) / self.config.dt
        self.total_jerk += jerk ** 2
        self.prev_accel = self.accel
        self.accel = accel

        # Update speed
        self.speed = float(np.clip(
            self.speed + self.accel * self.config.dt,
            0.0,
            self.config.max_speed
        ))
        
        if self.speed < 0.5:
            self.wait_steps += 1

        # Advance along path
        ds = self.speed * self.config.dt
        self.s += ds

        # Check if route is finished
        if self.s >= self.total_path_length:
            self.completed = True
            self.active = False
            self.x = self.path[-1, 0]
            self.y = self.path[-1, 1]
            return

        # Interpolate position along waypoint path
        cum_dist = 0.0
        for i, seg_len in enumerate(self.segment_lengths):
            if cum_dist + seg_len >= self.s:
                ratio = (self.s - cum_dist) / max(seg_len, 1e-6)
                p0 = self.path[i]
                p1 = self.path[i + 1]
                self.x = p0[0] + ratio * (p1[0] - p0[0])
                self.y = p0[1] + ratio * (p1[1] - p0[1])
                self.heading = self._compute_heading(i)
                self.path_index = i
                break
            cum_dist += seg_len

    def get_circles(self) -> List[Tuple[float, float, float]]:
        """
        Two-circle bounding volume decomposition along vehicle centerline
        for accurate, fast collision detection.
        Returns: list of (center_x, center_y, radius)
        """
        r = self.config.width / 2.0
        offset = (self.config.length / 4.0)
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)

        c1 = (self.x + offset * cos_h, self.y + offset * sin_h, r)
        c2 = (self.x - offset * cos_h, self.y - offset * sin_h, r)
        return [c1, c2]


def check_collision(v1: Vehicle, v2: Vehicle) -> bool:
    """Check if two vehicles collide using circle-decomposition distance."""
    if not v1.active or not v2.active or v1.completed or v2.completed:
        return False
        
    circles1 = v1.get_circles()
    circles2 = v2.get_circles()
    
    for c1 in circles1:
        for c2 in circles2:
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dist_sq = dx * dx + dy * dy
            min_dist = c1[2] + c2[2]
            if dist_sq < (min_dist * min_dist):
                return True
    return False


def build_v2v_graph(vehicles: List[Vehicle], comm_range: float) -> np.ndarray:
    """
    Build dynamic adjacency matrix A for active vehicles.
    A[i, j] = 1 if Euclidean distance between active vehicles i and j <= comm_range, else 0.
    """
    n = len(vehicles)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if not vehicles[i].active or vehicles[i].completed:
            continue
        for j in range(i + 1, n):
            if not vehicles[j].active or vehicles[j].completed:
                continue
            dx = vehicles[i].x - vehicles[j].x
            dy = vehicles[i].y - vehicles[j].y
            if (dx * dx + dy * dy) <= (comm_range * comm_range):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        adj[i, i] = 1.0  # self-loop
    return adj
