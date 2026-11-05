"""
intersection_env.py
Multi-Agent Industrial Intersection Environment for Autonomous Logistics Fleets.
Formulates the coordination problem as a Dec-POMDP with dynamic V2V communication graphs.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from src.env.kinematics import Vehicle, VehicleConfig, check_collision, build_v2v_graph


class IntersectionConfig:
    """Industrial intersection physical layout and operational parameters."""
    road_length: float = 60.0       # Half-length of each approach arm (meters)
    lane_width: float = 3.5         # Lane width (meters)
    max_vehicles: int = 8           # Maximum concurrent vehicles in the system
    spawn_interval: int = 15        # Steps between potential arrivals
    max_steps: int = 400            # Max steps per episode (40.0 simulated seconds)
    
    # Reward weights (Multi-objective ISE formulation)
    w_prog: float = 1.0             # Forward progress / velocity alignment
    w_safe: float = 50.0            # Penalty for collision / severe proximity breach
    w_jerk: float = 0.05            # Energy / acceleration change penalty
    w_wait: float = 0.2             # Penalty for queue stalling (< 0.5 m/s)
    w_complete: float = 20.0        # Bonus for clearing the intersection bottleneck safely


class IndustrialIntersectionEnv:
    """
    Multi-Agent Environment modeling an unsignalized 4-way intersection
    in an automated industrial freight facility.
    """
    
    def __init__(self, config: Optional[IntersectionConfig] = None, packet_drop_rate: float = 0.0):
        self.config = config or IntersectionConfig()
        self.packet_drop_rate = packet_drop_rate
        self.vehicle_config = VehicleConfig()
        
        # Define routes (centerlines of approaches and departures)
        self.routes = self._build_routes()
        
        # State containers
        self.vehicles: List[Vehicle] = []
        self.step_count = 0
        self.total_completed = 0
        self.total_collisions = 0
        self.next_agent_id = 0
        
        # Observation and Action dimensions per agent
        # Self features: 7
        # Leading vehicle features: 2
        # K=3 closest surrounding vehicles: 5 * 3 = 15
        # Total local observation dimension: 24
        self.obs_dim = 24
        self.action_dim = 1  # Continuous acceleration [-1, 1]

    def _build_routes(self) -> Dict[int, np.ndarray]:
        """
        Build waypoints for 4 straight crossing routes.
        Route 0: South to North (x = +lane_width/2)
        Route 1: North to South (x = -lane_width/2)
        Route 2: West to East   (y = -lane_width/2)
        Route 3: East to West   (y = +lane_width/2)
        """
        L = self.config.road_length
        lw = self.config.lane_width / 2.0
        num_pts = 60
        
        routes = {}
        # Route 0: South -> North
        y_pts = np.linspace(-L, L, num_pts)
        x_pts = np.full(num_pts, lw)
        routes[0] = np.column_stack([x_pts, y_pts])

        # Route 1: North -> South
        y_pts = np.linspace(L, -L, num_pts)
        x_pts = np.full(num_pts, -lw)
        routes[1] = np.column_stack([x_pts, y_pts])

        # Route 2: West -> East
        x_pts = np.linspace(-L, L, num_pts)
        y_pts = np.full(num_pts, -lw)
        routes[2] = np.column_stack([x_pts, y_pts])

        # Route 3: East -> West
        x_pts = np.linspace(L, -L, num_pts)
        y_pts = np.full(num_pts, lw)
        routes[3] = np.column_stack([x_pts, y_pts])

        return routes

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state with first batch of vehicles."""
        if seed is not None:
            np.random.seed(seed)
            
        self.step_count = 0
        self.total_completed = 0
        self.total_collisions = 0
        self.next_agent_id = 0
        self.vehicles.clear()
        
        # Spawn initial vehicles at staggered distances on different routes
        for route_id in range(4):
            self._spawn_vehicle(route_id, initial_speed=np.random.uniform(5.0, 7.5))
            
        obs, adj = self._get_observations_and_graph()
        info = self._get_info()
        return obs, adj, info

    def _spawn_vehicle(self, route_id: int, initial_speed: float = 6.0) -> bool:
        """Spawn a vehicle if lane entry is clear."""
        if len([v for v in self.vehicles if v.active]) >= self.config.max_vehicles:
            return False
            
        route_path = self.routes[route_id]
        spawn_pos = route_path[0]
        
        # Check if entry point is sufficiently clear of existing active vehicles
        for v in self.vehicles:
            if v.active and v.route_id == route_id:
                dist_to_spawn = np.linalg.norm(np.array([v.x, v.y]) - spawn_pos)
                if dist_to_spawn < 12.0:
                    return False
                    
        v = Vehicle(
            agent_id=self.next_agent_id,
            route_id=route_id,
            path_waypoints=route_path,
            initial_speed=initial_speed,
            config=self.vehicle_config
        )
        self.vehicles.append(v)
        self.next_agent_id += 1
        return True

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute actions for all active vehicles.
        actions: shape (N_max, 1), continuous acceleration values in [-1, 1].
        Returns:
            obs: (N_max, obs_dim)
            adj: (N_max, N_max) V2V adjacency matrix (with packet drop applied)
            rewards: (N_max,)
            terminated: (N_max,) bool
            info: dictionary of telemetry and ISE metrics
        """
        self.step_count += 1
        active_vehicles = [v for v in self.vehicles if v.active]
        
        # 1. Apply actions to active vehicles
        for idx, v in enumerate(active_vehicles):
            if idx < len(actions):
                action_val = float(actions[idx, 0] if actions.ndim > 1 else actions[idx])
                v.step(action_val)

        # 2. Check collisions between all active pairs
        current_collisions = 0
        for i in range(len(active_vehicles)):
            for j in range(i + 1, len(active_vehicles)):
                v1 = active_vehicles[i]
                v2 = active_vehicles[j]
                if check_collision(v1, v2):
                    v1.collided = True
                    v2.collided = True
                    v1.active = False
                    v2.active = False
                    current_collisions += 1

        self.total_collisions += current_collisions

        # 3. Handle newly completed vehicles
        newly_completed = 0
        for v in active_vehicles:
            if v.completed:
                newly_completed += 1
        self.total_completed += newly_completed

        # 4. Stochastic replenishment of vehicles at entry gates
        if (self.step_count % self.config.spawn_interval == 0) and (len([v for v in self.vehicles if v.active]) < self.config.max_vehicles):
            available_routes = list(range(4))
            np.random.shuffle(available_routes)
            for r_id in available_routes:
                if self._spawn_vehicle(r_id, initial_speed=np.random.uniform(5.0, 7.5)):
                    break

        # 5. Compute rewards and observations
        obs, adj = self._get_observations_and_graph()
        rewards = self._compute_rewards(active_vehicles)
        
        # Terminations
        episode_done = self.step_count >= self.config.max_steps
        terminated = np.array([v.completed or v.collided or episode_done for v in active_vehicles], dtype=bool)

        # Pad rewards and terminated to max_vehicles size
        padded_rewards = np.zeros(self.config.max_vehicles, dtype=np.float32)
        padded_terminated = np.ones(self.config.max_vehicles, dtype=bool)
        
        for idx in range(min(len(rewards), self.config.max_vehicles)):
            padded_rewards[idx] = rewards[idx]
            padded_terminated[idx] = terminated[idx]

        info = self._get_info()
        info["newly_completed"] = newly_completed
        info["newly_collided"] = current_collisions
        info["episode_done"] = episode_done
        
        return obs, adj, padded_rewards, padded_terminated, info

    def _get_observations_and_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract vectorized observations and communication graph."""
        active_vehicles = [v for v in self.vehicles if v.active]
        N_max = self.config.max_vehicles
        
        obs_matrix = np.zeros((N_max, self.obs_dim), dtype=np.float32)
        adj_matrix = np.zeros((N_max, N_max), dtype=np.float32)

        # Build raw geometric adjacency
        raw_adj = build_v2v_graph(active_vehicles, self.vehicle_config.comm_range)

        # Apply packet drop channel disturbance (cyber-physical network layer)
        if self.packet_drop_rate > 0.0:
            drop_mask = (np.random.rand(*raw_adj.shape) > self.packet_drop_rate).astype(np.float32)
            np.fill_diagonal(drop_mask, 1.0)  # Agent never drops its own internal state
            raw_adj = raw_adj * drop_mask

        n_active = len(active_vehicles)
        if n_active > 0:
            adj_matrix[:n_active, :n_active] = raw_adj[:n_active, :n_active]

        for i, vi in enumerate(active_vehicles[:N_max]):
            # Self features (7)
            norm_speed = vi.speed / self.vehicle_config.max_speed
            norm_accel = vi.accel / self.vehicle_config.max_accel
            dist_to_center = np.sqrt(vi.x**2 + vi.y**2) / self.config.road_length
            heading_sin = np.sin(vi.heading)
            heading_cos = np.cos(vi.heading)
            dist_to_goal = max(0.0, vi.total_path_length - vi.s) / vi.total_path_length
            route_norm = vi.route_id / 4.0
            
            self_feats = [norm_speed, norm_accel, dist_to_center, heading_sin, heading_cos, dist_to_goal, route_norm]

            # Leader vehicle in same lane (2 features: rel_dist, rel_speed)
            leader_dist = 1.0
            leader_rel_speed = 0.0
            for vj in active_vehicles:
                if vj.id != vi.id and vj.route_id == vi.route_id and vj.s > vi.s:
                    delta_s = vj.s - vi.s
                    if delta_s < self.vehicle_config.sensor_range:
                        leader_dist = delta_s / self.vehicle_config.sensor_range
                        leader_rel_speed = (vj.speed - vi.speed) / self.vehicle_config.max_speed
                        break
            leader_feats = [leader_dist, leader_rel_speed]

            # K=3 closest other vehicles (15 features: rel_x, rel_y, rel_vx, rel_vy, route_id)
            other_vehicles = []
            for j, vj in enumerate(active_vehicles):
                if i != j:
                    dx = vj.x - vi.x
                    dy = vj.y - vi.y
                    dist = np.hypot(dx, dy)
                    if dist <= self.vehicle_config.sensor_range:
                        other_vehicles.append((dist, dx, dy, vj.speed * np.cos(vj.heading) - vi.speed * np.cos(vi.heading),
                                              vj.speed * np.sin(vj.heading) - vi.speed * np.sin(vi.heading), vj.route_id))
            
            other_vehicles.sort(key=lambda item: item[0])
            cross_feats = []
            for k in range(3):
                if k < len(other_vehicles):
                    _, dx, dy, dvx, dvy, r_id = other_vehicles[k]
                    cross_feats.extend([
                        dx / self.vehicle_config.sensor_range,
                        dy / self.vehicle_config.sensor_range,
                        dvx / self.vehicle_config.max_speed,
                        dvy / self.vehicle_config.max_speed,
                        r_id / 4.0
                    ])
                else:
                    cross_feats.extend([1.0, 1.0, 0.0, 0.0, 0.0])

            agent_obs = np.array(self_feats + leader_feats + cross_feats, dtype=np.float32)
            obs_matrix[i, :] = agent_obs

        return obs_matrix, adj_matrix

    def _compute_rewards(self, active_vehicles: List[Vehicle]) -> np.ndarray:
        """Calculate multi-objective industrial reward vector."""
        rewards = []
        for vi in active_vehicles:
            # 1. Forward progress reward
            r_prog = vi.speed / self.vehicle_config.target_speed
            
            # 2. Safety penalty
            r_safe = 0.0
            if vi.collided:
                r_safe = -1.0
            else:
                # Proximity warning if dangerously close to another vehicle
                for vj in active_vehicles:
                    if vi.id != vj.id:
                        dist = np.hypot(vi.x - vj.x, vi.y - vj.y)
                        if dist < self.vehicle_config.safe_distance:
                            r_safe = -0.5 * (1.0 - dist / self.vehicle_config.safe_distance)
                            break
                            
            # 3. Energy / jerk penalty
            r_jerk = - (vi.accel - vi.prev_accel)**2 / (self.vehicle_config.max_accel**2)
            
            # 4. Delay / waiting penalty
            r_wait = -1.0 if vi.speed < 0.5 else 0.0
            
            # 5. Bottleneck completion bonus
            r_comp = 1.0 if vi.completed else 0.0

            total_r = (
                self.config.w_prog * r_prog
                + self.config.w_safe * r_safe
                + self.config.w_jerk * r_jerk
                + self.config.w_wait * r_wait
                + self.config.w_complete * r_comp
            )
            rewards.append(total_r)
        return np.array(rewards, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Aggregate industrial performance indicators."""
        active_vehicles = [v for v in self.vehicles if v.active]
        total_vehicles = len(self.vehicles)
        avg_wait = np.mean([v.wait_steps for v in self.vehicles]) if total_vehicles > 0 else 0.0
        avg_speed = np.mean([v.speed for v in active_vehicles]) if active_vehicles else 0.0
        avg_jerk = np.mean([v.total_jerk for v in self.vehicles]) if total_vehicles > 0 else 0.0

        return {
            "step": self.step_count,
            "active_vehicles": len(active_vehicles),
            "completed_vehicles": self.total_completed,
            "total_collisions": self.total_collisions,
            "average_speed": avg_speed,
            "average_wait_steps": avg_wait,
            "average_jerk": avg_jerk,
            "throughput_veh_per_step": self.total_completed / max(1, self.step_count)
        }
