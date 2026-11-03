"""
renderer.py
Real-time 2D visualizer and animated GIF/video generator for the
Industrial Autonomous Intersection Environment.
Supports both interactive Pygame display and headless rendering for slides/demos.
"""

import math
from typing import List, Optional
import numpy as np

# Try importing pygame; fallback gracefully if not yet installed
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class IntersectionRenderer:
    """Renders 4-way intersection, vehicles, dynamic V2V links, and telemetry HUD."""
    
    def __init__(self, width: int = 700, height: int = 700, road_scale: float = 5.0):
        self.width = width
        self.height = height
        self.scale = road_scale  # Pixels per meter
        self.center_x = width // 2
        self.center_y = height // 2
        self.screen = None
        self.clock = None
        
        # Color Palette (Dark modern industrial cyber-physical theme)
        self.COLOR_BG = (24, 28, 36)
        self.COLOR_ROAD = (40, 44, 52)
        self.COLOR_ROAD_MARK = (100, 110, 125)
        self.COLOR_LANE_DIVIDER = (220, 180, 50)
        self.COLOR_ZONE = (60, 68, 80)
        
        # Vehicle & Network colors
        self.COLOR_VEHICLE_BODY = (66, 133, 244)   # Google Blue / CAV cyan
        self.COLOR_VEHICLE_ALERT = (234, 67, 53)   # Red if collided / near-miss
        self.COLOR_VEHICLE_HEAD = (255, 255, 255)
        self.COLOR_V2V_LINK = (0, 230, 150)       # Cyber neon green for active V2V
        self.COLOR_SENSOR_HALO = (0, 180, 255, 35) # Translucent LiDAR circle
        self.COLOR_TEXT = (230, 235, 245)

    def _world_to_screen(self, x: float, y: float) -> (int, int):
        """Transform simulation coordinates (meters) to screen pixel coordinates."""
        sx = int(self.center_x + x * self.scale)
        sy = int(self.center_y - y * self.scale)  # Invert Y for screen coords
        return sx, sy

    def init_display(self, window_title: str = "Distributed CAV Network - Industrial Bottleneck"):
        """Initialize Pygame display."""
        if not PYGAME_AVAILABLE:
            return
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption(window_title)
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def render_frame(self, env, render_mode: str = "rgb_array") -> Optional[np.ndarray]:
        """
        Draw one simulation frame.
        render_mode: 'human' (interactive window) or 'rgb_array' (numpy array for GIF/video).
        """
        if not PYGAME_AVAILABLE:
            return None

        if self.screen is None:
            if render_mode == "human":
                self.init_display()
            else:
                pygame.init()
                self.screen = pygame.Surface((self.width, self.height))

        # 1. Clear background
        self.screen.fill(self.COLOR_BG)

        # 2. Draw roads & intersection
        lw = env.config.lane_width * self.scale
        half_lw = lw
        
        # Road surfaces
        # Vertical road
        v_rect = pygame.Rect(self.center_x - half_lw, 0, half_lw * 2, self.height)
        pygame.draw.rect(self.screen, self.COLOR_ROAD, v_rect)
        # Horizontal road
        h_rect = pygame.Rect(0, self.center_y - half_lw, self.width, half_lw * 2)
        pygame.draw.rect(self.screen, self.COLOR_ROAD, h_rect)

        # Conflict intersection zone
        zone_rect = pygame.Rect(self.center_x - half_lw, self.center_y - half_lw, half_lw * 2, half_lw * 2)
        pygame.draw.rect(self.screen, self.COLOR_ZONE, zone_rect)
        pygame.draw.rect(self.screen, (80, 90, 105), zone_rect, 2)

        # Lane markings (dashed center lines)
        dash_len = 12
        for y in range(0, self.height, dash_len * 2):
            if abs(y - self.center_y) > half_lw:
                pygame.draw.line(self.screen, self.COLOR_LANE_DIVIDER, (self.center_x, y), (self.center_x, y + dash_len), 2)
        for x in range(0, self.width, dash_len * 2):
            if abs(x - self.center_x) > half_lw:
                pygame.draw.line(self.screen, self.COLOR_LANE_DIVIDER, (x, self.center_y), (x + dash_len, self.center_y), 2)

        # 3. Draw dynamic V2V communication links
        active_vehicles = [v for v in env.vehicles if v.active]
        comm_range_px = env.vehicle_config.comm_range * self.scale
        
        for i in range(len(active_vehicles)):
            for j in range(i + 1, len(active_vehicles)):
                v1 = active_vehicles[i]
                v2 = active_vehicles[j]
                dist = np.hypot(v1.x - v2.x, v1.y - v2.y)
                if dist <= env.vehicle_config.comm_range:
                    p1 = self._world_to_screen(v1.x, v1.y)
                    p2 = self._world_to_screen(v2.x, v2.y)
                    # Draw glowing V2V link line
                    pygame.draw.line(self.screen, self.COLOR_V2V_LINK, p1, p2, 2)

        # 4. Draw vehicles
        veh_len_px = env.vehicle_config.length * self.scale
        veh_wid_px = env.vehicle_config.width * self.scale

        for v in active_vehicles:
            pos = self._world_to_screen(v.x, v.y)
            color = self.COLOR_VEHICLE_ALERT if v.collided else self.COLOR_VEHICLE_BODY
            
            # Create rotated vehicle rectangle surface
            veh_surf = pygame.Surface((veh_len_px, veh_wid_px), pygame.SRCALPHA)
            pygame.draw.rect(veh_surf, color, (0, 0, veh_len_px, veh_wid_px), border_radius=4)
            # Front headlight indicator
            pygame.draw.circle(veh_surf, self.COLOR_VEHICLE_HEAD, (veh_len_px - 3, veh_wid_px / 2), 3)

            # Pygame rotates counter-clockwise in degrees
            angle_deg = np.degrees(v.heading)
            rotated_surf = pygame.transform.rotate(veh_surf, angle_deg)
            rot_rect = rotated_surf.get_rect(center=pos)
            self.screen.blit(rotated_surf, rot_rect.topleft)

            # Vehicle speed text
            font = pygame.font.SysFont("monospace", 10, bold=True)
            txt = font.render(f"#{v.id}:{v.speed:.1f}m/s", True, self.COLOR_TEXT)
            self.screen.blit(txt, (pos[0] - 18, pos[1] - 18))

        # 5. Draw Industrial Telemetry HUD
        hud_font = pygame.font.SysFont("sans-serif", 13)
        title_font = pygame.font.SysFont("sans-serif", 15, bold=True)
        
        info = env._get_info()
        hud_lines = [
            f"Step: {info['step']} / {env.config.max_steps}",
            f"Active CAVs: {info['active_vehicles']}",
            f"Completed Vehicles: {info['completed_vehicles']}",
            f"Collisions: {info['total_collisions']}",
            f"Avg Speed: {info['average_speed']:.2f} m/s",
            f"Throughput: {info['throughput_veh_per_step']:.3f} veh/step",
            f"Packet Drop Rate: {env.packet_drop_rate*100:.0f}%"
        ]

        # HUD Panel background
        hud_rect = pygame.Rect(10, 10, 240, 155)
        pygame.draw.rect(self.screen, (15, 20, 28, 220), hud_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 75, 95), hud_rect, 1, border_radius=6)
        
        # HUD Title
        t_surf = title_font.render("Distributed CAV Analytics", True, (0, 220, 255))
        self.screen.blit(t_surf, (20, 16))
        
        for idx, line in enumerate(hud_lines):
            line_surf = hud_font.render(line, True, self.COLOR_TEXT)
            self.screen.blit(line_surf, (20, 36 + idx * 16))

        # Display or export
        if render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)
            return None
        else:
            # Convert Pygame surface to RGB numpy array for imageio/GIF
            view = pygame.surfarray.array3d(self.screen)
            view = view.transpose([1, 0, 2])
            return view

    def close(self):
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()
