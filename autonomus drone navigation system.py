#!/usr/bin/env python3
"""
Advanced Autonomous Drone Navigation System
============================================
Features:
- 3D Path Planning (RRT* + A* hybrid)
- SLAM-based Localization & Mapping
- Obstacle Avoidance (LiDAR + Stereo Vision fusion)
- PID Controller with feedforward
- Kalman Filter for state estimation
- Geofencing & Safety Protocols
- Mission Planning with waypoints
- Emergency RTL (Return to Launch)
- Sensor fusion (IMU, GPS, Barometer, Magnetometer)
- Dynamic replanning
"""

import numpy as np
import math
import random
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum, auto
from collections import deque
import threading
import json

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class DroneConfig:
    """Physical and operational parameters"""
    # Physical specs
    MAX_VELOCITY = 15.0          # m/s
    MAX_ACCELERATION = 5.0       # m/s²
    MAX_ANGULAR_VEL = 2.0        # rad/s
    MASS = 1.5                   # kg
    DRAG_COEFF = 0.25

    # Safety margins (FIXED: explicitly defined)
    SAFETY_MARGIN = 2.0          # meters around obstacles
    MIN_ALTITUDE = 2.0           # meters AGL
    MAX_ALTITUDE = 120.0         # meters (legal limit)
    GEO_FENCE_RADIUS = 500.0     # meters from home

    # Sensor specs
    LIDAR_RANGE = 50.0           # meters
    LIDAR_FOV = 360              # degrees horizontal
    LIDAR_RESOLUTION = 1.0       # degree
    CAMERA_FOV = 90              # degrees

    # Control
    PID_XY = (1.2, 0.05, 0.3)
    PID_Z = (2.0, 0.1, 0.5)
    PID_YAW = (1.5, 0.0, 0.3)

    # Planning
    RRT_STEP_SIZE = 5.0
    RRT_MAX_ITER = 3000
    A_STAR_GRID_RES = 2.0
    REPLAN_THRESHOLD = 10.0      # meters


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class FlightMode(Enum):
    DISARMED = auto()
    ARMED = auto()
    TAKEOFF = auto()
    HOLD = auto()
    MISSION = auto()
    RTL = auto()
    LANDING = auto()
    EMERGENCY = auto()

@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag < 1e-6:
            return Vector3D(0, 0, 0)
        return self / mag

    def distance_to(self, other):
        return (self - other).magnitude()

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr):
        return Vector3D(arr[0], arr[1], arr[2])

@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_euler(self):
        """Convert to roll, pitch, yaw (radians)"""
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def from_euler(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )

@dataclass
class SensorReading:
    timestamp: float
    gps_position: Vector3D
    gps_accuracy: float
    imu_accel: Vector3D
    imu_gyro: Vector3D
    baro_altitude: float
    magnetometer_heading: float
    lidar_points: List[Tuple[float, float, float]] = field(default_factory=list)
    camera_detections: List[Dict] = field(default_factory=list)
    battery_voltage: float = 16.8
    battery_percent: float = 100.0

@dataclass
class Waypoint:
    position: Vector3D
    heading: Optional[float] = None
    action: str = "pass"
    loiter_time: float = 0.0
    speed: float = 5.0


# ============================================================================
# KALMAN FILTER (Extended for 3D state estimation)
# ============================================================================

class ExtendedKalmanFilter:
    """EKF for state estimation fusing GPS, IMU, Barometer"""

    def __init__(self):
        # State: [x, y, z, vx, vy, vz, ax, ay, az, bias_ax, bias_ay, bias_az]
        self.state = np.zeros(12)
        self.covariance = np.eye(12) * 10.0

        # Process noise
        self.Q = np.eye(12)
        self.Q[0:3, 0:3] *= 0.1      # position noise
        self.Q[3:6, 3:6] *= 0.5      # velocity noise
        self.Q[6:9, 6:9] *= 1.0      # accel noise
        self.Q[9:12, 9:12] *= 0.01   # bias noise

        # Measurement noise
        self.R_gps = np.eye(3) * 2.0
        self.R_baro = np.array([[1.0]])
        self.R_mag = np.array([[0.5]])

        self.dt = 0.01  # 100Hz update rate
        self.last_update = time.time()

    def predict(self, imu_accel: Vector3D, imu_gyro: Vector3D):
        dt = time.time() - self.last_update
        if dt > 0.1:
            dt = 0.01

        # State transition matrix (simplified constant acceleration)
        F = np.eye(12)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        # Remove bias from IMU
        ax = imu_accel.x - self.state[9]
        ay = imu_accel.y - self.state[10]
        az = imu_accel.z - self.state[11]

        # Predict state
        self.state[0] += self.state[3] * dt + 0.5 * ax * dt**2
        self.state[1] += self.state[4] * dt + 0.5 * ay * dt**2
        self.state[2] += self.state[5] * dt + 0.5 * az * dt**2
        self.state[3] += ax * dt
        self.state[4] += ay * dt
        self.state[5] += az * dt

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
        self.last_update = time.time()

    def update_gps(self, position: Vector3D, accuracy: float):
        H = np.zeros((3, 12))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        R = self.R_gps * max(accuracy, 1.0)
        z = np.array([position.x, position.y, position.z])
        y = z - H @ self.state

        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state += K @ y
        self.covariance = (np.eye(12) - K @ H) @ self.covariance

    def update_baro(self, altitude: float):
        H = np.zeros((1, 12))
        H[0, 2] = 1

        z = np.array([altitude])
        y = z - H @ self.state

        S = H @ self.covariance @ H.T + self.R_baro
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state += K @ y.flatten()
        self.covariance = (np.eye(12) - K @ H) @ self.covariance

    def get_position(self) -> Vector3D:
        return Vector3D(self.state[0], self.state[1], self.state[2])

    def get_velocity(self) -> Vector3D:
        return Vector3D(self.state[3], self.state[4], self.state[5])


# ============================================================================
# SLAM - OCCUPANCY GRID MAPPER
# ============================================================================

class OccupancyGrid3D:
    """3D voxel grid for obstacle mapping"""

    def __init__(self, resolution=1.0, size=200):
        self.resolution = resolution
        self.size = size
        self.origin = Vector3D(-size/2, -size/2, 0)

        # Log-odds occupancy grid
        self.grid = np.zeros((size, size, size // 2), dtype=np.float32)
        self.log_odds_occ = 0.85
        self.log_odds_free = -0.4
        self.max_log_odds = 10.0
        self.min_log_odds = -10.0

        # Obstacle list for fast collision checking
        self.obstacles: Set[Tuple[int, int, int]] = set()

    def world_to_grid(self, pos: Vector3D) -> Tuple[int, int, int]:
        gx = int((pos.x - self.origin.x) / self.resolution)
        gy = int((pos.y - self.origin.y) / self.resolution)
        gz = int((pos.z - self.origin.z) / self.resolution)
        return (max(0, min(gx, self.size-1)),
                max(0, min(gy, self.size-1)),
                max(0, min(gz, self.size//2-1)))

    def grid_to_world(self, gx: int, gy: int, gz: int) -> Vector3D:
        return Vector3D(
            gx * self.resolution + self.origin.x,
            gy * self.resolution + self.origin.y,
            gz * self.resolution + self.origin.z
        )

    def update_line(self, start: Vector3D, end: Vector3D, hit: bool):
        """Bresenham-like 3D line update"""
        s = self.world_to_grid(start)
        e = self.world_to_grid(end)

        points = self._bresenham_3d(s, e)

        for i, p in enumerate(points):
            if 0 <= p[0] < self.size and 0 <= p[1] < self.size and 0 <= p[2] < self.size // 2:
                if i == len(points) - 1 and hit:
                    self.grid[p] += self.log_odds_occ
                    if self.grid[p] > 2.0:
                        self.obstacles.add(p)
                else:
                    self.grid[p] += self.log_odds_free

                self.grid[p] = np.clip(self.grid[p], self.min_log_odds, self.max_log_odds)

    def _bresenham_3d(self, start, end):
        """3D Bresenham line algorithm"""
        x1, y1, z1 = start
        x2, y2, z2 = end
        points = []

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)

        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                points.append((x1, y1, z1))
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                points.append((x1, y1, z1))
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                points.append((x1, y1, z1))
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx

        points.append((x2, y2, z2))
        return points

    def is_occupied(self, pos: Vector3D, safety_margin: float = 0.0) -> bool:
        """Check if position is occupied or within safety margin"""
        if safety_margin > 0:
            # Check surrounding voxels
            margin_steps = int(safety_margin / self.resolution) + 1
            center = self.world_to_grid(pos)
            for dx in range(-margin_steps, margin_steps + 1):
                for dy in range(-margin_steps, margin_steps + 1):
                    for dz in range(-margin_steps, margin_steps + 1):
                        check_pos = (center[0] + dx, center[1] + dy, center[2] + dz)
                        if check_pos in self.obstacles:
                            world_pos = self.grid_to_world(*check_pos)
                            if pos.distance_to(world_pos) < safety_margin:
                                return True
            return False
        else:
            g = self.world_to_grid(pos)
            return g in self.obstacles

    def get_nearest_obstacle(self, pos: Vector3D) -> Tuple[float, Vector3D]:
        """Return distance and position of nearest obstacle"""
        min_dist = float('inf')
        nearest = None

        for obs in self.obstacles:
            world_pos = self.grid_to_world(*obs)
            dist = pos.distance_to(world_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = world_pos

        return min_dist, nearest if nearest else Vector3D()


# ============================================================================
# PATH PLANNING - RRT* + A* HYBRID
# ============================================================================

class RRTStar:
    """RRT* for global path planning"""

    class Node:
        def __init__(self, position: Vector3D):
            self.position = position
            self.parent = None
            self.cost = 0.0
            self.children = []

    def __init__(self, start: Vector3D, goal: Vector3D, grid: OccupancyGrid3D):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.nodes: List[RRTStar.Node] = []
        self.max_iter = DroneConfig.RRT_MAX_ITER
        self.step_size = DroneConfig.RRT_STEP_SIZE
        self.neighbor_radius = 15.0
        self.safety_margin = DroneConfig.SAFETY_MARGIN  # Use config value

        self.start_node = self.Node(start)
        self.nodes.append(self.start_node)

    def plan(self) -> Optional[List[Vector3D]]:
        for _ in range(self.max_iter):
            # Sample random point
            if random.random() < 0.1:
                rand_pos = self.goal
            else:
                rand_pos = self._random_position()

            # Find nearest node
            nearest = self._nearest(rand_pos)

            # Steer towards random point
            new_pos = self._steer(nearest.position, rand_pos)

            if new_pos is None:
                continue

            # Check collision
            if self._check_collision(nearest.position, new_pos):
                continue

            # Find neighbors
            neighbors = self._near_neighbors(new_pos)

            # Choose best parent
            new_node = self.Node(new_pos)
            min_cost = nearest.cost + nearest.position.distance_to(new_pos)
            best_parent = nearest

            for neighbor in neighbors:
                cost = neighbor.cost + neighbor.position.distance_to(new_pos)
                if cost < min_cost and not self._check_collision(neighbor.position, new_pos):
                    min_cost = cost
                    best_parent = neighbor

            new_node.parent = best_parent
            new_node.cost = min_cost
            best_parent.children.append(new_node)
            self.nodes.append(new_node)

            # Rewire neighbors
            for neighbor in neighbors:
                if neighbor == best_parent:
                    continue
                new_cost = new_node.cost + new_node.position.distance_to(neighbor.position)
                if new_cost < neighbor.cost and not self._check_collision(new_node.position, neighbor.position):
                    # Remove from old parent
                    if neighbor.parent:
                        neighbor.parent.children.remove(neighbor)
                    neighbor.parent = new_node
                    neighbor.cost = new_cost
                    new_node.children.append(neighbor)
                    self._update_children_cost(neighbor)

            # Check goal reached
            if new_pos.distance_to(self.goal) < self.step_size:
                if not self._check_collision(new_pos, self.goal):
                    goal_node = self.Node(self.goal)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + new_pos.distance_to(self.goal)
                    new_node.children.append(goal_node)
                    self.nodes.append(goal_node)
                    return self._extract_path(goal_node)

        # Return best path found
        nearest_to_goal = self._nearest(self.goal)
        return self._extract_path(nearest_to_goal)

    def _random_position(self) -> Vector3D:
        # Sample within bounding box
        return Vector3D(
            random.uniform(self.start.x - 100, self.start.x + 100),
            random.uniform(self.start.y - 100, self.start.y + 100),
            random.uniform(DroneConfig.MIN_ALTITUDE, DroneConfig.MAX_ALTITUDE)
        )

    def _nearest(self, pos: Vector3D) -> 'RRTStar.Node':
        return min(self.nodes, key=lambda n: n.position.distance_to(pos))

    def _near_neighbors(self, pos: Vector3D) -> List['RRTStar.Node']:
        return [n for n in self.nodes if n.position.distance_to(pos) < self.neighbor_radius]

    def _steer(self, from_pos: Vector3D, to_pos: Vector3D) -> Optional[Vector3D]:
        dist = from_pos.distance_to(to_pos)
        if dist < 1e-6:
            return None

        if dist <= self.step_size:
            return to_pos

        direction = (to_pos - from_pos).normalize()
        return from_pos + direction * self.step_size

    def _check_collision(self, from_pos: Vector3D, to_pos: Vector3D) -> bool:
        # Check multiple points along line
        dist = from_pos.distance_to(to_pos)
        steps = max(1, int(dist / 0.5))

        for i in range(steps + 1):
            t = i / steps
            pos = Vector3D(
                from_pos.x + (to_pos.x - from_pos.x) * t,
                from_pos.y + (to_pos.y - from_pos.y) * t,
                from_pos.z + (to_pos.z - from_pos.z) * t
            )

            # Check bounds
            if pos.z < DroneConfig.MIN_ALTITUDE or pos.z > DroneConfig.MAX_ALTITUDE:
                return True

            if self.grid.is_occupied(pos, self.safety_margin):
                return True

        return False

    def _update_children_cost(self, node: 'RRTStar.Node'):
        for child in node.children:
            child.cost = node.cost + node.position.distance_to(child.position)
            self._update_children_cost(child)

    def _extract_path(self, goal_node: 'RRTStar.Node') -> List[Vector3D]:
        path = []
        current = goal_node
        while current:
            path.append(current.position)
            current = current.parent
        return list(reversed(path))


class AStar3D:
    """A* for local replanning on grid"""

    def __init__(self, grid: OccupancyGrid3D):
        self.grid = grid
        self.resolution = DroneConfig.A_STAR_GRID_RES

    def plan(self, start: Vector3D, goal: Vector3D) -> Optional[List[Vector3D]]:
        start_g = self.grid.world_to_grid(start)
        goal_g = self.grid.world_to_grid(goal)

        # Check if goal is occupied
        if self.grid.is_occupied(goal, DroneConfig.SAFETY_MARGIN):
            # Find nearest free cell
            goal_g = self._find_nearest_free(goal_g)

        open_set = [(0, start_g)]
        came_from = {}
        g_score = {start_g: 0}
        f_score = {start_g: self._heuristic(start_g, goal_g)}

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_g:
                return self._reconstruct_path(came_from, current, start)

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                world_pos = self.grid.grid_to_world(*neighbor)
                if self.grid.is_occupied(world_pos, DroneConfig.SAFETY_MARGIN):
                    continue

                tentative_g = g_score[current] + self._distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_g)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _find_nearest_free(self, pos: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Find nearest unoccupied grid cell"""
        for r in range(1, 20):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-r, r+1):
                        check = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
                        world = self.grid.grid_to_world(*check)
                        if not self.grid.is_occupied(world, DroneConfig.SAFETY_MARGIN):
                            return check
        return pos

    def _get_neighbors(self, pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    n = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
                    if 0 <= n[0] < self.grid.size and 0 <= n[1] < self.grid.size and 0 <= n[2] < self.grid.size // 2:
                        neighbors.append(n)
        return neighbors

    def _heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def _distance(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        return self._heuristic(a, b)

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int, int], start: Vector3D) -> List[Vector3D]:
        path = [self.grid.grid_to_world(*current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.grid.grid_to_world(*current))
        path.reverse()
        # Replace start with actual start position
        path[0] = start
        return path


# ============================================================================
# PID CONTROLLER
# ============================================================================

class PIDController:
    """PID controller with anti-windup and feedforward"""

    def __init__(self, kp: float, ki: float, kd: float, 
                 integral_limit: float = 10.0, output_limit: float = float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def compute(self, setpoint: float, measurement: float, 
                feedforward: float = 0.0) -> float:
        error = setpoint - measurement
        dt = time.time() - self.prev_time

        if dt < 1e-6:
            dt = 1e-6

        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i = self.ki * self.integral

        # Derivative (on measurement to avoid derivative kick)
        d = self.kd * (measurement - self.prev_error) / dt
        self.prev_error = measurement
        self.prev_time = time.time()

        output = p + i - d + feedforward

        # Output saturation with anti-windup
        if abs(output) > self.output_limit:
            output = math.copysign(self.output_limit, output)
            self.integral -= error * dt  # Back-calculate

        return output


# ============================================================================
# MAIN DRONE CLASS
# ============================================================================

class AutonomousDrone:
    """Main autonomous drone navigation system"""

    def __init__(self, drone_id: str = "UAV-001"):
        self.drone_id = drone_id

        # State
        self.position = Vector3D(0, 0, 0)
        self.velocity = Vector3D(0, 0, 0)
        self.attitude = Quaternion()
        self.mode = FlightMode.DISARMED

        # FIX: Initialize safety_margin explicitly
        self.safety_margin = DroneConfig.SAFETY_MARGIN

        # Home position
        self.home_position = Vector3D(0, 0, 0)

        # Mission
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        self.path: List[Vector3D] = []
        self.path_idx = 0

        # Subsystems
        self.ekf = ExtendedKalmanFilter()
        self.grid = OccupancyGrid3D(resolution=1.0, size=200)
        self.planner = None

        # Controllers
        self.pid_x = PIDController(*DroneConfig.PID_XY, output_limit=DroneConfig.MAX_VELOCITY)
        self.pid_y = PIDController(*DroneConfig.PID_XY, output_limit=DroneConfig.MAX_VELOCITY)
        self.pid_z = PIDController(*DroneConfig.PID_Z, output_limit=DroneConfig.MAX_VELOCITY)
        self.pid_yaw = PIDController(*DroneConfig.PID_YAW, output_limit=DroneConfig.MAX_ANGULAR_VEL)

        # Sensor data
        self.latest_reading: Optional[SensorReading] = None
        self.battery_percent = 100.0

        # Safety
        self.emergency_stop = False
        self.geofence_violations = 0
        self.last_obstacle_distance = float('inf')

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._control_thread = None

        # Telemetry log
        self.telemetry_log: List[Dict] = []

    def arm(self):
        """Arm the drone motors"""
        if self.mode == FlightMode.DISARMED:
            self.mode = FlightMode.ARMED
            self.home_position = Vector3D(self.position.x, self.position.y, self.position.z)
            print(f"[{self.drone_id}] ARMED - Home set to {self.home_position}")

    def disarm(self):
        """Disarm motors"""
        self.mode = FlightMode.DISARMED
        self._running = False
        print(f"[{self.drone_id}] DISARMED")

    def takeoff(self, target_altitude: float):
        """Takeoff to target altitude"""
        if self.mode != FlightMode.ARMED:
            print("Cannot takeoff - not armed")
            return

        self.mode = FlightMode.TAKEOFF
        target = Vector3D(self.position.x, self.position.y, target_altitude)
        self._set_target(target)
        print(f"[{self.drone_id}] TAKEOFF to {target_altitude}m")

    def upload_mission(self, waypoints: List[Waypoint]):
        """Upload waypoint mission"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        print(f"[{self.drone_id}] Mission uploaded: {len(waypoints)} waypoints")

    def start_mission(self):
        """Start executing mission"""
        if not self.waypoints:
            print("No mission uploaded")
            return

        self.mode = FlightMode.MISSION
        self._plan_path_to_waypoint()
        self._start_control_loop()
        print(f"[{self.drone_id}] MISSION STARTED")

    def rtl(self):
        """Return to Launch"""
        self.mode = FlightMode.RTL
        self._set_target(self.home_position)
        self._plan_path(self.position, self.home_position)
        print(f"[{self.drone_id}] RTL initiated")

    def land(self):
        """Land at current position"""
        self.mode = FlightMode.LANDING
        target = Vector3D(self.position.x, self.position.y, 0.5)
        self._set_target(target)
        print(f"[{self.drone_id}] LANDING")

    def emergency_stop_all(self):
        """Emergency stop"""
        self.emergency_stop = True
        self.mode = FlightMode.EMERGENCY
        self.velocity = Vector3D(0, 0, -2.0)  # Controlled descent
        print(f"[{self.drone_id}] EMERGENCY STOP!")

    def update_sensors(self, reading: SensorReading):
        """Process incoming sensor data"""
        with self._lock:
            self.latest_reading = reading

            # EKF update
            self.ekf.predict(reading.imu_accel, reading.imu_gyro)
            self.ekf.update_gps(reading.gps_position, reading.gps_accuracy)
            self.ekf.update_baro(reading.baro_altitude)

            # Update state
            self.position = self.ekf.get_position()
            self.velocity = self.ekf.get_velocity()

            # Update map
            self.process_lidar(reading.lidar_points)
            self.process_camera(reading.camera_detections)

            # Battery
            self.battery_percent = reading.battery_percent

            # Safety checks
            self._safety_checks()

    def process_lidar(self, points: List[Tuple[float, float, float]]):
        """Process LiDAR point cloud"""
        if not points:
            return

        # Convert to world coordinates and update grid
        for point in points:
            # Point is in drone body frame, transform to world
            world_point = self._body_to_world(Vector3D(*point))
            self.grid.update_line(self.position, world_point, hit=True)

        # Calculate obstacle proximity for avoidance
        distances = [self.position.distance_to(Vector3D(*p)) for p in points]
        if distances:
            self.last_obstacle_distance = min(distances)
            # Use self.safety_margin (FIXED - now properly initialized)
            radius = np.percentile(distances, 95) + self.safety_margin

            # Trigger replanning if obstacle too close
            if self.last_obstacle_distance < self.safety_margin * 2:
                if self.mode == FlightMode.MISSION:
                    self._dynamic_replan()

    def process_camera(self, detections: List[Dict]):
        """Process camera detections (dynamic obstacles)"""
        for det in detections:
            if det.get('class') in ['person', 'vehicle', 'animal']:
                pos = det.get('position', (0, 0, 0))
                confidence = det.get('confidence', 0)
                if confidence > 0.7:
                    world_pos = self._body_to_world(Vector3D(*pos))
                    # Mark as temporary obstacle
                    self.grid.update_line(self.position, world_pos, hit=True)

    def _body_to_world(self, body_pos: Vector3D) -> Vector3D:
        """Transform body-frame coordinates to world frame"""
        roll, pitch, yaw = self.attitude.to_euler()

        # Rotation matrix (simplified)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # Rotate by yaw (primary for navigation)
        x = body_pos.x * cy - body_pos.y * sy
        y = body_pos.x * sy + body_pos.y * cy
        z = body_pos.z

        return Vector3D(
            self.position.x + x,
            self.position.y + y,
            self.position.z + z
        )

    def _safety_checks(self):
        """Run safety checks"""
        # Geofence
        home_dist = self.position.distance_to(self.home_position)
        if home_dist > DroneConfig.GEO_FENCE_RADIUS:
            self.geofence_violations += 1
            if self.geofence_violations > 3:
                print("GEOFENCE BREACH - RTL")
                self.rtl()
        else:
            self.geofence_violations = max(0, self.geofence_violations - 1)

        # Altitude limits
        if self.position.z > DroneConfig.MAX_ALTITUDE:
            print("MAX ALTITUDE EXCEEDED - DESCENDING")
            self.pid_z.integral = 0
        elif self.position.z < DroneConfig.MIN_ALTITUDE and self.mode != FlightMode.LANDING:
            print("MIN ALTITUDE - CLIMBING")

        # Battery
        if self.battery_percent < 20 and self.mode != FlightMode.RTL:
            print("LOW BATTERY - RTL")
            self.rtl()
        elif self.battery_percent < 10:
            self.emergency_stop_all()

        # Collision imminent
        if self.last_obstacle_distance < self.safety_margin:
            print(f"OBSTACLE TOO CLOSE ({self.last_obstacle_distance:.1f}m) - HOLD")
            self.mode = FlightMode.HOLD

    def _plan_path(self, start: Vector3D, goal: Vector3D) -> List[Vector3D]:
        """Plan path using RRT*"""
        planner = RRTStar(start, goal, self.grid)
        path = planner.plan()

        if path and len(path) > 2:
            # Smooth path
            path = self._smooth_path(path)

        return path if path else [start, goal]

    def _plan_path_to_waypoint(self):
        """Plan path to current waypoint"""
        if self.current_waypoint_idx >= len(self.waypoints):
            return

        target = self.waypoints[self.current_waypoint_idx].position
        self.path = self._plan_path(self.position, target)
        self.path_idx = 0

    def _dynamic_replan(self):
        """Dynamic replanning when obstacles detected"""
        if not self.path or self.path_idx >= len(self.path):
            return

        # Use A* for fast local replanning
        astar = AStar3D(self.grid)
        current_target = self.path[min(self.path_idx + 5, len(self.path) - 1)]

        new_path = astar.plan(self.position, current_target)

        if new_path:
            # Insert new path segment
            self.path = self.path[:self.path_idx] + new_path + self.path[self.path_idx + 5:]
            print(f"[{self.drone_id}] Path replanned dynamically")

    def _smooth_path(self, path: List[Vector3D]) -> List[Vector3D]:
        """Path smoothing using gradient descent"""
        if len(path) < 3:
            return path

        smoothed = [p for p in path]

        for _ in range(50):  # iterations
            for i in range(1, len(smoothed) - 1):
                # Pull towards neighbors
                prev_pos = smoothed[i-1]
                next_pos = smoothed[i+1]
                current = smoothed[i]

                # Average of neighbors
                target = Vector3D(
                    (prev_pos.x + next_pos.x) / 2,
                    (prev_pos.y + next_pos.y) / 2,
                    (prev_pos.z + next_pos.z) / 2
                )

                # Check collision on new position
                if not self.grid.is_occupied(target, self.safety_margin):
                    smoothed[i] = target

        return smoothed

    def _set_target(self, target: Vector3D):
        """Set immediate target position"""
        self.path = [self.position, target]
        self.path_idx = 0

    def _start_control_loop(self):
        """Start control thread"""
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()

    def _control_loop(self):
        """Main control loop - runs at 50Hz"""
        while self._running and not self.emergency_stop:
            with self._lock:
                self._execute_control()

            time.sleep(0.02)  # 50Hz

    def _execute_control(self):
        """Execute one control step"""
        if not self.path or self.path_idx >= len(self.path):
            if self.mode == FlightMode.MISSION:
                self._advance_waypoint()
            elif self.mode == FlightMode.TAKEOFF:
                self.mode = FlightMode.HOLD
            elif self.mode == FlightMode.LANDING:
                self.mode = FlightMode.DISARMED
            return

        # Get target
        target = self.path[self.path_idx]

        # Check if reached waypoint
        if self.position.distance_to(target) < 2.0:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                return
            target = self.path[self.path_idx]

        # Compute velocity commands
        vx = self.pid_x.compute(target.x, self.position.x)
        vy = self.pid_y.compute(target.y, self.position.y)
        vz = self.pid_z.compute(target.z, self.position.z)

        # Heading control
        target_yaw = math.atan2(target.y - self.position.y, target.x - self.position.x)
        _, _, current_yaw = self.attitude.to_euler()
        yaw_rate = self.pid_yaw.compute(target_yaw, current_yaw)

        # Limit velocities
        v_horiz = math.sqrt(vx**2 + vy**2)
        if v_horiz > DroneConfig.MAX_VELOCITY:
            scale = DroneConfig.MAX_VELOCITY / v_horiz
            vx *= scale
            vy *= scale

        vz = max(-DroneConfig.MAX_VELOCITY, min(DroneConfig.MAX_VELOCITY, vz))

        # Set velocity (in real drone, send to flight controller)
        self.velocity = Vector3D(vx, vy, vz)

        # Update attitude for simulation
        self.attitude = Quaternion.from_euler(0, 0, current_yaw + yaw_rate * 0.02)

        # Log telemetry
        self._log_telemetry(target)

    def _advance_waypoint(self):
        """Move to next waypoint"""
        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.waypoints):
            print(f"[{self.drone_id}] MISSION COMPLETE")
            self.rtl()
            return

        print(f"[{self.drone_id}] Waypoint {self.current_waypoint_idx}/{len(self.waypoints)}")
        self._plan_path_to_waypoint()

    def _log_telemetry(self, target: Vector3D):
        """Log telemetry data"""
        self.telemetry_log.append({
            'timestamp': time.time(),
            'position': {'x': self.position.x, 'y': self.position.y, 'z': self.position.z},
            'velocity': {'x': self.velocity.x, 'y': self.velocity.y, 'z': self.velocity.z},
            'target': {'x': target.x, 'y': target.y, 'z': target.z},
            'mode': self.mode.name,
            'battery': self.battery_percent,
            'obstacle_dist': self.last_obstacle_distance
        })

    def get_status(self) -> Dict:
        """Get current drone status"""
        return {
            'drone_id': self.drone_id,
            'mode': self.mode.name,
            'position': {'x': self.position.x, 'y': self.position.y, 'z': self.position.z},
            'velocity': {'x': self.velocity.x, 'y': self.velocity.y, 'z': self.velocity.z},
            'battery': self.battery_percent,
            'waypoint': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'obstacles_mapped': len(self.grid.obstacles),
            'safety_margin': self.safety_margin
        }

    def save_telemetry(self, filename: str):
        """Save telemetry to file"""
        with open(filename, 'w') as f:
            json.dump(self.telemetry_log, f, indent=2)
        print(f"Telemetry saved to {filename}")


# ============================================================================
# SIMULATION ENVIRONMENT
# ============================================================================

class DroneSimulator:
    """Physics simulator for testing"""

    def __init__(self, drone: AutonomousDrone):
        self.drone = drone
        self.time = 0.0
        self.dt = 0.02

        # Simulated obstacles
        self.obstacles = self._generate_obstacles()

        # Wind
        self.wind = Vector3D(0.5, 0.3, 0)

    def _generate_obstacles(self) -> List[Vector3D]:
        """Generate random obstacles"""
        obstacles = []
        for _ in range(20):
            obs = Vector3D(
                random.uniform(-80, 80),
                random.uniform(-80, 80),
                random.uniform(5, 30)
            )
            obstacles.append(obs)
        return obstacles

    def run_mission(self, duration: float = 60.0) -> List[Dict]:
        """Run simulated mission"""
        positions = []

        while self.time < duration and self.drone.mode != FlightMode.DISARMED:
            # Generate sensor reading
            reading = self._generate_sensor_reading()

            # Update drone
            self.drone.update_sensors(reading)

            # Simulate physics
            self._simulate_physics()

            # Record
            positions.append({
                'time': self.time,
                'x': self.drone.position.x,
                'y': self.drone.position.y,
                'z': self.drone.position.z,
                'mode': self.drone.mode.name
            })

            self.time += self.dt

        return positions

    def _generate_sensor_reading(self) -> SensorReading:
        """Generate realistic sensor data"""
        # GPS with noise
        gps_noise = Vector3D(
            random.gauss(0, 0.5),
            random.gauss(0, 0.5),
            random.gauss(0, 1.0)
        )
        gps_pos = self.drone.position + gps_noise

        # IMU
        accel = Vector3D(
            random.gauss(0, 0.1),
            random.gauss(0, 0.1),
            random.gauss(-9.81, 0.05)
        )
        gyro = Vector3D(
            random.gauss(0, 0.01),
            random.gauss(0, 0.01),
            random.gauss(0, 0.01)
        )

        # LiDAR - simulate points around obstacles
        lidar_points = []
        for obs in self.obstacles:
            dist = self.drone.position.distance_to(obs)
            if dist < DroneConfig.LIDAR_RANGE:
                # Generate points on sphere around obstacle
                for _ in range(10):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = random.uniform(0, math.pi)
                    r = 2.0  # obstacle radius
                    px = obs.x + r * math.sin(phi) * math.cos(theta)
                    py = obs.y + r * math.sin(phi) * math.sin(theta)
                    pz = obs.z + r * math.cos(phi)

                    # Convert to body frame
                    dx = px - self.drone.position.x
                    dy = py - self.drone.position.y
                    dz = pz - self.drone.position.z
                    lidar_points.append((dx, dy, dz))

        # Battery drain
        battery = max(0, 100 - self.time * 0.5)

        return SensorReading(
            timestamp=self.time,
            gps_position=gps_pos,
            gps_accuracy=1.0,
            imu_accel=accel,
            imu_gyro=gyro,
            baro_altitude=self.drone.position.z + random.gauss(0, 0.3),
            magnetometer_heading=random.gauss(0, 0.05),
            lidar_points=lidar_points,
            camera_detections=[],
            battery_voltage=16.8 * (battery / 100),
            battery_percent=battery
        )

    def _simulate_physics(self):
        """Simulate drone physics"""
        # Simple physics: velocity integration
        self.drone.position = Vector3D(
            self.drone.position.x + self.drone.velocity.x * self.dt,
            self.drone.position.y + self.drone.velocity.y * self.dt,
            self.drone.position.z + self.drone.velocity.z * self.dt
        )

        # Add wind effect
        self.drone.position = Vector3D(
            self.drone.position.x + self.wind.x * self.dt * 0.1,
            self.drone.position.y + self.wind.y * self.dt * 0.1,
            self.drone.position.z
        )


# ============================================================================
# EXAMPLE USAGE / MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Autonomous Drone Navigation System")
    print("=" * 60)

    # Create drone
    drone = AutonomousDrone(drone_id="UAV-ALPHA")

    # Create mission waypoints
    mission = [
        Waypoint(Vector3D(50, 0, 10), speed=8),
        Waypoint(Vector3D(50, 50, 15), speed=8),
        Waypoint(Vector3D(0, 50, 10), speed=8),
        Waypoint(Vector3D(0, 0, 10), speed=5)
    ]

    # Upload and execute
    drone.upload_mission(mission)

    # Arm and takeoff
    drone.arm()
    drone.takeoff(10.0)

    # Start mission
    drone.start_mission()

    # Create simulator and run
    sim = DroneSimulator(drone)
    positions = sim.run_mission(duration=30.0)

    # Print results
    print(f"\nMission complete!")
    print(f"Total positions recorded: {len(positions)}")
    print(f"Final position: ({drone.position.x:.1f}, {drone.position.y:.1f}, {drone.position.z:.1f})")
    print(f"Obstacles mapped: {len(drone.grid.obstacles)}")
    print(f"Safety margin used: {drone.safety_margin}m")

    # Save telemetry
    drone.save_telemetry("drone_telemetry.json")

    print("\nSystem Status:")
    print(json.dumps(drone.get_status(), indent=2))
