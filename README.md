🚁 Autonomous Drone Navigation System

An advanced Python-based autonomous navigation system for UAVs, simulating real-world flight control, path planning, and sensor fusion pipelines used in production drone autopilots.

✨ Features


3D Path Planning — Hybrid RRT* + A* algorithm for obstacle-free route generation
SLAM-based Localization & Mapping — Real-time position and environment mapping
Obstacle Avoidance — Sensor fusion of LiDAR (360° FOV) + Stereo Vision
State Estimation — Extended Kalman Filter (EKF) fusing GPS, IMU, and Barometer data
Flight Control — PID controllers with feedforward for X/Y/Z and yaw axes
Safety Systems — Geofencing, minimum/maximum altitude limits, Emergency RTL (Return to Launch)
Mission Planning — Waypoint-based missions with configurable speed and loiter time
Dynamic Replanning — Recomputes path when new obstacles are detected mid-flight
Telemetry Logging — Full flight data exported to JSON for post-mission analysis


🛠️ Tech Stack


Python 3.9
NumPy — matrix operations for EKF state estimation
Standard library: math, heapq, dataclasses, enum, threading, json


📂 Project Structure

autonomous-drone-navigation/
├── autonomus drone navigation system.py   # Core system: EKF, RRT*, A*, PID, drone + simulator classes
└── README.md

⚙️ Core Components

ModuleDescriptionDroneConfigPhysical & operational parameters (velocity limits, safety margins, PID gains)ExtendedKalmanFilter12-state EKF for position/velocity/acceleration estimationVector3D / Quaternion3D math utilities for position and orientationAutonomousDroneMain drone class — flight modes, mission execution, sensor updatesDroneSimulatorPhysics simulator with randomized obstacles and wind for testing

🚀 Getting Started

Prerequisites

bashpip install numpy

Run the simulation

bashpython "autonomus drone navigation system.py"

This runs a demo mission with 4 waypoints, simulates 30 seconds of flight with randomized obstacles and wind, then saves telemetry to drone_telemetry.json.

📊 Sample Output

============================================================
Advanced Autonomous Drone Navigation System
============================================================
[UAV-ALPHA] RTL initiated
MIN ALTITUDE - CLIMBING
GEOFENCE BREACH - RTL
...

Mission complete!
Total positions recorded: 1501
Final position: (191.6, 157.2, -1209.5)
Obstacles mapped: 959
Safety margin used: 2.0m
Telemetry saved to drone_telemetry.json

System Status:
{
  "drone_id": "UAV-ALPHA",
  "mode": "RTL",
  "position": {
    "x": 191.57,
    "y": 157.23,
    "z": -1209.49
  },
  "velocity": {
    "x": -10.61,
    "y": -10.61,
    "z": 15.0
  },
  "battery": 85.0,
  "waypoint": 4,
  "total_waypoints": 4,
  "obstacles_mapped": 959,
  "safety_margin": 2.0
}

👤 Author

Jegadheesh — Final-year ECE student | Aspiring AI/ML Engineer


GitHub: @jegadheeshjega492-art
LinkedIn: jegadheesh-jega492

