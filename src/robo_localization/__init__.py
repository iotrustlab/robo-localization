"""
Robo Localization: 3D Rover Localization with Redundant Sensor Fusion

A scientific Python package for demonstrating multi-sensor fusion techniques
in mobile robot localization using Extended Kalman Filtering.

This package implements:
- Extended Kalman Filter for state estimation
- Sensor models for GPS, IMU, and wheel odometry
- 3D trajectory generation and rover simulation
- Real-time visualization and performance analysis

The system demonstrates how redundant sensors improve localization robustness
by automatically handling sensor failures and maintaining accuracy through
intelligent sensor fusion algorithms.
"""

from .simulation.rover import RoverSimulation
from .simulation.trajectory import TrajectoryGenerator
from .sensors.gps import GPSSensor
from .sensors.imu import IMUSensor
from .sensors.odometry import WheelOdometry
from .sensors.manager import SensorFusionManager
from .fusion.kalman import ExtendedKalmanFilter

# Optional visualization import (graceful failure if not available)
try:
    from .visualization.plotter import RealTimeVisualizer
    _has_visualization = True
except ImportError:
    RealTimeVisualizer = None
    _has_visualization = False

__version__ = "1.0.0"
__author__ = "Robo Localization Team"

__all__ = [
    "RoverSimulation",
    "TrajectoryGenerator", 
    "GPSSensor",
    "IMUSensor",
    "WheelOdometry",
    "SensorFusionManager",
    "ExtendedKalmanFilter"
]

# Add visualization to __all__ only if available
if _has_visualization:
    __all__.append("RealTimeVisualizer")