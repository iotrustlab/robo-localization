"""
Simulation components for robo localization.

This module contains comprehensive rover simulation, trajectory generation, and motion models
for testing and demonstrating localization algorithms with rigorous mathematical foundations.

Components:
    - TrajectoryGenerator: Advanced 3D trajectory generation with kinematic analysis
    - MotionModel: Differential drive dynamics with drag forces and coordinate transformations
    - RoverSimulation: Complete rover simulation environment with multiple control modes
    
Mathematical Models:
    - Parametric trajectory generation with analytical derivatives
    - Differential drive kinematics and dynamics
    - Coordinate frame transformations (body ↔ world)
    - Advanced control algorithms (PID + feedforward)
    - Comprehensive performance analysis and metrics
"""

# Core simulation classes
from .rover import RoverSimulation, ControlMode, ControlParameters, SimulationState, SimulationMetrics
from .trajectory import TrajectoryGenerator, TrajectoryParameters, TrajectoryAnalytics
from .motion import MotionModel, VehicleParameters, DragParameters, DragModel

# Utility functions
from .rover import compare_simulations, analyze_trajectory_feasibility

__all__ = [
    # Core classes
    "RoverSimulation",
    "TrajectoryGenerator", 
    "MotionModel",
    
    # Configuration classes
    "ControlMode",
    "ControlParameters",
    "SimulationState", 
    "SimulationMetrics",
    "TrajectoryParameters",
    "TrajectoryAnalytics",
    "VehicleParameters",
    "DragParameters",
    "DragModel",
    
    # Utility functions
    "compare_simulations",
    "analyze_trajectory_feasibility"
]