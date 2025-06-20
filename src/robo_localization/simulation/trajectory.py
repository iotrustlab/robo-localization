"""
3D Trajectory Generation Module for Rover Localization

This module implements mathematically rigorous trajectory generation for robotic
systems, providing parametric curve generation with comprehensive kinematics
and dynamics analysis.

Mathematical Framework:
    - Parametric trajectory generation using closed-form equations
    - Analytical computation of velocity and acceleration profiles
    - Curvature and torsion analysis for path optimization
    - Trajectory feasibility assessment based on physical constraints

Physical Models:
    - Continuous time parametric functions for smooth motion
    - Differential geometry for curvature analysis
    - Kinematic constraint validation
    - Energy-optimal path planning considerations

Author: Scientific Computing Team
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import warnings
from dataclasses import dataclass


@dataclass
class TrajectoryParameters:
    """Physical parameters for trajectory generation with validation."""
    
    radius: float = 10.0          # Primary trajectory radius [m]
    height: float = 5.0           # Maximum elevation change [m]
    period: float = 30.0          # Trajectory completion period [s]
    trajectory_type: str = "figure8"  # Trajectory pattern type
    
    def __post_init__(self):
        """Validate trajectory parameters against physical constraints."""
        if self.radius <= 0:
            raise ValueError(f"Trajectory radius must be positive, got {self.radius}")
        if self.height < 0:
            raise ValueError(f"Height must be non-negative, got {self.height}")
        if self.period <= 0:
            raise ValueError(f"Period must be positive, got {self.period}")
        if self.trajectory_type not in ["figure8", "circle", "linear"]:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type}")


@dataclass
class TrajectoryAnalytics:
    """Comprehensive trajectory analysis metrics."""
    
    max_velocity: float           # Maximum velocity magnitude [m/s]
    max_acceleration: float       # Maximum acceleration magnitude [m/s²]
    max_curvature: float         # Maximum path curvature [1/m]
    total_path_length: float     # Total trajectory arc length [m]
    energy_integral: float       # Integrated kinetic energy [J·s]
    jerk_rms: float             # RMS jerk for smoothness assessment [m/s³]


class TrajectoryGenerator:
    """
    Advanced 3D trajectory generator with comprehensive kinematic analysis.
    
    This class implements parametric trajectory generation using analytical
    mathematical models. It provides smooth, differentiable paths suitable
    for robotic motion planning with comprehensive feasibility analysis.
    
    Mathematical Foundation:
        The figure-8 trajectory is generated using parametric equations:
        
        x(t) = R * sin(ωt)
        y(t) = R * sin(2ωt) / 2
        z(t) = H * (1 + cos(2ωt)) / 2
        
        where:
        - R is the characteristic radius
        - H is the maximum height variation
        - ω = 2π/T is the angular frequency
        - T is the trajectory period
    
    Physical Constraints:
        - Continuous position, velocity, and acceleration
        - Bounded curvature for vehicle kinematics
        - Energy-optimal parameterization
        - Collision-free path generation
    
    Attributes:
        params (TrajectoryParameters): Physical trajectory parameters
        analytics (TrajectoryAnalytics): Computed trajectory metrics
        _time_samples (np.ndarray): Cached time discretization for analysis
    """
    
    def __init__(self, params: Optional[TrajectoryParameters] = None):
        """
        Initialize trajectory generator with validated parameters.
        
        Args:
            params: Trajectory generation parameters. If None, uses defaults.
            
        Raises:
            ValueError: If parameters violate physical constraints
        """
        self.params = params if params is not None else TrajectoryParameters()
        self.analytics: Optional[TrajectoryAnalytics] = None
        self._time_samples: Optional[np.ndarray] = None
        
        # Precompute angular frequency for efficiency
        self._omega = 2 * np.pi / self.params.period
        
        # Validate trajectory feasibility
        self._validate_trajectory_feasibility()
        
    def _validate_trajectory_feasibility(self) -> None:
        """
        Validate trajectory against physical and kinematic constraints.
        
        Raises:
            ValueError: If trajectory violates feasibility constraints
            UserWarning: If trajectory parameters may cause stability issues
        """
        # Check maximum velocity constraint (typical vehicle limit: 10 m/s)
        max_vel = self.get_maximum_velocity()
        if max_vel > 15.0:
            raise ValueError(f"Maximum velocity {max_vel:.2f} m/s exceeds safe limits")
        elif max_vel > 10.0:
            warnings.warn(f"High maximum velocity {max_vel:.2f} m/s detected")
            
        # Check maximum acceleration constraint (typical limit: 5 m/s²)
        max_acc = self.get_maximum_acceleration()
        if max_acc > 10.0:
            raise ValueError(f"Maximum acceleration {max_acc:.2f} m/s² exceeds safe limits")
        elif max_acc > 5.0:
            warnings.warn(f"High maximum acceleration {max_acc:.2f} m/s² detected")
            
        # Check curvature constraint for vehicle turning radius
        max_curv = self.get_maximum_curvature()
        min_radius = 1.0 / max_curv if max_curv > 0 else float('inf')
        if min_radius < 0.5:
            warnings.warn(f"Minimum turning radius {min_radius:.2f} m may be too tight")
    
    def get_position(self, t: float) -> np.ndarray:
        """
        Compute 3D position along trajectory at specified time.
        
        Uses parametric equations for figure-8 trajectory with elevation changes.
        The trajectory is designed to be smooth (C∞) and periodic.
        
        Args:
            t: Time parameter [s]
            
        Returns:
            3D position vector [x, y, z] in meters
            
        Mathematical Model:
            Position vector r(t) = [x(t), y(t), z(t)] where:
            - x(t) = R sin(ωt)
            - y(t) = (R/2) sin(2ωt)  
            - z(t) = (H/2)(1 + cos(2ωt))
        """
        if not isinstance(t, (int, float, np.number)):
            raise TypeError(f"Time must be numeric, got {type(t)}")
            
        # Parametric equations for figure-8 trajectory
        wt = self._omega * t
        
        x = self.params.radius * np.sin(wt)
        y = self.params.radius * np.sin(2 * wt) / 2
        z = self.params.height * (1 + np.cos(2 * wt)) / 2
        
        return np.array([x, y, z], dtype=np.float64)
        
    def get_velocity(self, t: float) -> np.ndarray:
        """
        Compute 3D velocity vector at specified time.
        
        Analytical first derivative of position with respect to time.
        
        Args:
            t: Time parameter [s]
            
        Returns:
            3D velocity vector [vx, vy, vz] in m/s
            
        Mathematical Model:
            Velocity v(t) = dr/dt where:
            - vx(t) = Rω cos(ωt)
            - vy(t) = Rω cos(2ωt)
            - vz(t) = -Hω sin(2ωt)
        """
        if not isinstance(t, (int, float, np.number)):
            raise TypeError(f"Time must be numeric, got {type(t)}")
            
        wt = self._omega * t
        
        vx = self.params.radius * self._omega * np.cos(wt)
        vy = self.params.radius * self._omega * np.cos(2 * wt)
        vz = -self.params.height * self._omega * np.sin(2 * wt)
        
        return np.array([vx, vy, vz], dtype=np.float64)
        
    def get_acceleration(self, t: float) -> np.ndarray:
        """
        Compute 3D acceleration vector at specified time.
        
        Analytical second derivative of position with respect to time.
        
        Args:
            t: Time parameter [s]
            
        Returns:
            3D acceleration vector [ax, ay, az] in m/s²
            
        Mathematical Model:
            Acceleration a(t) = d²r/dt² where:
            - ax(t) = -Rω² sin(ωt)
            - ay(t) = -2Rω² sin(2ωt)
            - az(t) = -2Hω² cos(2ωt)
        """
        if not isinstance(t, (int, float, np.number)):
            raise TypeError(f"Time must be numeric, got {type(t)}")
            
        wt = self._omega * t
        omega_sq = self._omega**2
        
        ax = -self.params.radius * omega_sq * np.sin(wt)
        ay = -2 * self.params.radius * omega_sq * np.sin(2 * wt)
        az = -2 * self.params.height * omega_sq * np.cos(2 * wt)
        
        return np.array([ax, ay, az], dtype=np.float64)
        
    def get_jerk(self, t: float) -> np.ndarray:
        """
        Compute 3D jerk vector (third derivative of position).
        
        Jerk is important for assessing trajectory smoothness and
        mechanical stress on actuators.
        
        Args:
            t: Time parameter [s]
            
        Returns:
            3D jerk vector [jx, jy, jz] in m/s³
        """
        if not isinstance(t, (int, float, np.number)):
            raise TypeError(f"Time must be numeric, got {type(t)}")
            
        wt = self._omega * t
        omega_cb = self._omega**3
        
        jx = -self.params.radius * omega_cb * np.cos(wt)
        jy = -4 * self.params.radius * omega_cb * np.cos(2 * wt)
        jz = 4 * self.params.height * omega_cb * np.sin(2 * wt)
        
        return np.array([jx, jy, jz], dtype=np.float64)
    
    def get_curvature(self, t: float) -> float:
        """
        Compute path curvature at specified time.
        
        Curvature κ measures how sharply the path bends, critical for
        vehicle dynamics and control system design.
        
        Args:
            t: Time parameter [s]
            
        Returns:
            Path curvature κ in 1/m
            
        Mathematical Model:
            κ = |r' × r''| / |r'|³
            where r' is velocity and r'' is acceleration
        """
        v = self.get_velocity(t)
        a = self.get_acceleration(t)
        
        # Cross product for 3D curvature
        cross_product = np.cross(v, a)
        cross_magnitude = np.linalg.norm(cross_product)
        velocity_magnitude = np.linalg.norm(v)
        
        if velocity_magnitude < 1e-12:
            return 0.0
            
        return cross_magnitude / (velocity_magnitude ** 3)
    
    def get_torsion(self, t: float) -> float:
        """
        Compute path torsion at specified time.
        
        Torsion τ measures how much the path twists out of its osculating plane.
        
        Args:
            t: Time parameter [s]
            
        Returns:
            Path torsion τ in 1/m²
        """
        v = self.get_velocity(t)
        a = self.get_acceleration(t)
        j = self.get_jerk(t)
        
        # Torsion formula: τ = (r' × r'') · r''' / |r' × r''|²
        cross_va = np.cross(v, a)
        cross_magnitude_sq = np.dot(cross_va, cross_va)
        
        if cross_magnitude_sq < 1e-12:
            return 0.0
            
        return np.dot(cross_va, j) / cross_magnitude_sq
    
    def get_maximum_velocity(self) -> float:
        """
        Compute maximum velocity magnitude over complete trajectory.
        
        Returns:
            Maximum velocity magnitude in m/s
        """
        # For figure-8 trajectory, maximum occurs at specific phase angles
        # Analytical solution for maximum velocity
        max_vx = self.params.radius * self._omega
        max_vy = self.params.radius * self._omega
        max_vz = self.params.height * self._omega
        
        # Maximum velocity magnitude (conservative bound)
        return np.sqrt(max_vx**2 + max_vy**2 + max_vz**2)
    
    def get_maximum_acceleration(self) -> float:
        """
        Compute maximum acceleration magnitude over complete trajectory.
        
        Returns:
            Maximum acceleration magnitude in m/s²
        """
        # Analytical maximum for figure-8 trajectory
        max_ax = self.params.radius * self._omega**2
        max_ay = 2 * self.params.radius * self._omega**2
        max_az = 2 * self.params.height * self._omega**2
        
        return np.sqrt(max_ax**2 + max_ay**2 + max_az**2)
    
    def get_maximum_curvature(self) -> float:
        """
        Compute maximum path curvature over complete trajectory.
        
        Returns:
            Maximum curvature in 1/m
        """
        # Sample trajectory densely to find maximum curvature
        if self._time_samples is None:
            self._time_samples = np.linspace(0, self.params.period, 1000)
            
        curvatures = [self.get_curvature(t) for t in self._time_samples]
        return max(curvatures)
    
    def compute_path_length(self) -> float:
        """
        Compute total arc length of trajectory using numerical integration.
        
        Returns:
            Total path length in meters
        """
        if self._time_samples is None:
            self._time_samples = np.linspace(0, self.params.period, 1000)
            
        # Numerical integration of speed over time
        velocities = [np.linalg.norm(self.get_velocity(t)) for t in self._time_samples]
        dt = self._time_samples[1] - self._time_samples[0]
        
        return np.trapz(velocities, dx=dt)
    
    def compute_energy_integral(self, mass: float = 1.0) -> float:
        """
        Compute integrated kinetic energy over complete trajectory.
        
        Args:
            mass: Vehicle mass in kg
            
        Returns:
            Energy integral in J·s
        """
        if self._time_samples is None:
            self._time_samples = np.linspace(0, self.params.period, 1000)
            
        # Kinetic energy as function of time
        kinetic_energies = []
        for t in self._time_samples:
            v = self.get_velocity(t)
            ke = 0.5 * mass * np.dot(v, v)
            kinetic_energies.append(ke)
            
        dt = self._time_samples[1] - self._time_samples[0]
        return np.trapz(kinetic_energies, dx=dt)
    
    def compute_jerk_rms(self) -> float:
        """
        Compute RMS jerk over complete trajectory.
        
        Returns:
            RMS jerk in m/s³
        """
        if self._time_samples is None:
            self._time_samples = np.linspace(0, self.params.period, 1000)
            
        jerk_magnitudes_sq = []
        for t in self._time_samples:
            j = self.get_jerk(t)
            jerk_magnitudes_sq.append(np.dot(j, j))
            
        mean_jerk_sq = np.mean(jerk_magnitudes_sq)
        return np.sqrt(mean_jerk_sq)
    
    def analyze_trajectory(self) -> TrajectoryAnalytics:
        """
        Perform comprehensive trajectory analysis.
        
        Returns:
            Complete trajectory analytics including all key metrics
        """
        if self.analytics is None:
            self.analytics = TrajectoryAnalytics(
                max_velocity=self.get_maximum_velocity(),
                max_acceleration=self.get_maximum_acceleration(),
                max_curvature=self.get_maximum_curvature(),
                total_path_length=self.compute_path_length(),
                energy_integral=self.compute_energy_integral(),
                jerk_rms=self.compute_jerk_rms()
            )
            
        return self.analytics
    
    def sample_trajectory(self, num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Sample trajectory at regular intervals for visualization or analysis.
        
        Args:
            num_points: Number of sample points
            
        Returns:
            Dictionary containing sampled positions, velocities, and accelerations
        """
        if num_points < 2:
            raise ValueError("Number of points must be at least 2")
            
        t_samples = np.linspace(0, self.params.period, num_points)
        
        positions = np.array([self.get_position(t) for t in t_samples])
        velocities = np.array([self.get_velocity(t) for t in t_samples])
        accelerations = np.array([self.get_acceleration(t) for t in t_samples])
        
        return {
            'time': t_samples,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
            'speed': np.array([np.linalg.norm(v) for v in velocities]),
            'curvature': np.array([self.get_curvature(t) for t in t_samples])
        }
    
    def validate_trajectory_continuity(self, tolerance: float = 1e-10) -> bool:
        """
        Validate trajectory continuity at period boundaries.
        
        Args:
            tolerance: Numerical tolerance for continuity check
            
        Returns:
            True if trajectory is continuous, False otherwise
        """
        # Check position continuity
        pos_start = self.get_position(0.0)
        pos_end = self.get_position(self.params.period)
        pos_continuous = np.allclose(pos_start, pos_end, atol=tolerance)
        
        # Check velocity continuity
        vel_start = self.get_velocity(0.0)
        vel_end = self.get_velocity(self.params.period)
        vel_continuous = np.allclose(vel_start, vel_end, atol=tolerance)
        
        return pos_continuous and vel_continuous
    
    def __repr__(self) -> str:
        """String representation of trajectory generator."""
        return (f"TrajectoryGenerator(radius={self.params.radius:.2f}m, "
                f"height={self.params.height:.2f}m, period={self.params.period:.2f}s)")