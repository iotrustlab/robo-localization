"""
Motion Model Module for Differential Drive Robots

This module implements comprehensive kinematic and dynamic models for differential
drive robotic systems, including coordinate transformations, drag forces, and
motion integration with rigorous mathematical foundations.

Mathematical Framework:
    - Differential drive kinematics with wheel constraints
    - Coordinate frame transformations (body ↔ world)  
    - Dynamic effects including drag and friction
    - Numerical integration with stability guarantees

Physical Models:
    - Instantaneous Center of Rotation (ICR) kinematics
    - Viscous and quadratic drag models
    - Slip and skid dynamics
    - Energy dissipation mechanisms

Author: Scientific Computing Team
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum


class DragModel(Enum):
    """Drag force models for vehicle dynamics."""
    NONE = "none"
    LINEAR = "linear"          # F_drag = -b*v
    QUADRATIC = "quadratic"    # F_drag = -c*v|v|
    COMBINED = "combined"      # F_drag = -b*v - c*v|v|


@dataclass
class VehicleParameters:
    """Physical parameters for differential drive vehicle."""
    
    wheel_base: float = 0.5       # Distance between wheels [m]
    wheel_radius: float = 0.1     # Wheel radius [m]
    vehicle_mass: float = 10.0    # Vehicle mass [kg]
    moment_inertia: float = 1.0   # Moment of inertia about vertical axis [kg·m²]
    max_wheel_speed: float = 10.0 # Maximum wheel angular velocity [rad/s]
    
    def __post_init__(self):
        """Validate vehicle parameters."""
        if self.wheel_base <= 0:
            raise ValueError(f"Wheel base must be positive, got {self.wheel_base}")
        if self.wheel_radius <= 0:
            raise ValueError(f"Wheel radius must be positive, got {self.wheel_radius}")
        if self.vehicle_mass <= 0:
            raise ValueError(f"Vehicle mass must be positive, got {self.vehicle_mass}")
        if self.moment_inertia <= 0:
            raise ValueError(f"Moment of inertia must be positive, got {self.moment_inertia}")


@dataclass
class DragParameters:
    """Drag force model parameters."""
    
    linear_coefficient: float = 0.1      # Linear drag coefficient [N·s/m]
    quadratic_coefficient: float = 0.01  # Quadratic drag coefficient [N·s²/m²]
    model_type: DragModel = DragModel.LINEAR
    
    def __post_init__(self):
        """Validate drag parameters."""
        if self.linear_coefficient < 0:
            raise ValueError(f"Linear drag coefficient must be non-negative")
        if self.quadratic_coefficient < 0:
            raise ValueError(f"Quadratic drag coefficient must be non-negative")


class MotionModel:
    """
    Comprehensive differential drive motion model with advanced physics.
    
    This class implements rigorous kinematics and dynamics for differential
    drive robots, including coordinate transformations, drag forces, and
    numerical integration with stability analysis.
    
    Mathematical Foundation:
        Differential drive kinematics are based on the instantaneous center
        of rotation (ICR) model:
        
        v = (v_L + v_R) / 2                    # Linear velocity
        ω = (v_R - v_L) / L                    # Angular velocity
        
        where:
        - v_L, v_R are left and right wheel velocities
        - L is the wheel base (distance between wheels)
        
        The motion is governed by:
        dx/dt = v cos(θ)
        dy/dt = v sin(θ)  
        dθ/dt = ω
        
        With drag forces:
        F_drag = -b*v - c*v|v|  (combined model)
    
    Coordinate Frames:
        - Body frame: x-forward, y-left, z-up (robot-centric)
        - World frame: x-east, y-north, z-up (global reference)
        
    Attributes:
        vehicle_params (VehicleParameters): Physical vehicle parameters
        drag_params (DragParameters): Drag force model parameters
    """
    
    def __init__(self, 
                 vehicle_params: Optional[VehicleParameters] = None,
                 drag_params: Optional[DragParameters] = None):
        """
        Initialize motion model with vehicle and drag parameters.
        
        Args:
            vehicle_params: Physical vehicle parameters
            drag_params: Drag force model parameters
        """
        self.vehicle_params = vehicle_params or VehicleParameters()
        self.drag_params = drag_params or DragParameters()
        
        # Precompute constants for efficiency
        self._half_wheelbase = self.vehicle_params.wheel_base / 2.0
        
        # Motion integration parameters
        self._integration_method = "rk4"  # Runge-Kutta 4th order
        self._min_dt = 1e-6  # Minimum time step for stability
        
    def compute_wheel_velocities_from_motion(self, 
                                           linear_velocity: float, 
                                           angular_velocity: float) -> Tuple[float, float]:
        """
        Compute required wheel velocities for desired motion.
        
        Inverse kinematics: given desired robot motion, compute wheel speeds.
        
        Args:
            linear_velocity: Desired linear velocity [m/s]
            angular_velocity: Desired angular velocity [rad/s]
            
        Returns:
            Tuple of (left_wheel_velocity, right_wheel_velocity) in m/s
            
        Mathematical Model:
            v_L = v - ω*L/2
            v_R = v + ω*L/2
        """
        left_velocity = linear_velocity - angular_velocity * self._half_wheelbase
        right_velocity = linear_velocity + angular_velocity * self._half_wheelbase
        
        return left_velocity, right_velocity
    
    def compute_motion_from_wheel_velocities(self, 
                                           left_wheel_velocity: float,
                                           right_wheel_velocity: float) -> Tuple[float, float]:
        """
        Compute robot motion from wheel velocities.
        
        Forward kinematics: given wheel speeds, compute robot motion.
        
        Args:
            left_wheel_velocity: Left wheel velocity [m/s]
            right_wheel_velocity: Right wheel velocity [m/s]
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
            
        Mathematical Model:
            v = (v_L + v_R) / 2
            ω = (v_R - v_L) / L
        """
        linear_velocity = (left_wheel_velocity + right_wheel_velocity) / 2.0
        angular_velocity = (right_wheel_velocity - left_wheel_velocity) / self.vehicle_params.wheel_base
        
        return linear_velocity, angular_velocity
    
    def compute_wheel_angular_velocities(self, 
                                       left_wheel_velocity: float,
                                       right_wheel_velocity: float) -> Tuple[float, float]:
        """
        Convert wheel linear velocities to angular velocities.
        
        Args:
            left_wheel_velocity: Left wheel linear velocity [m/s]
            right_wheel_velocity: Right wheel linear velocity [m/s]
            
        Returns:
            Tuple of (left_angular_velocity, right_angular_velocity) in rad/s
        """
        left_angular = left_wheel_velocity / self.vehicle_params.wheel_radius
        right_angular = right_wheel_velocity / self.vehicle_params.wheel_radius
        
        # Apply wheel speed limits
        max_angular = self.vehicle_params.max_wheel_speed
        left_angular = np.clip(left_angular, -max_angular, max_angular)
        right_angular = np.clip(right_angular, -max_angular, max_angular)
        
        return left_angular, right_angular
    
    def body_to_world_transformation_matrix(self, orientation: np.ndarray) -> np.ndarray:
        """
        Compute 3D transformation matrix from body to world frame.
        
        Args:
            orientation: Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            4x4 homogeneous transformation matrix
            
        Mathematical Model:
            Full 3D rotation matrix with ZYX Euler angle convention:
            R = R_z(yaw) * R_y(pitch) * R_x(roll)
        """
        roll, pitch, yaw = orientation
        
        # Trigonometric values
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # 3D rotation matrix (ZYX convention)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        
        # Homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        
        return T
    
    def body_to_world_velocity(self, 
                              body_velocity: np.ndarray, 
                              orientation: np.ndarray) -> np.ndarray:
        """
        Transform velocity vector from body frame to world frame.
        
        Args:
            body_velocity: Velocity in body frame [vx, vy, vz] or [vx, vy]
            orientation: Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            Velocity in world frame [vx_world, vy_world, vz_world]
            
        Mathematical Model:
            v_world = R_body_to_world * v_body
        """
        # Ensure 3D velocity vector
        if len(body_velocity) == 2:
            body_velocity = np.array([body_velocity[0], body_velocity[1], 0.0])
        elif len(body_velocity) != 3:
            raise ValueError(f"Body velocity must be 2D or 3D, got {len(body_velocity)}D")
            
        # Get rotation matrix
        T = self.body_to_world_transformation_matrix(orientation)
        R = T[:3, :3]
        
        # Transform velocity
        world_velocity = R @ body_velocity
        
        return world_velocity
    
    def world_to_body_velocity(self, 
                              world_velocity: np.ndarray, 
                              orientation: np.ndarray) -> np.ndarray:
        """
        Transform velocity vector from world frame to body frame.
        
        Args:
            world_velocity: Velocity in world frame [vx, vy, vz] or [vx, vy]
            orientation: Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            Velocity in body frame [vx_body, vy_body, vz_body]
        """
        # Ensure 3D velocity vector
        if len(world_velocity) == 2:
            world_velocity = np.array([world_velocity[0], world_velocity[1], 0.0])
            
        # Get inverse rotation matrix (transpose for rotation)
        T = self.body_to_world_transformation_matrix(orientation)
        R_inv = T[:3, :3].T
        
        # Transform velocity
        body_velocity = R_inv @ world_velocity
        
        return body_velocity
    
    def compute_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute drag force acting on vehicle.
        
        Args:
            velocity: Velocity vector [vx, vy, vz] in m/s
            
        Returns:
            Drag force vector [Fx, Fy, Fz] in N
            
        Mathematical Models:
            Linear: F_drag = -b * v
            Quadratic: F_drag = -c * v * |v|
            Combined: F_drag = -b * v - c * v * |v|
        """
        if self.drag_params.model_type == DragModel.NONE:
            return np.zeros_like(velocity)
            
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-12:  # Avoid division by zero
            return np.zeros_like(velocity)
            
        # Unit velocity vector
        velocity_unit = velocity / speed
        
        # Compute drag magnitude based on model type
        if self.drag_params.model_type == DragModel.LINEAR:
            drag_magnitude = self.drag_params.linear_coefficient * speed
            
        elif self.drag_params.model_type == DragModel.QUADRATIC:
            drag_magnitude = self.drag_params.quadratic_coefficient * speed**2
            
        elif self.drag_params.model_type == DragModel.COMBINED:
            linear_drag = self.drag_params.linear_coefficient * speed
            quadratic_drag = self.drag_params.quadratic_coefficient * speed**2
            drag_magnitude = linear_drag + quadratic_drag
            
        else:
            raise ValueError(f"Unknown drag model: {self.drag_params.model_type}")
        
        # Drag force opposes motion
        drag_force = -drag_magnitude * velocity_unit
        
        return drag_force
    
    def apply_drag_to_velocity(self, 
                              velocity: np.ndarray, 
                              dt: float) -> np.ndarray:
        """
        Apply drag effects to velocity over time step.
        
        Args:
            velocity: Current velocity vector [vx, vy, vz]
            dt: Time step [s]
            
        Returns:
            Updated velocity after drag application
            
        Physical Model:
            Drag acceleration: a_drag = F_drag / m
            Velocity update: v_new = v_old + a_drag * dt
            
            Special handling to prevent velocity reversal due to drag.
        """
        if dt <= 0:
            return velocity.copy()
            
        # Compute drag force and acceleration
        drag_force = self.compute_drag_force(velocity)
        drag_acceleration = drag_force / self.vehicle_params.vehicle_mass
        
        # Update velocity
        new_velocity = velocity + drag_acceleration * dt
        
        # Prevent velocity reversal - if drag would reverse motion, stop instead
        for i in range(len(velocity)):
            if abs(velocity[i]) > 1e-12:  # Only check non-zero components
                if np.sign(new_velocity[i]) != np.sign(velocity[i]):
                    new_velocity[i] = 0.0
        
        return new_velocity
    
    def compute_instantaneous_center_rotation(self, 
                                            linear_velocity: float,
                                            angular_velocity: float) -> Optional[Tuple[float, float]]:
        """
        Compute instantaneous center of rotation (ICR) coordinates.
        
        Args:
            linear_velocity: Robot linear velocity [m/s]
            angular_velocity: Robot angular velocity [rad/s]
            
        Returns:
            ICR coordinates (x_icr, y_icr) in body frame, or None for straight motion
            
        Mathematical Model:
            For differential drive: ICR_y = v / ω
            ICR is located on the line connecting the wheels
        """
        if abs(angular_velocity) < 1e-12:
            return None  # Straight line motion, ICR at infinity
            
        # ICR distance from robot center
        icr_distance = linear_velocity / angular_velocity
        
        # ICR coordinates in body frame (on y-axis)
        icr_x = 0.0
        icr_y = icr_distance
        
        return icr_x, icr_y
    
    def compute_wheel_slip_constraints(self, 
                                     velocity: np.ndarray,
                                     angular_velocity: float) -> Dict[str, float]:
        """
        Analyze kinematic constraints and potential wheel slip.
        
        Args:
            velocity: Robot velocity in body frame [vx, vy, vz]
            angular_velocity: Robot angular velocity [rad/s]
            
        Returns:
            Dictionary with constraint analysis results
        """
        # Extract body frame velocities
        vx_body = velocity[0] if len(velocity) > 0 else 0.0
        vy_body = velocity[1] if len(velocity) > 1 else 0.0
        
        # Ideal wheel velocities for no-slip condition
        ideal_left, ideal_right = self.compute_wheel_velocities_from_motion(
            vx_body, angular_velocity)
        
        # Lateral slip velocity (should be zero for pure rolling)
        lateral_slip = abs(vy_body)
        
        # Slip angle (angle between velocity and heading)
        slip_angle = np.arctan2(vy_body, vx_body) if abs(vx_body) > 1e-12 else 0.0
        
        return {
            'lateral_slip_velocity': lateral_slip,
            'slip_angle_rad': slip_angle,
            'slip_angle_deg': np.degrees(slip_angle),
            'ideal_left_wheel_velocity': ideal_left,
            'ideal_right_wheel_velocity': ideal_right,
            'constraint_violation': lateral_slip > 0.01  # 1 cm/s threshold
        }
    
    def integrate_motion_rk4(self, 
                           position: np.ndarray,
                           orientation: np.ndarray,
                           velocity: np.ndarray,
                           angular_velocity: float,
                           dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate motion using 4th-order Runge-Kutta method.
        
        Args:
            position: Current position [x, y, z]
            orientation: Current orientation [roll, pitch, yaw]
            velocity: Current velocity in world frame [vx, vy, vz]
            angular_velocity: Current angular velocity [rad/s]
            dt: Time step [s]
            
        Returns:
            Tuple of (new_position, new_orientation)
            
        Mathematical Method:
            4th-order Runge-Kutta integration for improved accuracy:
            
            k1 = f(t, y)
            k2 = f(t + dt/2, y + k1*dt/2)
            k3 = f(t + dt/2, y + k2*dt/2)
            k4 = f(t + dt, y + k3*dt)
            
            y_new = y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        """
        if dt < self._min_dt:
            warnings.warn(f"Time step {dt} below minimum {self._min_dt}, using Euler method")
            return self.integrate_motion_euler(position, orientation, velocity, 
                                             angular_velocity, dt)
        
        def motion_derivatives(pos, orient, vel, ang_vel):
            """Compute derivatives for motion integration."""
            # Position derivative is velocity
            pos_dot = vel.copy()
            
            # Orientation derivative (simplified - only yaw changes)
            orient_dot = np.zeros_like(orient)
            orient_dot[2] = ang_vel  # dyaw/dt = angular_velocity
            
            return pos_dot, orient_dot
        
        # RK4 integration
        dt_half = dt / 2.0
        
        # k1
        k1_pos, k1_orient = motion_derivatives(position, orientation, velocity, angular_velocity)
        
        # k2
        pos_k2 = position + k1_pos * dt_half
        orient_k2 = orientation + k1_orient * dt_half
        k2_pos, k2_orient = motion_derivatives(pos_k2, orient_k2, velocity, angular_velocity)
        
        # k3
        pos_k3 = position + k2_pos * dt_half
        orient_k3 = orientation + k2_orient * dt_half
        k3_pos, k3_orient = motion_derivatives(pos_k3, orient_k3, velocity, angular_velocity)
        
        # k4
        pos_k4 = position + k3_pos * dt
        orient_k4 = orientation + k3_orient * dt
        k4_pos, k4_orient = motion_derivatives(pos_k4, orient_k4, velocity, angular_velocity)
        
        # Final integration
        new_position = position + (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos) * dt / 6.0
        new_orientation = orientation + (k1_orient + 2*k2_orient + 2*k3_orient + k4_orient) * dt / 6.0
        
        # Normalize orientation angles
        new_orientation[2] = self._normalize_angle(new_orientation[2])
        
        return new_position, new_orientation
    
    def integrate_motion_euler(self, 
                             position: np.ndarray,
                             orientation: np.ndarray,
                             velocity: np.ndarray,
                             angular_velocity: float,
                             dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate motion using simple Euler method.
        
        Args:
            position: Current position [x, y, z]
            orientation: Current orientation [roll, pitch, yaw]
            velocity: Current velocity in world frame [vx, vy, vz]
            angular_velocity: Current angular velocity [rad/s]
            dt: Time step [s]
            
        Returns:
            Tuple of (new_position, new_orientation)
        """
        # Ensure 3D arrays
        if len(position) == 2:
            position = np.array([position[0], position[1], 0.0])
        if len(velocity) == 2:
            velocity = np.array([velocity[0], velocity[1], 0.0])
        if len(orientation) == 2:
            orientation = np.array([0.0, 0.0, orientation[0]])  # Assume only yaw
            
        # Simple Euler integration
        new_position = position + velocity * dt
        
        new_orientation = orientation.copy()
        new_orientation[2] += angular_velocity * dt  # Update yaw
        new_orientation[2] = self._normalize_angle(new_orientation[2])
        
        return new_position, new_orientation
    
    def integrate_motion(self, 
                        position: np.ndarray,
                        orientation: np.ndarray,
                        velocity: np.ndarray,
                        angular_velocity: float,
                        dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate motion using selected numerical method.
        
        Args:
            position: Current position [x, y, z]
            orientation: Current orientation [roll, pitch, yaw]
            velocity: Current velocity in world frame
            angular_velocity: Current angular velocity [rad/s]
            dt: Time step [s]
            
        Returns:
            Tuple of (new_position, new_orientation)
        """
        if self._integration_method == "rk4":
            return self.integrate_motion_rk4(position, orientation, velocity, 
                                           angular_velocity, dt)
        else:
            return self.integrate_motion_euler(position, orientation, velocity, 
                                             angular_velocity, dt)
    
    def compute_motion_energy(self, 
                            linear_velocity: float,
                            angular_velocity: float) -> Dict[str, float]:
        """
        Compute kinetic energy components of robot motion.
        
        Args:
            linear_velocity: Linear velocity magnitude [m/s]
            angular_velocity: Angular velocity [rad/s]
            
        Returns:
            Dictionary with energy components in Joules
        """
        # Translational kinetic energy
        translational_energy = 0.5 * self.vehicle_params.vehicle_mass * linear_velocity**2
        
        # Rotational kinetic energy  
        rotational_energy = 0.5 * self.vehicle_params.moment_inertia * angular_velocity**2
        
        # Total kinetic energy
        total_energy = translational_energy + rotational_energy
        
        return {
            'translational': translational_energy,
            'rotational': rotational_energy,
            'total': total_energy
        }
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def __repr__(self) -> str:
        """String representation of motion model."""
        return (f"MotionModel(wheelbase={self.vehicle_params.wheel_base:.3f}m, "
                f"drag={self.drag_params.model_type.value})")