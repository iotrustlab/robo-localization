"""
Rover simulation with 3D trajectory generation and motion dynamics.

This module implements:
- 3D figure-8 trajectory generation with elevation changes
- Differential drive motion model
- Physics-based dynamics with drag
- Coordinate frame transformations
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import time


class TrajectoryGenerator:
    """Generates 3D figure-8 trajectory with elevation changes"""
    
    def __init__(self, radius=10.0, height=5.0, period=30.0):
        self.radius = radius
        self.height = height
        self.period = period
        
    def get_position(self, t):
        """Get position at time t"""
        # Figure-8 parametric equations
        omega = 2 * np.pi / self.period
        
        # X coordinate - figure-8 in XY plane
        x = self.radius * np.sin(omega * t)
        
        # Y coordinate - figure-8 (double frequency)
        y = self.radius * np.sin(2 * omega * t) / 2
        
        # Z coordinate - elevation changes
        z = self.height * (1 + np.cos(2 * omega * t)) / 2
        
        return np.array([x, y, z])
        
    def get_velocity(self, t):
        """Get velocity at time t"""
        omega = 2 * np.pi / self.period
        
        # Derivatives of position
        vx = self.radius * omega * np.cos(omega * t)
        vy = self.radius * omega * np.cos(2 * omega * t)
        vz = -self.height * omega * np.sin(2 * omega * t)
        
        return np.array([vx, vy, vz])
        
    def get_acceleration(self, t):
        """Get acceleration at time t"""
        omega = 2 * np.pi / self.period
        
        # Second derivatives of position
        ax = -self.radius * omega**2 * np.sin(omega * t)
        ay = -2 * self.radius * omega**2 * np.sin(2 * omega * t)
        az = -2 * self.height * omega**2 * np.cos(2 * omega * t)
        
        return np.array([ax, ay, az])


class MotionModel:
    """Differential drive rover motion model with physics"""
    
    def __init__(self, wheel_base, drag_coefficient=0.1):
        self.wheel_base = wheel_base
        self.drag_coefficient = drag_coefficient
        
    def compute_velocities(self, left_wheel_speed, right_wheel_speed):
        """Compute linear and angular velocities from wheel speeds"""
        # Assuming wheel_radius is embedded in wheel_speed (already converted)
        average_speed = (left_wheel_speed + right_wheel_speed) / 2
        speed_difference = right_wheel_speed - left_wheel_speed
        
        linear_velocity = average_speed
        angular_velocity = speed_difference / self.wheel_base
        
        return linear_velocity, angular_velocity
        
    def body_to_world_velocity(self, body_velocity, orientation):
        """Transform velocity from body frame to world frame"""
        roll, pitch, yaw = orientation
        
        # Rotation matrix for yaw (simplified - only yaw rotation)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Transform velocity
        world_vx = body_velocity[0] * cos_yaw - body_velocity[1] * sin_yaw
        world_vy = body_velocity[0] * sin_yaw + body_velocity[1] * cos_yaw
        
        return np.array([world_vx, world_vy])
        
    def apply_drag(self, velocity, dt):
        """Apply drag forces to velocity"""
        # Simple drag model: F_drag = -c * v * |v|
        speed = np.linalg.norm(velocity)
        
        if speed > 1e-6:  # Avoid division by zero
            drag_force = -self.drag_coefficient * speed
            drag_acceleration = drag_force * velocity / speed
            
            # Apply drag
            new_velocity = velocity + drag_acceleration * dt
            
            # Ensure velocity doesn't reverse direction due to drag
            if np.dot(new_velocity, velocity) < 0:
                new_velocity = np.zeros_like(velocity)
                
            return new_velocity
        else:
            return velocity
            
    def integrate_motion(self, position, orientation, velocity, angular_velocity, dt):
        """Integrate motion over time step"""
        # Position update - ensure velocity is 3D
        if len(velocity) == 2:
            # Add zero z-velocity for 2D velocity input
            velocity_3d = np.array([velocity[0], velocity[1], 0.0])
        else:
            velocity_3d = velocity
            
        new_position = position + velocity_3d * dt
        
        # Orientation update (simplified - only yaw)
        new_orientation = orientation.copy()
        new_orientation[2] += angular_velocity * dt  # Update yaw
        
        return new_position, new_orientation


class RoverSimulation:
    """Complete rover simulation with trajectory following"""
    
    def __init__(self, wheel_base, wheel_radius, drag_coefficient=0.1):
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.drag_coefficient = drag_coefficient
        
        # Initialize components
        self.trajectory = TrajectoryGenerator()
        self.motion_model = MotionModel(wheel_base, drag_coefficient)
        
        # Rover state
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.velocity = np.zeros(2)     # [linear_vel, 0] in body frame
        self.angular_velocity = 0.0     # yaw rate
        
        # Control
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.enable_trajectory_following = True
        
        # Simulation state
        self.simulation_time = 0.0
        
    def set_wheel_speeds(self, left_speed, right_speed):
        """Set wheel speeds directly"""
        self.left_wheel_speed = left_speed
        self.right_wheel_speed = right_speed
        
    def update(self, dt):
        """Update rover simulation by one time step"""
        # Update simulation time
        self.simulation_time += dt
        
        # Trajectory following control (if enabled)
        if self.enable_trajectory_following:
            self._trajectory_following_control()
            
        # Convert wheel speeds to target velocities
        wheel_linear_vel = self.wheel_radius * (self.left_wheel_speed + self.right_wheel_speed) / 2
        wheel_angular_vel = self.wheel_radius * (self.right_wheel_speed - self.left_wheel_speed) / self.wheel_base
        
        # If wheel speeds are non-zero, set rover velocities directly (active control)
        # If wheel speeds are zero, let rover coast (maintain momentum)
        if abs(self.left_wheel_speed) > 1e-6 or abs(self.right_wheel_speed) > 1e-6:
            # Active control - set velocities from wheel speeds
            self.velocity[0] = wheel_linear_vel
            self.angular_velocity = wheel_angular_vel
        # else: coast with existing velocities
        
        # Apply drag to body frame velocity directly
        if self.drag_coefficient > 0:
            # Simple drag applied to body frame linear velocity
            speed = abs(self.velocity[0])
            if speed > 1e-6:
                drag_deceleration = -self.drag_coefficient * speed
                self.velocity[0] += drag_deceleration * dt
                # Ensure velocity doesn't reverse
                if self.velocity[0] * (self.velocity[0] - drag_deceleration * dt) < 0:
                    self.velocity[0] = 0.0
        
        # For motion integration, use world frame velocity
        world_velocity = self.motion_model.body_to_world_velocity(self.velocity, self.orientation)
        
        # Integrate motion
        self.position, self.orientation = self.motion_model.integrate_motion(
            self.position, self.orientation, world_velocity, self.angular_velocity, dt
        )
        
    def _trajectory_following_control(self):
        """Simple trajectory following control"""
        if self.trajectory is None:
            return
            
        # Get target position and velocity
        target_position = self.trajectory.get_position(self.simulation_time)
        target_velocity = self.trajectory.get_velocity(self.simulation_time)
        
        # Simple proportional control
        position_error = target_position - self.position
        
        # Compute desired body frame velocity
        yaw = self.orientation[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Transform position error to body frame
        body_error_x = position_error[0] * cos_yaw + position_error[1] * sin_yaw
        body_error_y = -position_error[0] * sin_yaw + position_error[1] * cos_yaw
        
        # Control gains
        kp_linear = 2.0
        kp_angular = 1.0
        
        # Compute desired velocities
        desired_linear_vel = kp_linear * body_error_x
        desired_angular_vel = kp_angular * body_error_y / max(0.1, abs(body_error_x))
        
        # Convert to wheel speeds
        desired_left_speed = (desired_linear_vel - desired_angular_vel * self.wheel_base / 2) / self.wheel_radius
        desired_right_speed = (desired_linear_vel + desired_angular_vel * self.wheel_base / 2) / self.wheel_radius
        
        # Limit wheel speeds
        max_wheel_speed = 5.0  # rad/s
        desired_left_speed = np.clip(desired_left_speed, -max_wheel_speed, max_wheel_speed)
        desired_right_speed = np.clip(desired_right_speed, -max_wheel_speed, max_wheel_speed)
        
        # Set wheel speeds
        self.set_wheel_speeds(desired_left_speed, desired_right_speed)
        
    def compute_kinetic_energy(self):
        """Compute current kinetic energy"""
        # Assume unit mass for simplicity
        linear_speed = np.linalg.norm(self.velocity)
        linear_energy = 0.5 * linear_speed**2
        
        # Rotational energy (assume unit moment of inertia)
        rotational_energy = 0.5 * self.angular_velocity**2
        
        return linear_energy + rotational_energy
        
    def get_simulation_info(self):
        """Get current simulation information"""
        trajectory_error = 0.0
        if self.trajectory is not None:
            target_pos = self.trajectory.get_position(self.simulation_time)
            trajectory_error = np.linalg.norm(self.position - target_pos)
            
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'wheel_speeds': [self.left_wheel_speed, self.right_wheel_speed],
            'trajectory_error': trajectory_error,
            'simulation_time': self.simulation_time
        } 