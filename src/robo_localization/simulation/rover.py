"""
Rover Simulation Module for Robotic Localization Systems

This module implements a comprehensive rover simulation environment combining
trajectory generation, motion dynamics, and control systems for robotic
localization research and development.

Mathematical Framework:
    - Complete vehicle dynamics simulation
    - Trajectory following control algorithms  
    - Multi-physics integration (kinematics + dynamics)
    - State estimation and sensor modeling
    - Performance analysis and metrics

Physical Models:
    - Differential drive vehicle dynamics
    - Environmental disturbances and uncertainties
    - Actuator limitations and constraints
    - Energy consumption modeling

Author: Scientific Computing Team
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Callable
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum

from .trajectory import TrajectoryGenerator, TrajectoryParameters
from .motion import MotionModel, VehicleParameters, DragParameters, DragModel


class ControlMode(Enum):
    """Robot control modes."""
    MANUAL = "manual"                    # Direct wheel speed control
    TRAJECTORY_FOLLOWING = "trajectory"  # Autonomous trajectory following
    VELOCITY_CONTROL = "velocity"        # Direct velocity commands
    POSITION_CONTROL = "position"        # Position-based control


@dataclass
class ControlParameters:
    """Control system parameters with physical validation."""
    
    # Trajectory following gains
    kp_linear: float = 2.0        # Linear position gain [1/s]
    kp_angular: float = 1.0       # Angular position gain [1/s]
    kd_linear: float = 0.5        # Linear velocity damping [1]
    kd_angular: float = 0.2       # Angular velocity damping [1]
    
    # Control limits
    max_linear_velocity: float = 5.0    # Maximum linear velocity [m/s]
    max_angular_velocity: float = 2.0   # Maximum angular velocity [rad/s]
    max_linear_acceleration: float = 3.0  # Maximum linear acceleration [m/s²]
    max_wheel_speed: float = 10.0       # Maximum wheel speed [rad/s]
    
    # Control tolerances
    position_tolerance: float = 0.1     # Position tracking tolerance [m]
    velocity_tolerance: float = 0.05    # Velocity tracking tolerance [m/s]
    
    def __post_init__(self):
        """Validate control parameters."""
        if self.kp_linear <= 0 or self.kp_angular <= 0:
            raise ValueError("Control gains must be positive")
        if any(param <= 0 for param in [self.max_linear_velocity, self.max_angular_velocity, 
                                       self.max_linear_acceleration, self.max_wheel_speed]):
            raise ValueError("Control limits must be positive")


@dataclass
class SimulationState:
    """Complete simulation state representation."""
    
    # Core state variables
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: float = 0.0
    
    # Control inputs
    left_wheel_speed: float = 0.0
    right_wheel_speed: float = 0.0
    
    # Simulation metadata
    simulation_time: float = 0.0
    control_mode: ControlMode = ControlMode.MANUAL
    
    # Performance metrics
    trajectory_error: float = 0.0
    energy_consumed: float = 0.0
    total_distance: float = 0.0


@dataclass
class SimulationMetrics:
    """Comprehensive simulation performance metrics."""
    
    # Trajectory tracking performance
    mean_tracking_error: float = 0.0
    max_tracking_error: float = 0.0
    rms_tracking_error: float = 0.0
    
    # Motion characteristics
    total_distance_traveled: float = 0.0
    average_speed: float = 0.0
    max_speed_achieved: float = 0.0
    
    # Energy analysis
    total_energy_consumed: float = 0.0
    energy_efficiency: float = 0.0  # distance/energy
    
    # Control performance
    control_effort_rms: float = 0.0
    wheel_speed_variance: float = 0.0
    
    # Stability metrics
    velocity_smoothness: float = 0.0
    acceleration_rms: float = 0.0


class RoverSimulation:
    """
    Comprehensive rover simulation environment with advanced dynamics.
    
    This class integrates trajectory generation, motion dynamics, and control
    systems to provide a complete simulation environment for robotic systems
    research and development.
    
    Features:
        - Multiple control modes (manual, trajectory following, velocity control)
        - Advanced motion dynamics with drag and friction
        - Comprehensive performance analysis and metrics
        - Real-time simulation capabilities
        - Extensive validation and error checking
        
    Mathematical Foundation:
        The simulation integrates multiple mathematical models:
        
        1. Trajectory Generation: Parametric curves with analytical derivatives
        2. Vehicle Dynamics: Differential drive kinematics with drag forces
        3. Control Systems: PID-based trajectory following with feedforward
        4. State Integration: Numerical methods for temporal evolution
        
    Coordinate Systems:
        - World Frame: Fixed global reference (x-east, y-north, z-up)
        - Body Frame: Robot-centered (x-forward, y-left, z-up)
        - Sensor Frame: Individual sensor orientations
        
    Attributes:
        vehicle_params (VehicleParameters): Physical vehicle parameters
        control_params (ControlParameters): Control system configuration
        trajectory_params (TrajectoryParameters): Trajectory generation settings
        state (SimulationState): Current simulation state
        metrics (SimulationMetrics): Performance analysis results
    """
    
    def __init__(self, 
                 vehicle_params: Optional[VehicleParameters] = None,
                 control_params: Optional[ControlParameters] = None,
                 trajectory_params: Optional[TrajectoryParameters] = None,
                 drag_params: Optional[DragParameters] = None):
        """
        Initialize rover simulation with comprehensive parameter validation.
        
        Args:
            vehicle_params: Physical vehicle parameters
            control_params: Control system parameters
            trajectory_params: Trajectory generation parameters
            drag_params: Drag force model parameters
            
        Raises:
            ValueError: If parameter combinations are physically invalid
        """
        # Initialize parameters with defaults
        self.vehicle_params = vehicle_params or VehicleParameters()
        self.control_params = control_params or ControlParameters()
        self.trajectory_params = trajectory_params or TrajectoryParameters()
        self.drag_params = drag_params or DragParameters()
        
        # Initialize subsystems
        self.trajectory_generator = TrajectoryGenerator(self.trajectory_params)
        self.motion_model = MotionModel(self.vehicle_params, self.drag_params)
        
        # Initialize simulation state
        self.state = SimulationState()
        self.metrics = SimulationMetrics()
        
        # Control system state
        self._previous_position_error = np.zeros(3)
        self._previous_velocity_error = np.zeros(3)
        self._control_integral = np.zeros(3)
        
        # Data logging for analysis
        self._trajectory_errors: List[float] = []
        self._velocities: List[np.ndarray] = []
        self._accelerations: List[np.ndarray] = []
        self._control_inputs: List[Tuple[float, float]] = []
        self._energy_history: List[float] = []
        
        # Simulation parameters
        self._integration_dt = 0.01  # Internal integration time step
        self._enable_logging = True
        
        # Validate system compatibility
        self._validate_system_configuration()
    
    def _validate_system_configuration(self) -> None:
        """
        Validate compatibility between vehicle and control parameters.
        
        Raises:
            ValueError: If configuration is physically invalid
            UserWarning: If configuration may cause performance issues
        """
        # Check if control limits are compatible with vehicle capabilities
        max_theoretical_speed = (self.control_params.max_wheel_speed * 
                                self.vehicle_params.wheel_radius)
        
        if self.control_params.max_linear_velocity > max_theoretical_speed:
            raise ValueError(
                f"Maximum linear velocity {self.control_params.max_linear_velocity:.2f} m/s "
                f"exceeds vehicle capability {max_theoretical_speed:.2f} m/s"
            )
        
        # Check trajectory compatibility
        trajectory_analytics = self.trajectory_generator.analyze_trajectory()
        
        if trajectory_analytics.max_velocity > self.control_params.max_linear_velocity:
            warnings.warn(
                f"Trajectory max velocity {trajectory_analytics.max_velocity:.2f} m/s "
                f"exceeds control limit {self.control_params.max_linear_velocity:.2f} m/s"
            )
        
        if trajectory_analytics.max_acceleration > self.control_params.max_linear_acceleration:
            warnings.warn(
                f"Trajectory max acceleration {trajectory_analytics.max_acceleration:.2f} m/s² "
                f"exceeds control limit {self.control_params.max_linear_acceleration:.2f} m/s²"
            )
    
    def reset_simulation(self, 
                        initial_position: Optional[np.ndarray] = None,
                        initial_orientation: Optional[np.ndarray] = None) -> None:
        """
        Reset simulation to initial conditions.
        
        Args:
            initial_position: Initial position [x, y, z]. Defaults to origin.
            initial_orientation: Initial orientation [roll, pitch, yaw]. Defaults to zeros.
        """
        # Reset state
        self.state.position = initial_position if initial_position is not None else np.zeros(3)
        self.state.orientation = initial_orientation if initial_orientation is not None else np.zeros(3)
        self.state.velocity = np.zeros(3)
        self.state.angular_velocity = 0.0
        self.state.left_wheel_speed = 0.0
        self.state.right_wheel_speed = 0.0
        self.state.simulation_time = 0.0
        self.state.trajectory_error = 0.0
        self.state.energy_consumed = 0.0
        self.state.total_distance = 0.0
        
        # Reset control system
        self._previous_position_error = np.zeros(3)
        self._previous_velocity_error = np.zeros(3)
        self._control_integral = np.zeros(3)
        
        # Clear logging data
        self._trajectory_errors.clear()
        self._velocities.clear()
        self._accelerations.clear()
        self._control_inputs.clear()
        self._energy_history.clear()
        
        # Reset metrics
        self.metrics = SimulationMetrics()
    
    def set_control_mode(self, mode: ControlMode) -> None:
        """
        Set robot control mode.
        
        Args:
            mode: Desired control mode
        """
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Invalid control mode: {mode}")
            
        self.state.control_mode = mode
        
        # Reset control system when changing modes
        self._previous_position_error = np.zeros(3)
        self._previous_velocity_error = np.zeros(3)
        self._control_integral = np.zeros(3)
    
    def set_wheel_speeds(self, left_speed: float, right_speed: float) -> None:
        """
        Set wheel speeds directly (manual control mode).
        
        Args:
            left_speed: Left wheel angular velocity [rad/s]
            right_speed: Right wheel angular velocity [rad/s]
            
        Raises:
            ValueError: If wheel speeds exceed limits
        """
        # Validate wheel speed limits
        max_speed = self.control_params.max_wheel_speed
        
        if abs(left_speed) > max_speed or abs(right_speed) > max_speed:
            raise ValueError(
                f"Wheel speeds ({left_speed:.2f}, {right_speed:.2f}) exceed limit {max_speed:.2f} rad/s"
            )
        
        self.state.left_wheel_speed = left_speed
        self.state.right_wheel_speed = right_speed
    
    def set_velocity_command(self, linear_velocity: float, angular_velocity: float) -> None:
        """
        Set desired velocities (velocity control mode).
        
        Args:
            linear_velocity: Desired linear velocity [m/s]
            angular_velocity: Desired angular velocity [rad/s]
        """
        # Validate velocity limits
        if abs(linear_velocity) > self.control_params.max_linear_velocity:
            linear_velocity = np.sign(linear_velocity) * self.control_params.max_linear_velocity
            
        if abs(angular_velocity) > self.control_params.max_angular_velocity:
            angular_velocity = np.sign(angular_velocity) * self.control_params.max_angular_velocity
        
        # Convert to wheel speeds
        left_wheel_vel, right_wheel_vel = self.motion_model.compute_wheel_velocities_from_motion(
            linear_velocity, angular_velocity)
        
        # Convert to angular velocities
        left_angular, right_angular = self.motion_model.compute_wheel_angular_velocities(
            left_wheel_vel, right_wheel_vel)
        
        self.set_wheel_speeds(left_angular, right_angular)
    
    def update(self, dt: float) -> None:
        """
        Update simulation by one time step with comprehensive physics integration.
        
        Args:
            dt: Time step [s]
            
        Raises:
            ValueError: If time step is invalid
        """
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        if dt > 0.1:
            warnings.warn(f"Large time step {dt:.3f}s may cause numerical instability")
        
        # Update simulation time
        self.state.simulation_time += dt
        
        # Store previous state for differentiation
        previous_position = self.state.position.copy()
        previous_velocity = self.state.velocity.copy()
        
        # Execute control algorithm based on mode
        if self.state.control_mode == ControlMode.TRAJECTORY_FOLLOWING:
            self._trajectory_following_control(dt)
        elif self.state.control_mode == ControlMode.VELOCITY_CONTROL:
            # Wheel speeds already set by set_velocity_command
            pass
        elif self.state.control_mode == ControlMode.MANUAL:
            # Wheel speeds set manually
            pass
        else:
            raise ValueError(f"Unsupported control mode: {self.state.control_mode}")
        
        # Convert wheel speeds to body frame velocities
        wheel_left_vel = self.state.left_wheel_speed * self.vehicle_params.wheel_radius
        wheel_right_vel = self.state.right_wheel_speed * self.vehicle_params.wheel_radius
        
        linear_vel, angular_vel = self.motion_model.compute_motion_from_wheel_velocities(
            wheel_left_vel, wheel_right_vel)
        
        # Update velocities (with potential for coasting if wheels stop)
        if abs(self.state.left_wheel_speed) > 1e-6 or abs(self.state.right_wheel_speed) > 1e-6:
            # Active control - set velocities from wheel commands
            body_velocity = np.array([linear_vel, 0.0, 0.0])  # No lateral motion in body frame
            self.state.angular_velocity = angular_vel
        else:
            # Coasting - maintain current body frame velocity
            body_velocity = self.motion_model.world_to_body_velocity(
                self.state.velocity, self.state.orientation)
        
        # Apply drag forces to body frame velocity
        body_velocity = self.motion_model.apply_drag_to_velocity(body_velocity, dt)
        
        # Transform to world frame for integration
        world_velocity = self.motion_model.body_to_world_velocity(
            body_velocity, self.state.orientation)
        
        # Integrate motion using numerical method
        self.state.position, self.state.orientation = self.motion_model.integrate_motion(
            self.state.position, self.state.orientation, world_velocity, 
            self.state.angular_velocity, dt)
        
        # Update world frame velocity
        self.state.velocity = world_velocity
        
        # Compute performance metrics
        self._update_metrics(dt, previous_position, previous_velocity)
        
        # Log data for analysis
        if self._enable_logging:
            self._log_simulation_data()
    
    def _trajectory_following_control(self, dt: float) -> None:
        """
        Advanced trajectory following control with PID and feedforward.
        
        Args:
            dt: Control loop time step [s]
            
        Mathematical Model:
            PID control with feedforward compensation:
            
            u = K_p * e + K_d * ė + K_i * ∫e dt + u_ff
            
            where:
            - e is position error
            - ė is velocity error  
            - u_ff is feedforward term from trajectory
        """
        if self.trajectory_generator is None:
            return
        
        # Get trajectory targets
        target_position = self.trajectory_generator.get_position(self.state.simulation_time)
        target_velocity = self.trajectory_generator.get_velocity(self.state.simulation_time)
        target_acceleration = self.trajectory_generator.get_acceleration(self.state.simulation_time)
        
        # Compute position error
        position_error = target_position - self.state.position
        
        # Compute velocity error
        velocity_error = target_velocity - self.state.velocity
        
        # Transform errors to body frame for control
        body_position_error = self.motion_model.world_to_body_velocity(
            position_error, self.state.orientation)
        body_velocity_error = self.motion_model.world_to_body_velocity(
            velocity_error, self.state.orientation)
        
        # PID control law
        # Proportional term
        control_p = self.control_params.kp_linear * body_position_error[0]
        
        # Derivative term (velocity error)
        control_d = self.control_params.kd_linear * body_velocity_error[0]
        
        # Integral term (accumulate error)
        self._control_integral[0] += body_position_error[0] * dt
        control_i = 0.1 * self._control_integral[0]  # Small integral gain
        
        # Feedforward term from trajectory acceleration
        body_target_accel = self.motion_model.world_to_body_velocity(
            target_acceleration, self.state.orientation)
        control_ff = body_target_accel[0] / self.control_params.kp_linear
        
        # Combined control
        desired_linear_velocity = control_p + control_d + control_i + control_ff
        
        # Angular control (simplified)
        lateral_error = body_position_error[1]
        angular_error_rate = body_velocity_error[1]
        
        desired_angular_velocity = (self.control_params.kp_angular * lateral_error + 
                                  self.control_params.kd_angular * angular_error_rate)
        
        # Apply velocity limits
        desired_linear_velocity = np.clip(
            desired_linear_velocity, 
            -self.control_params.max_linear_velocity,
            self.control_params.max_linear_velocity
        )
        
        desired_angular_velocity = np.clip(
            desired_angular_velocity,
            -self.control_params.max_angular_velocity, 
            self.control_params.max_angular_velocity
        )
        
        # Convert to wheel speeds
        left_wheel_vel, right_wheel_vel = self.motion_model.compute_wheel_velocities_from_motion(
            desired_linear_velocity, desired_angular_velocity)
        
        left_angular, right_angular = self.motion_model.compute_wheel_angular_velocities(
            left_wheel_vel, right_wheel_vel)
        
        # Set wheel speeds
        self.set_wheel_speeds(left_angular, right_angular)
        
        # Update trajectory error for metrics
        self.state.trajectory_error = np.linalg.norm(position_error)
        
        # Store previous errors for derivative computation
        self._previous_position_error = body_position_error.copy()
        self._previous_velocity_error = body_velocity_error.copy()
    
    def _update_metrics(self, dt: float, previous_position: np.ndarray, 
                       previous_velocity: np.ndarray) -> None:
        """
        Update comprehensive simulation performance metrics.
        
        Args:
            dt: Time step [s]
            previous_position: Position at previous time step
            previous_velocity: Velocity at previous time step
        """
        # Distance traveled this step
        distance_step = np.linalg.norm(self.state.position - previous_position)
        self.state.total_distance += distance_step
        
        # Energy consumption (simplified model)
        speed = np.linalg.norm(self.state.velocity)
        power = (abs(self.state.left_wheel_speed) + abs(self.state.right_wheel_speed)) * 0.1
        energy_step = power * dt
        self.state.energy_consumed += energy_step
        
        # Update trajectory tracking metrics
        if self.state.control_mode == ControlMode.TRAJECTORY_FOLLOWING:
            self._trajectory_errors.append(self.state.trajectory_error)
            
        # Log other quantities for final analysis
        self._velocities.append(self.state.velocity.copy())
        self._control_inputs.append((self.state.left_wheel_speed, self.state.right_wheel_speed))
        self._energy_history.append(energy_step)
        
        # Compute acceleration for smoothness metrics
        if len(self._velocities) > 1:
            acceleration = (self.state.velocity - previous_velocity) / dt
            self._accelerations.append(acceleration)
    
    def _log_simulation_data(self) -> None:
        """Log current simulation data for post-analysis."""
        # Data logging is handled in _update_metrics
        pass
    
    def compute_final_metrics(self) -> SimulationMetrics:
        """
        Compute comprehensive simulation metrics from logged data.
        
        Returns:
            Complete simulation performance metrics
        """
        metrics = SimulationMetrics()
        
        # Trajectory tracking metrics
        if self._trajectory_errors:
            errors = np.array(self._trajectory_errors)
            metrics.mean_tracking_error = np.mean(errors)
            metrics.max_tracking_error = np.max(errors)
            metrics.rms_tracking_error = np.sqrt(np.mean(errors**2))
        
        # Motion characteristics
        metrics.total_distance_traveled = self.state.total_distance
        
        if self._velocities:
            speeds = [np.linalg.norm(v) for v in self._velocities]
            metrics.average_speed = np.mean(speeds)
            metrics.max_speed_achieved = np.max(speeds)
            
            # Velocity smoothness (inverse of velocity variation)
            if len(speeds) > 1:
                speed_variance = np.var(speeds)
                metrics.velocity_smoothness = 1.0 / (1.0 + speed_variance)
        
        # Energy analysis
        metrics.total_energy_consumed = self.state.energy_consumed
        if metrics.total_energy_consumed > 0:
            metrics.energy_efficiency = metrics.total_distance_traveled / metrics.total_energy_consumed
        
        # Control performance
        if self._control_inputs:
            control_array = np.array(self._control_inputs)
            left_speeds = control_array[:, 0]
            right_speeds = control_array[:, 1]
            
            # RMS control effort
            control_effort = np.sqrt(left_speeds**2 + right_speeds**2)
            metrics.control_effort_rms = np.sqrt(np.mean(control_effort**2))
            
            # Wheel speed variance
            metrics.wheel_speed_variance = np.var(left_speeds) + np.var(right_speeds)
        
        # Acceleration metrics
        if self._accelerations:
            accel_magnitudes = [np.linalg.norm(a) for a in self._accelerations]
            metrics.acceleration_rms = np.sqrt(np.mean(np.array(accel_magnitudes)**2))
        
        self.metrics = metrics
        return metrics
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get complete current simulation state.
        
        Returns:
            Dictionary containing all simulation state variables
        """
        # Get motion analysis
        slip_analysis = self.motion_model.compute_wheel_slip_constraints(
            self.state.velocity, self.state.angular_velocity)
        
        # Get energy analysis
        speed = np.linalg.norm(self.state.velocity[:2])  # 2D speed
        energy_analysis = self.motion_model.compute_motion_energy(
            speed, self.state.angular_velocity)
        
        return {
            # Core state
            'position': self.state.position.copy(),
            'orientation': self.state.orientation.copy(),
            'velocity': self.state.velocity.copy(),
            'angular_velocity': self.state.angular_velocity,
            
            # Control state
            'wheel_speeds': [self.state.left_wheel_speed, self.state.right_wheel_speed],
            'control_mode': self.state.control_mode,
            
            # Performance metrics
            'trajectory_error': self.state.trajectory_error,
            'total_distance': self.state.total_distance,
            'energy_consumed': self.state.energy_consumed,
            'simulation_time': self.state.simulation_time,
            
            # Analysis
            'slip_analysis': slip_analysis,
            'energy_analysis': energy_analysis,
            
            # Trajectory information (if following)
            'target_position': (self.trajectory_generator.get_position(self.state.simulation_time) 
                              if self.state.control_mode == ControlMode.TRAJECTORY_FOLLOWING 
                              else None),
            'target_velocity': (self.trajectory_generator.get_velocity(self.state.simulation_time)
                              if self.state.control_mode == ControlMode.TRAJECTORY_FOLLOWING
                              else None)
        }
    
    def run_simulation(self, 
                      duration: float, 
                      dt: float = 0.01,
                      progress_callback: Optional[Callable[[float], None]] = None) -> SimulationMetrics:
        """
        Run complete simulation for specified duration.
        
        Args:
            duration: Simulation duration [s]
            dt: Time step [s]
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final simulation metrics
            
        Raises:
            ValueError: If simulation parameters are invalid
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if dt <= 0 or dt > duration:
            raise ValueError(f"Invalid time step {dt} for duration {duration}")
        
        start_time = time.time()
        steps = int(duration / dt)
        
        try:
            for step in range(steps):
                self.update(dt)
                
                # Progress callback
                if progress_callback and step % max(1, steps // 100) == 0:
                    progress = step / steps
                    progress_callback(progress)
                    
        except KeyboardInterrupt:
            print(f"Simulation interrupted at t={self.state.simulation_time:.2f}s")
        
        # Compute final metrics
        final_metrics = self.compute_final_metrics()
        
        # Simulation summary
        real_time = time.time() - start_time
        sim_ratio = self.state.simulation_time / real_time if real_time > 0 else float('inf')
        
        print(f"Simulation completed:")
        print(f"  Duration: {self.state.simulation_time:.2f}s")
        print(f"  Real time: {real_time:.2f}s (ratio: {sim_ratio:.1f}x)")
        print(f"  Distance: {final_metrics.total_distance_traveled:.2f}m")
        print(f"  Energy: {final_metrics.total_energy_consumed:.2f}J")
        
        return final_metrics
    
    def enable_data_logging(self, enabled: bool = True) -> None:
        """Enable or disable data logging for performance analysis."""
        self._enable_logging = enabled
        
        if not enabled:
            # Clear existing data to save memory
            self._trajectory_errors.clear()
            self._velocities.clear()
            self._accelerations.clear()
            self._control_inputs.clear()
            self._energy_history.clear()
    
    def export_simulation_data(self) -> Dict[str, np.ndarray]:
        """
        Export all logged simulation data for external analysis.
        
        Returns:
            Dictionary containing all logged simulation time series
        """
        data = {}
        
        if self._trajectory_errors:
            data['trajectory_errors'] = np.array(self._trajectory_errors)
            
        if self._velocities:
            data['velocities'] = np.array(self._velocities)
            
        if self._accelerations:
            data['accelerations'] = np.array(self._accelerations)
            
        if self._control_inputs:
            data['control_inputs'] = np.array(self._control_inputs)
            
        if self._energy_history:
            data['energy_history'] = np.array(self._energy_history)
            
        return data
    
    def __repr__(self) -> str:
        """String representation of rover simulation."""
        return (f"RoverSimulation(mode={self.state.control_mode.value}, "
                f"t={self.state.simulation_time:.2f}s, "
                f"pos=[{self.state.position[0]:.2f}, {self.state.position[1]:.2f}])")


# Utility functions for simulation analysis

def compare_simulations(simulations: List[RoverSimulation]) -> Dict[str, Any]:
    """
    Compare performance metrics across multiple simulations.
    
    Args:
        simulations: List of completed simulations
        
    Returns:
        Comparative analysis results
    """
    if not simulations:
        return {}
    
    metrics_list = [sim.compute_final_metrics() for sim in simulations]
    
    comparison = {
        'num_simulations': len(simulations),
        'tracking_error': {
            'mean': [m.mean_tracking_error for m in metrics_list],
            'max': [m.max_tracking_error for m in metrics_list],
            'rms': [m.rms_tracking_error for m in metrics_list]
        },
        'energy_efficiency': [m.energy_efficiency for m in metrics_list],
        'total_distance': [m.total_distance_traveled for m in metrics_list],
        'control_effort': [m.control_effort_rms for m in metrics_list]
    }
    
    return comparison


def analyze_trajectory_feasibility(trajectory_gen: TrajectoryGenerator,
                                 vehicle_params: VehicleParameters,
                                 control_params: ControlParameters) -> Dict[str, Any]:
    """
    Analyze trajectory feasibility for given vehicle and control parameters.
    
    Args:
        trajectory_gen: Trajectory generator
        vehicle_params: Vehicle physical parameters
        control_params: Control system parameters
        
    Returns:
        Feasibility analysis results
    """
    analytics = trajectory_gen.analyze_trajectory()
    
    # Check velocity feasibility
    max_vehicle_speed = control_params.max_wheel_speed * vehicle_params.wheel_radius
    velocity_feasible = analytics.max_velocity <= max_vehicle_speed
    
    # Check acceleration feasibility
    acceleration_feasible = analytics.max_acceleration <= control_params.max_linear_acceleration
    
    # Check curvature feasibility (minimum turning radius)
    min_turn_radius = 1.0 / analytics.max_curvature if analytics.max_curvature > 0 else float('inf')
    curvature_feasible = min_turn_radius >= vehicle_params.wheel_base / 2.0
    
    return {
        'overall_feasible': velocity_feasible and acceleration_feasible and curvature_feasible,
        'velocity_feasible': velocity_feasible,
        'acceleration_feasible': acceleration_feasible,
        'curvature_feasible': curvature_feasible,
        'max_trajectory_velocity': analytics.max_velocity,
        'max_vehicle_velocity': max_vehicle_speed,
        'max_trajectory_acceleration': analytics.max_acceleration,
        'max_control_acceleration': control_params.max_linear_acceleration,
        'min_turning_radius': min_turn_radius,
        'vehicle_wheelbase': vehicle_params.wheel_base
    }