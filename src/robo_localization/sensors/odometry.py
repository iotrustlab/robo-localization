"""
Wheel odometry sensor implementation for differential drive robots with comprehensive
error modeling and validation.

This module implements wheel odometry based on differential drive kinematics,
incorporating realistic error sources including encoder noise, wheel slip,
and systematic errors. The implementation follows established robotics principles
for dead reckoning navigation.

Mathematical Model:
    The differential drive kinematic model relates wheel velocities to robot motion:
    
    v = (r/2) * (ω_left + ω_right)        # linear velocity
    ω = (r/L) * (ω_right - ω_left)        # angular velocity
    
    Where:
    - r: wheel radius
    - L: wheelbase (distance between wheels)
    - ω_left, ω_right: left and right wheel angular velocities
    
    Position integration:
    Δx = v * cos(θ) * Δt
    Δy = v * sin(θ) * Δt  
    Δθ = ω * Δt

References:
    - Siegwart, R., Nourbakhsh, I. R. (2004). Introduction to Autonomous Mobile Robots
    - Thrun, S., Burgard, W., Fox, D. (2005). Probabilistic Robotics
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .health import SensorHealth


@dataclass
class OdometryMeasurement:
    """
    Odometry measurement containing pose change and uncertainty estimates.
    
    Attributes:
        delta_x: Change in x-position (meters)
        delta_y: Change in y-position (meters) 
        delta_theta: Change in orientation (radians)
        timestamp: Measurement timestamp
        uncertainty: Estimated measurement uncertainty
        wheel_speeds: Raw wheel speed measurements (rad/s)
    """
    delta_x: float
    delta_y: float
    delta_theta: float
    timestamp: float
    uncertainty: Dict[str, float]
    wheel_speeds: Tuple[float, float]


class WheelOdometry:
    """
    Wheel odometry sensor implementation for differential drive robots.
    
    This class simulates realistic wheel odometry with comprehensive error modeling
    including encoder noise, wheel slip, systematic errors, and calibration
    uncertainties. The implementation provides both raw measurements and
    uncertainty estimates for sensor fusion applications.
    
    Error Sources Modeled:
    - Encoder quantization noise
    - Wheel slip effects
    - Wheelbase calibration errors
    - Wheel radius calibration errors
    - Temperature-dependent systematic errors
    - Integration drift over time
    
    Attributes:
        wheelbase: Distance between left and right wheels (meters)
        wheel_radius: Effective wheel radius (meters)
        encoder_resolution: Encoder pulses per revolution
        encoder_noise_std: Standard deviation of encoder noise (rad/s)
        slip_factor: Wheel slip coefficient (0-1, where 0 is no slip)
        systematic_error_std: Standard deviation of systematic errors
        health: Sensor health monitoring instance
    """
    
    def __init__(
        self,
        wheelbase: float,
        wheel_radius: float,
        encoder_resolution: int = 1024,
        encoder_noise_std: float = 0.05,
        slip_factor: float = 0.0,
        systematic_error_std: float = 0.01,
        wheelbase_uncertainty: float = 0.005,
        radius_uncertainty: float = 0.002
    ):
        """
        Initialize wheel odometry sensor with specified parameters.
        
        Args:
            wheelbase: Distance between wheels in meters (typical: 0.3-1.0m)
            wheel_radius: Wheel radius in meters (typical: 0.05-0.2m)
            encoder_resolution: Encoder pulses per revolution (typical: 1024-4096)
            encoder_noise_std: Encoder noise standard deviation (rad/s)
            slip_factor: Wheel slip coefficient (0=no slip, 1=complete slip)
            systematic_error_std: Systematic error standard deviation
            wheelbase_uncertainty: Wheelbase calibration uncertainty (meters)
            radius_uncertainty: Wheel radius calibration uncertainty (meters)
            
        Raises:
            ValueError: If input parameters are invalid or out of physical range
        """
        # Input validation
        self._validate_parameters(
            wheelbase, wheel_radius, encoder_resolution, encoder_noise_std,
            slip_factor, systematic_error_std, wheelbase_uncertainty, radius_uncertainty
        )
        
        # Core kinematic parameters
        self.wheelbase = wheelbase
        self.wheel_radius = wheel_radius
        self.encoder_resolution = encoder_resolution
        
        # Noise and error parameters
        self.encoder_noise_std = encoder_noise_std
        self.slip_factor = slip_factor
        self.systematic_error_std = systematic_error_std
        
        # Calibration uncertainties
        self.wheelbase_uncertainty = wheelbase_uncertainty
        self.radius_uncertainty = radius_uncertainty
        
        # Effective parameters with calibration errors
        self._effective_wheelbase = wheelbase + np.random.normal(0, wheelbase_uncertainty)
        self._effective_radius = wheel_radius + np.random.normal(0, radius_uncertainty)
        
        # Systematic bias terms (simulate manufacturing tolerances)
        self._left_wheel_scale_factor = 1.0 + np.random.normal(0, 0.01)
        self._right_wheel_scale_factor = 1.0 + np.random.normal(0, 0.01)
        
        # Health monitoring
        self.health = SensorHealth()
        
        # State tracking
        self._total_distance = 0.0
        self._total_rotations = 0.0
        self._initialization_time = time.time()
        
    def _validate_parameters(
        self,
        wheelbase: float,
        wheel_radius: float,
        encoder_resolution: int,
        encoder_noise_std: float,
        slip_factor: float,
        systematic_error_std: float,
        wheelbase_uncertainty: float,
        radius_uncertainty: float
    ) -> None:
        """
        Validate input parameters for physical plausibility.
        
        Args:
            All constructor parameters
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if wheelbase <= 0 or wheelbase > 5.0:
            raise ValueError(f"Wheelbase must be positive and <= 5.0m, got {wheelbase}")
            
        if wheel_radius <= 0 or wheel_radius > 1.0:
            raise ValueError(f"Wheel radius must be positive and <= 1.0m, got {wheel_radius}")
            
        if encoder_resolution < 1 or encoder_resolution > 65536:
            raise ValueError(f"Encoder resolution must be 1-65536, got {encoder_resolution}")
            
        if encoder_noise_std < 0 or encoder_noise_std > 1.0:
            raise ValueError(f"Encoder noise std must be 0-1.0, got {encoder_noise_std}")
            
        if slip_factor < 0 or slip_factor > 1.0:
            raise ValueError(f"Slip factor must be 0-1.0, got {slip_factor}")
            
        if systematic_error_std < 0 or systematic_error_std > 0.1:
            raise ValueError(f"Systematic error std must be 0-0.1, got {systematic_error_std}")
            
        if wheelbase_uncertainty < 0 or wheelbase_uncertainty > wheelbase * 0.1:
            raise ValueError(f"Wheelbase uncertainty too large: {wheelbase_uncertainty}")
            
        if radius_uncertainty < 0 or radius_uncertainty > wheel_radius * 0.1:
            raise ValueError(f"Radius uncertainty too large: {radius_uncertainty}")
    
    def compute_delta_pose(
        self,
        left_wheel_speed: float,
        right_wheel_speed: float,
        dt: float
    ) -> Optional[OdometryMeasurement]:
        """
        Compute pose change from wheel speeds using differential drive kinematics.
        
        This method implements the complete odometry computation chain including:
        1. Encoder noise simulation
        2. Wheel slip modeling
        3. Systematic error injection
        4. Kinematic computation
        5. Uncertainty estimation
        
        Args:
            left_wheel_speed: Left wheel angular velocity (rad/s)
            right_wheel_speed: Right wheel angular velocity (rad/s)
            dt: Time step (seconds)
            
        Returns:
            OdometryMeasurement with pose change and uncertainty, or None if failed
            
        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Input validation
            if dt <= 0 or dt > 1.0:
                raise ValueError(f"Time step must be positive and <= 1.0s, got {dt}")
            
            if abs(left_wheel_speed) > 100.0 or abs(right_wheel_speed) > 100.0:
                raise ValueError("Wheel speeds exceed physical limits (>100 rad/s)")
            
            timestamp = time.time()
            
            # Add encoder quantization noise
            encoder_noise_left = np.random.normal(0, self.encoder_noise_std)
            encoder_noise_right = np.random.normal(0, self.encoder_noise_std)
            
            # Apply systematic scaling errors
            left_speed_measured = (left_wheel_speed + encoder_noise_left) * self._left_wheel_scale_factor
            right_speed_measured = (right_wheel_speed + encoder_noise_right) * self._right_wheel_scale_factor
            
            # Compute kinematic velocities using effective parameters
            linear_velocity = (self._effective_radius / 2.0) * (left_speed_measured + right_speed_measured)
            angular_velocity = (self._effective_radius / self._effective_wheelbase) * (right_speed_measured - left_speed_measured)
            
            # Apply slip effects (reduces linear motion, increases uncertainty)
            slip_noise = np.random.normal(0, self.slip_factor * abs(linear_velocity))
            linear_velocity_corrected = linear_velocity * (1.0 - self.slip_factor) + slip_noise
            
            # Add systematic errors (temperature effects, wear, etc.)
            systematic_linear_error = np.random.normal(0, self.systematic_error_std)
            systematic_angular_error = np.random.normal(0, self.systematic_error_std * 0.1)
            
            linear_velocity_final = linear_velocity_corrected + systematic_linear_error
            angular_velocity_final = angular_velocity + systematic_angular_error
            
            # Compute pose change in robot frame
            delta_x = linear_velocity_final * dt
            delta_y = 0.0  # No lateral motion in differential drive
            delta_theta = angular_velocity_final * dt
            
            # Update state tracking
            self._total_distance += abs(delta_x)
            self._total_rotations += abs(delta_theta)
            
            # Compute measurement uncertainty
            uncertainty = self._compute_uncertainty(
                linear_velocity_final, angular_velocity_final, dt
            )
            
            # Create measurement object
            measurement = OdometryMeasurement(
                delta_x=delta_x,
                delta_y=delta_y,
                delta_theta=delta_theta,
                timestamp=timestamp,
                uncertainty=uncertainty,
                wheel_speeds=(left_wheel_speed, right_wheel_speed)
            )
            
            self.health.record_success()
            return measurement
            
        except Exception as e:
            self.health.record_failure()
            return None
    
    def _compute_uncertainty(
        self,
        linear_velocity: float,
        angular_velocity: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Compute measurement uncertainty based on error propagation theory.
        
        The uncertainty model accounts for:
        - Encoder noise propagation
        - Calibration parameter uncertainties
        - Slip-induced errors
        - Integration over time
        
        Args:
            linear_velocity: Computed linear velocity (m/s)
            angular_velocity: Computed angular velocity (rad/s)
            dt: Time step (seconds)
            
        Returns:
            Dictionary containing uncertainty estimates for each pose component
        """
        # Base encoder uncertainty
        encoder_uncertainty = self.encoder_noise_std * self._effective_radius * dt
        
        # Calibration parameter uncertainties
        radius_contribution = abs(linear_velocity) * (self.radius_uncertainty / self._effective_radius) * dt
        wheelbase_contribution = abs(angular_velocity) * (self.wheelbase_uncertainty / self._effective_wheelbase) * dt
        
        # Slip-induced uncertainty
        slip_contribution = self.slip_factor * abs(linear_velocity) * dt
        
        # Systematic error contribution
        systematic_contribution = self.systematic_error_std * dt
        
        # Total uncertainty (RSS combination)
        position_uncertainty = np.sqrt(
            encoder_uncertainty**2 +
            radius_contribution**2 +
            slip_contribution**2 +
            systematic_contribution**2
        )
        
        orientation_uncertainty = np.sqrt(
            (encoder_uncertainty / self._effective_wheelbase)**2 +
            wheelbase_contribution**2 +
            (systematic_contribution * 0.1)**2
        )
        
        # Scale uncertainty with accumulated distance (drift effects)
        drift_factor = 1.0 + 0.01 * (self._total_distance / 100.0)  # 1% per 100m
        
        return {
            'delta_x': position_uncertainty * drift_factor,
            'delta_y': position_uncertainty * 0.1,  # Minimal lateral uncertainty
            'delta_theta': orientation_uncertainty * drift_factor
        }
    
    def get_calibration_parameters(self) -> Dict[str, float]:
        """
        Get current calibration parameters including effective values.
        
        Returns:
            Dictionary containing all calibration parameters
        """
        return {
            'nominal_wheelbase': self.wheelbase,
            'nominal_radius': self.wheel_radius,
            'effective_wheelbase': self._effective_wheelbase,
            'effective_radius': self._effective_radius,
            'left_scale_factor': self._left_wheel_scale_factor,
            'right_scale_factor': self._right_wheel_scale_factor,
            'encoder_resolution': self.encoder_resolution,
            'slip_factor': self.slip_factor
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for sensor evaluation.
        
        Returns:
            Dictionary containing performance statistics
        """
        current_time = time.time()
        uptime = current_time - self._initialization_time
        
        return {
            'total_distance': self._total_distance,
            'total_rotations': self._total_rotations,
            'uptime_seconds': uptime,
            'average_speed': self._total_distance / max(uptime, 1e-6),
            'reliability': self.health.reliability,
            'failure_rate': self.health.failure_count / max(uptime, 1e-6)
        }
    
    def reset_calibration(
        self,
        wheelbase: Optional[float] = None,
        wheel_radius: Optional[float] = None
    ) -> None:
        """
        Reset calibration parameters with new values.
        
        Args:
            wheelbase: New wheelbase value (meters)
            wheel_radius: New wheel radius value (meters)
        """
        if wheelbase is not None:
            if wheelbase <= 0 or wheelbase > 5.0:
                raise ValueError(f"Invalid wheelbase: {wheelbase}")
            self.wheelbase = wheelbase
            self._effective_wheelbase = wheelbase + np.random.normal(0, self.wheelbase_uncertainty)
        
        if wheel_radius is not None:
            if wheel_radius <= 0 or wheel_radius > 1.0:
                raise ValueError(f"Invalid wheel radius: {wheel_radius}")
            self.wheel_radius = wheel_radius
            self._effective_radius = wheel_radius + np.random.normal(0, self.radius_uncertainty)