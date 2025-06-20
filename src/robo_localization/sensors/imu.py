"""
IMU sensor simulation with bias drift and multiple failure modes.

This module implements an Inertial Measurement Unit (IMU) model with realistic
error characteristics including bias drift, scale factor errors, noise, and
various failure modes commonly observed in MEMS sensors.

IMU Measurement Model:
    z_accel = (1 + s_a) * (a_true + b_a) + n_a
    z_gyro = (1 + s_g) * (w_true + b_g) + n_g
    
    where:
    - a_true, w_true: True acceleration and angular velocity
    - b_a, b_g: Time-varying bias vectors
    - s_a, s_g: Scale factor errors (typically small)
    - n_a, n_g: Zero-mean Gaussian noise

Bias Drift Model:
    b(t+1) = b(t) + drift_rate * dt + w_drift
    where w_drift ~ N(0, σ_drift²)
    
Typical MEMS IMU Characteristics:
    - Accelerometer noise: 0.01-0.1 m/s² (1σ)
    - Gyroscope noise: 0.01-0.1 rad/s (1σ)
    - Bias stability: 0.1-10 mg (accelerometer), 1-100 deg/hr (gyroscope)
    - Temperature sensitivity: bias varies with thermal conditions

Failure Modes:
    - Stuck values: sensor output becomes constant
    - Increased noise: degraded measurement quality
    - Bias jumps: sudden changes in systematic error
    - Scale factor drift: gradual change in sensitivity
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from .health import SensorHealth


class IMUSensor:
    """
    IMU sensor with bias drift and multiple failure modes.
    
    This class simulates a 6-DOF IMU (3-axis accelerometer + 3-axis gyroscope)
    with realistic error models including bias drift, noise, and various failure modes.
    
    The bias drift model captures the slow variation of sensor offsets due to
    temperature changes, aging, and other environmental factors.
    
    Attributes:
        accel_noise_std: Accelerometer noise standard deviation (m/s²)
        gyro_noise_std: Gyroscope noise standard deviation (rad/s)
        bias_drift_rate: Rate of bias drift per time step
        sensor_id: Unique identifier for this IMU
        accel_bias: Current accelerometer bias vector [x, y, z]
        gyro_bias: Current gyroscope bias vector [x, y, z]
        health: SensorHealth object tracking reliability
    """
    
    def __init__(self, accel_noise_std: float = 0.1, gyro_noise_std: float = 0.05,
                 bias_drift_rate: float = 0.001, sensor_id: int = 1):
        """
        Initialize IMU sensor with specified noise characteristics.
        
        Args:
            accel_noise_std: Accelerometer noise standard deviation (m/s²)
            gyro_noise_std: Gyroscope noise standard deviation (rad/s)
            bias_drift_rate: Bias drift rate per time step (sensor units/s)
            sensor_id: Unique identifier for this IMU
        """
        if accel_noise_std <= 0 or gyro_noise_std <= 0:
            raise ValueError("IMU noise standard deviations must be positive")
        if bias_drift_rate < 0:
            raise ValueError("Bias drift rate must be non-negative")
            
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_rate = bias_drift_rate
        self.sensor_id = sensor_id
        
        # Initialize biases with small random values
        # Accelerometer bias typically 0.01-0.05 m/s² (10-50 mg)
        self.accel_bias = np.random.normal(0, 0.02, 3)
        # Gyroscope bias typically 0.01-0.1 rad/s (0.5-5 deg/s)
        self.gyro_bias = np.random.normal(0, 0.01, 3)
        
        # Health monitoring
        self.health = SensorHealth()
        
        # Failure mode state
        self.failure_mode = None
        self.stuck_values = None
        self._original_noise_std = (accel_noise_std, gyro_noise_std)
        
        # Scale factor errors (typically < 1%)
        self.accel_scale_error = np.random.normal(0, 0.001, 3)  # 0.1% scale error
        self.gyro_scale_error = np.random.normal(0, 0.001, 3)
        
        # Temperature effects (simplified model)
        self._temperature_offset = 25.0  # Celsius
        self._temp_coefficient_accel = np.random.normal(0, 0.0001, 3)  # bias/°C
        self._temp_coefficient_gyro = np.random.normal(0, 0.0001, 3)
        
    def update_bias_drift(self, dt: float = 0.1, temperature: Optional[float] = None) -> None:
        """
        Update sensor biases due to drift and temperature effects.
        
        Args:
            dt: Time step in seconds
            temperature: Current temperature in Celsius (optional)
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")
            
        # Random walk bias drift
        drift_accel = np.random.normal(0, self.bias_drift_rate * dt, 3)
        drift_gyro = np.random.normal(0, self.bias_drift_rate * dt, 3)
        
        self.accel_bias += drift_accel
        self.gyro_bias += drift_gyro
        
        # Temperature effects on bias
        if temperature is not None:
            temp_delta = temperature - self._temperature_offset
            self.accel_bias += self._temp_coefficient_accel * temp_delta * dt
            self.gyro_bias += self._temp_coefficient_gyro * temp_delta * dt
            
        # Limit bias drift to reasonable ranges
        self.accel_bias = np.clip(self.accel_bias, -0.5, 0.5)  # ±0.5 m/s²
        self.gyro_bias = np.clip(self.gyro_bias, -0.2, 0.2)   # ±0.2 rad/s
        
    def simulate_failure(self, mode: str = 'stuck') -> None:
        """
        Simulate different IMU failure modes.
        
        Args:
            mode: Failure mode - 'stuck', 'noisy', 'bias_jump', 'scale_drift'
        """
        valid_modes = ['stuck', 'noisy', 'bias_jump', 'scale_drift']
        if mode not in valid_modes:
            raise ValueError(f"Invalid failure mode. Must be one of {valid_modes}")
            
        self.failure_mode = mode
        
        if mode == 'stuck':
            # Record current values as stuck values
            self.stuck_values = {
                'acceleration': np.array([0.0, 0.0, -9.81]),  # Gravity only
                'angular_velocity': np.array([0.0, 0.0, 0.0])  # No rotation
            }
            
        elif mode == 'noisy':
            # Significantly increase noise levels
            self.accel_noise_std *= 10
            self.gyro_noise_std *= 10
            
        elif mode == 'bias_jump':
            # Sudden bias change (e.g., due to shock or temperature change)
            self.accel_bias += np.random.normal(0, 0.1, 3)
            self.gyro_bias += np.random.normal(0, 0.05, 3)
            
        elif mode == 'scale_drift':
            # Gradual scale factor change
            self.accel_scale_error += np.random.normal(0, 0.01, 3)
            self.gyro_scale_error += np.random.normal(0, 0.01, 3)
            
    def clear_failure(self) -> None:
        """
        Clear current failure mode and restore normal operation.
        """
        self.failure_mode = None
        self.stuck_values = None
        
        # Restore original noise levels
        self.accel_noise_std, self.gyro_noise_std = self._original_noise_std
        
    def get_measurement(self, true_acceleration: np.ndarray, 
                      true_angular_velocity: np.ndarray,
                      dt: float = 0.1) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate IMU measurement with noise, bias, and potential failures.
        
        Args:
            true_acceleration: True acceleration [x, y, z] in m/s²
            true_angular_velocity: True angular velocity [x, y, z] in rad/s
            dt: Time step for bias update (seconds)
            
        Returns:
            Dictionary with 'acceleration' and 'angular_velocity' measurements,
            or None if complete sensor failure
            
        Raises:
            ValueError: If input vectors are not 3D
        """
        if len(true_acceleration) != 3 or len(true_angular_velocity) != 3:
            raise ValueError("IMU requires 3D acceleration and angular velocity inputs")
            
        # Update bias drift
        self.update_bias_drift(dt)
        
        # Handle failure modes
        if self.failure_mode == 'stuck' and self.stuck_values is not None:
            self.health.record_failure()
            return self.stuck_values
            
        # Apply systematic biases
        accel_with_bias = true_acceleration + self.accel_bias
        gyro_with_bias = true_angular_velocity + self.gyro_bias
        
        # Apply scale factor errors
        accel_with_scale = accel_with_bias * (1 + self.accel_scale_error)
        gyro_with_scale = gyro_with_bias * (1 + self.gyro_scale_error)
        
        # Add measurement noise
        accel_noise = np.random.normal(0, self.accel_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        measurement = {
            'acceleration': accel_with_scale + accel_noise,
            'angular_velocity': gyro_with_scale + gyro_noise
        }
        
        # Check for measurement validity (simple sanity check)
        if self._is_measurement_valid(measurement):
            self.health.record_success()
            return measurement
        else:
            self.health.record_failure()
            return None
            
    def _is_measurement_valid(self, measurement: Dict[str, np.ndarray]) -> bool:
        """
        Validate measurement for reasonable physical values.
        
        Args:
            measurement: IMU measurement dictionary
            
        Returns:
            True if measurement passes validity checks
        """
        accel = measurement['acceleration']
        gyro = measurement['angular_velocity']
        
        # Check for reasonable acceleration magnitude (including gravity)
        accel_mag = np.linalg.norm(accel)
        if accel_mag > 50.0:  # 5g seems reasonable for rover
            return False
            
        # Check for reasonable angular velocity
        gyro_mag = np.linalg.norm(gyro)
        if gyro_mag > 10.0:  # 10 rad/s maximum
            return False
            
        # Check for NaN or infinite values
        if np.any(~np.isfinite(accel)) or np.any(~np.isfinite(gyro)):
            return False
            
        return True
        
    def get_measurement_covariance(self) -> Dict[str, np.ndarray]:
        """
        Get measurement noise covariance matrices.
        
        Returns:
            Dictionary with accelerometer and gyroscope covariance matrices
        """
        accel_cov = np.eye(3) * (self.accel_noise_std ** 2)
        gyro_cov = np.eye(3) * (self.gyro_noise_std ** 2)
        
        return {
            'acceleration': accel_cov,
            'angular_velocity': gyro_cov
        }
        
    def calibrate_bias(self, accel_measurements: np.ndarray, 
                      gyro_measurements: np.ndarray,
                      expected_accel: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate sensor biases using static measurements.
        
        Args:
            accel_measurements: Array of accelerometer measurements (N x 3)
            gyro_measurements: Array of gyroscope measurements (N x 3)
            expected_accel: Expected acceleration during calibration (default: gravity)
            
        Returns:
            Tuple of (estimated_accel_bias, estimated_gyro_bias)
        """
        if expected_accel is None:
            expected_accel = np.array([0.0, 0.0, -9.81])  # Gravity in NED frame
            
        # Estimate biases as mean of measurements minus expected values
        accel_bias_est = np.mean(accel_measurements, axis=0) - expected_accel
        gyro_bias_est = np.mean(gyro_measurements, axis=0)  # Expect zero angular velocity
        
        # Update internal bias estimates
        self.accel_bias = accel_bias_est
        self.gyro_bias = gyro_bias_est
        
        return accel_bias_est, gyro_bias_est
        
    def get_uncertainty_estimate(self) -> float:
        """
        Get current measurement uncertainty estimate.
        
        Returns:
            Scalar uncertainty estimate based on noise characteristics and health
        """
        # Combined uncertainty from accelerometer and gyroscope
        accel_uncertainty = self.accel_noise_std + np.linalg.norm(self.accel_bias) * 0.1
        gyro_uncertainty = self.gyro_noise_std + np.linalg.norm(self.gyro_bias) * 0.1
        
        # Base uncertainty is the RMS of both components
        base_uncertainty = np.sqrt(accel_uncertainty**2 + gyro_uncertainty**2)
        
        # Scale by reliability (lower reliability = higher uncertainty)
        reliability_factor = 1.0 / max(0.1, self.health.reliability)
        
        # Additional uncertainty from failure modes
        failure_uncertainty = 0.0
        if self.failure_mode == 'noisy':
            failure_uncertainty = base_uncertainty * 5.0
        elif self.failure_mode == 'stuck':
            failure_uncertainty = base_uncertainty * 10.0
        elif self.failure_mode == 'bias_jump':
            failure_uncertainty = base_uncertainty * 2.0
        
        return base_uncertainty * reliability_factor + failure_uncertainty
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """
        Get comprehensive sensor information for monitoring.
        
        Returns:
            Dictionary containing sensor configuration and status
        """
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': 'IMU',
            'accel_noise_std': self.accel_noise_std,
            'gyro_noise_std': self.gyro_noise_std,
            'bias_drift_rate': self.bias_drift_rate,
            'accel_bias': self.accel_bias.tolist(),
            'gyro_bias': self.gyro_bias.tolist(),
            'failure_mode': self.failure_mode,
            'health': self.health.get_health_summary(),
            'covariance': {
                'acceleration': self.get_measurement_covariance()['acceleration'].tolist(),
                'angular_velocity': self.get_measurement_covariance()['angular_velocity'].tolist()
            }
        }


# Maintain backward compatibility with original naming  
IMU = IMUSensor
