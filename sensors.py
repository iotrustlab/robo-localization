"""
Sensor simulation components for 3D rover localization.

This module implements realistic sensor models including:
- Multi-GPS units with different noise characteristics and dropout patterns
- Multi-IMU sensors with bias drift and failure modes
- Wheel odometry with slip modeling
- Sensor health monitoring and failure detection
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    """Sensor type enumeration."""
    GPS = "gps"
    IMU = "imu"
    ODOMETRY = "odometry"


@dataclass
class SensorHealth:
    """Tracks sensor health, reliability, and failure recovery"""
    
    def __init__(self):
        self.is_operational = True
        self.reliability = 1.0
        self.failure_count = 0
        self.recovery_count = 0
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
    def record_failure(self):
        """Record a sensor failure event"""
        self.failure_count += 1
        self.consecutive_failures += 1
        
        # Reduce reliability based on failures - be more aggressive
        reliability_reduction = min(0.15 * self.consecutive_failures, 0.9)
        self.reliability = max(0.0, 1.0 - reliability_reduction)
        
        # Disable sensor after too many consecutive failures
        if self.consecutive_failures >= 5:
            self.is_operational = False
            
    def record_success(self):
        """Record a successful sensor measurement"""
        self.recovery_count += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        # Improve reliability on successful measurements
        self.reliability = min(1.0, self.reliability + 0.05)
        
        # Re-enable sensor if it was disabled but now working
        if not self.is_operational and self.reliability > 0.5:
            self.is_operational = True


class GPS:
    """GPS sensor simulation with configurable noise, bias, and dropout"""
    
    def __init__(self, noise_std=2.0, dropout_prob=0.1, sensor_id=1):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.sensor_id = sensor_id
        
        # Random bias for this GPS unit
        self.position_bias = np.random.normal(0, 1.0, 3)
        
        # Health monitoring
        self.health = SensorHealth()
        
    def get_measurement(self, true_position):
        """Get GPS position measurement with noise and potential dropout"""
        # Check for dropout
        if np.random.random() < self.dropout_prob:
            self.health.record_failure()
            return None
            
        # Add bias and noise
        noise = np.random.normal(0, self.noise_std, 3)
        measurement = true_position + self.position_bias + noise
        
        self.health.record_success()
        return measurement


class IMU:
    """IMU sensor simulation with bias drift and multiple failure modes"""
    
    def __init__(self, accel_noise_std=0.1, gyro_noise_std=0.05, bias_drift_rate=0.001, sensor_id=1):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_rate = bias_drift_rate
        self.sensor_id = sensor_id
        
        # Initial bias values
        self.accel_bias = np.random.normal(0, 0.02, 3)
        self.gyro_bias = np.random.normal(0, 0.01, 3)
        
        # Health monitoring
        self.health = SensorHealth()
        
        # Failure mode state
        self.failure_mode = None
        self.stuck_values = None
        
    def update_bias_drift(self, dt=0.1):
        """Update sensor bias due to drift over time"""
        drift_accel = np.random.normal(0, self.bias_drift_rate * dt, 3)
        drift_gyro = np.random.normal(0, self.bias_drift_rate * dt, 3)
        
        self.accel_bias += drift_accel
        self.gyro_bias += drift_gyro
        
    def simulate_failure(self, mode='stuck'):
        """Simulate different failure modes"""
        self.failure_mode = mode
        if mode == 'stuck':
            # Record current values as stuck values
            self.stuck_values = {
                'acceleration': np.array([0.0, 0.0, -9.81]),
                'angular_velocity': np.array([0.0, 0.0, 0.0])
            }
        elif mode == 'noisy':
            # Increase noise significantly
            self.accel_noise_std *= 10
            self.gyro_noise_std *= 10
        elif mode == 'dropout':
            # Increase dropout probability
            self.dropout_prob = 0.8
        
    def get_measurement(self, true_acceleration, true_angular_velocity):
        """Get IMU measurement with noise, bias, and potential failures"""
        # Update bias drift
        self.update_bias_drift()
        
        # Handle failure modes
        if self.failure_mode == 'stuck' and self.stuck_values is not None:
            self.health.record_failure()
            return self.stuck_values
            
        if self.failure_mode == 'dropout':
            if np.random.random() < 0.8:  # High dropout probability
                self.health.record_failure()
                return None
                
        # Add bias
        accel_with_bias = true_acceleration + self.accel_bias
        gyro_with_bias = true_angular_velocity + self.gyro_bias
        
        # Add noise
        accel_noise = np.random.normal(0, self.accel_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        measurement = {
            'acceleration': accel_with_bias + accel_noise,
            'angular_velocity': gyro_with_bias + gyro_noise
        }
        
        self.health.record_success()
        return measurement


class WheelOdometry:
    """Wheel odometry sensor with differential drive kinematics"""
    
    def __init__(self, wheel_base, wheel_radius, encoder_noise_std=0.05, slip_factor=0.0):
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.encoder_noise_std = encoder_noise_std
        self.slip_factor = slip_factor
        
        # Health monitoring
        self.health = SensorHealth()
        
    def compute_delta_pose(self, left_wheel_speed, right_wheel_speed, dt):
        """Compute change in pose from wheel speeds using differential drive kinematics"""
        # Add encoder noise
        left_noise = np.random.normal(0, self.encoder_noise_std)
        right_noise = np.random.normal(0, self.encoder_noise_std)
        
        left_speed_noisy = left_wheel_speed + left_noise
        right_speed_noisy = right_wheel_speed + right_noise
        
        # Compute linear and angular velocities
        linear_velocity = self.wheel_radius * (left_speed_noisy + right_speed_noisy) / 2
        angular_velocity = self.wheel_radius * (left_speed_noisy - right_speed_noisy) / self.wheel_base
        
        # Apply slip factor (reduces reported motion)
        linear_velocity *= (1 - self.slip_factor)
        
        # Add additional noise to motion computation (simulating integration errors)
        motion_noise = np.random.normal(0, self.encoder_noise_std * 0.1)
        linear_velocity += motion_noise
        
        # Compute pose change
        delta_pose = {
            'dx': linear_velocity * dt,
            'dy': 0.0,  # No lateral motion in body frame
            'dtheta': angular_velocity * dt
        }
        
        self.health.record_success()
        return delta_pose


class SensorFusionManager:
    """Manages multiple sensors and coordinates measurements"""
    
    def __init__(self):
        # Create redundant sensors with different characteristics
        self.gps_sensors = [
            GPS(noise_std=2.0, dropout_prob=0.1, sensor_id=1),
            GPS(noise_std=3.0, dropout_prob=0.15, sensor_id=2)
        ]
        
        self.imu_sensors = [
            IMU(accel_noise_std=0.1, gyro_noise_std=0.05, bias_drift_rate=0.001, sensor_id=1),
            IMU(accel_noise_std=0.15, gyro_noise_std=0.08, bias_drift_rate=0.002, sensor_id=2)
        ]
        
        self.wheel_odometry = WheelOdometry(wheel_base=0.6, wheel_radius=0.15)
        
        # Failure simulation state
        self.gps_constellation_failure = False
        self.failure_start_time = None
        
    def get_all_measurements(self, position, acceleration, angular_velocity, wheel_speeds):
        """Collect measurements from all sensors"""
        measurements = {
            'gps': [],
            'imu': [],
            'odometry': None
        }
        
        # GPS measurements
        for gps in self.gps_sensors:
            if not self.gps_constellation_failure:
                gps_measurement = gps.get_measurement(position)
                if gps_measurement is not None:
                    measurements['gps'].append({
                        'position': gps_measurement,
                        'sensor_id': gps.sensor_id
                    })
                    
        # IMU measurements
        for imu in self.imu_sensors:
            imu_measurement = imu.get_measurement(acceleration, angular_velocity)
            if imu_measurement is not None:
                measurements['imu'].append({
                    'measurement': imu_measurement,
                    'sensor_id': imu.sensor_id
                })
                
        # Wheel odometry
        left_speed, right_speed = wheel_speeds
        odometry_measurement = self.wheel_odometry.compute_delta_pose(left_speed, right_speed, 0.1)
        measurements['odometry'] = odometry_measurement
        
        return measurements
        
    def simulate_gps_constellation_failure(self, duration=5.0):
        """Simulate GPS constellation dropout affecting all GPS sensors"""
        self.gps_constellation_failure = True
        self.failure_start_time = time.time()
        
        # This would be called periodically to check if failure should end
        
    def simulate_coordinated_imu_failure(self):
        """Simulate failure affecting multiple IMU sensors"""
        for imu in self.imu_sensors:
            if np.random.random() < 0.3:  # 30% chance each IMU fails
                failure_modes = ['stuck', 'noisy', 'dropout']
                mode = np.random.choice(failure_modes)
                imu.simulate_failure(mode)
                
    def get_sensor_health(self):
        """Get health status of all sensors"""
        health_status = {
            'gps': [],
            'imu': [],
            'odometry': None
        }
        
        for gps in self.gps_sensors:
            health_status['gps'].append({
                'operational': gps.health.is_operational,
                'reliability': gps.health.reliability,
                'failure_count': gps.health.failure_count,
                'sensor_id': gps.sensor_id
            })
            
        for imu in self.imu_sensors:
            health_status['imu'].append({
                'operational': imu.health.is_operational,
                'reliability': imu.health.reliability,
                'failure_count': imu.health.failure_count,
                'sensor_id': imu.sensor_id
            })
            
        health_status['odometry'] = {
            'operational': self.wheel_odometry.health.is_operational,
            'reliability': self.wheel_odometry.health.reliability,
            'failure_count': self.wheel_odometry.health.failure_count
        }
        
        return health_status 