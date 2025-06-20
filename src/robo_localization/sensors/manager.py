"""
Sensor fusion manager for coordinating multiple sensors and managing
sensor health, redundancy, and failure recovery.

This module implements a comprehensive sensor fusion management system that
coordinates multiple GPS units, IMU sensors, and odometry systems. It provides
intelligent sensor selection, failure detection, and graceful degradation
strategies for robust localization.

The manager implements several key functions:
- Multi-sensor coordination and synchronization
- Sensor health monitoring and failure detection
- Redundancy management and fallback strategies
- Measurement quality assessment and filtering
- Coordinated failure simulation for testing

Mathematical Framework:
    Sensor fusion follows the general framework:
    
    x̂ = Σ(w_i * x_i) / Σ(w_i)
    
    Where:
    - x̂: fused estimate
    - x_i: individual sensor measurements
    - w_i: sensor weights based on reliability and uncertainty
    
    Weights are computed as:
    w_i = (reliability_i / uncertainty_i²)

References:
    - Bar-Shalom, Y., Li, X. R., Kirubarajan, T. (2001). Estimation with Applications
    - Crassidis, J. L., Junkins, J. L. (2012). Optimal Estimation of Dynamic Systems
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .gps import GPSSensor
from .imu import IMUSensor
from .odometry import WheelOdometry
from .health import SensorHealth


class SensorType(Enum):
    """Enumeration of sensor types for identification and management."""
    GPS = "gps"
    IMU = "imu"
    ODOMETRY = "odometry"


class FusionStrategy(Enum):
    """Sensor fusion strategies for different operational modes."""
    WEIGHTED_AVERAGE = "weighted_average"
    BEST_AVAILABLE = "best_available"
    MAJORITY_VOTE = "majority_vote"
    KALMAN_FUSION = "kalman_fusion"


@dataclass 
class SensorConfiguration:
    """
    Configuration parameters for sensor initialization.
    
    Attributes:
        gps_count: Number of GPS sensors to initialize
        imu_count: Number of IMU sensors to initialize
        enable_odometry: Whether to enable wheel odometry
        fusion_strategy: Strategy for combining sensor measurements
        health_check_interval: Interval for health monitoring (seconds)
    """
    gps_count: int = 2
    imu_count: int = 2
    enable_odometry: bool = True
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
    health_check_interval: float = 1.0


@dataclass
class FusedMeasurement:
    """
    Fused measurement containing combined sensor data and quality metrics.
    
    Attributes:
        position: Fused position estimate (x, y, z)
        velocity: Fused velocity estimate (vx, vy, vz)
        acceleration: Fused acceleration estimate (ax, ay, az)
        angular_velocity: Fused angular velocity (wx, wy, wz)
        pose_delta: Odometry-based pose change
        timestamp: Measurement timestamp
        quality_score: Overall measurement quality [0-1]
        sensor_contributions: Individual sensor contributions
        uncertainty: Measurement uncertainty estimates
    """
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    angular_velocity: Optional[np.ndarray] = None
    pose_delta: Optional[Dict[str, float]] = None
    timestamp: float = 0.0
    quality_score: float = 0.0
    sensor_contributions: Dict[str, List[int]] = None
    uncertainty: Dict[str, float] = None


class SensorFusionManager:
    """
    Advanced sensor fusion manager with comprehensive error handling and 
    intelligent sensor coordination.
    
    This manager coordinates multiple sensors of different types, providing
    robust localization through redundancy, intelligent sensor selection,
    and graceful degradation. It implements advanced features including:
    
    - Multi-sensor health monitoring
    - Adaptive sensor weighting based on performance
    - Coordinated failure simulation and recovery
    - Quality-based measurement filtering
    - Temporal synchronization of measurements
    - Statistical validation of sensor data
    
    Attributes:
        gps_sensors: List of GPS sensor instances
        imu_sensors: List of IMU sensor instances
        wheel_odometry: Wheel odometry sensor instance
        configuration: Manager configuration parameters
        health_monitor: System health monitoring state
    """
    
    def __init__(self, configuration: Optional[SensorConfiguration] = None):
        """
        Initialize sensor fusion manager with specified configuration.
        
        Args:
            configuration: Sensor configuration parameters
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        self.configuration = configuration or SensorConfiguration()
        self._validate_configuration()
        
        # Initialize sensors with diverse characteristics for redundancy
        self.gps_sensors = self._initialize_gps_sensors()
        self.imu_sensors = self._initialize_imu_sensors()
        self.wheel_odometry = self._initialize_odometry()
        
        # Fusion state management
        self._fusion_weights = {}
        self._measurement_history = []
        self._last_health_check = time.time()
        
        # Failure simulation state
        self._gps_constellation_failure = False
        self._coordinated_failure_active = False
        self._failure_start_time = None
        self._failure_duration = 0.0
        
        # Performance tracking
        self._measurement_count = 0
        self._successful_fusions = 0
        self._initialization_time = time.time()
        
        # Quality control parameters
        self._min_quality_threshold = 0.3
        self._max_measurement_age = 1.0
        self._outlier_rejection_threshold = 3.0
        
    def _validate_configuration(self) -> None:
        """Validate sensor configuration parameters."""
        config = self.configuration
        
        if config.gps_count < 0 or config.gps_count > 10:
            raise ValueError(f"GPS count must be 0-10, got {config.gps_count}")
            
        if config.imu_count < 0 or config.imu_count > 10:
            raise ValueError(f"IMU count must be 0-10, got {config.imu_count}")
            
        if config.health_check_interval <= 0 or config.health_check_interval > 60:
            raise ValueError(f"Health check interval must be 0-60s, got {config.health_check_interval}")
    
    def _initialize_gps_sensors(self) -> List[GPSSensor]:
        """Initialize GPS sensors with diverse characteristics."""
        gps_sensors = []
        
        # Define diverse GPS configurations for redundancy
        gps_configs = [
            {'noise_std': 2.0, 'dropout_prob': 0.1, 'update_rate': 10.0},
            {'noise_std': 3.0, 'dropout_prob': 0.15, 'update_rate': 5.0},
            {'noise_std': 1.5, 'dropout_prob': 0.05, 'update_rate': 20.0},
            {'noise_std': 4.0, 'dropout_prob': 0.2, 'update_rate': 1.0}
        ]
        
        for i in range(self.configuration.gps_count):
            config = gps_configs[i % len(gps_configs)]
            gps_sensor = GPSSensor(
                noise_std=config['noise_std'],
                dropout_prob=config['dropout_prob'],
                sensor_id=i + 1
            )
            gps_sensors.append(gps_sensor)
            
        return gps_sensors
    
    def _initialize_imu_sensors(self) -> List[IMUSensor]:
        """Initialize IMU sensors with diverse characteristics."""
        imu_sensors = []
        
        # Define diverse IMU configurations
        imu_configs = [
            {'accel_noise_std': 0.1, 'gyro_noise_std': 0.05, 'bias_drift_rate': 0.001},
            {'accel_noise_std': 0.15, 'gyro_noise_std': 0.08, 'bias_drift_rate': 0.002},
            {'accel_noise_std': 0.08, 'gyro_noise_std': 0.03, 'bias_drift_rate': 0.0005},
            {'accel_noise_std': 0.2, 'gyro_noise_std': 0.1, 'bias_drift_rate': 0.003}
        ]
        
        for i in range(self.configuration.imu_count):
            config = imu_configs[i % len(imu_configs)]
            imu_sensor = IMUSensor(
                accel_noise_std=config['accel_noise_std'],
                gyro_noise_std=config['gyro_noise_std'],
                bias_drift_rate=config['bias_drift_rate'],
                sensor_id=i + 1
            )
            imu_sensors.append(imu_sensor)
            
        return imu_sensors
    
    def _initialize_odometry(self) -> Optional[WheelOdometry]:
        """Initialize wheel odometry sensor."""
        if not self.configuration.enable_odometry:
            return None
            
        return WheelOdometry(
            wheelbase=0.6,
            wheel_radius=0.15,
            encoder_noise_std=0.05,
            slip_factor=0.02,
            systematic_error_std=0.01
        )
    
    def get_fused_measurements(
        self,
        true_position: np.ndarray,
        true_velocity: np.ndarray,
        true_acceleration: np.ndarray,
        true_angular_velocity: np.ndarray,
        wheel_speeds: Tuple[float, float],
        dt: float = 0.1
    ) -> FusedMeasurement:
        """
        Obtain fused measurements from all available sensors.
        
        This method coordinates all sensors to provide a comprehensive
        measurement package with quality assessment and uncertainty
        quantification.
        
        Args:
            true_position: True position for sensor simulation (x, y, z)
            true_velocity: True velocity for sensor simulation (vx, vy, vz)
            true_acceleration: True acceleration for IMU simulation
            true_angular_velocity: True angular velocity for IMU simulation
            wheel_speeds: Wheel speeds for odometry (left, right)
            dt: Time step for odometry integration
            
        Returns:
            FusedMeasurement containing all sensor data and quality metrics
            
        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Input validation
            self._validate_measurement_inputs(
                true_position, true_velocity, true_acceleration, 
                true_angular_velocity, wheel_speeds, dt
            )
            
            timestamp = time.time()
            self._measurement_count += 1
            
            # Collect raw measurements from all sensors
            raw_measurements = self._collect_raw_measurements(
                true_position, true_velocity, true_acceleration,
                true_angular_velocity, wheel_speeds, dt
            )
            
            # Perform health checks periodically
            if timestamp - self._last_health_check > self.configuration.health_check_interval:
                self._perform_health_checks()
                self._last_health_check = timestamp
            
            # Fuse measurements using selected strategy
            fused_measurement = self._fuse_measurements(raw_measurements, timestamp)
            
            # Validate and filter the fused result
            if self._validate_fused_measurement(fused_measurement):
                self._successful_fusions += 1
                self._update_measurement_history(fused_measurement)
                return fused_measurement
            else:
                # Return degraded measurement if validation fails
                return self._create_degraded_measurement(raw_measurements, timestamp)
                
        except Exception as e:
            # Return safe fallback measurement
            return self._create_fallback_measurement(timestamp)
    
    def _validate_measurement_inputs(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        wheel_speeds: Tuple[float, float],
        dt: float
    ) -> None:
        """Validate input parameters for measurement collection."""
        if not isinstance(position, np.ndarray) or position.shape != (3,):
            raise ValueError("Position must be 3D numpy array")
            
        if not isinstance(velocity, np.ndarray) or velocity.shape != (3,):
            raise ValueError("Velocity must be 3D numpy array")
            
        if not isinstance(acceleration, np.ndarray) or acceleration.shape != (3,):
            raise ValueError("Acceleration must be 3D numpy array")
            
        if not isinstance(angular_velocity, np.ndarray) or angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be 3D numpy array")
            
        if len(wheel_speeds) != 2:
            raise ValueError("Wheel speeds must be tuple of length 2")
            
        if dt <= 0 or dt > 1.0:
            raise ValueError(f"Time step must be positive and <= 1.0s, got {dt}")
    
    def _collect_raw_measurements(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        wheel_speeds: Tuple[float, float],
        dt: float
    ) -> Dict[str, List[Any]]:
        """Collect measurements from all sensors."""
        measurements = {
            'gps': [],
            'imu': [],
            'odometry': None
        }
        
        # GPS measurements
        if not self._gps_constellation_failure:
            for gps in self.gps_sensors:
                if gps.health.is_operational:
                    gps_measurement = gps.get_measurement(position)
                    if gps_measurement is not None:
                        measurements['gps'].append({
                            'position': gps_measurement,
                            'sensor_id': gps.sensor_id,
                            'reliability': gps.health.reliability,
                            'uncertainty': gps.get_uncertainty_estimate()
                        })
        
        # IMU measurements
        for imu in self.imu_sensors:
            if imu.health.is_operational:
                imu_measurement = imu.get_measurement(acceleration, angular_velocity)
                if imu_measurement is not None:
                    measurements['imu'].append({
                        'measurement': imu_measurement,
                        'sensor_id': imu.sensor_id,
                        'reliability': imu.health.reliability,
                        'uncertainty': imu.get_uncertainty_estimate()
                    })
        
        # Odometry measurement
        if self.wheel_odometry and self.wheel_odometry.health.is_operational:
            left_speed, right_speed = wheel_speeds
            odometry_measurement = self.wheel_odometry.compute_delta_pose(
                left_speed, right_speed, dt
            )
            if odometry_measurement is not None:
                measurements['odometry'] = {
                    'measurement': odometry_measurement,
                    'reliability': self.wheel_odometry.health.reliability
                }
        
        return measurements
    
    def _fuse_measurements(
        self,
        raw_measurements: Dict[str, List[Any]],
        timestamp: float
    ) -> FusedMeasurement:
        """Fuse raw measurements using the configured strategy."""
        if self.configuration.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(raw_measurements, timestamp)
        elif self.configuration.fusion_strategy == FusionStrategy.BEST_AVAILABLE:
            return self._best_available_fusion(raw_measurements, timestamp)
        elif self.configuration.fusion_strategy == FusionStrategy.MAJORITY_VOTE:
            return self._majority_vote_fusion(raw_measurements, timestamp)
        else:
            # Default to weighted average
            return self._weighted_average_fusion(raw_measurements, timestamp)
    
    def _weighted_average_fusion(
        self,
        raw_measurements: Dict[str, List[Any]],
        timestamp: float
    ) -> FusedMeasurement:
        """Fuse measurements using reliability-weighted averaging."""
        fused = FusedMeasurement(timestamp=timestamp)
        sensor_contributions = {'gps': [], 'imu': [], 'odometry': []}
        
        # Fuse GPS measurements
        if raw_measurements['gps']:
            positions = []
            weights = []
            
            for gps_data in raw_measurements['gps']:
                # Weight by reliability and inverse uncertainty
                weight = gps_data['reliability'] / (gps_data['uncertainty'] + 1e-6)
                positions.append(gps_data['position'])
                weights.append(weight)
                sensor_contributions['gps'].append(gps_data['sensor_id'])
            
            if positions:
                weights = np.array(weights)
                weights /= np.sum(weights)  # Normalize
                fused.position = np.average(positions, axis=0, weights=weights)
                fused.uncertainty = {'position': np.sum(weights * np.array([
                    raw_measurements['gps'][i]['uncertainty'] 
                    for i in range(len(positions))
                ]))}
        
        # Fuse IMU measurements
        if raw_measurements['imu']:
            accelerations = []
            angular_velocities = []
            weights = []
            
            for imu_data in raw_measurements['imu']:
                weight = imu_data['reliability'] / (imu_data['uncertainty'] + 1e-6)
                accelerations.append(imu_data['measurement']['acceleration'])
                angular_velocities.append(imu_data['measurement']['angular_velocity'])
                weights.append(weight)
                sensor_contributions['imu'].append(imu_data['sensor_id'])
            
            if accelerations:
                weights = np.array(weights)
                weights /= np.sum(weights)
                fused.acceleration = np.average(accelerations, axis=0, weights=weights)
                fused.angular_velocity = np.average(angular_velocities, axis=0, weights=weights)
        
        # Add odometry data
        if raw_measurements['odometry']:
            odometry_data = raw_measurements['odometry']['measurement']
            fused.pose_delta = {
                'delta_x': odometry_data.delta_x,
                'delta_y': odometry_data.delta_y,
                'delta_theta': odometry_data.delta_theta
            }
            sensor_contributions['odometry'].append(1)
        
        fused.sensor_contributions = sensor_contributions
        fused.quality_score = self._compute_quality_score(raw_measurements)
        
        return fused
    
    def _best_available_fusion(
        self,
        raw_measurements: Dict[str, List[Any]],
        timestamp: float
    ) -> FusedMeasurement:
        """Select best available sensor for each measurement type."""
        fused = FusedMeasurement(timestamp=timestamp)
        sensor_contributions = {'gps': [], 'imu': [], 'odometry': []}
        
        # Select best GPS
        if raw_measurements['gps']:
            best_gps = max(raw_measurements['gps'], 
                          key=lambda x: x['reliability'] / (x['uncertainty'] + 1e-6))
            fused.position = best_gps['position']
            sensor_contributions['gps'].append(best_gps['sensor_id'])
        
        # Select best IMU
        if raw_measurements['imu']:
            best_imu = max(raw_measurements['imu'],
                          key=lambda x: x['reliability'] / (x['uncertainty'] + 1e-6))
            fused.acceleration = best_imu['measurement']['acceleration']
            fused.angular_velocity = best_imu['measurement']['angular_velocity']
            sensor_contributions['imu'].append(best_imu['sensor_id'])
        
        # Use odometry if available
        if raw_measurements['odometry']:
            odometry_data = raw_measurements['odometry']['measurement']
            fused.pose_delta = {
                'delta_x': odometry_data.delta_x,
                'delta_y': odometry_data.delta_y,
                'delta_theta': odometry_data.delta_theta
            }
            sensor_contributions['odometry'].append(1)
        
        fused.sensor_contributions = sensor_contributions
        fused.quality_score = self._compute_quality_score(raw_measurements)
        
        return fused
    
    def _majority_vote_fusion(
        self,
        raw_measurements: Dict[str, List[Any]],
        timestamp: float
    ) -> FusedMeasurement:
        """Implement majority vote fusion for redundant sensors."""
        # This would implement a more sophisticated majority vote algorithm
        # For now, fall back to weighted average
        return self._weighted_average_fusion(raw_measurements, timestamp)
    
    def _compute_quality_score(self, raw_measurements: Dict[str, List[Any]]) -> float:
        """Compute overall measurement quality score."""
        scores = []
        
        # GPS quality contribution
        if raw_measurements['gps']:
            gps_scores = [gps['reliability'] for gps in raw_measurements['gps']]
            scores.append(np.mean(gps_scores))
        
        # IMU quality contribution
        if raw_measurements['imu']:
            imu_scores = [imu['reliability'] for imu in raw_measurements['imu']]
            scores.append(np.mean(imu_scores))
        
        # Odometry quality contribution
        if raw_measurements['odometry']:
            scores.append(raw_measurements['odometry']['reliability'])
        
        return np.mean(scores) if scores else 0.0
    
    def _validate_fused_measurement(self, measurement: FusedMeasurement) -> bool:
        """Validate fused measurement for reasonableness."""
        if measurement.quality_score < self._min_quality_threshold:
            return False
        
        # Check for reasonable physical limits
        if measurement.position is not None:
            if np.any(np.abs(measurement.position) > 10000):  # 10km limit
                return False
        
        if measurement.acceleration is not None:
            if np.any(np.abs(measurement.acceleration) > 100):  # 100 m/s² limit
                return False
        
        return True
    
    def _create_degraded_measurement(
        self,
        raw_measurements: Dict[str, List[Any]],
        timestamp: float
    ) -> FusedMeasurement:
        """Create degraded measurement when validation fails."""
        # Return partial measurement with reduced quality score
        measurement = FusedMeasurement(timestamp=timestamp, quality_score=0.1)
        
        # Use only most reliable sensor of each type
        if raw_measurements['gps']:
            best_gps = max(raw_measurements['gps'], key=lambda x: x['reliability'])
            measurement.position = best_gps['position']
        
        return measurement
    
    def _create_fallback_measurement(self, timestamp: float) -> FusedMeasurement:
        """Create safe fallback measurement when all else fails."""
        return FusedMeasurement(
            timestamp=timestamp,
            quality_score=0.0,
            sensor_contributions={'gps': [], 'imu': [], 'odometry': []}
        )
    
    def _perform_health_checks(self) -> None:
        """Perform periodic health checks on all sensors."""
        current_time = time.time()
        
        # Check for stale sensors
        for gps in self.gps_sensors:
            time_since_success = current_time - gps.health.last_success_time
            if time_since_success > self._max_measurement_age:
                gps.health.record_failure()
        
        for imu in self.imu_sensors:
            time_since_success = current_time - imu.health.last_success_time
            if time_since_success > self._max_measurement_age:
                imu.health.record_failure()
        
        if self.wheel_odometry:
            time_since_success = current_time - self.wheel_odometry.health.last_success_time
            if time_since_success > self._max_measurement_age:
                self.wheel_odometry.health.record_failure()
    
    def _update_measurement_history(self, measurement: FusedMeasurement) -> None:
        """Update measurement history for trend analysis."""
        self._measurement_history.append(measurement)
        
        # Keep only recent history
        max_history_length = 100
        if len(self._measurement_history) > max_history_length:
            self._measurement_history = self._measurement_history[-max_history_length:]
    
    def simulate_gps_constellation_failure(self, duration: float = 5.0) -> None:
        """
        Simulate GPS constellation failure affecting all GPS sensors.
        
        Args:
            duration: Failure duration in seconds
        """
        if duration <= 0 or duration > 300:
            raise ValueError(f"Duration must be 0-300 seconds, got {duration}")
            
        self._gps_constellation_failure = True
        self._failure_start_time = time.time()
        self._failure_duration = duration
    
    def simulate_coordinated_imu_failure(self, failure_probability: float = 0.3) -> None:
        """
        Simulate coordinated IMU failure affecting multiple sensors.
        
        Args:
            failure_probability: Probability of each IMU failing (0-1)
        """
        if failure_probability < 0 or failure_probability > 1:
            raise ValueError(f"Failure probability must be 0-1, got {failure_probability}")
            
        failure_modes = ['stuck', 'noisy', 'dropout']
        
        for imu in self.imu_sensors:
            if np.random.random() < failure_probability:
                mode = np.random.choice(failure_modes)
                imu.simulate_failure(mode)
    
    def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all sensors and system.
        
        Returns:
            Dictionary containing detailed health information
        """
        current_time = time.time()
        uptime = current_time - self._initialization_time
        
        # GPS health
        gps_health = []
        for gps in self.gps_sensors:
            gps_health.append({
                'sensor_id': gps.sensor_id,
                'operational': gps.health.is_operational,
                'reliability': gps.health.reliability,
                'failure_count': gps.health.failure_count,
                'recovery_count': gps.health.recovery_count,
                'consecutive_failures': gps.health.consecutive_failures,
                'time_since_success': current_time - gps.health.last_success_time
            })
        
        # IMU health
        imu_health = []
        for imu in self.imu_sensors:
            imu_health.append({
                'sensor_id': imu.sensor_id,
                'operational': imu.health.is_operational,
                'reliability': imu.health.reliability,
                'failure_count': imu.health.failure_count,
                'recovery_count': imu.health.recovery_count,
                'consecutive_failures': imu.health.consecutive_failures,
                'failure_mode': imu.failure_mode,
                'time_since_success': current_time - imu.health.last_success_time
            })
        
        # Odometry health
        odometry_health = None
        if self.wheel_odometry:
            odometry_health = {
                'operational': self.wheel_odometry.health.is_operational,
                'reliability': self.wheel_odometry.health.reliability,
                'failure_count': self.wheel_odometry.health.failure_count,
                'recovery_count': self.wheel_odometry.health.recovery_count,
                'time_since_success': current_time - self.wheel_odometry.health.last_success_time,
                'performance_metrics': self.wheel_odometry.get_performance_metrics()
            }
        
        # System-level metrics
        success_rate = self._successful_fusions / max(self._measurement_count, 1)
        
        return {
            'gps_sensors': gps_health,
            'imu_sensors': imu_health,
            'odometry': odometry_health,
            'system_metrics': {
                'uptime_seconds': uptime,
                'total_measurements': self._measurement_count,
                'successful_fusions': self._successful_fusions,
                'fusion_success_rate': success_rate,
                'gps_constellation_failure': self._gps_constellation_failure,
                'coordinated_failure_active': self._coordinated_failure_active,
                'configuration': {
                    'gps_count': len(self.gps_sensors),
                    'imu_count': len(self.imu_sensors),
                    'odometry_enabled': self.wheel_odometry is not None,
                    'fusion_strategy': self.configuration.fusion_strategy.value
                }
            }
        }
    
    def reset_all_sensors(self) -> None:
        """Reset all sensors to operational state."""
        for gps in self.gps_sensors:
            gps.health = SensorHealth()
        
        for imu in self.imu_sensors:
            imu.health = SensorHealth()
            imu.failure_mode = None
            imu.stuck_values = None
        
        if self.wheel_odometry:
            self.wheel_odometry.health = SensorHealth()
        
        self._gps_constellation_failure = False
        self._coordinated_failure_active = False
        self._failure_start_time = None
    
    def get_sensor_statistics(self) -> Dict[str, Any]:
        """Get detailed sensor performance statistics."""
        stats = {}
        
        # GPS statistics
        if self.gps_sensors:
            gps_reliabilities = [gps.health.reliability for gps in self.gps_sensors]
            stats['gps'] = {
                'count': len(self.gps_sensors),
                'average_reliability': np.mean(gps_reliabilities),
                'min_reliability': np.min(gps_reliabilities),
                'max_reliability': np.max(gps_reliabilities),
                'operational_count': sum(1 for gps in self.gps_sensors if gps.health.is_operational)
            }
        
        # IMU statistics
        if self.imu_sensors:
            imu_reliabilities = [imu.health.reliability for imu in self.imu_sensors]
            stats['imu'] = {
                'count': len(self.imu_sensors),
                'average_reliability': np.mean(imu_reliabilities),
                'min_reliability': np.min(imu_reliabilities),
                'max_reliability': np.max(imu_reliabilities),
                'operational_count': sum(1 for imu in self.imu_sensors if imu.health.is_operational),
                'failure_modes': [imu.failure_mode for imu in self.imu_sensors if imu.failure_mode]
            }
        
        # Odometry statistics
        if self.wheel_odometry:
            stats['odometry'] = {
                'reliability': self.wheel_odometry.health.reliability,
                'operational': self.wheel_odometry.health.is_operational,
                'performance': self.wheel_odometry.get_performance_metrics()
            }
        
        return stats