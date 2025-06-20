"""
GPS sensor simulation with realistic noise models and failure modes.

This module implements a GPS sensor model that simulates real-world behavior
including measurement noise, systematic biases, signal dropouts, and multipath effects.

GPS Measurement Model:
    z_gps = p_true + b_gps + n_gps + dropout_mask
    
    where:
    - p_true: True 3D position [x, y, z] in meters
    - b_gps: Systematic bias vector (constant per GPS unit)
    - n_gps: Zero-mean Gaussian noise ~ N(0, σ²I)
    - dropout_mask: Binary mask for signal loss events

Noise Characteristics:
    - Horizontal accuracy: typically 2-5m (1σ)
    - Vertical accuracy: typically 1.5-3x horizontal accuracy
    - Bias stability: varies with atmospheric conditions and satellite geometry
    - Dropout probability: increases in urban environments, under foliage

Failure Modes:
    - Signal loss: temporary or extended dropout periods
    - Degraded accuracy: increased noise during poor satellite geometry
    - Multipath: systematic errors from signal reflections
    - Jamming/interference: complete signal loss or erroneous measurements
"""

import numpy as np
from typing import Optional, Tuple
from .health import SensorHealth


class GPSSensor:
    """
    GPS sensor with configurable noise, bias, and dropout characteristics.
    
    This class simulates a single GPS receiver with realistic error sources
    including thermal noise, atmospheric delays, and signal blockage.
    
    The sensor maintains its own systematic bias that remains constant over
    the simulation period, representing fixed errors from atmospheric delays
    and receiver clock offsets.
    
    Attributes:
        noise_std: Standard deviation of measurement noise (meters)
        dropout_prob: Probability of measurement dropout [0.0, 1.0]
        sensor_id: Unique identifier for this GPS unit
        position_bias: Systematic bias vector [x, y, z] in meters
        health: SensorHealth object tracking reliability
    """
    
    def __init__(self, noise_std: float = 2.0, dropout_prob: float = 0.1, 
                 sensor_id: int = 1, bias_std: float = 1.0):
        """
        Initialize GPS sensor with specified characteristics.
        
        Args:
            noise_std: Standard deviation of Gaussian measurement noise (meters)
            dropout_prob: Probability of measurement dropout per time step
            sensor_id: Unique identifier for this GPS receiver
            bias_std: Standard deviation for systematic bias initialization
        """
        if noise_std <= 0:
            raise ValueError("GPS noise standard deviation must be positive")
        if not 0 <= dropout_prob <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
            
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.sensor_id = sensor_id
        
        # Initialize systematic bias (remains constant during simulation)
        # Represents atmospheric delays, clock errors, and receiver biases
        self.position_bias = np.random.normal(0, bias_std, 3)
        
        # Health monitoring
        self.health = SensorHealth()
        
        # Internal state for modeling time-correlated errors
        self._atmospheric_delay = np.random.normal(0, 0.5, 3)
        self._multipath_errors = np.zeros(3)
        
    def get_measurement(self, true_position: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate GPS position measurement with realistic error sources.
        
        Args:
            true_position: True 3D position [x, y, z] in meters
            
        Returns:
            GPS position measurement with noise and bias, or None if dropout occurs
            
        Raises:
            ValueError: If true_position is not a 3D vector
        """
        if len(true_position) != 3:
            raise ValueError("GPS requires 3D position input [x, y, z]")
            
        # Check for signal dropout (simulate signal blockage/interference)
        if np.random.random() < self.dropout_prob:
            self.health.record_failure()
            return None
            
        # Apply systematic bias (atmospheric delays, clock errors)
        measurement = true_position + self.position_bias
        
        # Add time-correlated atmospheric delay errors
        measurement += self._atmospheric_delay
        
        # Add uncorrelated thermal noise
        thermal_noise = np.random.normal(0, self.noise_std, 3)
        measurement += thermal_noise
        
        # Add multipath errors (more significant in urban environments)
        multipath_noise = np.random.normal(0, self.noise_std * 0.2, 3)
        measurement += multipath_noise
        
        # Update atmospheric delay for next measurement (slowly varying)
        self._atmospheric_delay += np.random.normal(0, 0.01, 3)
        self._atmospheric_delay = np.clip(self._atmospheric_delay, -2.0, 2.0)
        
        self.health.record_success()
        return measurement
        
    def simulate_degraded_accuracy(self, degradation_factor: float = 3.0) -> None:
        """
        Simulate degraded GPS accuracy due to poor satellite geometry.
        
        Args:
            degradation_factor: Factor by which to increase noise (> 1.0)
        """
        if degradation_factor < 1.0:
            raise ValueError("Degradation factor must be >= 1.0")
            
        self.noise_std *= degradation_factor
        
    def simulate_jamming(self, jam_probability: float = 0.9) -> None:
        """
        Simulate GPS jamming by increasing dropout probability.
        
        Args:
            jam_probability: Probability of dropout during jamming
        """
        if not 0 <= jam_probability <= 1:
            raise ValueError("Jam probability must be between 0 and 1")
            
        self.dropout_prob = jam_probability
        
    def get_measurement_covariance(self) -> np.ndarray:
        """
        Get measurement noise covariance matrix.
        
        Returns:
            3x3 covariance matrix for GPS measurements
        """
        # GPS typically has slightly worse vertical accuracy
        horizontal_var = self.noise_std ** 2
        vertical_var = (self.noise_std * 1.5) ** 2
        
        covariance = np.diag([horizontal_var, horizontal_var, vertical_var])
        return covariance
        
    def reset_bias(self, bias_std: float = 1.0) -> None:
        """
        Reset systematic bias (useful for testing different scenarios).
        
        Args:
            bias_std: Standard deviation for new bias initialization
        """
        self.position_bias = np.random.normal(0, bias_std, 3)
        self._atmospheric_delay = np.random.normal(0, 0.5, 3)
        
    def get_uncertainty_estimate(self) -> float:
        """
        Get current measurement uncertainty estimate.
        
        Returns:
            Scalar uncertainty estimate based on noise characteristics and health
        """
        # Base uncertainty from noise characteristics
        base_uncertainty = self.noise_std
        
        # Scale by reliability (lower reliability = higher uncertainty) 
        reliability_factor = 1.0 / max(0.1, self.health.reliability)
        
        # Add atmospheric delay uncertainty
        atmospheric_uncertainty = np.linalg.norm(self._atmospheric_delay) * 0.5
        
        return base_uncertainty * reliability_factor + atmospheric_uncertainty
    
    def get_sensor_info(self) -> dict:
        """
        Get comprehensive sensor information for monitoring.
        
        Returns:
            Dictionary containing sensor configuration and status
        """
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': 'GPS',
            'noise_std': self.noise_std,
            'dropout_prob': self.dropout_prob,
            'position_bias': self.position_bias.tolist(),
            'health': self.health.get_health_summary(),
            'covariance': self.get_measurement_covariance().tolist()
        }


# Maintain backward compatibility with original naming
GPS = GPSSensor
