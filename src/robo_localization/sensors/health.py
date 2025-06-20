"""
Sensor health monitoring for robust multi-sensor fusion.

This module implements health tracking, failure detection, and recovery monitoring
for individual sensors in the localization system. The health monitoring enables
autonomous fault detection and provides reliability metrics for sensor fusion.

The SensorHealth class tracks:
- Operational status (functional/failed)
- Reliability metrics (0.0 to 1.0)
- Failure/recovery counts and timing
- Consecutive failure tracking for fault isolation

Mathematical Model:
    reliability(t) = max(0.0, 1.0 - failure_penalty * consecutive_failures)
    where failure_penalty is calibrated based on sensor characteristics

Failure Detection:
    A sensor is marked as failed when consecutive failures exceed threshold
    or when measurement validation fails (e.g., Mahalanobis distance test)

Recovery Model:
    reliability(t+1) = min(1.0, reliability(t) + recovery_rate)
    Applied when successful measurements are received after failures
"""

import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    """Enumeration of supported sensor types."""
    GPS = "gps"
    IMU = "imu" 
    ODOMETRY = "odometry"


@dataclass
class SensorHealth:
    """
    Tracks sensor health, reliability, and failure recovery.
    
    This class implements a statistical model for sensor reliability based on
    observed failure patterns. The reliability score decreases with consecutive
    failures and recovers with successful measurements.
    
    Attributes:
        is_operational: Boolean flag indicating if sensor is functional
        reliability: Float [0.0, 1.0] indicating measurement trustworthiness
        failure_count: Total number of failures observed
        recovery_count: Total number of recoveries observed
        consecutive_failures: Current consecutive failure streak
        last_success_time: Timestamp of last successful measurement
    
    Mathematical Model:
        The reliability decreases exponentially with consecutive failures:
        reliability = max(0.0, base_reliability * exp(-failure_rate * consecutive_failures))
        
        Recovery follows a linear model:
        reliability = min(1.0, reliability + recovery_rate)
    """
    
    def __init__(self, failure_threshold: int = 5, reliability_decay: float = 0.15):
        """
        Initialize sensor health monitor.
        
        Args:
            failure_threshold: Number of consecutive failures before marking inoperational
            reliability_decay: Rate of reliability decrease per consecutive failure
        """
        self.is_operational = True
        self.reliability = 1.0
        self.failure_count = 0
        self.recovery_count = 0
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        # Configuration parameters
        self._failure_threshold = failure_threshold
        self._reliability_decay = reliability_decay
        self._recovery_rate = 0.05
        
    def record_failure(self) -> None:
        """
        Record a sensor failure event.
        
        Updates failure statistics and reduces reliability score based on
        consecutive failure count. Marks sensor as inoperational if failure
        threshold is exceeded.
        
        Mathematical Update:
            reliability = max(0.0, 1.0 - decay_rate * consecutive_failures)
        """
        self.failure_count += 1
        self.consecutive_failures += 1
        
        # Apply reliability penalty based on consecutive failures
        # More aggressive penalty for repeated failures
        penalty = min(0.9, self._reliability_decay * self.consecutive_failures)
        self.reliability = max(0.0, 1.0 - penalty)
        
        # Mark as inoperational if threshold exceeded
        if self.consecutive_failures >= self._failure_threshold:
            self.is_operational = False
            
    def record_success(self) -> None:
        """
        Record a successful sensor measurement.
        
        Resets consecutive failure count and improves reliability score.
        Re-enables sensor if it was previously marked as failed but is now working.
        
        Mathematical Update:
            reliability = min(1.0, reliability + recovery_rate)
        """
        self.recovery_count += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        # Improve reliability with successful measurements
        self.reliability = min(1.0, self.reliability + self._recovery_rate)
        
        # Re-enable sensor if reliability is sufficient
        if not self.is_operational and self.reliability > 0.5:
            self.is_operational = True
            
    def get_reliability_score(self) -> float:
        """
        Get current reliability score [0.0, 1.0].
        
        Returns:
            Float reliability score where 1.0 is fully reliable, 0.0 is unreliable
        """
        return self.reliability
        
    def is_sensor_functional(self) -> bool:
        """
        Check if sensor is currently functional.
        
        Returns:
            True if sensor is operational and reliable enough for use
        """
        return self.is_operational and self.reliability > 0.3
        
    def get_failure_rate(self) -> float:
        """
        Calculate failure rate as percentage of total measurements.
        
        Returns:
            Failure rate [0.0, 1.0] if measurements have been made, 0.0 otherwise
        """
        total_measurements = self.failure_count + self.recovery_count
        if total_measurements == 0:
            return 0.0
        return self.failure_count / total_measurements
        
    def time_since_last_success(self) -> float:
        """
        Get time elapsed since last successful measurement.
        
        Returns:
            Time in seconds since last successful measurement
        """
        return time.time() - self.last_success_time
        
    def reset_health(self) -> None:
        """
        Reset health statistics to initial state.
        
        Used for testing or when reinitializing sensor systems.
        """
        self.is_operational = True
        self.reliability = 1.0
        self.failure_count = 0
        self.recovery_count = 0
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
    def get_health_summary(self) -> dict:
        """
        Get comprehensive health status summary.
        
        Returns:
            Dictionary containing all health metrics and statistics
        """
        return {
            'operational': self.is_operational,
            'reliability': self.reliability,
            'failure_count': self.failure_count,
            'recovery_count': self.recovery_count,
            'consecutive_failures': self.consecutive_failures,
            'failure_rate': self.get_failure_rate(),
            'time_since_success': self.time_since_last_success()
        }
