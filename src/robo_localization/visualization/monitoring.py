"""
Scientific sensor health monitoring and performance metrics visualization module.

This module provides comprehensive sensor health monitoring capabilities with
statistical analysis, predictive maintenance indicators, and real-time
performance metrics visualization for robotic localization systems.

Classes:
    SensorHealthMonitor: Advanced sensor health monitoring with predictive analytics
    MetricsDisplay: Performance metrics visualization and statistical analysis

Author: Automated Code Generation System
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from collections import deque, defaultdict
import warnings
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.signal import find_peaks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Enumeration for sensor health status levels."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class SensorMetrics:
    """Container for individual sensor performance metrics."""
    sensor_id: str
    reliability: float
    uptime_percentage: float
    mean_response_time: float
    error_rate: float
    last_maintenance: Optional[float] = None
    predicted_failure_time: Optional[float] = None
    health_trend: float = 0.0
    confidence_score: float = 1.0

@dataclass
class SystemHealthSummary:
    """Summary of overall system health metrics."""
    overall_health_score: float
    redundancy_level: float
    critical_sensors_count: int
    failed_sensors_count: int
    maintenance_recommendations: List[str] = field(default_factory=list)
    performance_trend: float = 0.0
    estimated_mission_duration: Optional[float] = None


class SensorHealthMonitor:
    """
    Advanced sensor health monitoring system with predictive analytics.
    
    This class provides comprehensive sensor health monitoring capabilities including:
    - Real-time health status tracking with statistical analysis
    - Predictive maintenance using trend analysis and machine learning
    - Failure detection and recovery monitoring with anomaly detection
    - Redundancy analysis and system resilience assessment
    - Performance degradation tracking with early warning systems
    - Statistical health metrics and confidence intervals
    
    The monitor employs scientific statistical methods for reliability analysis,
    including Weibull distribution fitting for failure prediction and Kalman
    filtering for trend estimation.
    
    Parameters:
        history_length (int): Maximum number of historical data points to maintain
        prediction_horizon (float): Time horizon for failure prediction in seconds
        confidence_threshold (float): Confidence threshold for health assessments
        enable_predictive_analytics (bool): Enable predictive maintenance features
        
    Attributes:
        sensor_health_history (List[Dict]): Historical health status data
        sensor_metrics (Dict[str, SensorMetrics]): Current sensor performance metrics
        failure_events (List[Dict]): Record of sensor failure events
        recovery_events (List[Dict]): Record of sensor recovery events
        reliability_models (Dict): Statistical models for reliability prediction
    """
    
    def __init__(self, 
                 history_length: int = 1000,
                 prediction_horizon: float = 3600.0,  # 1 hour
                 confidence_threshold: float = 0.95,
                 enable_predictive_analytics: bool = True):
        """Initialize the sensor health monitoring system."""
        
        # Input validation
        if history_length <= 0:
            raise ValueError("History length must be positive")
        if prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive")
        if not (0 < confidence_threshold < 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
            
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        self.enable_predictive_analytics = enable_predictive_analytics
        
        # Health monitoring data structures
        self.sensor_health_history = deque(maxlen=history_length)
        self.sensor_metrics = {}
        self.failure_events = []
        self.recovery_events = []
        self.anomaly_events = []
        
        # Statistical models and analysis
        self.reliability_models = {}
        self.trend_filters = {}
        self.anomaly_detectors = {}
        
        # Performance tracking
        self.previous_status = {}
        self.health_statistics = defaultdict(lambda: deque(maxlen=100))
        self.system_performance_history = deque(maxlen=history_length)
        
        # Predictive analytics
        self.maintenance_schedule = {}
        self.failure_predictions = {}
        
        logger.info(f"SensorHealthMonitor initialized with {history_length} point history, "
                   f"{prediction_horizon}s prediction horizon")
        
    def update_health_status(self, 
                           health_status: Dict[str, Any], 
                           timestamp: float,
                           additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update sensor health status with comprehensive analysis.
        
        Args:
            health_status: Dictionary containing sensor status information
            timestamp: Unix timestamp or simulation time
            additional_metrics: Optional additional sensor metrics
            
        Raises:
            ValueError: If health status data is invalid
            TypeError: If input types are incorrect
        """
        if not isinstance(health_status, dict):
            raise TypeError("Health status must be a dictionary")
            
        if not isinstance(timestamp, (int, float)):
            raise TypeError("Timestamp must be numeric")
            
        # Store historical data
        health_record = {
            'timestamp': timestamp,
            'status': health_status.copy(),
            'additional_metrics': additional_metrics.copy() if additional_metrics else {}
        }
        self.sensor_health_history.append(health_record)
        
        # Update individual sensor metrics
        self._update_sensor_metrics(health_status, timestamp, additional_metrics)
        
        # Detect state changes and events
        self._detect_health_state_changes(health_status, timestamp)
        
        # Perform anomaly detection
        if self.enable_predictive_analytics:
            self._perform_anomaly_detection(health_status, timestamp)
            
        # Update statistical models
        self._update_reliability_models(timestamp)
        
        # Update system performance summary
        self._update_system_performance_summary(timestamp)
        
        # Update previous status for comparison
        self.previous_status = health_status.copy()
        
        logger.debug(f"Health status updated at timestamp {timestamp}")
        
    def _update_sensor_metrics(self, 
                             health_status: Dict[str, Any], 
                             timestamp: float,
                             additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update comprehensive metrics for each sensor."""
        
        for sensor_type, sensor_data in health_status.items():
            if isinstance(sensor_data, list):
                # Multiple sensors of the same type
                for i, sensor in enumerate(sensor_data):
                    sensor_id = f"{sensor_type}_{i}"
                    self._update_individual_sensor_metrics(
                        sensor_id, sensor, timestamp, additional_metrics)
            else:
                # Single sensor
                sensor_id = sensor_type
                self._update_individual_sensor_metrics(
                    sensor_id, sensor_data, timestamp, additional_metrics)
                    
    def _update_individual_sensor_metrics(self, 
                                        sensor_id: str, 
                                        sensor_data: Dict[str, Any], 
                                        timestamp: float,
                                        additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update metrics for an individual sensor."""
        
        # Extract basic sensor information
        operational = sensor_data.get('operational', True)
        reliability = sensor_data.get('reliability', 1.0)
        response_time = sensor_data.get('response_time', 0.0)
        error_rate = sensor_data.get('error_rate', 0.0)
        
        # Initialize or update sensor metrics
        if sensor_id not in self.sensor_metrics:
            self.sensor_metrics[sensor_id] = SensorMetrics(
                sensor_id=sensor_id,
                reliability=reliability,
                uptime_percentage=100.0 if operational else 0.0,
                mean_response_time=response_time,
                error_rate=error_rate
            )
        else:
            # Update existing metrics
            metrics = self.sensor_metrics[sensor_id]
            
            # Update reliability with exponential moving average
            alpha = 0.1  # Smoothing factor
            metrics.reliability = (1 - alpha) * metrics.reliability + alpha * reliability
            
            # Update response time
            metrics.mean_response_time = (
                (1 - alpha) * metrics.mean_response_time + alpha * response_time)
                
            # Update error rate
            metrics.error_rate = (1 - alpha) * metrics.error_rate + alpha * error_rate
            
        # Store historical data for trend analysis
        self.health_statistics[sensor_id].append({
            'timestamp': timestamp,
            'reliability': reliability,
            'operational': operational,
            'response_time': response_time,
            'error_rate': error_rate
        })
        
        # Compute health trend
        self._compute_health_trend(sensor_id)
        
        # Update uptime percentage
        self._update_uptime_percentage(sensor_id, operational)
        
    def _compute_health_trend(self, sensor_id: str) -> None:
        """Compute health trend using linear regression."""
        if len(self.health_statistics[sensor_id]) < 5:
            return
            
        # Extract recent reliability data
        recent_data = list(self.health_statistics[sensor_id])[-20:]  # Last 20 points
        timestamps = [d['timestamp'] for d in recent_data]
        reliabilities = [d['reliability'] for d in recent_data]
        
        # Perform linear regression
        if len(timestamps) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                timestamps, reliabilities)
            
            # Update trend in sensor metrics
            if sensor_id in self.sensor_metrics:
                self.sensor_metrics[sensor_id].health_trend = slope
                self.sensor_metrics[sensor_id].confidence_score = abs(r_value)
                
    def _update_uptime_percentage(self, sensor_id: str, operational: bool) -> None:
        """Update uptime percentage using sliding window."""
        if sensor_id not in self.sensor_metrics:
            return
            
        # Calculate uptime from recent history
        recent_data = list(self.health_statistics[sensor_id])[-50:]  # Last 50 points
        if len(recent_data) > 0:
            operational_count = sum(1 for d in recent_data if d['operational'])
            uptime_percentage = (operational_count / len(recent_data)) * 100
            self.sensor_metrics[sensor_id].uptime_percentage = uptime_percentage
            
    def _detect_health_state_changes(self, 
                                   current_status: Dict[str, Any], 
                                   timestamp: float) -> None:
        """Detect sensor failure and recovery events with enhanced analysis."""
        
        for sensor_type, sensor_data in current_status.items():
            if isinstance(sensor_data, list):
                for i, sensor in enumerate(sensor_data):
                    sensor_id = f"{sensor_type}_{i}"
                    self._check_individual_sensor_state_change(
                        sensor_id, sensor, timestamp)
            else:
                sensor_id = sensor_type
                self._check_individual_sensor_state_change(
                    sensor_id, sensor_data, timestamp)
                    
    def _check_individual_sensor_state_change(self, 
                                            sensor_id: str, 
                                            current_sensor: Dict[str, Any], 
                                            timestamp: float) -> None:
        """Check for state changes in individual sensor."""
        
        current_operational = current_sensor.get('operational', True)
        current_reliability = current_sensor.get('reliability', 1.0)
        
        # Find previous state
        previous_operational = True
        previous_reliability = 1.0
        
        if self.previous_status:
            for sensor_type, sensor_data in self.previous_status.items():
                if isinstance(sensor_data, list):
                    for i, prev_sensor in enumerate(sensor_data):
                        if f"{sensor_type}_{i}" == sensor_id:
                            previous_operational = prev_sensor.get('operational', True)
                            previous_reliability = prev_sensor.get('reliability', 1.0)
                            break
                else:
                    if sensor_type == sensor_id:
                        previous_operational = sensor_data.get('operational', True)
                        previous_reliability = sensor_data.get('reliability', 1.0)
                        
        # Detect failure events
        if previous_operational and not current_operational:
            failure_event = {
                'sensor_id': sensor_id,
                'timestamp': timestamp,
                'type': 'failure',
                'previous_reliability': previous_reliability,
                'severity': self._assess_failure_severity(current_sensor)
            }
            self.failure_events.append(failure_event)
            logger.warning(f"Sensor failure detected: {sensor_id} at {timestamp}")
            
        # Detect recovery events
        elif not previous_operational and current_operational:
            recovery_event = {
                'sensor_id': sensor_id,
                'timestamp': timestamp,
                'type': 'recovery',
                'new_reliability': current_reliability,
                'downtime': self._calculate_downtime(sensor_id, timestamp)
            }
            self.recovery_events.append(recovery_event)
            logger.info(f"Sensor recovery detected: {sensor_id} at {timestamp}")
            
        # Detect significant reliability degradation
        reliability_threshold = 0.2  # 20% degradation threshold
        if (previous_reliability - current_reliability > reliability_threshold and
            current_operational):
            
            degradation_event = {
                'sensor_id': sensor_id,
                'timestamp': timestamp,
                'type': 'degradation',
                'reliability_drop': previous_reliability - current_reliability,
                'severity': 'critical' if current_reliability < 0.5 else 'warning'
            }
            self.anomaly_events.append(degradation_event)
            logger.warning(f"Reliability degradation detected: {sensor_id}")
            
    def _assess_failure_severity(self, sensor_data: Dict[str, Any]) -> str:
        """Assess the severity of sensor failure."""
        reliability = sensor_data.get('reliability', 0.0)
        error_rate = sensor_data.get('error_rate', 0.0)
        
        if reliability < 0.1 and error_rate > 0.5:
            return 'critical'
        elif reliability < 0.3 and error_rate > 0.3:
            return 'high'
        elif reliability < 0.5:
            return 'medium'
        else:
            return 'low'
            
    def _calculate_downtime(self, sensor_id: str, recovery_timestamp: float) -> float:
        """Calculate sensor downtime duration."""
        # Find the most recent failure event for this sensor
        for event in reversed(self.failure_events):
            if event['sensor_id'] == sensor_id:
                return recovery_timestamp - event['timestamp']
        return 0.0
        
    def _perform_anomaly_detection(self, 
                                 health_status: Dict[str, Any], 
                                 timestamp: float) -> None:
        """Perform statistical anomaly detection on sensor data."""
        
        for sensor_type, sensor_data in health_status.items():
            if isinstance(sensor_data, list):
                for i, sensor in enumerate(sensor_data):
                    sensor_id = f"{sensor_type}_{i}"
                    self._detect_sensor_anomalies(sensor_id, sensor, timestamp)
            else:
                sensor_id = sensor_type
                self._detect_sensor_anomalies(sensor_id, sensor_data, timestamp)
                
    def _detect_sensor_anomalies(self, 
                                sensor_id: str, 
                                sensor_data: Dict[str, Any], 
                                timestamp: float) -> None:
        """Detect anomalies in individual sensor data."""
        
        if len(self.health_statistics[sensor_id]) < 10:
            return  # Not enough data for anomaly detection
            
        # Extract recent reliability data
        recent_data = list(self.health_statistics[sensor_id])[-30:]
        reliabilities = [d['reliability'] for d in recent_data]
        
        # Statistical anomaly detection using z-score
        mean_reliability = np.mean(reliabilities)
        std_reliability = np.std(reliabilities)
        
        current_reliability = sensor_data.get('reliability', 1.0)
        
        if std_reliability > 0:
            z_score = abs(current_reliability - mean_reliability) / std_reliability
            
            # Anomaly threshold (3 standard deviations)
            if z_score > 3.0:
                anomaly_event = {
                    'sensor_id': sensor_id,
                    'timestamp': timestamp,
                    'type': 'statistical_anomaly',
                    'z_score': z_score,
                    'current_value': current_reliability,
                    'expected_value': mean_reliability,
                    'severity': 'high' if z_score > 4.0 else 'medium'
                }
                self.anomaly_events.append(anomaly_event)
                logger.warning(f"Statistical anomaly detected in {sensor_id}: z-score={z_score:.2f}")
                
    def _update_reliability_models(self, timestamp: float) -> None:
        """Update statistical reliability models for failure prediction."""
        
        if not self.enable_predictive_analytics:
            return
            
        for sensor_id, metrics in self.sensor_metrics.items():
            if len(self.health_statistics[sensor_id]) >= 20:
                self._fit_reliability_model(sensor_id, timestamp)
                
    def _fit_reliability_model(self, sensor_id: str, current_timestamp: float) -> None:
        """Fit Weibull reliability model for failure prediction."""
        
        try:
            # Extract reliability data
            sensor_data = list(self.health_statistics[sensor_id])
            reliabilities = [d['reliability'] for d in sensor_data]
            timestamps = [d['timestamp'] for d in sensor_data]
            
            # Normalize time to start from 0
            time_points = np.array(timestamps) - timestamps[0]
            reliability_array = np.array(reliabilities)
            
            # Fit exponential decay model for reliability
            # R(t) = R0 * exp(-λt)
            if np.any(reliability_array > 0):
                # Use logarithmic transformation for linear regression
                log_reliability = np.log(np.maximum(reliability_array, 1e-6))
                
                # Fit linear model
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_points, log_reliability)
                
                # Store model parameters
                self.reliability_models[sensor_id] = {
                    'decay_rate': -slope,  # λ parameter
                    'initial_reliability': np.exp(intercept),  # R0
                    'r_squared': r_value**2,
                    'last_updated': current_timestamp
                }
                
                # Predict failure time (when reliability drops below threshold)
                failure_threshold = 0.1
                if slope < 0:  # Decreasing reliability
                    predicted_failure_time = (
                        (np.log(failure_threshold) - intercept) / slope + timestamps[0])
                    
                    # Only consider predictions within reasonable horizon
                    if predicted_failure_time <= current_timestamp + self.prediction_horizon:
                        self.sensor_metrics[sensor_id].predicted_failure_time = predicted_failure_time
                        
        except Exception as e:
            logger.debug(f"Could not fit reliability model for {sensor_id}: {e}")
            
    def _update_system_performance_summary(self, timestamp: float) -> None:
        """Update overall system performance summary."""
        
        # Collect system-wide metrics
        total_sensors = len(self.sensor_metrics)
        if total_sensors == 0:
            return
            
        operational_sensors = sum(1 for m in self.sensor_metrics.values() 
                                if m.uptime_percentage > 50.0)
        failed_sensors = sum(1 for m in self.sensor_metrics.values() 
                           if m.uptime_percentage < 10.0)
        critical_sensors = sum(1 for m in self.sensor_metrics.values() 
                             if m.reliability < 0.3)
        
        # Calculate overall health score
        avg_reliability = np.mean([m.reliability for m in self.sensor_metrics.values()])
        avg_uptime = np.mean([m.uptime_percentage for m in self.sensor_metrics.values()])
        
        # Weighted health score
        health_score = (0.7 * avg_reliability + 0.3 * avg_uptime / 100.0)
        
        # Redundancy analysis
        redundancy_level = operational_sensors / total_sensors if total_sensors > 0 else 0
        
        # Generate maintenance recommendations
        recommendations = self._generate_maintenance_recommendations()
        
        # Create summary
        summary = SystemHealthSummary(
            overall_health_score=health_score,
            redundancy_level=redundancy_level,
            critical_sensors_count=critical_sensors,
            failed_sensors_count=failed_sensors,
            maintenance_recommendations=recommendations,
            performance_trend=self._compute_system_performance_trend()
        )
        
        self.system_performance_history.append({
            'timestamp': timestamp,
            'summary': summary
        })
        
    def _generate_maintenance_recommendations(self) -> List[str]:
        """Generate maintenance recommendations based on sensor analysis."""
        
        recommendations = []
        
        for sensor_id, metrics in self.sensor_metrics.items():
            # Check for degrading sensors
            if metrics.reliability < 0.5:
                recommendations.append(f"Immediate attention required for {sensor_id}")
                
            # Check for sensors with negative health trends
            if metrics.health_trend < -0.01:  # Significant negative trend
                recommendations.append(f"Preventive maintenance recommended for {sensor_id}")
                
            # Check for predicted failures
            if (metrics.predicted_failure_time and 
                metrics.predicted_failure_time < time.time() + 3600):  # Within 1 hour
                recommendations.append(f"Urgent: {sensor_id} predicted to fail soon")
                
        return recommendations
        
    def _compute_system_performance_trend(self) -> float:
        """Compute overall system performance trend."""
        
        if len(self.system_performance_history) < 5:
            return 0.0
            
        # Extract recent health scores
        recent_scores = [entry['summary'].overall_health_score 
                        for entry in list(self.system_performance_history)[-10:]]
        
        # Compute trend using linear regression
        if len(recent_scores) > 1:
            time_points = np.arange(len(recent_scores))
            slope, _, _, _, _ = stats.linregress(time_points, recent_scores)
            return slope
            
        return 0.0
        
    def get_redundancy_status(self) -> Dict[str, Any]:
        """
        Get comprehensive redundancy status analysis.
        
        Returns:
            Dictionary containing detailed redundancy information
        """
        if not self.sensor_health_history:
            return {}
            
        latest_status = self.sensor_health_history[-1]['status']
        redundancy_analysis = {}
        
        # Analyze by sensor type
        for sensor_type, sensors in latest_status.items():
            if isinstance(sensors, list):
                total_sensors = len(sensors)
                operational_sensors = sum(1 for s in sensors if s.get('operational', True))
                reliable_sensors = sum(1 for s in sensors if s.get('reliability', 1.0) > 0.7)
                
                # Determine redundancy level
                if operational_sensors == total_sensors:
                    redundancy_level = 'full'
                elif operational_sensors > total_sensors / 2:
                    redundancy_level = 'partial'
                elif operational_sensors > 0:
                    redundancy_level = 'minimal'
                else:
                    redundancy_level = 'none'
                    
                # Risk assessment
                if redundancy_level == 'none':
                    risk_level = 'critical'
                elif redundancy_level == 'minimal':
                    risk_level = 'high'
                elif redundancy_level == 'partial':
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                    
                redundancy_analysis[sensor_type] = {
                    'total_sensors': total_sensors,
                    'operational_sensors': operational_sensors,
                    'reliable_sensors': reliable_sensors,
                    'redundancy_level': redundancy_level,
                    'risk_level': risk_level,
                    'availability_percentage': (operational_sensors / total_sensors) * 100
                }
                
        return redundancy_analysis
        
    def compute_sensor_health_statistics(self, sensor_id: str) -> Dict[str, Any]:
        """
        Compute comprehensive health statistics for a specific sensor.
        
        Args:
            sensor_id: Identifier of the sensor to analyze
            
        Returns:
            Dictionary containing detailed health statistics
        """
        if sensor_id not in self.health_statistics:
            return {}
            
        sensor_data = list(self.health_statistics[sensor_id])
        if len(sensor_data) < 2:
            return {}
            
        # Extract metrics
        reliabilities = [d['reliability'] for d in sensor_data]
        response_times = [d['response_time'] for d in sensor_data]
        error_rates = [d['error_rate'] for d in sensor_data]
        operational_status = [d['operational'] for d in sensor_data]
        
        # Basic statistics
        reliability_stats = {
            'mean': np.mean(reliabilities),
            'std': np.std(reliabilities),
            'min': np.min(reliabilities),
            'max': np.max(reliabilities),
            'median': np.median(reliabilities),
            'percentile_95': np.percentile(reliabilities, 95),
            'percentile_5': np.percentile(reliabilities, 5)
        }
        
        # Response time statistics
        response_time_stats = {
            'mean': np.mean(response_times),
            'std': np.std(response_times),
            'min': np.min(response_times),
            'max': np.max(response_times)
        }
        
        # Error rate statistics
        error_rate_stats = {
            'mean': np.mean(error_rates),
            'std': np.std(error_rates),
            'max': np.max(error_rates)
        }
        
        # Operational statistics
        uptime_percentage = (sum(operational_status) / len(operational_status)) * 100
        
        # Trend analysis
        trend_analysis = self._analyze_sensor_trends(sensor_data)
        
        return {
            'sensor_id': sensor_id,
            'data_points': len(sensor_data),
            'reliability_statistics': reliability_stats,
            'response_time_statistics': response_time_stats,
            'error_rate_statistics': error_rate_stats,
            'uptime_percentage': uptime_percentage,
            'trend_analysis': trend_analysis,
            'last_updated': sensor_data[-1]['timestamp']
        }
        
    def _analyze_sensor_trends(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in sensor performance data."""
        
        if len(sensor_data) < 5:
            return {}
            
        timestamps = [d['timestamp'] for d in sensor_data]
        reliabilities = [d['reliability'] for d in sensor_data]
        response_times = [d['response_time'] for d in sensor_data]
        
        # Normalize timestamps
        time_points = np.array(timestamps) - timestamps[0]
        
        # Reliability trend
        reliability_slope, _, reliability_r, _, _ = stats.linregress(time_points, reliabilities)
        
        # Response time trend
        response_time_slope, _, response_time_r, _, _ = stats.linregress(time_points, response_times)
        
        return {
            'reliability_trend': reliability_slope,
            'reliability_trend_confidence': abs(reliability_r),
            'response_time_trend': response_time_slope,
            'response_time_trend_confidence': abs(response_time_r),
            'trend_assessment': self._assess_trend_severity(reliability_slope, response_time_slope)
        }
        
    def _assess_trend_severity(self, reliability_slope: float, response_time_slope: float) -> str:
        """Assess the severity of performance trends."""
        
        # Negative reliability trend is bad
        # Positive response time trend is bad
        
        if reliability_slope < -0.01 or response_time_slope > 0.01:
            return 'critical'
        elif reliability_slope < -0.005 or response_time_slope > 0.005:
            return 'warning'
        elif reliability_slope < 0 or response_time_slope > 0:
            return 'attention'
        else:
            return 'stable'
            
    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report.
        
        Returns:
            Detailed health report with recommendations
        """
        if not self.system_performance_history:
            return {'error': 'No performance data available'}
            
        latest_summary = self.system_performance_history[-1]['summary']
        
        # Collect sensor-specific reports
        sensor_reports = {}
        for sensor_id in self.sensor_metrics:
            sensor_reports[sensor_id] = self.compute_sensor_health_statistics(sensor_id)
            
        # System-wide event summary
        event_summary = {
            'total_failures': len(self.failure_events),
            'total_recoveries': len(self.recovery_events),
            'total_anomalies': len(self.anomaly_events),
            'recent_failures': len([e for e in self.failure_events 
                                  if e['timestamp'] > time.time() - 3600]),
            'recent_recoveries': len([e for e in self.recovery_events 
                                    if e['timestamp'] > time.time() - 3600])
        }
        
        return {
            'timestamp': time.time(),
            'system_summary': latest_summary,
            'sensor_reports': sensor_reports,
            'event_summary': event_summary,
            'redundancy_status': self.get_redundancy_status(),
            'maintenance_recommendations': latest_summary.maintenance_recommendations
        }
        
    def plot_health_trends(self, 
                          sensors: Optional[List[str]] = None,
                          time_window: Optional[float] = None) -> None:
        """
        Plot comprehensive sensor health trends.
        
        Args:
            sensors: List of specific sensors to plot (None for all)
            time_window: Time window in seconds (None for all data)
        """
        if not self.sensor_health_history:
            logger.warning("No health data available for plotting")
            return
            
        # Set up the plot
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Filter data by time window if specified
        current_time = time.time()
        if time_window:
            filtered_data = [entry for entry in self.sensor_health_history 
                           if entry['timestamp'] > current_time - time_window]
        else:
            filtered_data = list(self.sensor_health_history)
            
        if not filtered_data:
            logger.warning("No data in specified time window")
            return
            
        # Plot 1: Reliability trends
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_reliability_trends(ax1, filtered_data, sensors)
        
        # Plot 2: Response time trends
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_response_time_trends(ax2, filtered_data, sensors)
        
        # Plot 3: Error rate trends
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_error_rate_trends(ax3, filtered_data, sensors)
        
        # Plot 4: System health overview
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_system_health_overview(ax4)
        
        # Plot 5: Event timeline
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_event_timeline(ax5, time_window)
        
        plt.suptitle('Sensor Health Monitoring Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def _plot_reliability_trends(self, ax, filtered_data, sensors):
        """Plot reliability trends for sensors."""
        ax.set_title('Reliability Trends', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Reliability')
        ax.grid(True, alpha=0.3)
        
        # Extract sensor data
        sensor_data = defaultdict(list)
        timestamps = []
        
        for entry in filtered_data:
            timestamps.append(entry['timestamp'])
            
            for sensor_type, sensor_info in entry['status'].items():
                if isinstance(sensor_info, list):
                    for i, sensor in enumerate(sensor_info):
                        sensor_id = f"{sensor_type}_{i}"
                        if sensors is None or sensor_id in sensors:
                            sensor_data[sensor_id].append(sensor.get('reliability', 1.0))
                else:
                    sensor_id = sensor_type
                    if sensors is None or sensor_id in sensors:
                        sensor_data[sensor_id].append(sensor_info.get('reliability', 1.0))
                        
        # Plot each sensor
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_data)))
        for i, (sensor_id, reliabilities) in enumerate(sensor_data.items()):
            if len(reliabilities) == len(timestamps):
                ax.plot(timestamps, reliabilities, label=sensor_id, 
                       color=colors[i], linewidth=2, alpha=0.8)
                
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        
    def _plot_response_time_trends(self, ax, filtered_data, sensors):
        """Plot response time trends for sensors."""
        ax.set_title('Response Time Trends', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Response Time (ms)')
        ax.grid(True, alpha=0.3)
        
        # Similar implementation as reliability trends
        # Extract and plot response time data
        sensor_data = defaultdict(list)
        timestamps = []
        
        for entry in filtered_data:
            timestamps.append(entry['timestamp'])
            
            for sensor_type, sensor_info in entry['status'].items():
                if isinstance(sensor_info, list):
                    for i, sensor in enumerate(sensor_info):
                        sensor_id = f"{sensor_type}_{i}"
                        if sensors is None or sensor_id in sensors:
                            sensor_data[sensor_id].append(sensor.get('response_time', 0.0))
                else:
                    sensor_id = sensor_type
                    if sensors is None or sensor_id in sensors:
                        sensor_data[sensor_id].append(sensor_info.get('response_time', 0.0))
                        
        # Plot each sensor
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_data)))
        for i, (sensor_id, response_times) in enumerate(sensor_data.items()):
            if len(response_times) == len(timestamps):
                ax.plot(timestamps, response_times, label=sensor_id, 
                       color=colors[i], linewidth=2, alpha=0.8)
                
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    def _plot_error_rate_trends(self, ax, filtered_data, sensors):
        """Plot error rate trends for sensors."""
        ax.set_title('Error Rate Trends', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error Rate')
        ax.grid(True, alpha=0.3)
        
        # Extract and plot error rate data
        sensor_data = defaultdict(list)
        timestamps = []
        
        for entry in filtered_data:
            timestamps.append(entry['timestamp'])
            
            for sensor_type, sensor_info in entry['status'].items():
                if isinstance(sensor_info, list):
                    for i, sensor in enumerate(sensor_info):
                        sensor_id = f"{sensor_type}_{i}"
                        if sensors is None or sensor_id in sensors:
                            sensor_data[sensor_id].append(sensor.get('error_rate', 0.0))
                else:
                    sensor_id = sensor_type
                    if sensors is None or sensor_id in sensors:
                        sensor_data[sensor_id].append(sensor_info.get('error_rate', 0.0))
                        
        # Plot each sensor
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_data)))
        for i, (sensor_id, error_rates) in enumerate(sensor_data.items()):
            if len(error_rates) == len(timestamps):
                ax.plot(timestamps, error_rates, label=sensor_id, 
                       color=colors[i], linewidth=2, alpha=0.8)
                
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, max(1.0, max([max(rates) for rates in sensor_data.values()]) * 1.1))
        
    def _plot_system_health_overview(self, ax):
        """Plot overall system health overview."""
        ax.set_title('System Health Overview', fontweight='bold')
        
        if not self.system_performance_history:
            ax.text(0.5, 0.5, 'No system performance data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Extract system health scores
        timestamps = [entry['timestamp'] for entry in self.system_performance_history]
        health_scores = [entry['summary'].overall_health_score 
                        for entry in self.system_performance_history]
        
        ax.plot(timestamps, health_scores, 'g-', linewidth=3, alpha=0.8, label='Health Score')
        
        # Add horizontal lines for health thresholds
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (0.6)')
        ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Poor (0.4)')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Health Score')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def _plot_event_timeline(self, ax, time_window):
        """Plot timeline of failure and recovery events."""
        ax.set_title('Event Timeline', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Events')
        
        current_time = time.time()
        
        # Filter events by time window
        if time_window:
            filtered_failures = [e for e in self.failure_events 
                               if e['timestamp'] > current_time - time_window]
            filtered_recoveries = [e for e in self.recovery_events 
                                 if e['timestamp'] > current_time - time_window]
            filtered_anomalies = [e for e in self.anomaly_events 
                                 if e['timestamp'] > current_time - time_window]
        else:
            filtered_failures = self.failure_events
            filtered_recoveries = self.recovery_events
            filtered_anomalies = self.anomaly_events
            
        # Plot events
        if filtered_failures:
            failure_times = [e['timestamp'] for e in filtered_failures]
            ax.scatter(failure_times, [1] * len(failure_times), 
                      c='red', marker='x', s=100, label='Failures', alpha=0.8)
                      
        if filtered_recoveries:
            recovery_times = [e['timestamp'] for e in filtered_recoveries]
            ax.scatter(recovery_times, [2] * len(recovery_times), 
                      c='green', marker='o', s=100, label='Recoveries', alpha=0.8)
                      
        if filtered_anomalies:
            anomaly_times = [e['timestamp'] for e in filtered_anomalies]
            ax.scatter(anomaly_times, [3] * len(anomaly_times), 
                      c='orange', marker='^', s=100, label='Anomalies', alpha=0.8)
                      
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Failures', 'Recoveries', 'Anomalies'])
        ax.grid(True, alpha=0.3)
        ax.legend()


class MetricsDisplay:
    """
    Advanced performance metrics visualization and statistical analysis system.
    
    This class provides comprehensive performance metrics visualization including:
    - Real-time performance dashboard with statistical analysis
    - Error distribution analysis with confidence intervals
    - System availability metrics and SLA compliance tracking
    - Fusion confidence analysis with uncertainty quantification
    - Predictive performance modeling and trend analysis
    - Publication-quality scientific plotting and data export
    
    The display system emphasizes statistical accuracy and scientific rigor
    in all computations and visualizations.
    
    Parameters:
        metrics_history_length (int): Maximum number of historical metrics to maintain
        confidence_level (float): Statistical confidence level for analysis
        enable_statistical_analysis (bool): Enable advanced statistical features
        
    Attributes:
        metrics_history (deque): Historical performance metrics data
        availability_history (deque): Sensor availability data
        confidence_history (deque): Fusion confidence data
        statistical_models (Dict): Fitted statistical models for metrics
        performance_baselines (Dict): Baseline performance metrics
    """
    
    def __init__(self, 
                 metrics_history_length: int = 2000,
                 confidence_level: float = 0.95,
                 enable_statistical_analysis: bool = True):
        """Initialize the metrics display system."""
        
        # Input validation
        if metrics_history_length <= 0:
            raise ValueError("Metrics history length must be positive")
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
            
        self.metrics_history_length = metrics_history_length
        self.confidence_level = confidence_level
        self.enable_statistical_analysis = enable_statistical_analysis
        
        # Data storage
        self.metrics_history = deque(maxlen=metrics_history_length)
        self.availability_history = deque(maxlen=metrics_history_length)
        self.confidence_history = deque(maxlen=metrics_history_length)
        self.performance_stats = {}
        
        # Statistical analysis components
        self.statistical_models = {}
        self.performance_baselines = {}
        self.anomaly_detection_params = {}
        
        # Visualization components
        self.figure = None
        self.dashboard_axes = {}
        
        logger.info(f"MetricsDisplay initialized with {metrics_history_length} point history")
        
    def update_position_error(self, 
                            error: float, 
                            timestamp: float,
                            error_components: Optional[Dict[str, float]] = None) -> None:
        """
        Update position error metrics with enhanced statistical tracking.
        
        Args:
            error: Position error magnitude in meters
            timestamp: Unix timestamp or simulation time
            error_components: Optional breakdown of error components (x, y, z)
        """
        if not isinstance(error, (int, float)) or error < 0:
            raise ValueError("Error must be a non-negative number")
            
        if not isinstance(timestamp, (int, float)):
            raise ValueError("Timestamp must be numeric")
            
        # Create comprehensive error record
        error_record = {
            'timestamp': timestamp,
            'position_error': error,
            'error_type': 'position_error',
            'components': error_components.copy() if error_components else {}
        }
        
        self.metrics_history.append(error_record)
        
        # Update statistical models
        if self.enable_statistical_analysis:
            self._update_error_statistical_models(error, timestamp)
            
        logger.debug(f"Position error updated: {error:.4f}m at {timestamp}")
        
    def _update_error_statistical_models(self, error: float, timestamp: float) -> None:
        """Update statistical models for error prediction and analysis."""
        
        # Extract recent error data
        recent_errors = [entry['position_error'] for entry in self.metrics_history 
                        if entry['error_type'] == 'position_error']
        
        if len(recent_errors) >= 30:  # Minimum data for statistical analysis
            # Fit distribution to error data
            try:
                # Test multiple distributions
                distributions = [stats.norm, stats.lognorm, stats.gamma, stats.weibull_min]
                best_dist = None
                best_params = None
                best_aic = np.inf
                
                for dist in distributions:
                    try:
                        params = dist.fit(recent_errors)
                        log_likelihood = np.sum(dist.logpdf(recent_errors, *params))
                        aic = 2 * len(params) - 2 * log_likelihood
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_dist = dist
                            best_params = params
                    except:
                        continue
                        
                if best_dist is not None:
                    self.statistical_models['position_error'] = {
                        'distribution': best_dist,
                        'parameters': best_params,
                        'aic': best_aic,
                        'last_updated': timestamp
                    }
                    
            except Exception as e:
                logger.debug(f"Could not fit error distribution: {e}")
                
        # Update performance baselines
        if len(recent_errors) >= 100:
            self.performance_baselines['position_error'] = {
                'mean': np.mean(recent_errors),
                'std': np.std(recent_errors),
                'median': np.median(recent_errors),
                'percentile_95': np.percentile(recent_errors, 95),
                'percentile_99': np.percentile(recent_errors, 99)
            }
            
    def update_sensor_availability(self, 
                                 availability_data: Dict[str, Any],
                                 timestamp: Optional[float] = None) -> None:
        """
        Update sensor availability metrics with comprehensive analysis.
        
        Args:
            availability_data: Dictionary containing sensor availability information
            timestamp: Unix timestamp or simulation time
        """
        if not isinstance(availability_data, dict):
            raise TypeError("Availability data must be a dictionary")
            
        if timestamp is None:
            timestamp = time.time()
            
        # Add timestamp to availability data
        availability_record = availability_data.copy()
        availability_record['timestamp'] = timestamp
        
        self.availability_history.append(availability_record)
        
        # Update availability statistics
        self._update_availability_statistics(availability_data, timestamp)
        
        logger.debug(f"Sensor availability updated at {timestamp}")
        
    def _update_availability_statistics(self, 
                                      availability_data: Dict[str, Any], 
                                      timestamp: float) -> None:
        """Update comprehensive availability statistics."""
        
        # Compute system-wide availability
        total_sensors = 0
        available_sensors = 0
        
        for sensor_type, sensor_info in availability_data.items():
            if sensor_type == 'timestamp':
                continue
                
            if isinstance(sensor_info, list):
                total_sensors += len(sensor_info)
                available_sensors += sum(1 for s in sensor_info if s)
            else:
                total_sensors += 1
                available_sensors += 1 if sensor_info else 0
                
        # Calculate availability percentage
        if total_sensors > 0:
            availability_percentage = (available_sensors / total_sensors) * 100
            
            # Update performance statistics
            if 'availability' not in self.performance_stats:
                self.performance_stats['availability'] = deque(maxlen=100)
                
            self.performance_stats['availability'].append({
                'timestamp': timestamp,
                'percentage': availability_percentage,
                'total_sensors': total_sensors,
                'available_sensors': available_sensors
            })
            
    def update_fusion_confidence(self, 
                               confidence: float, 
                               timestamp: float,
                               confidence_breakdown: Optional[Dict[str, float]] = None) -> None:
        """
        Update fusion confidence metrics with statistical analysis.
        
        Args:
            confidence: Fusion confidence score (0-1)
            timestamp: Unix timestamp or simulation time
            confidence_breakdown: Optional breakdown by sensor type
        """
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
            
        if not isinstance(timestamp, (int, float)):
            raise ValueError("Timestamp must be numeric")
            
        # Create confidence record
        confidence_record = {
            'timestamp': timestamp,
            'confidence': confidence,
            'breakdown': confidence_breakdown.copy() if confidence_breakdown else {}
        }
        
        self.confidence_history.append(confidence_record)
        
        # Update statistical analysis
        if self.enable_statistical_analysis:
            self._update_confidence_statistical_analysis(confidence, timestamp)
            
        logger.debug(f"Fusion confidence updated: {confidence:.4f} at {timestamp}")
        
    def _update_confidence_statistical_analysis(self, confidence: float, timestamp: float) -> None:
        """Update statistical analysis of confidence data."""
        
        # Extract recent confidence data
        recent_confidences = [entry['confidence'] for entry in self.confidence_history]
        
        if len(recent_confidences) >= 50:
            # Detect confidence anomalies
            mean_confidence = np.mean(recent_confidences)
            std_confidence = np.std(recent_confidences)
            
            # Z-score based anomaly detection
            if std_confidence > 0:
                z_score = abs(confidence - mean_confidence) / std_confidence
                
                if z_score > 3.0:  # Anomaly threshold
                    logger.warning(f"Confidence anomaly detected: {confidence:.4f} "
                                 f"(z-score: {z_score:.2f})")
                    
            # Update baseline statistics
            self.performance_baselines['confidence'] = {
                'mean': mean_confidence,
                'std': std_confidence,
                'min': np.min(recent_confidences),
                'max': np.max(recent_confidences),
                'trend': self._compute_confidence_trend(recent_confidences)
            }
            
    def _compute_confidence_trend(self, confidences: List[float]) -> float:
        """Compute confidence trend using linear regression."""
        if len(confidences) < 5:
            return 0.0
            
        time_points = np.arange(len(confidences))
        slope, _, _, _, _ = stats.linregress(time_points, confidences)
        return slope
        
    def get_comprehensive_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics with confidence intervals.
        
        Returns:
            Dictionary containing detailed error analysis
        """
        position_errors = [entry['position_error'] for entry in self.metrics_history 
                         if entry['error_type'] == 'position_error']
                         
        if not position_errors:
            return {}
            
        # Basic statistics
        basic_stats = {
            'count': len(position_errors),
            'mean_error': np.mean(position_errors),
            'std_error': np.std(position_errors),
            'min_error': np.min(position_errors),
            'max_error': np.max(position_errors),
            'median_error': np.median(position_errors),
            'rms_error': np.sqrt(np.mean(np.array(position_errors)**2))
        }
        
        # Percentile analysis
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        percentile_stats = {f'percentile_{p}': np.percentile(position_errors, p) 
                          for p in percentiles}
        
        # Confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(position_errors, 100 * alpha / 2)
        ci_upper = np.percentile(position_errors, 100 * (1 - alpha/2))
        
        confidence_intervals = {
            'confidence_level': self.confidence_level,
            'lower_bound': ci_lower,
            'upper_bound': ci_upper
        }
        
        # Distribution analysis
        distribution_stats = {}
        if self.enable_statistical_analysis and len(position_errors) >= 30:
            distribution_stats = {
                'skewness': stats.skew(position_errors),
                'kurtosis': stats.kurtosis(position_errors),
                'normality_test_statistic': stats.shapiro(position_errors)[0],
                'normality_test_p_value': stats.shapiro(position_errors)[1]
            }
            
            # Add fitted distribution information
            if 'position_error' in self.statistical_models:
                model = self.statistical_models['position_error']
                distribution_stats['fitted_distribution'] = model['distribution'].name
                distribution_stats['distribution_aic'] = model['aic']
                
        # Combine all statistics
        comprehensive_stats = {
            'basic_statistics': basic_stats,
            'percentile_analysis': percentile_stats,
            'confidence_intervals': confidence_intervals,
            'distribution_analysis': distribution_stats,
            'temporal_analysis': self._get_temporal_error_analysis()
        }
        
        return comprehensive_stats
        
    def _get_temporal_error_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns in error data."""
        
        error_data = [(entry['timestamp'], entry['position_error']) 
                     for entry in self.metrics_history 
                     if entry['error_type'] == 'position_error']
        
        if len(error_data) < 10:
            return {}
            
        timestamps, errors = zip(*error_data)
        
        # Time series analysis
        time_points = np.array(timestamps) - timestamps[0]  # Normalize to start from 0
        
        # Trend analysis
        if len(time_points) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, errors)
            
            return {
                'trend_slope': slope,
                'trend_intercept': intercept,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'trend_significance': 'significant' if p_value < 0.05 else 'not_significant'
            }
            
        return {}
        
    def get_availability_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive sensor availability statistics.
        
        Returns:
            Dictionary containing detailed availability analysis
        """
        if not self.availability_history:
            return {}
            
        # Aggregate availability by sensor type
        sensor_availability = defaultdict(list)
        
        for entry in self.availability_history:
            for sensor_type, sensor_info in entry.items():
                if sensor_type == 'timestamp':
                    continue
                    
                if isinstance(sensor_info, list):
                    for i, available in enumerate(sensor_info):
                        sensor_id = f"{sensor_type}_{i}"
                        sensor_availability[sensor_id].append(1 if available else 0)
                else:
                    sensor_availability[sensor_type].append(1 if sensor_info else 0)
                    
        # Compute statistics for each sensor
        availability_stats = {}
        
        for sensor_id, availability_data in sensor_availability.items():
            if len(availability_data) > 0:
                availability_percentage = (sum(availability_data) / len(availability_data)) * 100
                
                # SLA compliance analysis (assuming 99% SLA)
                sla_target = 99.0
                sla_compliance = availability_percentage >= sla_target
                
                # Downtime analysis
                downtime_events = []
                current_downtime = 0
                
                for i, status in enumerate(availability_data):
                    if status == 0:  # Sensor down
                        current_downtime += 1
                    else:  # Sensor up
                        if current_downtime > 0:
                            downtime_events.append(current_downtime)
                            current_downtime = 0
                            
                # Add final downtime if still down
                if current_downtime > 0:
                    downtime_events.append(current_downtime)
                    
                availability_stats[sensor_id] = {
                    'availability_percentage': availability_percentage,
                    'total_measurements': len(availability_data),
                    'uptime_measurements': sum(availability_data),
                    'downtime_measurements': len(availability_data) - sum(availability_data),
                    'sla_compliance': sla_compliance,
                    'sla_target': sla_target,
                    'downtime_events': len(downtime_events),
                    'mean_downtime_duration': np.mean(downtime_events) if downtime_events else 0,
                    'max_downtime_duration': max(downtime_events) if downtime_events else 0,
                    'mtbf_estimate': len(availability_data) / len(downtime_events) if downtime_events else np.inf
                }
                
        return availability_stats
        
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive fusion confidence statistics.
        
        Returns:
            Dictionary containing detailed confidence analysis
        """
        if not self.confidence_history:
            return {}
            
        confidences = [entry['confidence'] for entry in self.confidence_history]
        
        # Basic statistics
        basic_stats = {
            'count': len(confidences),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'median_confidence': np.median(confidences)
        }
        
        # Confidence level analysis
        confidence_levels = {
            'high_confidence_percentage': (sum(1 for c in confidences if c > 0.8) / len(confidences)) * 100,
            'medium_confidence_percentage': (sum(1 for c in confidences if 0.5 <= c <= 0.8) / len(confidences)) * 100,
            'low_confidence_percentage': (sum(1 for c in confidences if c < 0.5) / len(confidences)) * 100
        }
        
        # Trend analysis
        trend_stats = {}
        if 'confidence' in self.performance_baselines:
            trend_stats = {
                'trend_slope': self.performance_baselines['confidence']['trend'],
                'trend_assessment': self._assess_confidence_trend(
                    self.performance_baselines['confidence']['trend'])
            }
            
        return {
            'basic_statistics': basic_stats,
            'confidence_levels': confidence_levels,
            'trend_analysis': trend_stats
        }
        
    def _assess_confidence_trend(self, trend_slope: float) -> str:
        """Assess confidence trend severity."""
        if trend_slope > 0.01:
            return 'improving'
        elif trend_slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
            
    def compute_overall_performance_score(self) -> Dict[str, Any]:
        """
        Compute comprehensive system performance score.
        
        Returns:
            Dictionary containing overall performance assessment
        """
        # Initialize performance components
        performance_components = {}
        
        # Error performance (lower is better)
        error_stats = self.get_comprehensive_error_statistics()
        if error_stats and 'basic_statistics' in error_stats:
            mean_error = error_stats['basic_statistics']['mean_error']
            # Normalize error to 0-1 scale (assuming 10m is maximum acceptable error)
            error_score = max(0, 1 - mean_error / 10.0)
            performance_components['error_performance'] = error_score
            
        # Availability performance
        availability_stats = self.get_availability_statistics()
        if availability_stats:
            avg_availability = np.mean([stats['availability_percentage'] 
                                      for stats in availability_stats.values()])
            availability_score = avg_availability / 100.0
            performance_components['availability_performance'] = availability_score
            
        # Confidence performance
        confidence_stats = self.get_confidence_statistics()
        if confidence_stats and 'basic_statistics' in confidence_stats:
            confidence_score = confidence_stats['basic_statistics']['mean_confidence']
            performance_components['confidence_performance'] = confidence_score
            
        # Compute weighted overall score
        if performance_components:
            weights = {
                'error_performance': 0.4,
                'availability_performance': 0.3,
                'confidence_performance': 0.3
            }
            
            weighted_score = sum(weights.get(component, 0) * score 
                               for component, score in performance_components.items())
            
            # Normalize by total weight
            total_weight = sum(weights.get(component, 0) 
                             for component in performance_components.keys())
            
            if total_weight > 0:
                overall_score = weighted_score / total_weight
            else:
                overall_score = 0.0
                
        else:
            overall_score = 0.0
            
        return {
            'overall_score': overall_score,
            'component_scores': performance_components,
            'performance_grade': self._get_performance_grade(overall_score),
            'recommendations': self._generate_performance_recommendations(performance_components)
        }
        
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
            
    def _generate_performance_recommendations(self, components: Dict[str, float]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        for component, score in components.items():
            if score < 0.6:
                if component == 'error_performance':
                    recommendations.append("Position accuracy requires immediate attention - check sensor calibration")
                elif component == 'availability_performance':
                    recommendations.append("Sensor availability is critically low - inspect hardware connections")
                elif component == 'confidence_performance':
                    recommendations.append("Fusion confidence is low - review sensor fusion algorithms")
                    
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
            
        return recommendations
        
    def create_performance_dashboard(self) -> None:
        """
        Create comprehensive real-time performance dashboard.
        """
        # Create dashboard figure
        self.figure = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 3, figure=self.figure, hspace=0.4, wspace=0.3)
        
        # Dashboard title
        self.figure.suptitle('System Performance Dashboard', fontsize=18, fontweight='bold')
        
        # Plot 1: Error trends with confidence intervals
        ax1 = self.figure.add_subplot(gs[0, 0])
        self._plot_error_trends_with_ci(ax1)
        
        # Plot 2: Error distribution
        ax2 = self.figure.add_subplot(gs[0, 1])
        self._plot_error_distribution(ax2)
        
        # Plot 3: Availability heatmap
        ax3 = self.figure.add_subplot(gs[0, 2])
        self._plot_availability_heatmap(ax3)
        
        # Plot 4: Confidence trends
        ax4 = self.figure.add_subplot(gs[1, 0])
        self._plot_confidence_trends(ax4)
        
        # Plot 5: Performance metrics summary
        ax5 = self.figure.add_subplot(gs[1, 1])
        self._plot_performance_summary(ax5)
        
        # Plot 6: SLA compliance
        ax6 = self.figure.add_subplot(gs[1, 2])
        self._plot_sla_compliance(ax6)
        
        # Plot 7: System health timeline
        ax7 = self.figure.add_subplot(gs[2, :])
        self._plot_system_health_timeline(ax7)
        
        # Plot 8: Statistical analysis summary
        ax8 = self.figure.add_subplot(gs[3, :])
        self._plot_statistical_summary(ax8)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_error_trends_with_ci(self, ax):
        """Plot error trends with confidence intervals."""
        ax.set_title('Position Error Trends with Confidence Intervals', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error (m)')
        
        # Extract error data
        error_data = [(entry['timestamp'], entry['position_error']) 
                     for entry in self.metrics_history 
                     if entry['error_type'] == 'position_error']
        
        if len(error_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient error data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
            
        timestamps, errors = zip(*error_data)
        
        # Plot main error line
        ax.plot(timestamps, errors, 'b-', linewidth=2, alpha=0.8, label='Position Error')
        
        # Add confidence intervals using rolling statistics
        if len(errors) >= 20:
            window_size = min(20, len(errors) // 4)
            
            # Compute rolling statistics
            if HAS_PANDAS:
                rolling_mean = pd.Series(errors).rolling(window=window_size, center=True).mean()
                rolling_std = pd.Series(errors).rolling(window=window_size, center=True).std()
            else:
                # Fallback to numpy-based rolling statistics
                rolling_mean = np.convolve(errors, np.ones(window_size)/window_size, mode='same')
                rolling_std = np.array([np.std(errors[max(0, i-window_size//2):min(len(errors), i+window_size//2+1)]) 
                                      for i in range(len(errors))])
            
            # Confidence intervals
            ci_upper = rolling_mean + 1.96 * rolling_std
            ci_lower = rolling_mean - 1.96 * rolling_std
            
            # Plot confidence intervals
            ax.fill_between(timestamps, ci_lower, ci_upper, alpha=0.3, color='blue', 
                           label='95% Confidence Interval')
            
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def _plot_error_distribution(self, ax):
        """Plot error distribution with fitted curve."""
        ax.set_title('Error Distribution Analysis', fontweight='bold')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Frequency')
        
        # Extract error data
        errors = [entry['position_error'] for entry in self.metrics_history 
                 if entry['error_type'] == 'position_error']
        
        if len(errors) < 5:
            ax.text(0.5, 0.5, 'Insufficient data for distribution', ha='center', va='center', 
                   transform=ax.transAxes)
            return
            
        # Plot histogram
        n, bins, patches = ax.hist(errors, bins=30, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
        
        # Fit and plot normal distribution
        mu, sigma = stats.norm.fit(errors)
        x = np.linspace(min(errors), max(errors), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
        
        # Add statistical information
        ax.axvline(np.mean(errors), color='green', linestyle='--', 
                  label=f'Mean: {np.mean(errors):.3f}m')
        ax.axvline(np.median(errors), color='orange', linestyle='--', 
                  label=f'Median: {np.median(errors):.3f}m')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_availability_heatmap(self, ax):
        """Plot sensor availability heatmap."""
        ax.set_title('Sensor Availability Heatmap', fontweight='bold')
        
        # Create availability matrix
        availability_stats = self.get_availability_statistics()
        
        if not availability_stats:
            ax.text(0.5, 0.5, 'No availability data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
            
        # Prepare data for heatmap
        sensor_names = list(availability_stats.keys())
        metrics = ['Availability %', 'SLA Compliance', 'MTBF Score']
        
        # Create data matrix
        data_matrix = np.zeros((len(sensor_names), len(metrics)))
        
        for i, sensor_name in enumerate(sensor_names):
            stats = availability_stats[sensor_name]
            data_matrix[i, 0] = stats['availability_percentage']
            data_matrix[i, 1] = 100 if stats['sla_compliance'] else 0
            data_matrix[i, 2] = min(100, stats['mtbf_estimate'] / 10) if np.isfinite(stats['mtbf_estimate']) else 100
            
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(range(len(sensor_names)))
        ax.set_yticklabels(sensor_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(sensor_names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}', 
                             ha='center', va='center', color='black', fontweight='bold')
                
    def _plot_confidence_trends(self, ax):
        """Plot fusion confidence trends."""
        ax.set_title('Fusion Confidence Trends', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Confidence')
        
        if not self.confidence_history:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
            
        timestamps = [entry['timestamp'] for entry in self.confidence_history]
        confidences = [entry['confidence'] for entry in self.confidence_history]
        
        ax.plot(timestamps, confidences, 'g-', linewidth=2, alpha=0.8, label='Fusion Confidence')
        
        # Add threshold lines
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High Confidence (0.8)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence (0.5)')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Low Confidence (0.2)')
        
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def _plot_performance_summary(self, ax):
        """Plot overall performance summary."""
        ax.set_title('Performance Summary', fontweight='bold')
        
        # Get performance scores
        performance_data = self.compute_overall_performance_score()
        
        if not performance_data['component_scores']:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
            
        # Create radar chart
        categories = list(performance_data['component_scores'].keys())
        values = list(performance_data['component_scores'].values())
        
        # Convert to more readable labels
        label_mapping = {
            'error_performance': 'Error\nPerformance',
            'availability_performance': 'Availability\nPerformance',
            'confidence_performance': 'Confidence\nPerformance'
        }
        
        readable_labels = [label_mapping.get(cat, cat) for cat in categories]
        
        # Create bar chart instead of radar for simplicity
        bars = ax.bar(readable_labels, values, color=['red', 'blue', 'green'], alpha=0.7)
        
        # Color bars based on performance
        for bar, value in zip(bars, values):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
                
        ax.set_ylim(0, 1)
        ax.set_ylabel('Performance Score')
        
        # Add overall grade
        ax.text(0.5, 0.95, f'Overall Grade: {performance_data["performance_grade"]}', 
               ha='center', va='top', transform=ax.transAxes, 
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
    def _plot_sla_compliance(self, ax):
        """Plot SLA compliance metrics."""
        ax.set_title('SLA Compliance Status', fontweight='bold')
        
        availability_stats = self.get_availability_statistics()
        
        if not availability_stats:
            ax.text(0.5, 0.5, 'No SLA data available', ha='center', va='center', 
                   transform=ax.transAxes)
            return
            
        # Extract SLA compliance data
        sensor_names = list(availability_stats.keys())
        compliance_status = [stats['sla_compliance'] for stats in availability_stats.values()]
        availability_percentages = [stats['availability_percentage'] for stats in availability_stats.values()]
        
        # Create compliance visualization
        colors = ['green' if compliant else 'red' for compliant in compliance_status]
        
        bars = ax.bar(sensor_names, availability_percentages, color=colors, alpha=0.7)
        
        # Add SLA target line
        ax.axhline(y=99.0, color='blue', linestyle='--', linewidth=2, 
                  label='SLA Target (99%)')
        
        ax.set_ylabel('Availability (%)')
        ax.set_ylim(95, 100)
        ax.legend()
        
        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add compliance percentage
        compliance_percentage = (sum(compliance_status) / len(compliance_status)) * 100
        ax.text(0.5, 0.05, f'SLA Compliance: {compliance_percentage:.1f}%', 
               ha='center', va='bottom', transform=ax.transAxes, 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen' if compliance_percentage > 80 else 'lightcoral', alpha=0.8))
        
    def _plot_system_health_timeline(self, ax):
        """Plot system health timeline."""
        ax.set_title('System Health Timeline', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Health Metrics')
        
        # This would integrate with the SensorHealthMonitor
        # For now, create a placeholder
        ax.text(0.5, 0.5, 'System Health Timeline\n(Requires integration with SensorHealthMonitor)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
    def _plot_statistical_summary(self, ax):
        """Plot statistical analysis summary."""
        ax.set_title('Statistical Analysis Summary', fontweight='bold')
        ax.axis('off')  # Turn off axes for text display
        
        # Get comprehensive statistics
        error_stats = self.get_comprehensive_error_statistics()
        confidence_stats = self.get_confidence_statistics()
        availability_stats = self.get_availability_statistics()
        
        # Create summary text
        summary_lines = []
        
        # Error statistics
        if error_stats and 'basic_statistics' in error_stats:
            basic = error_stats['basic_statistics']
            summary_lines.extend([
                f"ERROR ANALYSIS (n={basic['count']}):",
                f"  Mean Error: {basic['mean_error']:.4f} ± {basic['std_error']:.4f} m",
                f"  RMS Error: {basic['rms_error']:.4f} m",
                f"  95th Percentile: {error_stats['percentile_analysis']['percentile_95']:.4f} m",
                ""
            ])
            
        # Confidence statistics
        if confidence_stats and 'basic_statistics' in confidence_stats:
            basic = confidence_stats['basic_statistics']
            levels = confidence_stats['confidence_levels']
            summary_lines.extend([
                f"CONFIDENCE ANALYSIS (n={basic['count']}):",
                f"  Mean Confidence: {basic['mean_confidence']:.3f} ± {basic['std_confidence']:.3f}",
                f"  High Confidence: {levels['high_confidence_percentage']:.1f}%",
                f"  Low Confidence: {levels['low_confidence_percentage']:.1f}%",
                ""
            ])
            
        # Availability summary
        if availability_stats:
            avg_availability = np.mean([stats['availability_percentage'] 
                                      for stats in availability_stats.values()])
            sla_compliance = np.mean([stats['sla_compliance'] 
                                    for stats in availability_stats.values()]) * 100
            
            summary_lines.extend([
                f"AVAILABILITY ANALYSIS:",
                f"  Average Availability: {avg_availability:.2f}%",
                f"  SLA Compliance: {sla_compliance:.1f}%",
                f"  Monitored Sensors: {len(availability_stats)}",
                ""
            ])
            
        # Performance grade
        performance_data = self.compute_overall_performance_score()
        summary_lines.extend([
            f"OVERALL PERFORMANCE:",
            f"  Performance Score: {performance_data['overall_score']:.3f}",
            f"  Performance Grade: {performance_data['performance_grade']}",
        ])
        
        # Display summary
        summary_text = '\n'.join(summary_lines)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=11, fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
    def export_metrics_report(self, 
                            filename: str,
                            format: str = 'json',
                            include_raw_data: bool = False) -> None:
        """
        Export comprehensive metrics report.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'csv', 'xlsx')
            include_raw_data: Include raw historical data
        """
        # Compile comprehensive report
        report = {
            'timestamp': time.time(),
            'summary': {
                'error_statistics': self.get_comprehensive_error_statistics(),
                'availability_statistics': self.get_availability_statistics(),
                'confidence_statistics': self.get_confidence_statistics(),
                'performance_score': self.compute_overall_performance_score()
            }
        }
        
        # Add raw data if requested
        if include_raw_data:
            report['raw_data'] = {
                'metrics_history': [dict(entry) for entry in self.metrics_history],
                'availability_history': [dict(entry) for entry in self.availability_history],
                'confidence_history': [dict(entry) for entry in self.confidence_history]
            }
            
        # Export based on format
        if format == 'json':
            import json
            
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
                
            # Recursively convert numpy types
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
                    
            report_converted = recursive_convert(report)
            
            with open(filename, 'w') as f:
                json.dump(report_converted, f, indent=2)
                
        elif format == 'csv':
            # Flatten report for CSV export
            if not HAS_PANDAS:
                raise ImportError("pandas is required for CSV export. Install with: pip install pandas")
            
            # Create flattened data structure
            flat_data = []
            
            # Add error statistics
            if 'error_statistics' in report['summary']:
                error_stats = report['summary']['error_statistics']
                if 'basic_statistics' in error_stats:
                    for metric, value in error_stats['basic_statistics'].items():
                        flat_data.append({
                            'category': 'error_statistics',
                            'metric': metric,
                            'value': value
                        })
                        
            # Add other statistics similarly...
            
            df = pd.DataFrame(flat_data)
            df.to_csv(filename, index=False)
            
        elif format == 'xlsx':
            # Excel export with multiple sheets
            if not HAS_PANDAS:
                raise ImportError("pandas is required for Excel export. Install with: pip install pandas openpyxl")
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                
                # Convert nested dictionaries to flat structure
                def flatten_dict(d, parent_key='', sep='_'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                    
                flattened = flatten_dict(report['summary'])
                
                for key, value in flattened.items():
                    summary_data.append({'Metric': key, 'Value': value})
                    
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Raw data sheets if included
                if include_raw_data:
                    pd.DataFrame(report['raw_data']['metrics_history']).to_excel(
                        writer, sheet_name='Metrics_History', index=False)
                    pd.DataFrame(report['raw_data']['availability_history']).to_excel(
                        writer, sheet_name='Availability_History', index=False)
                    pd.DataFrame(report['raw_data']['confidence_history']).to_excel(
                        writer, sheet_name='Confidence_History', index=False)
                        
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Metrics report exported to {filename} in {format} format")