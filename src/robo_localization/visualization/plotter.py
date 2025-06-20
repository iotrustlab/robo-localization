"""
Scientific visualization module for real-time trajectory plotting and analysis.

This module provides comprehensive real-time 3D visualization capabilities for 
robotic localization systems, with emphasis on statistical analysis, error 
quantification, and scientific data visualization.

Classes:
    RealTimeVisualizer: Real-time 3D trajectory visualization with statistical overlays
    TrajectoryPlotter: Advanced trajectory analysis and comparison tools

Author: Automated Code Generation System
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from collections import deque
import scipy.ndimage
import scipy.stats
import warnings
from dataclasses import dataclass
import logging

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryStatistics:
    """Container for trajectory statistical data."""
    rmse: float
    max_error: float
    mean_error: float
    std_error: float
    median_error: float
    percentile_95: float
    trajectory_length: float
    velocity_statistics: Dict[str, float]
    acceleration_statistics: Dict[str, float]


class RealTimeVisualizer:
    """
    Advanced real-time 3D visualization system for robotic trajectory analysis.
    
    This class provides comprehensive real-time visualization capabilities including:
    - Multi-trajectory comparison with uncertainty quantification
    - Statistical error analysis and confidence intervals
    - Adaptive viewport management and data-driven axis scaling
    - Sensor health monitoring with reliability trend analysis
    - Performance metrics overlay with real-time computation
    
    The visualizer employs scientific plotting standards with proper error bars,
    confidence ellipses, and statistical overlays for quantitative analysis.
    
    Parameters:
        update_rate (float): Visualization update frequency in Hz. Default: 10.0
        trail_length (int): Maximum number of trajectory points to display. Default: 100
        confidence_level (float): Statistical confidence level for uncertainty visualization. Default: 0.95
        enable_statistics (bool): Enable real-time statistical analysis overlay. Default: True
        adaptive_scaling (bool): Enable adaptive viewport scaling. Default: True
        
    Attributes:
        position_history (deque): Circular buffer for position data
        ground_truth_history (deque): Ground truth trajectory buffer
        estimated_history (deque): Estimated trajectory buffer
        uncertainty_history (deque): Uncertainty covariance data
        velocity_history (deque): Velocity vector data
        acceleration_history (deque): Acceleration vector data
    """
    
    def __init__(self, 
                 update_rate: float = 10.0, 
                 trail_length: int = 100,
                 confidence_level: float = 0.95,
                 enable_statistics: bool = True,
                 adaptive_scaling: bool = True):
        """Initialize the real-time visualization system."""
        
        # Input validation
        if update_rate <= 0:
            raise ValueError("Update rate must be positive")
        if trail_length <= 0:
            raise ValueError("Trail length must be positive")
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
            
        self.update_rate = update_rate
        self.trail_length = trail_length
        self.confidence_level = confidence_level
        self.enable_statistics = enable_statistics
        self.adaptive_scaling = adaptive_scaling
        
        # Trajectory data buffers
        self.position_history = deque(maxlen=trail_length)
        self.ground_truth_history = deque(maxlen=trail_length)
        self.estimated_history = deque(maxlen=trail_length)
        self.uncertainty_history = deque(maxlen=trail_length)
        self.velocity_history = deque(maxlen=trail_length)
        self.acceleration_history = deque(maxlen=trail_length)
        
        # Temporal data
        self.timestamp_history = deque(maxlen=trail_length)
        
        # Sensor status and health monitoring
        self.sensor_status = {}
        self.sensor_reliability_history = {}
        
        # Performance metrics
        self.error_statistics = {}
        self.computation_times = deque(maxlen=50)
        
        # Initialize matplotlib components
        plt.ion()  # Enable interactive mode
        self._initialize_visualization()
        
        logger.info(f"RealTimeVisualizer initialized with update_rate={update_rate}Hz, "
                   f"trail_length={trail_length}, confidence_level={confidence_level}")
        
    def _initialize_visualization(self):
        """Initialize matplotlib figure and subplot layout."""
        self.figure = plt.figure(figsize=(16, 12))
        
        # Create subplot grid for comprehensive display
        gs = gridspec.GridSpec(3, 3, figure=self.figure, hspace=0.3, wspace=0.3)
        
        # Main 3D trajectory plot
        self.axes_3d = self.figure.add_subplot(gs[:2, :2], projection='3d')
        
        # Statistical analysis plots
        self.axes_error = self.figure.add_subplot(gs[0, 2])
        self.axes_velocity = self.figure.add_subplot(gs[1, 2])
        self.axes_metrics = self.figure.add_subplot(gs[2, :])
        
        self._setup_3d_plot()
        self._setup_analysis_plots()
        
    def _setup_3d_plot(self):
        """Configure the main 3D trajectory plot with scientific styling."""
        self.axes_3d.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        self.axes_3d.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        self.axes_3d.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
        self.axes_3d.set_title('Real-Time 3D Trajectory Analysis', fontsize=14, fontweight='bold')
        
        # Set scientific axis formatting
        self.axes_3d.ticklabel_format(style='scientific', scilimits=(-2, 2))
        
        # Initialize with reasonable limits
        self.axes_3d.set_xlim([-20, 20])
        self.axes_3d.set_ylim([-20, 20])
        self.axes_3d.set_zlim([0, 10])
        
        # Grid for better depth perception
        self.axes_3d.grid(True, alpha=0.3)
        
    def _setup_analysis_plots(self):
        """Configure statistical analysis subplot panels."""
        # Error analysis plot
        self.axes_error.set_title('Position Error Analysis', fontsize=10, fontweight='bold')
        self.axes_error.set_xlabel('Time (s)')
        self.axes_error.set_ylabel('Error (m)')
        self.axes_error.grid(True, alpha=0.3)
        
        # Velocity analysis plot
        self.axes_velocity.set_title('Velocity Profile', fontsize=10, fontweight='bold')
        self.axes_velocity.set_xlabel('Time (s)')
        self.axes_velocity.set_ylabel('Speed (m/s)')
        self.axes_velocity.grid(True, alpha=0.3)
        
        # Performance metrics plot
        self.axes_metrics.set_title('System Performance Metrics', fontsize=10, fontweight='bold')
        self.axes_metrics.set_xlabel('Metric Type')
        self.axes_metrics.set_ylabel('Normalized Score')
        self.axes_metrics.grid(True, alpha=0.3)
        
    def update_position(self, 
                       position: np.ndarray, 
                       estimated: bool = True,
                       covariance: Optional[np.ndarray] = None,
                       timestamp: Optional[float] = None) -> None:
        """
        Update position data with comprehensive validation and processing.
        
        Args:
            position: 3D position vector [x, y, z] in meters
            estimated: True for estimated position, False for ground truth
            covariance: 3x3 covariance matrix for uncertainty quantification
            timestamp: Unix timestamp or simulation time
            
        Raises:
            ValueError: If position vector is invalid
            TypeError: If input types are incorrect
        """
        start_time = time.perf_counter()
        
        # Input validation
        if not isinstance(position, np.ndarray):
            position = np.array(position, dtype=np.float64)
            
        if position.shape != (3,):
            raise ValueError(f"Position must be 3D vector, got shape {position.shape}")
            
        if not np.all(np.isfinite(position)):
            raise ValueError("Position contains invalid values (inf/nan)")
            
        # Timestamp handling
        if timestamp is None:
            timestamp = time.time()
            
        # Store position data
        if estimated:
            self.estimated_history.append(position.copy())
            self.position_history.append(position.copy())  # Backward compatibility
            
            # Store uncertainty data if provided
            if covariance is not None:
                if covariance.shape != (3, 3):
                    raise ValueError(f"Covariance must be 3x3 matrix, got {covariance.shape}")
                self.uncertainty_history.append(covariance.copy())
            else:
                # Default uncertainty
                self.uncertainty_history.append(np.eye(3) * 0.1)
                
        else:
            self.ground_truth_history.append(position.copy())
            
        self.timestamp_history.append(timestamp)
        
        # Compute velocity and acceleration
        self._compute_kinematic_derivatives()
        
        # Update error statistics
        if estimated and len(self.ground_truth_history) > 0:
            self._update_error_statistics()
            
        # Track computation time
        computation_time = time.perf_counter() - start_time
        self.computation_times.append(computation_time)
        
    def _compute_kinematic_derivatives(self) -> None:
        """Compute velocity and acceleration from position history using numerical differentiation."""
        if len(self.estimated_history) < 2 or len(self.timestamp_history) < 2:
            return
            
        # Compute velocity (first derivative)
        pos_current = self.estimated_history[-1]
        pos_previous = self.estimated_history[-2] if len(self.estimated_history) > 1 else pos_current
        
        time_current = self.timestamp_history[-1]
        time_previous = self.timestamp_history[-2] if len(self.timestamp_history) > 1 else time_current
        
        dt = time_current - time_previous
        if dt > 0:
            velocity = (pos_current - pos_previous) / dt
            self.velocity_history.append(velocity)
            
            # Compute acceleration (second derivative)
            if len(self.velocity_history) >= 2:
                vel_current = self.velocity_history[-1]
                vel_previous = self.velocity_history[-2]
                acceleration = (vel_current - vel_previous) / dt
                self.acceleration_history.append(acceleration)
                
    def _update_error_statistics(self) -> None:
        """Update comprehensive error statistics between estimated and ground truth trajectories."""
        if len(self.estimated_history) == 0 or len(self.ground_truth_history) == 0:
            return
            
        # Align trajectories by length
        min_length = min(len(self.estimated_history), len(self.ground_truth_history))
        if min_length == 0:
            return
            
        estimated_array = np.array(list(self.estimated_history)[-min_length:])
        ground_truth_array = np.array(list(self.ground_truth_history)[-min_length:])
        
        # Compute position errors
        position_errors = np.linalg.norm(estimated_array - ground_truth_array, axis=1)
        
        # Comprehensive error statistics
        self.error_statistics = {
            'rmse': np.sqrt(np.mean(position_errors**2)),
            'max_error': np.max(position_errors),
            'mean_error': np.mean(position_errors),
            'std_error': np.std(position_errors),
            'median_error': np.median(position_errors),
            'percentile_95': np.percentile(position_errors, 95),
            'current_error': position_errors[-1] if len(position_errors) > 0 else 0.0
        }
        
    def update_sensor_status(self, sensor_status: Dict[str, Any]) -> None:
        """
        Update sensor health status with reliability tracking.
        
        Args:
            sensor_status: Dictionary containing sensor operational status and reliability metrics
        """
        if not isinstance(sensor_status, dict):
            raise TypeError("Sensor status must be a dictionary")
            
        self.sensor_status = sensor_status.copy()
        
        # Track sensor reliability trends
        current_time = time.time()
        for sensor_type, sensor_data in sensor_status.items():
            if sensor_type not in self.sensor_reliability_history:
                self.sensor_reliability_history[sensor_type] = deque(maxlen=100)
                
            if isinstance(sensor_data, list):
                # Multiple sensors of same type
                for i, sensor in enumerate(sensor_data):
                    sensor_id = f"{sensor_type}_{i}"
                    reliability = sensor.get('reliability', 1.0)
                    self.sensor_reliability_history[sensor_id] = self.sensor_reliability_history.get(
                        sensor_id, deque(maxlen=100))
                    self.sensor_reliability_history[sensor_id].append({
                        'timestamp': current_time,
                        'reliability': reliability
                    })
            else:
                # Single sensor
                reliability = sensor_data.get('reliability', 1.0)
                self.sensor_reliability_history[sensor_type].append({
                    'timestamp': current_time,
                    'reliability': reliability
                })
                
    def update_plot(self) -> None:
        """
        Update all visualization components with comprehensive data analysis.
        
        This method performs real-time updates of:
        - 3D trajectory visualization with uncertainty ellipses
        - Error analysis plots with statistical overlays
        - Velocity and acceleration profiles
        - System performance metrics dashboard
        """
        try:
            start_time = time.perf_counter()
            
            # Clear and update 3D plot
            self._update_3d_trajectory_plot()
            
            # Update analysis plots if statistics enabled
            if self.enable_statistics:
                self._update_error_analysis_plot()
                self._update_velocity_analysis_plot()
                self._update_performance_metrics_plot()
                
            # Update sensor status display
            self._display_comprehensive_sensor_status()
            
            # Refresh display
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            plt.pause(0.001)  # Minimal pause for display update
            
            # Track update performance
            update_time = time.perf_counter() - start_time
            if update_time > 1.0 / self.update_rate:
                logger.warning(f"Update time {update_time:.3f}s exceeds target rate")
                
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            
    def _update_3d_trajectory_plot(self) -> None:
        """Update the main 3D trajectory plot with uncertainty visualization."""
        self.axes_3d.clear()
        self._setup_3d_plot()
        
        # Plot ground truth trajectory
        if len(self.ground_truth_history) > 1:
            gt_positions = np.array(self.ground_truth_history)
            self.axes_3d.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                            'g-', linewidth=3, label='Ground Truth', alpha=0.9)
                            
        # Plot estimated trajectory with uncertainty
        if len(self.estimated_history) > 1:
            est_positions = np.array(self.estimated_history)
            
            # Main trajectory line
            self.axes_3d.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 
                            'r-', linewidth=3, label='Estimated', alpha=0.9)
                            
            # Uncertainty visualization (every 5th point to avoid clutter)
            if len(self.uncertainty_history) > 0:
                self._plot_uncertainty_ellipses(est_positions)
                
        # Current position marker
        if len(self.estimated_history) > 0:
            current_pos = self.estimated_history[-1]
            self.axes_3d.scatter(current_pos[0], current_pos[1], current_pos[2], 
                               c='red', s=150, marker='o', label='Current Position',
                               edgecolors='black', linewidth=2)
                               
        # Velocity vector at current position
        if len(self.velocity_history) > 0 and len(self.estimated_history) > 0:
            current_pos = self.estimated_history[-1]
            current_vel = self.velocity_history[-1]
            vel_magnitude = np.linalg.norm(current_vel)
            
            if vel_magnitude > 0.1:  # Only show significant velocities
                # Scale velocity vector for visibility
                vel_scaled = current_vel / vel_magnitude * 2.0
                self.axes_3d.quiver(current_pos[0], current_pos[1], current_pos[2],
                                  vel_scaled[0], vel_scaled[1], vel_scaled[2],
                                  color='blue', alpha=0.7, arrow_length_ratio=0.3,
                                  label=f'Velocity ({vel_magnitude:.2f} m/s)')
                                  
        # Adaptive axis scaling
        if self.adaptive_scaling:
            self._update_adaptive_axis_limits()
            
        # Enhanced legend
        legend = self.axes_3d.legend(loc='upper left', fontsize=10)
        legend.set_frame_on(True)
        legend.get_frame().set_alpha(0.9)
        
    def _plot_uncertainty_ellipses(self, positions: np.ndarray) -> None:
        """Plot uncertainty ellipses at trajectory points."""
        if len(self.uncertainty_history) == 0:
            return
            
        # Sample every 5th point to avoid visual clutter
        sample_indices = range(0, len(positions), 5)
        
        for i in sample_indices:
            if i < len(self.uncertainty_history):
                pos = positions[i]
                cov = self.uncertainty_history[i]
                
                # Project covariance to XY plane for visualization
                cov_2d = cov[:2, :2]
                
                # Compute confidence ellipse parameters
                eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
                
                # Chi-square value for confidence level
                chi2_val = scipy.stats.chi2.ppf(self.confidence_level, df=2)
                
                # Ellipse parameters
                width = 2 * np.sqrt(eigenvals[0] * chi2_val)
                height = 2 * np.sqrt(eigenvals[1] * chi2_val)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Create ellipse (projected to XY plane)
                ellipse = Ellipse((pos[0], pos[1]), width, height, angle=angle,
                                facecolor='red', alpha=0.1, edgecolor='red', linewidth=0.5)
                
                # Add to 3D plot (requires conversion to 3D patch)
                self.axes_3d.add_patch(ellipse)
                
    def _update_adaptive_axis_limits(self) -> None:
        """Update axis limits dynamically based on trajectory data."""
        all_positions = []
        
        if len(self.ground_truth_history) > 0:
            all_positions.extend(self.ground_truth_history)
        if len(self.estimated_history) > 0:
            all_positions.extend(self.estimated_history)
            
        if len(all_positions) > 0:
            positions = np.array(all_positions)
            
            # Compute adaptive margins based on data spread
            position_std = np.std(positions, axis=0)
            margins = np.maximum(position_std * 2, [2.0, 2.0, 1.0])  # Minimum margins
            
            # Axis limits with margins
            x_center, y_center, z_center = np.mean(positions, axis=0)
            x_range = positions[:, 0].max() - positions[:, 0].min() + 2 * margins[0]
            y_range = positions[:, 1].max() - positions[:, 1].min() + 2 * margins[1]
            z_range = positions[:, 2].max() - positions[:, 2].min() + 2 * margins[2]
            
            self.axes_3d.set_xlim([x_center - x_range/2, x_center + x_range/2])
            self.axes_3d.set_ylim([y_center - y_range/2, y_center + y_range/2])
            self.axes_3d.set_zlim([max(0, z_center - z_range/2), z_center + z_range/2])
            
    def _update_error_analysis_plot(self) -> None:
        """Update error analysis subplot with statistical information."""
        self.axes_error.clear()
        self.axes_error.set_title('Position Error Analysis', fontsize=10, fontweight='bold')
        self.axes_error.set_xlabel('Time (s)')
        self.axes_error.set_ylabel('Error (m)')
        self.axes_error.grid(True, alpha=0.3)
        
        if len(self.timestamp_history) > 1 and len(self.estimated_history) > 0 and len(self.ground_truth_history) > 0:
            # Compute errors over time
            min_length = min(len(self.estimated_history), len(self.ground_truth_history), 
                           len(self.timestamp_history))
            
            if min_length > 1:
                timestamps = list(self.timestamp_history)[-min_length:]
                estimated = np.array(list(self.estimated_history)[-min_length:])
                ground_truth = np.array(list(self.ground_truth_history)[-min_length:])
                
                errors = np.linalg.norm(estimated - ground_truth, axis=1)
                
                # Plot error time series
                self.axes_error.plot(timestamps, errors, 'b-', linewidth=2, alpha=0.8)
                
                # Add statistical information
                if len(errors) > 5:
                    # Moving average
                    window_size = min(10, len(errors) // 3)
                    if window_size >= 3:
                        moving_avg = np.convolve(errors, np.ones(window_size)/window_size, mode='valid')
                        timestamps_avg = timestamps[window_size-1:]
                        self.axes_error.plot(timestamps_avg, moving_avg, 'r--', 
                                           linewidth=2, alpha=0.7, label='Moving Average')
                        
                    # Statistical bounds
                    mean_error = np.mean(errors)
                    std_error = np.std(errors)
                    self.axes_error.axhline(y=mean_error, color='g', linestyle=':', 
                                          alpha=0.7, label=f'Mean: {mean_error:.3f}m')
                    self.axes_error.axhline(y=mean_error + std_error, color='orange', 
                                          linestyle=':', alpha=0.5, label=f'+1σ: {mean_error + std_error:.3f}m')
                    
                self.axes_error.legend(fontsize=8)
                
    def _update_velocity_analysis_plot(self) -> None:
        """Update velocity analysis subplot."""
        self.axes_velocity.clear()
        self.axes_velocity.set_title('Velocity Profile', fontsize=10, fontweight='bold')
        self.axes_velocity.set_xlabel('Time (s)')
        self.axes_velocity.set_ylabel('Speed (m/s)')
        self.axes_velocity.grid(True, alpha=0.3)
        
        if len(self.velocity_history) > 1 and len(self.timestamp_history) > 1:
            # Velocity magnitude over time
            min_length = min(len(self.velocity_history), len(self.timestamp_history))
            timestamps = list(self.timestamp_history)[-min_length:]
            velocities = np.array(list(self.velocity_history)[-min_length:])
            
            speeds = np.linalg.norm(velocities, axis=1)
            
            self.axes_velocity.plot(timestamps, speeds, 'g-', linewidth=2, alpha=0.8)
            
            # Add statistics
            if len(speeds) > 0:
                mean_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                self.axes_velocity.axhline(y=mean_speed, color='orange', linestyle='--', 
                                         alpha=0.7, label=f'Mean: {mean_speed:.2f} m/s')
                self.axes_velocity.text(0.7, 0.9, f'Max: {max_speed:.2f} m/s', 
                                      transform=self.axes_velocity.transAxes, fontsize=8)
                self.axes_velocity.legend(fontsize=8)
                
    def _update_performance_metrics_plot(self) -> None:
        """Update system performance metrics display."""
        self.axes_metrics.clear()
        self.axes_metrics.set_title('System Performance Metrics', fontsize=10, fontweight='bold')
        
        # Collect performance metrics
        metrics = {}
        
        # Error metrics
        if self.error_statistics:
            metrics['RMSE'] = min(1.0, 1.0 / (1.0 + self.error_statistics['rmse']))
            metrics['Max Error'] = min(1.0, 1.0 / (1.0 + self.error_statistics['max_error']))
            
        # Computation performance
        if len(self.computation_times) > 0:
            avg_compute_time = np.mean(self.computation_times)
            target_time = 1.0 / self.update_rate
            metrics['Compute Performance'] = min(1.0, target_time / (avg_compute_time + 1e-6))
            
        # Sensor availability
        if self.sensor_status:
            total_sensors = 0
            operational_sensors = 0
            
            for sensor_type, sensors in self.sensor_status.items():
                if isinstance(sensors, list):
                    total_sensors += len(sensors)
                    operational_sensors += sum(1 for s in sensors if s.get('operational', True))
                else:
                    total_sensors += 1
                    operational_sensors += 1 if sensors.get('operational', True) else 0
                    
            if total_sensors > 0:
                metrics['Sensor Availability'] = operational_sensors / total_sensors
                
        # Plot metrics as bar chart
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = self.axes_metrics.bar(metric_names, metric_values, alpha=0.7)
            
            # Color coding based on performance
            for bar, value in zip(bars, metric_values):
                if value > 0.8:
                    bar.set_color('green')
                elif value > 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
                    
            self.axes_metrics.set_ylim([0, 1])
            self.axes_metrics.set_ylabel('Performance Score')
            
            # Rotate labels for readability
            plt.setp(self.axes_metrics.get_xticklabels(), rotation=45, ha='right')
            
    def _display_comprehensive_sensor_status(self) -> None:
        """Display comprehensive sensor status with reliability trends."""
        if not self.sensor_status:
            return
            
        status_lines = ["System Status:"]
        
        for sensor_type, sensors in self.sensor_status.items():
            if isinstance(sensors, list):
                for i, sensor in enumerate(sensors):
                    operational = "OPERATIONAL" if sensor.get('operational', True) else "FAILED"
                    reliability = sensor.get('reliability', 1.0)
                    
                    # Color coding based on reliability
                    if reliability > 0.9:
                        status_color = 'green'
                    elif reliability > 0.7:
                        status_color = 'orange'
                    else:
                        status_color = 'red'
                        
                    status_lines.append(f"{sensor_type.upper()}{i+1}: {operational} "
                                      f"(R={reliability:.3f})")
            else:
                operational = "OPERATIONAL" if sensors.get('operational', True) else "FAILED"
                reliability = sensors.get('reliability', 1.0)
                status_lines.append(f"{sensor_type.upper()}: {operational} (R={reliability:.3f})")
                
        # Add system-wide statistics
        if self.error_statistics:
            status_lines.append(f"Current Error: {self.error_statistics.get('current_error', 0):.3f}m")
            status_lines.append(f"RMSE: {self.error_statistics.get('rmse', 0):.3f}m")
            
        status_text = '\n'.join(status_lines)
        
        # Display in the 3D plot area
        self.axes_3d.text2D(0.02, 0.98, status_text, transform=self.axes_3d.transAxes,
                          verticalalignment='top', fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                                  alpha=0.8, edgecolor='navy'))


class TrajectoryPlotter:
    """
    Advanced trajectory analysis and comparison toolkit for scientific evaluation.
    
    This class provides comprehensive tools for trajectory analysis including:
    - Multi-trajectory comparison with statistical significance testing
    - Advanced error analysis with confidence intervals
    - Trajectory smoothing and filtering algorithms
    - Velocity and acceleration profile analysis
    - Path similarity metrics and geometric analysis
    - Publication-quality scientific plotting
    
    The plotter emphasizes statistical rigor and scientific accuracy in all
    computations and visualizations.
    
    Attributes:
        trajectories (List[Dict]): Repository of trajectory data with metadata
        figure (matplotlib.figure.Figure): Current figure handle
        axes (matplotlib.axes.Axes): Current axes handle
        analysis_results (Dict): Cached analysis results
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (15, 10)):
        """
        Initialize the trajectory analysis toolkit.
        
        Args:
            figure_size: Matplotlib figure size in inches (width, height)
        """
        self.trajectories = []
        self.figure = None
        self.axes = None
        self.figure_size = figure_size
        self.analysis_results = {}
        
        # Scientific plotting configuration
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        logger.info("TrajectoryPlotter initialized with scientific plotting configuration")
        
    def add_trajectory(self, 
                      positions: Union[np.ndarray, List[List[float]]], 
                      label: str = "Trajectory", 
                      color: str = 'blue',
                      timestamps: Optional[Union[np.ndarray, List[float]]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add trajectory data to the analysis repository with comprehensive validation.
        
        Args:
            positions: Nx3 array of position coordinates [x, y, z]
            label: Descriptive label for the trajectory
            color: Matplotlib color specification
            timestamps: Optional temporal data for velocity analysis
            metadata: Additional trajectory metadata
            
        Raises:
            ValueError: If trajectory data is invalid
            TypeError: If input types are incorrect
        """
        # Input validation and conversion
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype=np.float64)
            
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Positions must be Nx3 array, got shape {positions.shape}")
            
        if not np.all(np.isfinite(positions)):
            raise ValueError("Trajectory contains invalid values (inf/nan)")
            
        if len(positions) < 2:
            raise ValueError("Trajectory must contain at least 2 points")
            
        # Timestamp validation
        if timestamps is not None:
            timestamps = np.array(timestamps, dtype=np.float64)
            if len(timestamps) != len(positions):
                raise ValueError("Timestamps length must match positions length")
            if not np.all(np.diff(timestamps) > 0):
                logger.warning("Timestamps are not strictly increasing")
                
        # Create trajectory record
        trajectory_record = {
            'positions': positions.copy(),
            'label': str(label),
            'color': color,
            'timestamps': timestamps.copy() if timestamps is not None else None,
            'metadata': metadata.copy() if metadata is not None else {},
            'statistics': self._compute_trajectory_statistics(positions, timestamps)
        }
        
        self.trajectories.append(trajectory_record)
        
        logger.info(f"Added trajectory '{label}' with {len(positions)} points")
        
    def _compute_trajectory_statistics(self, 
                                     positions: np.ndarray, 
                                     timestamps: Optional[np.ndarray] = None) -> TrajectoryStatistics:
        """
        Compute comprehensive trajectory statistics.
        
        Args:
            positions: Trajectory position data
            timestamps: Optional temporal data
            
        Returns:
            TrajectoryStatistics object with computed metrics
        """
        # Basic geometric statistics
        trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        # Velocity analysis
        velocity_stats = {}
        acceleration_stats = {}
        
        if timestamps is not None and len(timestamps) > 1:
            dt = np.diff(timestamps)
            velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
            speeds = np.linalg.norm(velocities, axis=1)
            
            velocity_stats = {
                'mean_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'std_speed': np.std(speeds),
                'median_speed': np.median(speeds)
            }
            
            # Acceleration analysis
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
                accel_magnitudes = np.linalg.norm(accelerations, axis=1)
                
                acceleration_stats = {
                    'mean_acceleration': np.mean(accel_magnitudes),
                    'max_acceleration': np.max(accel_magnitudes),
                    'std_acceleration': np.std(accel_magnitudes)
                }
                
        return TrajectoryStatistics(
            rmse=0.0,  # Will be computed during comparison
            max_error=0.0,
            mean_error=0.0,
            std_error=0.0,
            median_error=0.0,
            percentile_95=0.0,
            trajectory_length=trajectory_length,
            velocity_statistics=velocity_stats,
            acceleration_statistics=acceleration_stats
        )
        
    def plot_all_trajectories(self, 
                            show_uncertainty: bool = False,
                            show_velocities: bool = False,
                            subplot_layout: Optional[Tuple[int, int]] = None) -> None:
        """
        Generate comprehensive multi-trajectory visualization.
        
        Args:
            show_uncertainty: Display uncertainty regions if available
            show_velocities: Show velocity vectors along trajectories
            subplot_layout: Optional subplot arrangement (rows, cols)
        """
        if not self.trajectories:
            logger.warning("No trajectories to plot")
            return
            
        # Determine subplot layout
        if subplot_layout is None:
            if len(self.trajectories) <= 4:
                subplot_layout = (2, 2)
            else:
                subplot_layout = (2, 3)
                
        self.figure = plt.figure(figsize=self.figure_size)
        
        # Main 3D trajectory comparison
        self.axes = self.figure.add_subplot(2, 2, 1, projection='3d')
        
        for traj in self.trajectories:
            positions = traj['positions']
            self.axes.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                          color=traj['color'], label=traj['label'], 
                          linewidth=2.5, alpha=0.8)
                          
            # Mark start and end points
            self.axes.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                            color=traj['color'], marker='o', s=100, alpha=0.9)
            self.axes.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                            color=traj['color'], marker='s', s=100, alpha=0.9)
                            
        self._configure_3d_axes(self.axes, "Trajectory Comparison")
        
        # 2D projections
        self._plot_2d_projections()
        
        # Statistical analysis
        self._plot_trajectory_statistics()
        
        plt.tight_layout()
        plt.show()
        
    def _configure_3d_axes(self, axes, title: str) -> None:
        """Configure 3D axes with scientific styling."""
        axes.set_xlabel('X Position (m)', fontweight='bold')
        axes.set_ylabel('Y Position (m)', fontweight='bold')
        axes.set_zlabel('Z Position (m)', fontweight='bold')
        axes.set_title(title, fontweight='bold')
        axes.legend()
        axes.grid(True, alpha=0.3)
        
        # Equal aspect ratio for accurate geometric representation
        # Note: This is a simplified approach; full equal aspect in 3D is complex
        all_positions = np.vstack([traj['positions'] for traj in self.trajectories])
        max_range = np.ptp(all_positions, axis=0).max() / 2.0
        center = np.mean(all_positions, axis=0)
        
        axes.set_xlim(center[0] - max_range, center[0] + max_range)
        axes.set_ylim(center[1] - max_range, center[1] + max_range)
        axes.set_zlim(center[2] - max_range, center[2] + max_range)
        
    def _plot_2d_projections(self) -> None:
        """Plot 2D projections of trajectories for detailed analysis."""
        # XY projection
        ax_xy = self.figure.add_subplot(2, 2, 2)
        for traj in self.trajectories:
            positions = traj['positions']
            ax_xy.plot(positions[:, 0], positions[:, 1], 
                      color=traj['color'], label=traj['label'], linewidth=2)
                      
        ax_xy.set_xlabel('X Position (m)')
        ax_xy.set_ylabel('Y Position (m)')
        ax_xy.set_title('XY Projection')
        ax_xy.grid(True, alpha=0.3)
        ax_xy.legend()
        ax_xy.set_aspect('equal')
        
        # XZ projection
        ax_xz = self.figure.add_subplot(2, 2, 3)
        for traj in self.trajectories:
            positions = traj['positions']
            ax_xz.plot(positions[:, 0], positions[:, 2], 
                      color=traj['color'], label=traj['label'], linewidth=2)
                      
        ax_xz.set_xlabel('X Position (m)')
        ax_xz.set_ylabel('Z Position (m)')
        ax_xz.set_title('XZ Projection')
        ax_xz.grid(True, alpha=0.3)
        ax_xz.legend()
        
    def _plot_trajectory_statistics(self) -> None:
        """Plot statistical summary of trajectory characteristics."""
        ax_stats = self.figure.add_subplot(2, 2, 4)
        
        # Collect statistics
        lengths = [traj['statistics'].trajectory_length for traj in self.trajectories]
        labels = [traj['label'] for traj in self.trajectories]
        colors = [traj['color'] for traj in self.trajectories]
        
        # Bar plot of trajectory lengths
        bars = ax_stats.bar(labels, lengths, color=colors, alpha=0.7)
        ax_stats.set_ylabel('Trajectory Length (m)')
        ax_stats.set_title('Trajectory Statistics')
        ax_stats.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, length in zip(bars, lengths):
            height = bar.get_height()
            ax_stats.text(bar.get_x() + bar.get_width()/2., height,
                         f'{length:.1f}m', ha='center', va='bottom')
                         
    def compare_trajectories(self, 
                           reference_label: str, 
                           comparison_label: str,
                           detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive statistical comparison between two trajectories.
        
        Args:
            reference_label: Label of reference (ground truth) trajectory
            comparison_label: Label of trajectory to compare against reference
            detailed_analysis: Enable detailed statistical analysis
            
        Returns:
            Dictionary containing comprehensive comparison metrics
            
        Raises:
            ValueError: If specified trajectories are not found
        """
        # Find trajectories
        ref_traj = None
        comp_traj = None
        
        for traj in self.trajectories:
            if traj['label'] == reference_label:
                ref_traj = traj
            elif traj['label'] == comparison_label:
                comp_traj = traj
                
        if ref_traj is None:
            raise ValueError(f"Reference trajectory '{reference_label}' not found")
        if comp_traj is None:
            raise ValueError(f"Comparison trajectory '{comparison_label}' not found")
            
        ref_positions = ref_traj['positions']
        comp_positions = comp_traj['positions']
        
        # Align trajectories by interpolation to common length
        aligned_ref, aligned_comp = self._align_trajectories(ref_positions, comp_positions)
        
        # Compute position errors
        position_errors = np.linalg.norm(aligned_ref - aligned_comp, axis=1)
        
        # Basic error statistics
        basic_stats = {
            'rmse': np.sqrt(np.mean(position_errors**2)),
            'max_error': np.max(position_errors),
            'mean_error': np.mean(position_errors),
            'std_error': np.std(position_errors),
            'median_error': np.median(position_errors),
            'percentile_95': np.percentile(position_errors, 95),
            'percentile_5': np.percentile(position_errors, 5)
        }
        
        # Initialize results
        comparison_results = basic_stats.copy()
        
        if detailed_analysis:
            # Advanced statistical analysis
            advanced_stats = self._compute_advanced_error_statistics(
                position_errors, aligned_ref, aligned_comp)
            comparison_results.update(advanced_stats)
            
            # Geometric analysis
            geometric_stats = self._compute_geometric_analysis(aligned_ref, aligned_comp)
            comparison_results.update(geometric_stats)
            
            # Temporal analysis if timestamps available
            if (ref_traj['timestamps'] is not None and 
                comp_traj['timestamps'] is not None):
                temporal_stats = self._compute_temporal_analysis(
                    ref_traj, comp_traj, aligned_ref, aligned_comp)
                comparison_results.update(temporal_stats)
                
        # Cache results
        self.analysis_results[f"{reference_label}_vs_{comparison_label}"] = comparison_results
        
        logger.info(f"Trajectory comparison completed: RMSE = {basic_stats['rmse']:.4f}m")
        
        return comparison_results
        
    def _align_trajectories(self, 
                          traj1: np.ndarray, 
                          traj2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two trajectories by interpolating to common parameterization.
        
        Args:
            traj1: First trajectory positions
            traj2: Second trajectory positions
            
        Returns:
            Tuple of aligned trajectory arrays
        """
        # Use the trajectory with fewer points as reference length
        target_length = min(len(traj1), len(traj2))
        
        # Simple alignment by linear interpolation
        if len(traj1) != target_length:
            indices = np.linspace(0, len(traj1) - 1, target_length)
            traj1_aligned = np.array([np.interp(indices, np.arange(len(traj1)), traj1[:, i]) 
                                    for i in range(3)]).T
        else:
            traj1_aligned = traj1[:target_length]
            
        if len(traj2) != target_length:
            indices = np.linspace(0, len(traj2) - 1, target_length)
            traj2_aligned = np.array([np.interp(indices, np.arange(len(traj2)), traj2[:, i]) 
                                    for i in range(3)]).T
        else:
            traj2_aligned = traj2[:target_length]
            
        return traj1_aligned, traj2_aligned
        
    def _compute_advanced_error_statistics(self, 
                                         errors: np.ndarray,
                                         ref_traj: np.ndarray, 
                                         comp_traj: np.ndarray) -> Dict[str, float]:
        """Compute advanced error statistics and confidence intervals."""
        # Confidence intervals
        alpha = 0.05  # 95% confidence
        ci_lower = np.percentile(errors, 100 * alpha / 2)
        ci_upper = np.percentile(errors, 100 * (1 - alpha/2))
        
        # Error distribution analysis
        skewness = scipy.stats.skew(errors)
        kurtosis = scipy.stats.kurtosis(errors)
        
        # Normality test
        _, normality_p_value = scipy.stats.shapiro(errors)
        is_normal = normality_p_value > 0.05
        
        # Directional error analysis
        error_vectors = comp_traj - ref_traj
        x_errors = error_vectors[:, 0]
        y_errors = error_vectors[:, 1]
        z_errors = error_vectors[:, 2]
        
        return {
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'error_skewness': skewness,
            'error_kurtosis': kurtosis,
            'normality_test_passed': is_normal,
            'x_error_mean': np.mean(x_errors),
            'y_error_mean': np.mean(y_errors),
            'z_error_mean': np.mean(z_errors),
            'x_error_std': np.std(x_errors),
            'y_error_std': np.std(y_errors),
            'z_error_std': np.std(z_errors)
        }
        
    def _compute_geometric_analysis(self, 
                                  ref_traj: np.ndarray, 
                                  comp_traj: np.ndarray) -> Dict[str, float]:
        """Compute geometric similarity metrics."""
        # Path length comparison
        ref_length = np.sum(np.linalg.norm(np.diff(ref_traj, axis=0), axis=1))
        comp_length = np.sum(np.linalg.norm(np.diff(comp_traj, axis=0), axis=1))
        length_ratio = comp_length / ref_length if ref_length > 0 else 1.0
        
        # Hausdorff distance (simplified approximation)
        hausdorff_dist = self._approximate_hausdorff_distance(ref_traj, comp_traj)
        
        # Area between curves (simplified for 3D)
        area_between = self._compute_area_between_curves(ref_traj, comp_traj)
        
        return {
            'path_length_ratio': length_ratio,
            'hausdorff_distance': hausdorff_dist,
            'area_between_curves': area_between
        }
        
    def _approximate_hausdorff_distance(self, 
                                      traj1: np.ndarray, 
                                      traj2: np.ndarray) -> float:
        """Compute approximate Hausdorff distance between trajectories."""
        # For computational efficiency, use a sampled approximation
        sample_size = min(50, len(traj1), len(traj2))
        
        indices1 = np.linspace(0, len(traj1) - 1, sample_size, dtype=int)
        indices2 = np.linspace(0, len(traj2) - 1, sample_size, dtype=int)
        
        sample1 = traj1[indices1]
        sample2 = traj2[indices2]
        
        # Compute pairwise distances
        distances = np.linalg.norm(sample1[:, np.newaxis] - sample2[np.newaxis, :], axis=2)
        
        # Hausdorff distance
        hausdorff_1_to_2 = np.max(np.min(distances, axis=1))
        hausdorff_2_to_1 = np.max(np.min(distances, axis=0))
        
        return max(hausdorff_1_to_2, hausdorff_2_to_1)
        
    def _compute_area_between_curves(self, 
                                   traj1: np.ndarray, 
                                   traj2: np.ndarray) -> float:
        """Compute approximate area between trajectory curves."""
        # Simplified calculation using trapezoidal rule in 3D
        distances = np.linalg.norm(traj1 - traj2, axis=1)
        
        # Parameterization lengths
        param_lengths = np.linspace(0, 1, len(distances))
        
        # Trapezoidal integration
        if len(distances) > 1:
            area = np.trapz(distances, param_lengths)
        else:
            area = 0.0
            
        return area
        
    def _compute_temporal_analysis(self, 
                                 ref_traj: Dict, 
                                 comp_traj: Dict,
                                 aligned_ref: np.ndarray, 
                                 aligned_comp: np.ndarray) -> Dict[str, float]:
        """Compute temporal characteristics comparison."""
        ref_times = ref_traj['timestamps']
        comp_times = comp_traj['timestamps']
        
        # Time span comparison
        ref_duration = ref_times[-1] - ref_times[0] if len(ref_times) > 1 else 0
        comp_duration = comp_times[-1] - comp_times[0] if len(comp_times) > 1 else 0
        
        duration_ratio = comp_duration / ref_duration if ref_duration > 0 else 1.0
        
        return {
            'reference_duration': ref_duration,
            'comparison_duration': comp_duration,
            'duration_ratio': duration_ratio
        }
        
    def smooth_trajectory(self, 
                         label: str, 
                         method: str = 'gaussian',
                         window_size: int = 5,
                         sigma: float = 1.0) -> np.ndarray:
        """
        Apply advanced smoothing algorithms to trajectory data.
        
        Args:
            label: Trajectory label to smooth
            method: Smoothing method ('gaussian', 'savgol', 'moving_average')
            window_size: Smoothing window size
            sigma: Gaussian sigma parameter
            
        Returns:
            Smoothed trajectory positions
            
        Raises:
            ValueError: If trajectory not found or invalid parameters
        """
        # Find trajectory
        target_traj = None
        for traj in self.trajectories:
            if traj['label'] == label:
                target_traj = traj
                break
                
        if target_traj is None:
            raise ValueError(f"Trajectory '{label}' not found")
            
        positions = target_traj['positions']
        
        if window_size >= len(positions):
            raise ValueError("Window size must be smaller than trajectory length")
            
        if method == 'gaussian':
            # Gaussian smoothing
            smoothed = np.zeros_like(positions)
            for i in range(3):  # For each coordinate
                smoothed[:, i] = scipy.ndimage.gaussian_filter1d(
                    positions[:, i], sigma=sigma, mode='nearest')
                    
        elif method == 'savgol':
            # Savitzky-Golay smoothing
            from scipy.signal import savgol_filter
            
            # Ensure odd window size
            if window_size % 2 == 0:
                window_size += 1
                
            poly_order = min(3, window_size - 1)  # Polynomial order
            
            smoothed = np.zeros_like(positions)
            for i in range(3):
                smoothed[:, i] = savgol_filter(
                    positions[:, i], window_size, poly_order, mode='nearest')
                    
        elif method == 'moving_average':
            # Moving average smoothing
            smoothed = np.zeros_like(positions)
            half_window = window_size // 2
            
            for i in range(len(positions)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(positions), i + half_window + 1)
                smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
                
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
            
        logger.info(f"Applied {method} smoothing to trajectory '{label}'")
        
        return smoothed
        
    def export_analysis_results(self, 
                              filename: str,
                              format: str = 'json') -> None:
        """
        Export analysis results to file for further processing.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'csv')
        """
        if format == 'json':
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            exportable_results = {}
            for key, value in self.analysis_results.items():
                exportable_results[key] = {}
                for metric, val in value.items():
                    if isinstance(val, np.ndarray):
                        exportable_results[key][metric] = val.tolist()
                    elif isinstance(val, (np.integer, np.floating)):
                        exportable_results[key][metric] = float(val)
                    else:
                        exportable_results[key][metric] = val
                        
            with open(filename, 'w') as f:
                json.dump(exportable_results, f, indent=2)
                
        elif format == 'csv':
            import pandas as pd
            
            # Flatten results for CSV export
            rows = []
            for comparison, metrics in self.analysis_results.items():
                row = {'comparison': comparison}
                row.update(metrics)
                rows.append(row)
                
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Analysis results exported to {filename}")