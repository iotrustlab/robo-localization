"""
Real-time 3D visualization for rover localization system.

This module implements:
- Real-time 3D trajectory plotting
- Sensor health monitoring displays
- Performance metrics visualization
- Interactive plots for system analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import deque
import scipy.ndimage
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class RealTimeVisualizer:
    """Real-time 3D visualization for rover trajectory and sensor status"""
    
    def __init__(self, update_rate=10.0, trail_length=100):
        self.update_rate = update_rate
        self.trail_length = trail_length
        
        # Position history
        self.position_history = deque(maxlen=trail_length)
        self.ground_truth_history = deque(maxlen=trail_length)
        self.estimated_history = deque(maxlen=trail_length)
        
        # Initialize matplotlib figure
        plt.ion()  # Interactive mode
        self.figure = plt.figure(figsize=(12, 8))
        self.axes = self.figure.add_subplot(111, projection='3d')
        
        # Plot elements
        self.ground_truth_line = None
        self.estimated_line = None
        self.current_pos_marker = None
        
        # Sensor status display
        self.sensor_status = {}
        
        self._setup_plot()
        
    def _setup_plot(self):
        """Setup the 3D plot"""
        self.axes.set_xlabel('X (m)')
        self.axes.set_ylabel('Y (m)')
        self.axes.set_zlabel('Z (m)')
        self.axes.set_title('Rover 3D Trajectory')
        
        # Set initial limits
        self.axes.set_xlim([-20, 20])
        self.axes.set_ylim([-20, 20])
        self.axes.set_zlim([0, 10])
        
    def update_position(self, position, estimated=True):
        """Update position in the visualizer"""
        if estimated:
            self.estimated_history.append(position.copy())
            self.position_history.append(position.copy())  # For compatibility
        else:
            self.ground_truth_history.append(position.copy())
            
    def update_sensor_status(self, sensor_status):
        """Update sensor status display"""
        self.sensor_status = sensor_status
        
    def update_plot(self):
        """Update the 3D plot"""
        # Clear previous plots
        self.axes.clear()
        self._setup_plot()
        
        # Plot ground truth trajectory
        if len(self.ground_truth_history) > 1:
            gt_positions = np.array(self.ground_truth_history)
            self.axes.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                          'g-', linewidth=2, label='Ground Truth', alpha=0.8)
                          
        # Plot estimated trajectory  
        if len(self.estimated_history) > 1:
            est_positions = np.array(self.estimated_history)
            self.axes.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 
                          'r-', linewidth=2, label='Estimated', alpha=0.8)
                          
        # Plot current position marker
        if len(self.estimated_history) > 0:
            current_pos = self.estimated_history[-1]
            self.axes.scatter(current_pos[0], current_pos[1], current_pos[2], 
                            c='red', s=100, marker='o', label='Current Position')
                            
        # Update axis limits based on data
        self._update_axis_limits()
        
        # Add legend
        self.axes.legend()
        
        # Display sensor status as text
        self._display_sensor_status()
        
        # Refresh plot
        plt.pause(0.01)
        
    def _update_axis_limits(self):
        """Update axis limits to encompass all data"""
        all_positions = []
        
        if len(self.ground_truth_history) > 0:
            all_positions.extend(self.ground_truth_history)
        if len(self.estimated_history) > 0:
            all_positions.extend(self.estimated_history)
            
        if len(all_positions) > 0:
            positions = np.array(all_positions)
            
            # Add margin
            margin = 5.0
            x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
            y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
            z_min, z_max = positions[:, 2].min() - margin, positions[:, 2].max() + margin
            
            self.axes.set_xlim([x_min, x_max])
            self.axes.set_ylim([y_min, y_max])
            self.axes.set_zlim([max(0, z_min), z_max])
            
    def _display_sensor_status(self):
        """Display sensor status information"""
        if not self.sensor_status:
            return
            
        status_text = "Sensor Status:\n"
        
        # GPS status
        if 'gps' in self.sensor_status:
            for i, gps in enumerate(self.sensor_status['gps']):
                operational = "OK" if gps.get('operational', True) else "FAIL"
                reliability = gps.get('reliability', 1.0)
                status_text += f"GPS{i+1}: {operational} ({reliability:.2f})\n"
                
        # IMU status
        if 'imu' in self.sensor_status:
            for i, imu in enumerate(self.sensor_status['imu']):
                operational = "OK" if imu.get('operational', True) else "FAIL"
                reliability = imu.get('reliability', 1.0)
                status_text += f"IMU{i+1}: {operational} ({reliability:.2f})\n"
                
        # Add text to plot
        self.axes.text2D(0.02, 0.98, status_text, transform=self.axes.transAxes,
                        verticalalignment='top', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


class TrajectoryPlotter:
    """Tools for plotting and comparing trajectories"""
    
    def __init__(self):
        self.trajectories = []
        self.figure = None
        self.axes = None
        
    def add_trajectory(self, positions, label="Trajectory", color='blue'):
        """Add trajectory for plotting"""
        self.trajectories.append({
            'positions': np.array(positions),
            'label': label,
            'color': color
        })
        
    def plot_all_trajectories(self):
        """Plot all added trajectories"""
        if not self.trajectories:
            return
            
        self.figure = plt.figure(figsize=(10, 8))
        self.axes = self.figure.add_subplot(111, projection='3d')
        
        for traj in self.trajectories:
            positions = traj['positions']
            self.axes.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                          color=traj['color'], label=traj['label'], linewidth=2)
                          
        self.axes.set_xlabel('X (m)')
        self.axes.set_ylabel('Y (m)')
        self.axes.set_zlabel('Z (m)')
        self.axes.set_title('Trajectory Comparison')
        self.axes.legend()
        
        plt.show()
        
    def compare_trajectories(self, label1, label2):
        """Compare two trajectories and return error statistics"""
        traj1 = None
        traj2 = None
        
        for traj in self.trajectories:
            if traj['label'] == label1:
                traj1 = traj['positions']
            elif traj['label'] == label2:
                traj2 = traj['positions']
                
        if traj1 is None or traj2 is None:
            raise ValueError("One or both trajectories not found")
            
        # Ensure same length (interpolate if necessary)
        min_length = min(len(traj1), len(traj2))
        traj1 = traj1[:min_length]
        traj2 = traj2[:min_length]
        
        # Compute errors
        errors = np.linalg.norm(traj1 - traj2, axis=1)
        
        return {
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
        
    def compute_error_statistics(self, reference_label, comparison_label):
        """Compute detailed error statistics between trajectories"""
        return self.compare_trajectories(reference_label, comparison_label)
        
    def smooth_trajectory(self, label, window_size=5):
        """Apply smoothing to a trajectory"""
        for traj in self.trajectories:
            if traj['label'] == label:
                positions = traj['positions']
                
                # Simple moving average smoothing
                smoothed = np.zeros_like(positions)
                half_window = window_size // 2
                
                for i in range(len(positions)):
                    start = max(0, i - half_window)
                    end = min(len(positions), i + half_window + 1)
                    smoothed[i] = np.mean(positions[start:end], axis=0)
                    
                return smoothed
                
        raise ValueError(f"Trajectory with label '{label}' not found")


class SensorHealthMonitor:
    """Monitor and visualize sensor health over time"""
    
    def __init__(self):
        self.sensor_health_history = []
        self.failure_events = []
        self.recovery_events = []
        self.previous_status = {}
        
    def update_health_status(self, health_status, timestamp):
        """Update sensor health status"""
        # Store health status
        self.sensor_health_history.append({
            'timestamp': timestamp,
            'status': health_status.copy()
        })
        
        # Detect failure and recovery events
        self._detect_state_changes(health_status, timestamp)
        
        # Update previous status
        self.previous_status = health_status.copy()
        
    def _detect_state_changes(self, current_status, timestamp):
        """Detect sensor failures and recoveries"""
        for sensor_type in current_status:
            if sensor_type not in self.previous_status:
                continue
                
            if isinstance(current_status[sensor_type], list):
                # Multiple sensors of this type
                for i, sensor in enumerate(current_status[sensor_type]):
                    sensor_id = f"{sensor_type}_{i}"
                    self._check_sensor_state_change(sensor, sensor_id, timestamp)
            else:
                # Single sensor
                sensor_id = sensor_type
                self._check_sensor_state_change(current_status[sensor_type], sensor_id, timestamp)
                
    def _check_sensor_state_change(self, current_sensor, sensor_id, timestamp):
        """Check for state changes in a single sensor"""
        # Find previous status
        prev_operational = True
        if self.previous_status:
            for sensor_type in self.previous_status:
                if isinstance(self.previous_status[sensor_type], list):
                    for i, prev_sensor in enumerate(self.previous_status[sensor_type]):
                        if f"{sensor_type}_{i}" == sensor_id:
                            prev_operational = prev_sensor.get('operational', True)
                            break
                else:
                    if sensor_type == sensor_id:
                        prev_operational = self.previous_status[sensor_type].get('operational', True)
                        
        current_operational = current_sensor.get('operational', True)
        
        # Detect failure
        if prev_operational and not current_operational:
            self.failure_events.append({
                'sensor': sensor_id,
                'timestamp': timestamp,
                'type': 'failure'
            })
            
        # Detect recovery
        if not prev_operational and current_operational:
            self.recovery_events.append({
                'sensor': sensor_id,
                'timestamp': timestamp,
                'type': 'recovery'
            })
            
    def get_redundancy_status(self):
        """Get current redundancy status"""
        if not self.sensor_health_history:
            return {}
            
        latest_status = self.sensor_health_history[-1]['status']
        redundancy = {}
        
        for sensor_type, sensors in latest_status.items():
            if isinstance(sensors, list):
                total = len(sensors)
                operational = sum(1 for s in sensors if s.get('operational', True))
                
                if operational == total:
                    level = 'full'
                elif operational > 0:
                    level = 'partial'
                else:
                    level = 'none'
                    
                redundancy[sensor_type] = {
                    'total_sensors': total,
                    'operational_sensors': operational,
                    'redundancy_level': level
                }
                
        return redundancy
        
    def compute_health_statistics(self, sensor_id):
        """Compute health statistics for a specific sensor"""
        reliabilities = []
        
        for entry in self.sensor_health_history:
            # Find the sensor
            found = False
            for sensor_type, sensors in entry['status'].items():
                if isinstance(sensors, list):
                    for i, sensor in enumerate(sensors):
                        if f"{sensor_type}_{i}" == sensor_id:
                            reliabilities.append(sensor.get('reliability', 1.0))
                            found = True
                            break
                else:
                    if sensor_type == sensor_id:
                        reliabilities.append(sensors.get('reliability', 1.0))
                        found = True
                        
                if found:
                    break
                    
        if not reliabilities:
            return {}
            
        # Compute trend
        if len(reliabilities) > 1:
            time_points = np.arange(len(reliabilities))
            trend = np.polyfit(time_points, reliabilities, 1)[0]
        else:
            trend = 0.0
            
        return {
            'mean_reliability': np.mean(reliabilities),
            'reliability_trend': trend,
            'uptime_percentage': np.mean([r > 0.5 for r in reliabilities]) * 100
        }
        
    def plot_health_trends(self):
        """Plot sensor health trends over time"""
        if not self.sensor_health_history:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Extract time series for each sensor
        sensor_data = {}
        timestamps = [entry['timestamp'] for entry in self.sensor_health_history]
        
        for entry in self.sensor_health_history:
            for sensor_type, sensors in entry['status'].items():
                if isinstance(sensors, list):
                    for i, sensor in enumerate(sensors):
                        sensor_id = f"{sensor_type}_{i}"
                        if sensor_id not in sensor_data:
                            sensor_data[sensor_id] = []
                        sensor_data[sensor_id].append(sensor.get('reliability', 1.0))
                else:
                    sensor_id = sensor_type
                    if sensor_id not in sensor_data:
                        sensor_data[sensor_id] = []
                    sensor_data[sensor_id].append(sensors.get('reliability', 1.0))
                    
        # Plot each sensor
        for sensor_id, reliabilities in sensor_data.items():
            plt.plot(timestamps, reliabilities, label=sensor_id, linewidth=2)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Reliability')
        plt.title('Sensor Health Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_failure_timeline(self):
        """Plot timeline of sensor failures and recoveries"""
        plt.figure(figsize=(12, 4))
        
        # Plot failure events
        failure_times = [event['timestamp'] for event in self.failure_events]
        failure_sensors = [event['sensor'] for event in self.failure_events]
        
        if failure_times:
            plt.scatter(failure_times, range(len(failure_times)), 
                       c='red', marker='x', s=100, label='Failures')
                       
        # Plot recovery events
        recovery_times = [event['timestamp'] for event in self.recovery_events]
        recovery_sensors = [event['sensor'] for event in self.recovery_events]
        
        if recovery_times:
            plt.scatter(recovery_times, range(len(recovery_times)), 
                       c='green', marker='o', s=100, label='Recoveries')
                       
        plt.xlabel('Time (s)')
        plt.ylabel('Event Index')
        plt.title('Sensor Failure/Recovery Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class MetricsDisplay:
    """Display and track performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_stats = {}
        self.availability_history = []
        self.confidence_history = []
        
    def update_position_error(self, error, timestamp):
        """Update position error metric"""
        self.metrics_history.append({
            'timestamp': timestamp,
            'position_error': error,
            'type': 'position_error'
        })
        
    def update_sensor_availability(self, availability_data):
        """Update sensor availability metrics"""
        self.availability_history.append(availability_data)
        
    def update_fusion_confidence(self, confidence, timestamp):
        """Update fusion confidence metric"""
        self.confidence_history.append({
            'timestamp': timestamp,
            'confidence': confidence
        })
        
    def get_error_statistics(self):
        """Get position error statistics"""
        errors = [entry['position_error'] for entry in self.metrics_history 
                 if entry['type'] == 'position_error']
                 
        if not errors:
            return {}
            
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'rms_error': np.sqrt(np.mean(np.array(errors)**2)),
            'std_error': np.std(errors)
        }
        
    def get_availability_statistics(self):
        """Get sensor availability statistics"""
        if not self.availability_history:
            return {}
            
        # Aggregate availability by sensor type
        availability_stats = {}
        
        for data in self.availability_history:
            for sensor_type, sensors in data.items():
                if sensor_type == 'timestamp':
                    continue
                    
                if sensor_type not in availability_stats:
                    availability_stats[sensor_type] = {}
                    
                if isinstance(sensors, list):
                    for i, available in enumerate(sensors):
                        if i not in availability_stats[sensor_type]:
                            availability_stats[sensor_type][i] = []
                        availability_stats[sensor_type][i].append(available)
                else:
                    if 0 not in availability_stats[sensor_type]:
                        availability_stats[sensor_type][0] = []
                    availability_stats[sensor_type][0].append(sensors)
                    
        # Compute availability percentages
        for sensor_type in availability_stats:
            for sensor_id in availability_stats[sensor_type]:
                availability_stats[sensor_type][sensor_id] = np.mean(availability_stats[sensor_type][sensor_id])
                
        return availability_stats
        
    def get_confidence_statistics(self):
        """Get fusion confidence statistics"""
        if not self.confidence_history:
            return {}
            
        confidences = [entry['confidence'] for entry in self.confidence_history]
        
        return {
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences)
        }
        
    def compute_overall_performance(self):
        """Compute overall system performance metrics"""
        error_stats = self.get_error_statistics()
        confidence_stats = self.get_confidence_statistics()
        
        # Compute trends
        error_trend = 0.0
        confidence_trend = 0.0
        
        if len(self.metrics_history) > 1:
            errors = [entry['position_error'] for entry in self.metrics_history 
                     if entry['type'] == 'position_error']
            if len(errors) > 1:
                time_points = np.arange(len(errors))
                error_trend = np.polyfit(time_points, errors, 1)[0]
                
        if len(self.confidence_history) > 1:
            confidences = [entry['confidence'] for entry in self.confidence_history]
            time_points = np.arange(len(confidences))
            confidence_trend = np.polyfit(time_points, confidences, 1)[0]
            
        # Overall score (simplified)
        overall_score = 0.0
        if error_stats and confidence_stats:
            # Lower error is better, higher confidence is better
            error_score = max(0, 1 - error_stats['mean_error'] / 10.0)  # Normalize by 10m
            confidence_score = confidence_stats['mean_confidence']
            overall_score = (error_score + confidence_score) / 2
            
        return {
            'error_trend': error_trend,
            'confidence_trend': confidence_trend,
            'overall_score': overall_score
        }
        
    def plot_error_trends(self):
        """Plot error trends over time"""
        if not self.metrics_history:
            return
            
        plt.figure(figsize=(10, 6))
        
        timestamps = [entry['timestamp'] for entry in self.metrics_history 
                     if entry['type'] == 'position_error']
        errors = [entry['position_error'] for entry in self.metrics_history 
                 if entry['type'] == 'position_error']
                 
        plt.plot(timestamps, errors, 'b-', linewidth=2, label='Position Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.title('Position Error Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
    def plot_availability_trends(self):
        """Plot sensor availability trends"""
        # Simplified implementation
        plt.figure(figsize=(10, 6))
        plt.title('Sensor Availability Trends')
        plt.xlabel('Time')
        plt.ylabel('Availability')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_confidence_trends(self):
        """Plot fusion confidence trends"""
        if not self.confidence_history:
            return
            
        plt.figure(figsize=(10, 6))
        
        timestamps = [entry['timestamp'] for entry in self.confidence_history]
        confidences = [entry['confidence'] for entry in self.confidence_history]
        
        plt.plot(timestamps, confidences, 'g-', linewidth=2, label='Fusion Confidence')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence')
        plt.title('Fusion Confidence Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
    def update_dashboard(self, current_metrics):
        """Update real-time dashboard display"""
        # Update all metrics from current data
        self.update_position_error(current_metrics['position_error'], 
                                  current_metrics['timestamp'])
        self.update_fusion_confidence(current_metrics['fusion_confidence'], 
                                    current_metrics['timestamp'])
        self.update_sensor_availability(current_metrics['sensor_availability']) 