import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viz import RealTimeVisualizer, TrajectoryPlotter, SensorHealthMonitor, MetricsDisplay


class TestRealTimeVisualizer:
    @pytest.fixture
    def visualizer(self):
        """Create a real-time visualizer for testing."""
        return RealTimeVisualizer(
            update_rate=10.0,  # 10 Hz
            trail_length=100
        )
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initializes correctly."""
        assert visualizer.update_rate == 10.0
        assert visualizer.trail_length == 100
        assert visualizer.sensor_count['gps'] == 2
        assert visualizer.sensor_count['imu'] == 2
        assert visualizer.sensor_count['odometry'] == 1
        
        # Check internal state initialization
        assert len(visualizer.position_history) == 0
        assert len(visualizer.ground_truth_history) == 0
        assert visualizer.current_time == 0.0
    
    @patch('matplotlib.pyplot.figure')
    def test_visualizer_setup(self, mock_figure, visualizer):
        """Test visualizer plot setup."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        visualizer.setup_plots()
        
        # Should create figure and 3D axis
        mock_figure.assert_called_once()
        mock_fig.add_subplot.assert_called_once()
        
        # Should set up axis properties
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.set_zlabel.called
        assert mock_ax.set_title.called
    
    def test_position_data_update(self, visualizer):
        """Test position data updates correctly."""
        # Add position data
        estimated_pos = np.array([1.0, 2.0, 3.0])
        ground_truth_pos = np.array([1.1, 1.9, 3.1])
        timestamp = 1.5
        
        visualizer.update_position(estimated_pos, ground_truth_pos, timestamp)
        
        # Check data was stored
        assert len(visualizer.position_history) == 1
        assert len(visualizer.ground_truth_history) == 1
        assert np.array_equal(visualizer.position_history[0], estimated_pos)
        assert np.array_equal(visualizer.ground_truth_history[0], ground_truth_pos)
        assert visualizer.current_time == timestamp
    
    def test_trail_length_limit(self, visualizer):
        """Test position history respects trail length limit."""
        # Add more positions than trail length
        for i in range(visualizer.trail_length + 50):
            pos = np.array([i, i * 0.5, i * 0.1])
            visualizer.update_position(pos, pos + 0.1, i * 0.1)
        
        # Should only keep trail_length positions
        assert len(visualizer.position_history) == visualizer.trail_length
        assert len(visualizer.ground_truth_history) == visualizer.trail_length
        
        # Should keep the most recent positions
        latest_pos = visualizer.position_history[-1]
        expected_latest = np.array([visualizer.trail_length + 49, 
                                   (visualizer.trail_length + 49) * 0.5,
                                   (visualizer.trail_length + 49) * 0.1])
        assert np.array_equal(latest_pos, expected_latest)
    
    def test_sensor_data_update(self, visualizer):
        """Test sensor data updates correctly."""
        # Update GPS sensor data
        gps_data = {
            'sensor_id': 1,
            'position': np.array([5.0, 3.0, 1.0]),
            'noise_level': 0.5,
            'health': True
        }
        visualizer.update_sensor_data('gps', gps_data)
        
        # Check data was stored
        assert 'gps' in visualizer.sensor_data
        assert visualizer.sensor_data['gps'][1] == gps_data
    
    @patch('matplotlib.pyplot.pause')
    def test_real_time_update(self, mock_pause, visualizer):
        """Test real-time update cycle."""
        # Setup mock plots
        visualizer.ax = Mock()
        visualizer.position_line = Mock()
        visualizer.ground_truth_line = Mock()
        
        # Add some test data
        visualizer.update_position(np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]), 1.0)
        
        # Perform update
        visualizer.update_plots()
        
        # Should update line data
        assert visualizer.position_line.set_data_3d.called
        assert visualizer.ground_truth_line.set_data_3d.called
        
        # Should update axis limits
        assert visualizer.ax.set_xlim.called
        assert visualizer.ax.set_ylim.called
        assert visualizer.ax.set_zlim.called
    
    def test_position_error_calculation(self, visualizer):
        """Test position error calculation is correct."""
        estimated = np.array([1.0, 2.0, 3.0])
        ground_truth = np.array([1.2, 1.8, 3.1])
        
        error = visualizer.calculate_position_error(estimated, ground_truth)
        expected_error = np.linalg.norm(estimated - ground_truth)
        
        assert np.isclose(error, expected_error, atol=1e-10)
    
    def test_axis_scaling(self, visualizer):
        """Test automatic axis scaling works correctly."""
        # Add positions with known range
        positions = [
            np.array([0, 0, 0]),
            np.array([10, 5, 2]),
            np.array([-5, -3, -1])
        ]
        
        for i, pos in enumerate(positions):
            visualizer.update_position(pos, pos + 0.1, i * 0.1)
        
        x_range, y_range, z_range = visualizer.get_axis_ranges()
        
        # Check ranges are correct
        assert x_range[0] <= -5 and x_range[1] >= 10
        assert y_range[0] <= -3 and y_range[1] >= 5
        assert z_range[0] <= -1 and z_range[1] >= 2


class TestTrajectoryPlotter:
    @pytest.fixture
    def plotter(self):
        """Create a trajectory plotter for testing."""
        return TrajectoryPlotter()
    
    def test_plotter_initialization(self, plotter):
        """Test trajectory plotter initializes correctly."""
        assert plotter.figure_size == (12, 8)
        assert plotter.line_width == 2.0
        assert plotter.alpha == 0.8
    
    def test_trajectory_data_validation(self, plotter):
        """Test trajectory data validation."""
        # Valid data
        valid_trajectory = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert plotter.validate_trajectory_data(valid_trajectory)
        
        # Invalid data
        invalid_trajectory = np.array([[1, 2], [3, 4]])  # Wrong dimensions
        assert not plotter.validate_trajectory_data(invalid_trajectory)
        
        # Empty data
        empty_trajectory = np.array([])
        assert not plotter.validate_trajectory_data(empty_trajectory)
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_trajectory_comparison(self, mock_figure, plotter):
        """Test trajectory comparison plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Test data
        estimated_trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        ground_truth_trajectory = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1]])
        
        plotter.plot_trajectory_comparison(estimated_trajectory, ground_truth_trajectory)
        
        # Should create figure and axis
        mock_figure.assert_called_once()
        
        # Should plot both trajectories
        assert mock_ax.plot.call_count >= 2  # At least estimated and ground truth
        
        # Should set labels and title
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.set_zlabel.called
        assert mock_ax.set_title.called
        assert mock_ax.legend.called
    
    def test_error_statistics_calculation(self, plotter):
        """Test error statistics calculation."""
        estimated = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        ground_truth = np.array([[0.1, 0.1, 0.1], [1.2, 0.8, 1.1], [1.9, 2.1, 2.0]])
        
        stats = plotter.calculate_error_statistics(estimated, ground_truth)
        
        # Check required statistics are present
        assert 'mean_error' in stats
        assert 'max_error' in stats
        assert 'std_error' in stats
        assert 'rmse' in stats
        
        # Verify calculations are reasonable
        assert stats['mean_error'] > 0
        assert stats['max_error'] >= stats['mean_error']
        assert stats['std_error'] >= 0
        assert stats['rmse'] > 0
    
    def test_trajectory_smoothing(self, plotter):
        """Test trajectory smoothing functionality."""
        # Noisy trajectory
        t = np.linspace(0, 10, 50)
        clean_traj = np.column_stack([t, np.sin(t), np.cos(t)])
        noise = np.random.normal(0, 0.1, clean_traj.shape)
        noisy_traj = clean_traj + noise
        
        smoothed_traj = plotter.smooth_trajectory(noisy_traj, window_size=5)
        
        # Smoothed trajectory should have same shape
        assert smoothed_traj.shape == noisy_traj.shape
        
        # Smoothed trajectory should be closer to clean trajectory
        original_error = np.mean(np.linalg.norm(noisy_traj - clean_traj, axis=1))
        smoothed_error = np.mean(np.linalg.norm(smoothed_traj - clean_traj, axis=1))
        assert smoothed_error < original_error


class TestSensorHealthMonitor:
    @pytest.fixture
    def health_monitor(self):
        """Create a sensor health monitor for testing."""
        return SensorHealthMonitor()
    
    def test_health_monitor_initialization(self, health_monitor):
        """Test sensor health monitor initializes correctly."""
        assert health_monitor.sensor_types == ['gps', 'imu', 'odometry']
        assert health_monitor.update_interval == 1.0
        assert health_monitor.failure_threshold == 5
        
        # Check initial health status
        for sensor_type in health_monitor.sensor_types:
            assert sensor_type in health_monitor.sensor_health
    
    def test_sensor_health_update(self, health_monitor):
        """Test sensor health status updates."""
        # Update GPS health
        health_data = {
            'sensor_id': 1,
            'is_healthy': True,
            'failure_count': 0,
            'last_update': 10.5
        }
        
        health_monitor.update_sensor_health('gps', 1, health_data)
        
        # Check data was stored
        assert health_monitor.sensor_health['gps'][1] == health_data
    
    def test_failure_detection(self, health_monitor):
        """Test sensor failure detection logic."""
        # Simulate sensor failures
        for i in range(health_monitor.failure_threshold + 2):
            health_data = {
                'sensor_id': 1,
                'is_healthy': False,
                'failure_count': i + 1,
                'last_update': i * 0.1
            }
            health_monitor.update_sensor_health('gps', 1, health_data)
        
        # Should detect failure
        is_failed = health_monitor.is_sensor_failed('gps', 1)
        assert is_failed
    
    def test_redundancy_status(self, health_monitor):
        """Test redundancy status calculation."""
        # Set up GPS sensors - one healthy, one failed
        health_monitor.update_sensor_health('gps', 1, {
            'sensor_id': 1, 'is_healthy': True, 'failure_count': 0, 'last_update': 10.0
        })
        health_monitor.update_sensor_health('gps', 2, {
            'sensor_id': 2, 'is_healthy': False, 'failure_count': 10, 'last_update': 10.0
        })
        
        # Should still have redundancy (1 healthy sensor remaining)
        has_redundancy = health_monitor.has_redundancy('gps')
        assert has_redundancy
        
        # Fail the second GPS
        health_monitor.update_sensor_health('gps', 1, {
            'sensor_id': 1, 'is_healthy': False, 'failure_count': 10, 'last_update': 10.0
        })
        
        # Should lose redundancy (no healthy sensors)
        has_redundancy = health_monitor.has_redundancy('gps')
        assert not has_redundancy
    
    @patch('matplotlib.pyplot.figure')
    def test_health_status_plotting(self, mock_figure, health_monitor):
        """Test sensor health status plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Add some health data
        health_monitor.update_sensor_health('gps', 1, {
            'sensor_id': 1, 'is_healthy': True, 'failure_count': 0, 'last_update': 10.0
        })
        
        health_monitor.plot_health_status()
        
        # Should create figure
        mock_figure.assert_called_once()
        
        # Should create visualization elements
        assert mock_ax.bar.called or mock_ax.scatter.called
    
    def test_health_history_tracking(self, health_monitor):
        """Test health history tracking over time."""
        # Add health updates over time
        times = np.linspace(0, 10, 20)
        for i, t in enumerate(times):
            is_healthy = i % 3 != 0  # Intermittent failures
            health_data = {
                'sensor_id': 1,
                'is_healthy': is_healthy,
                'failure_count': max(0, i - 15),
                'last_update': t
            }
            health_monitor.update_sensor_health('gps', 1, health_data)
        
        # Get health history
        history = health_monitor.get_health_history('gps', 1)
        
        # Should have recorded history
        assert len(history) > 0
        assert len(history) <= len(times)  # May be limited by history length


class TestMetricsDisplay:
    @pytest.fixture
    def metrics_display(self):
        """Create a metrics display for testing."""
        return MetricsDisplay()
    
    def test_metrics_display_initialization(self, metrics_display):
        """Test metrics display initializes correctly."""
        assert metrics_display.update_rate == 2.0
        assert metrics_display.precision == 3
        assert metrics_display.units['position'] == 'm'
        assert metrics_display.units['velocity'] == 'm/s'
        assert metrics_display.units['time'] == 's'
    
    def test_position_error_tracking(self, metrics_display):
        """Test position error metrics tracking."""
        # Add position error data
        errors = [0.5, 1.2, 0.8, 1.5, 0.3]
        times = [1, 2, 3, 4, 5]
        
        for error, time in zip(errors, times):
            metrics_display.update_position_error(error, time)
        
        # Calculate statistics
        stats = metrics_display.get_position_error_stats()
        
        assert 'mean_error' in stats
        assert 'max_error' in stats
        assert 'current_error' in stats
        assert np.isclose(stats['max_error'], max(errors))
        assert np.isclose(stats['current_error'], errors[-1])
    
    def test_sensor_availability_tracking(self, metrics_display):
        """Test sensor availability metrics."""
        # Simulate sensor availability over time
        sensor_status = {
            'gps_1': [True, True, False, True, True],
            'gps_2': [True, False, False, True, True],
            'imu_1': [True, True, True, True, False],
            'imu_2': [True, True, True, True, True]
        }
        
        for i in range(5):
            availability = {}
            for sensor, status_list in sensor_status.items():
                availability[sensor] = status_list[i]
            metrics_display.update_sensor_availability(availability, i + 1)
        
        # Get availability statistics
        stats = metrics_display.get_availability_stats()
        
        # Check GPS redundancy (should be available when at least one GPS works)
        gps_availability = stats['gps_redundancy']
        expected_gps = [s1 or s2 for s1, s2 in zip(sensor_status['gps_1'], sensor_status['gps_2'])]
        assert gps_availability == np.mean(expected_gps)
    
    def test_fusion_confidence_tracking(self, metrics_display):
        """Test fusion confidence metrics."""
        # Add confidence data
        confidences = [0.9, 0.85, 0.7, 0.95, 0.8]
        times = [1, 2, 3, 4, 5]
        
        for conf, time in zip(confidences, times):
            metrics_display.update_fusion_confidence(conf, time)
        
        # Get confidence statistics
        stats = metrics_display.get_confidence_stats()
        
        assert 'mean_confidence' in stats
        assert 'min_confidence' in stats
        assert 'current_confidence' in stats
        assert np.isclose(stats['min_confidence'], min(confidences))
        assert np.isclose(stats['current_confidence'], confidences[-1])
    
    @patch('matplotlib.pyplot.figure')
    def test_metrics_plotting(self, mock_figure, metrics_display):
        """Test metrics plotting functionality."""
        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(4)]  # 4 subplots
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)
        
        # Add some test data
        for i in range(10):
            metrics_display.update_position_error(i * 0.1, i)
            metrics_display.update_fusion_confidence(0.9 - i * 0.01, i)
        
        metrics_display.plot_metrics()
        
        # Should create figure with subplots
        mock_figure.assert_called_once()
        
        # Should plot on each axis
        for ax in mock_axes:
            assert ax.plot.called or ax.bar.called
    
    def test_metrics_text_formatting(self, metrics_display):
        """Test metrics text formatting."""
        # Test position error formatting
        error = 1.23456
        formatted = metrics_display.format_metric(error, 'position')
        expected = f"1.235 {metrics_display.units['position']}"
        assert formatted == expected
        
        # Test velocity formatting
        velocity = 2.6789
        formatted = metrics_display.format_metric(velocity, 'velocity')
        expected = f"2.679 {metrics_display.units['velocity']}"
        assert formatted == expected


class TestVisualizationIntegration:
    def test_complete_visualization_system(self):
        """Test complete visualization system integration."""
        # Create all components
        visualizer = RealTimeVisualizer(update_rate=1.0)
        health_monitor = SensorHealthMonitor()
        metrics_display = MetricsDisplay()
        
        # Simulate sensor data updates
        for i in range(10):
            time = i * 0.1
            
            # Update positions
            estimated_pos = np.array([i, i * 0.5, np.sin(i)])
            ground_truth_pos = estimated_pos + np.random.normal(0, 0.1, 3)
            visualizer.update_position(estimated_pos, ground_truth_pos, time)
            
            # Update sensor health
            gps_health = {'sensor_id': 1, 'is_healthy': i % 5 != 0, 'failure_count': max(0, i-5), 'last_update': time}
            health_monitor.update_sensor_health('gps', 1, gps_health)
            
            # Update metrics
            position_error = np.linalg.norm(estimated_pos - ground_truth_pos)
            metrics_display.update_position_error(position_error, time)
            metrics_display.update_fusion_confidence(0.9 - i * 0.01, time)
        
        # Verify all components have data
        assert len(visualizer.position_history) > 0
        assert len(health_monitor.sensor_health['gps']) > 0
        assert len(metrics_display.position_errors) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_synchronized_updates(self, mock_show):
        """Test synchronized updates across visualization components."""
        visualizer = RealTimeVisualizer(update_rate=1.0)
        # visualizer.setup_plots()  # This method doesn't exist, plot is setup in __init__
        
        # Mock the plot objects
        visualizer.ax = Mock()
        visualizer.position_line = Mock()
        visualizer.ground_truth_line = Mock()
        
        # Add data and update
        visualizer.update_position(np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]), 1.0)
        visualizer.update_plots()
        
        # Should update all plot elements
        assert visualizer.position_line.set_data_3d.called
        assert visualizer.ground_truth_line.set_data_3d.called
        assert visualizer.ax.set_xlim.called 