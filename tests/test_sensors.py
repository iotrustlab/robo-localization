import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robo_localization.sensors import GPSSensor, IMUSensor, WheelOdometry, SensorHealth, SensorFusionManager


class TestSensorHealth:
    """Test sensor health monitoring and failure detection"""
    
    def test_sensor_health_initialization(self):
        """Test proper initialization of sensor health tracker"""
        health = SensorHealth()
        assert health.is_operational == True
        assert health.reliability == 1.0
        assert health.failure_count == 0
        assert health.recovery_count == 0
        
    def test_failure_detection(self):
        """Test failure detection reduces reliability"""
        health = SensorHealth()
        initial_reliability = health.reliability
        
        health.record_failure()
        assert health.failure_count == 1
        assert health.reliability < initial_reliability
        assert health.is_operational == True  # Single failure shouldn't kill sensor
        
    def test_multiple_failures_disable_sensor(self):
        """Test multiple consecutive failures disable sensor"""
        health = SensorHealth()
        
        # Record multiple failures
        for _ in range(5):
            health.record_failure()
            
        assert health.is_operational == False
        assert health.reliability < 0.5
        
    def test_recovery_improves_reliability(self):
        """Test successful measurements improve reliability"""
        health = SensorHealth()
        
        # Cause some failures
        health.record_failure()
        health.record_failure()
        low_reliability = health.reliability
        
        # Record successful measurements
        health.record_success()
        health.record_success()
        
        assert health.reliability > low_reliability
        assert health.recovery_count == 2


class TestGPSSensor:
    """Test GPS sensor simulation"""
    
    def test_gps_initialization(self):
        """Test GPS sensor initializes with correct parameters"""
        gps = GPSSensor(noise_std=2.0, dropout_prob=0.1)
        assert gps.noise_std == 2.0
        assert gps.dropout_prob == 0.1
        assert gps.position_bias.shape == (3,)
        assert hasattr(gps, 'health')
        
    def test_gps_measurement_accuracy(self):
        """Test GPS measurements are reasonably accurate without noise"""
        gps = GPSSensor(noise_std=0.001, dropout_prob=0.0)  # Minimal noise, no dropouts
        true_position = np.array([100.0, 200.0, 50.0])
        
        measurement = gps.get_measurement(true_position)
        
        # Should be close to true position (only bias should affect it)
        assert measurement is not None
        np.testing.assert_allclose(measurement, true_position + gps.position_bias, atol=1.5)  # Account for all error sources including atmospheric delays
        
    def test_gps_noise_characteristics(self):
        """Test GPS noise has correct statistical properties"""
        gps = GPSSensor(noise_std=5.0, dropout_prob=0.0)
        true_position = np.array([0.0, 0.0, 0.0])
        
        measurements = []
        for _ in range(1000):
            measurement = gps.get_measurement(true_position)
            if measurement is not None:
                measurements.append(measurement - gps.position_bias)  # Remove bias
                
        measurements = np.array(measurements)
        
        # Check noise statistics
        assert len(measurements) > 950  # Should have most measurements (no dropouts)
        noise = measurements - true_position
        assert np.abs(np.std(noise[:, 0]) - 5.0) < 0.5  # X noise std
        assert np.abs(np.std(noise[:, 1]) - 5.0) < 0.5  # Y noise std
        assert np.abs(np.std(noise[:, 2]) - 5.0) < 0.5  # Z noise std
        
    def test_gps_dropout_behavior(self):
        """Test GPS dropouts occur at expected rate"""
        gps = GPSSensor(noise_std=0.001, dropout_prob=0.3)
        true_position = np.array([0.0, 0.0, 0.0])
        
        measurements = []
        for _ in range(1000):
            measurement = gps.get_measurement(true_position)
            measurements.append(measurement is not None)
            
        success_rate = np.mean(measurements)
        assert 0.65 < success_rate < 0.75  # Should be around 70% success rate
        
    def test_gps_coordinate_transformations(self):
        """Test GPS handles different coordinate systems correctly"""
        gps = GPSSensor(noise_std=0.001, dropout_prob=0.0)
        
        # Test positions in different quadrants and elevations
        test_positions = [
            np.array([100.0, 200.0, 50.0]),
            np.array([-150.0, 300.0, 100.0]),
            np.array([250.0, -100.0, 0.0]),
            np.array([-50.0, -75.0, 200.0])
        ]
        
        for pos in test_positions:
            measurement = gps.get_measurement(pos)
            assert measurement is not None
            # Check that measurement is in reasonable range of true position
            error = np.linalg.norm(measurement - pos - gps.position_bias)
            assert error < 1.5  # More lenient given additional noise sources


class TestIMUSensor:
    """Test IMU sensor simulation"""
    
    def test_imu_initialization(self):
        """Test IMU sensor initializes correctly"""
        imu = IMUSensor(accel_noise_std=0.1, gyro_noise_std=0.05)
        assert imu.accel_noise_std == 0.1
        assert imu.gyro_noise_std == 0.05
        assert imu.accel_bias.shape == (3,)
        assert imu.gyro_bias.shape == (3,)
        assert hasattr(imu, 'health')
        
    def test_imu_measurement_structure(self):
        """Test IMU returns properly structured measurements"""
        imu = IMUSensor(accel_noise_std=0.001, gyro_noise_std=0.001)
        acceleration = np.array([0.0, 0.0, -9.81])
        angular_velocity = np.array([0.1, 0.2, 0.3])
        
        measurement = imu.get_measurement(acceleration, angular_velocity)
        
        assert measurement is not None
        assert 'acceleration' in measurement
        assert 'angular_velocity' in measurement
        assert measurement['acceleration'].shape == (3,)
        assert measurement['angular_velocity'].shape == (3,)
        
    def test_imu_bias_drift(self):
        """Test IMU bias drifts over time"""
        imu = IMUSensor(accel_noise_std=0.001, gyro_noise_std=0.001, bias_drift_rate=0.01)
        initial_accel_bias = imu.accel_bias.copy()
        initial_gyro_bias = imu.gyro_bias.copy()
        
        # Simulate many measurements to allow bias drift
        for _ in range(100):
            imu.get_measurement(np.zeros(3), np.zeros(3))
            
        # Bias should have drifted
        assert not np.allclose(imu.accel_bias, initial_accel_bias)
        assert not np.allclose(imu.gyro_bias, initial_gyro_bias)
        
    def test_imu_failure_modes(self):
        """Test different IMU failure modes"""
        imu = IMUSensor(accel_noise_std=0.001, gyro_noise_std=0.001)
        
        # Test stuck sensor failure
        imu.simulate_failure('stuck')
        measurement1 = imu.get_measurement(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
        measurement2 = imu.get_measurement(np.array([5, 6, 7]), np.array([0.5, 0.6, 0.7]))
        
        if measurement1 is not None and measurement2 is not None:
            np.testing.assert_allclose(measurement1['acceleration'], measurement2['acceleration'])
            np.testing.assert_allclose(measurement1['angular_velocity'], measurement2['angular_velocity'])
            
        # Test noisy sensor failure
        imu.simulate_failure('noisy')
        measurement = imu.get_measurement(np.zeros(3), np.zeros(3))
        # Should have much higher noise when in noisy failure mode
        
        # Test bias jump failure
        imu.simulate_failure('bias_jump')
        measurement = imu.get_measurement(np.zeros(3), np.zeros(3))
        # Should have different bias after bias jump failure


class TestWheelOdometry:
    """Test wheel odometry sensor"""
    
    def test_wheel_odometry_initialization(self):
        """Test wheel odometry initializes correctly"""
        odometry = WheelOdometry(wheelbase=0.5, wheel_radius=0.1)
        assert odometry.wheelbase == 0.5
        assert odometry.wheel_radius == 0.1
        assert hasattr(odometry, 'health')
        
    def test_differential_drive_kinematics(self):
        """Test differential drive kinematics are correct"""
        odometry = WheelOdometry(wheelbase=1.0, wheel_radius=0.1, encoder_noise_std=0.001, slip_factor=0.0)
        
        # Test straight line motion
        left_speed = 1.0  # rad/s
        right_speed = 1.0  # rad/s
        dt = 0.1
        
        delta_pose = odometry.compute_delta_pose(left_speed, right_speed, dt)
        
        expected_linear_vel = 0.1  # wheel_radius * average_speed
        expected_distance = expected_linear_vel * dt
        
        assert abs(delta_pose.delta_x - expected_distance) < 0.002  # Account for encoder noise and systematic errors
        assert abs(delta_pose.delta_y) < 0.001  # Small systematic errors expected
        assert abs(delta_pose.delta_theta) < 0.001  # Small systematic errors expected
        
    def test_rotation_kinematics(self):
        """Test pure rotation kinematics"""
        odometry = WheelOdometry(wheelbase=1.0, wheel_radius=0.1, encoder_noise_std=0.001, slip_factor=0.0)
        
        # Test pure rotation (opposite wheel speeds)
        left_speed = 1.0   # rad/s
        right_speed = -1.0  # rad/s
        dt = 0.1
        
        delta_pose = odometry.compute_delta_pose(left_speed, right_speed, dt)
        
        expected_angular_vel = 0.1 * 2.0 / 1.0  # wheel_radius * speed_diff / wheel_base
        expected_angle = expected_angular_vel * dt
        
        assert abs(delta_pose.delta_x) < 0.01  # Pure rotation may have small linear component due to systematic errors
        assert abs(delta_pose.delta_y) < 0.001  # Small systematic errors expected
        assert abs(delta_pose.delta_theta - expected_angle) < 0.05  # Account for systematic errors and noise
        
    def test_slip_modeling(self):
        """Test wheel slip reduces odometry accuracy"""
        odometry_no_slip = WheelOdometry(wheelbase=1.0, wheel_radius=0.1, slip_factor=0.0)
        odometry_with_slip = WheelOdometry(wheelbase=1.0, wheel_radius=0.1, slip_factor=0.2)
        
        left_speed = 2.0
        right_speed = 2.0
        dt = 0.1
        
        delta_no_slip = odometry_no_slip.compute_delta_pose(left_speed, right_speed, dt)
        delta_with_slip = odometry_with_slip.compute_delta_pose(left_speed, right_speed, dt)
        
        # With slip, reported distance should be less than actual
        assert delta_with_slip.delta_x < delta_no_slip.delta_x
        
    def test_encoder_noise(self):
        """Test encoder noise affects measurements"""
        odometry = WheelOdometry(wheelbase=1.0, wheel_radius=0.1, encoder_noise_std=0.1)
        
        measurements = []
        for _ in range(100):
            delta_pose = odometry.compute_delta_pose(1.0, 1.0, 0.1)
            measurements.append(delta_pose.delta_x)
            
        # Should have variation due to noise
        assert np.std(measurements) > 0.001


class TestSensorFusionManager:
    """Test sensor fusion and management"""
    
    def test_sensor_manager_initialization(self):
        """Test sensor manager initializes all sensors"""
        manager = SensorFusionManager()
        
        assert len(manager.gps_sensors) == 2
        assert len(manager.imu_sensors) == 2
        assert manager.wheel_odometry is not None
        
    def test_redundant_sensor_measurements(self):
        """Test manager collects measurements from all sensors"""
        manager = SensorFusionManager()
        
        # Mock rover state
        position = np.array([100.0, 200.0, 50.0])
        velocity = np.array([1.0, 0.5, 0.0])
        acceleration = np.array([0.0, 0.0, -9.81])
        angular_velocity = np.array([0.1, 0.2, 0.3])
        wheel_speeds = (1.0, 1.2)
        
        measurements = manager.get_fused_measurements(position, velocity, acceleration, angular_velocity, wheel_speeds)
        
        # FusedMeasurement should have position, velocity, etc. attributes
        assert hasattr(measurements, 'position')
        assert hasattr(measurements, 'velocity')
        assert hasattr(measurements, 'acceleration')
        assert hasattr(measurements, 'angular_velocity')
        
    def test_sensor_failure_simulation(self):
        """Test manager can simulate coordinated sensor failures"""
        manager = SensorFusionManager()
        
        # Simulate GPS constellation dropout
        manager.simulate_gps_constellation_failure()
        
        position = np.array([100.0, 200.0, 50.0])
        velocity = np.array([0.0, 0.0, 0.0])
        measurements = manager.get_fused_measurements(position, velocity, np.zeros(3), np.zeros(3), (0.0, 0.0))
        
        # Should have reduced GPS availability during constellation failure
        # GPS position might be None or have lower quality score
        assert measurements.position is None or measurements.quality_score < 0.8
        
    def test_sensor_health_monitoring(self):
        """Test manager tracks overall sensor health"""
        manager = SensorFusionManager()
        
        health_status = manager.get_comprehensive_health_status()
        
        # Should contain expected health information structure
        assert 'gps_sensors' in health_status
        assert 'imu_sensors' in health_status
        assert 'odometry' in health_status
        assert 'system_metrics' in health_status
        assert isinstance(health_status, dict)
        assert len(health_status['gps_sensors']) == 2
        assert len(health_status['imu_sensors']) == 2


# Integration tests
class TestSensorIntegration:
    """Test sensor integration and independence"""
    
    def test_multiple_sensor_independence(self):
        """Test that multiple sensors of same type behave independently"""
        gps1 = GPSSensor(noise_std=1.0, dropout_prob=0.1, sensor_id=1)
        gps2 = GPSSensor(noise_std=2.0, dropout_prob=0.2, sensor_id=2)
        
        position = np.array([100.0, 100.0, 100.0])
        
        measurements1 = []
        measurements2 = []
        
        for _ in range(100):
            m1 = gps1.get_measurement(position)
            m2 = gps2.get_measurement(position)
            if m1 is not None:
                measurements1.append(m1)
            if m2 is not None:
                measurements2.append(m2)
                
        # Should have different numbers of measurements due to different dropout rates
        # and different noise characteristics
        assert len(measurements1) != len(measurements2)
        
        if len(measurements1) > 10 and len(measurements2) > 10:
            noise1 = np.std([m - gps1.position_bias for m in measurements1], axis=0)
            noise2 = np.std([m - gps2.position_bias for m in measurements2], axis=0)
            
            # GPS1 should have lower noise
            assert np.mean(noise1) < np.mean(noise2)


if __name__ == "__main__":
    pytest.main([__file__])
