import pytest
import numpy as np
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robo_localization.fusion import StateVector, CovarianceMatrix, ExtendedKalmanFilter


class TestStateVector:
    """Test 12D state vector for rover [position, velocity, orientation, angular_velocity]"""
    
    def test_state_vector_initialization(self):
        """Test state vector initializes with correct dimensions and values"""
        state = StateVector()
        
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.orientation.shape == (3,)
        assert state.angular_velocity.shape == (3,)
        assert state.to_array().shape == (12,)
        
    def test_state_vector_from_array(self):
        """Test state vector can be created from numpy array"""
        test_array = np.array([1, 2, 3, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.01, 0.02, 0.03])
        state = StateVector.from_array(test_array)
        
        np.testing.assert_allclose(state.position, [1, 2, 3])
        np.testing.assert_allclose(state.velocity, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state.orientation, [0.5, 0.6, 0.7])
        np.testing.assert_allclose(state.angular_velocity, [0.01, 0.02, 0.03])
        
    def test_state_vector_to_array(self):
        """Test state vector can be converted to numpy array"""
        state = StateVector()
        test_array = np.array([10, 20, 30, 1, 2, 3, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        state.update_from_array(test_array)
        
        array = state.to_array()
        expected = np.array([10, 20, 30, 1, 2, 3, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        
        np.testing.assert_allclose(array, expected)
        
    def test_state_vector_update(self):
        """Test state vector can be updated from array"""
        state = StateVector()
        new_values = np.array([5, 10, 15, 0.5, 1.0, 1.5, 0.2, 0.4, 0.6, 0.05, 0.1, 0.15])
        
        state.update_from_array(new_values)
        
        np.testing.assert_allclose(state.position, [5, 10, 15])
        np.testing.assert_allclose(state.velocity, [0.5, 1.0, 1.5])
        np.testing.assert_allclose(state.orientation, [0.2, 0.4, 0.6])
        np.testing.assert_allclose(state.angular_velocity, [0.05, 0.1, 0.15])


class TestCovarianceMatrix:
    """Test covariance matrix handling and properties"""
    
    def test_covariance_initialization(self):
        """Test covariance matrix initializes correctly"""
        cov = CovarianceMatrix(size=12)
        
        assert cov.matrix.shape == (12, 12)
        # Should be positive definite (all eigenvalues > 0)
        eigenvals = np.linalg.eigvals(cov.matrix)
        assert np.all(eigenvals > 0)
        
    def test_covariance_positive_definiteness(self):
        """Test covariance matrix maintains positive definiteness"""
        cov = CovarianceMatrix(size=3)
        
        # Add some noise to make it more realistic
        noise = np.random.randn(3, 3) * 0.1
        cov.matrix += noise @ noise.T  # Ensures positive semi-definite
        
        cov.ensure_positive_definite()
        
        eigenvals = np.linalg.eigvals(cov.matrix)
        assert np.all(eigenvals > 0)
        
    def test_covariance_uncertainty_tracking(self):
        """Test covariance properly tracks uncertainty"""
        cov = CovarianceMatrix(size=3, initial_uncertainty=0.1)
        
        # Initial uncertainty should be reflected in diagonal
        assert np.all(np.diag(cov.matrix) >= 0.1)
        
        # Adding process noise should increase uncertainty
        process_noise = np.eye(3) * 0.05
        initial_trace = np.trace(cov.matrix)
        
        cov.add_process_noise(process_noise)
        
        assert np.trace(cov.matrix) > initial_trace
        
    def test_covariance_update(self):
        """Test covariance matrix basic functionality"""
        cov = CovarianceMatrix(size=2, initial_uncertainty=1.0)
        
        # Test that matrix properties are maintained
        initial_trace = np.trace(cov.matrix)
        assert initial_trace > 0
        
        # Test condition number calculation
        condition_num = cov.get_condition_number()
        assert condition_num > 0
        
        # Test that matrix is well-conditioned initially
        assert cov.is_well_conditioned()


class TestExtendedKalmanFilter:
    """Test Extended Kalman Filter implementation"""
    
    def test_ekf_initialization(self):
        """Test EKF initializes with correct state and covariance"""
        ekf = ExtendedKalmanFilter()
        
        assert ekf.state.get_full_state().shape == (12,)
        assert ekf.covariance.matrix.shape == (12, 12)
        assert hasattr(ekf, 'process_noise')
        assert hasattr(ekf, 'measurement_noise')
        
    def test_ekf_prediction_step(self):
        """Test EKF prediction step updates state and covariance"""
        ekf = ExtendedKalmanFilter()
        
        # Set initial state
        initial_state = ekf.state.get_full_state().copy()
        initial_covariance_trace = np.trace(ekf.covariance.matrix)
        
        dt = 0.1
        ekf.predict(dt)
        
        # State should change due to motion model
        new_state = ekf.state.get_full_state()
        # Position should update based on velocity
        position_change = new_state[:3] - initial_state[:3]
        expected_change = initial_state[3:6] * dt  # velocity * dt
        
        np.testing.assert_allclose(position_change, expected_change, atol=0.01)
        
        # Covariance should increase due to process noise
        assert np.trace(ekf.covariance.matrix) > initial_covariance_trace
        
    def test_ekf_gps_update(self):
        """Test EKF GPS measurement update"""
        ekf = ExtendedKalmanFilter()
        
        # Set known position
        ekf.state.position = np.array([100.0, 200.0, 50.0])
        initial_uncertainty = np.trace(ekf.covariance.matrix)
        
        # GPS measurement close to true position
        gps_measurement = np.array([101.0, 199.0, 51.0])
        gps_noise = np.eye(3) * 2.0  # 2m standard deviation
        
        ekf.update_gps(gps_measurement, gps_noise)
        
        # Position should be updated toward measurement
        updated_position = ekf.state.position
        assert np.linalg.norm(updated_position - gps_measurement) < 2.0
        
        # Uncertainty should be reduced
        assert np.trace(ekf.covariance.matrix) < initial_uncertainty
        
    def test_ekf_imu_update(self):
        """Test EKF IMU measurement update"""
        ekf = ExtendedKalmanFilter()
        
        # Set known velocity and angular velocity
        ekf.state.velocity = np.array([1.0, 0.0, 0.0])
        ekf.state.angular_velocity = np.array([0.0, 0.0, 0.1])
        
        # IMU measurement
        imu_measurement = {
            'acceleration': np.array([0.1, 0.0, -9.81]),
            'angular_velocity': np.array([0.0, 0.0, 0.09])
        }
        imu_noise = {
            'acceleration': np.eye(3) * 0.1,
            'angular_velocity': np.eye(3) * 0.05
        }
        
        initial_uncertainty = np.trace(ekf.covariance.matrix)
        ekf.update_imu(imu_measurement, imu_noise)
        
        # Angular velocity should be updated toward measurement
        updated_angular_vel = ekf.state.angular_velocity
        assert abs(updated_angular_vel[2] - 0.09) < 0.02
        
        # Uncertainty should be reduced
        assert np.trace(ekf.covariance.matrix) < initial_uncertainty
        
    def test_ekf_odometry_update(self):
        """Test EKF wheel odometry measurement update"""
        ekf = ExtendedKalmanFilter()
        
        # Set known position and orientation
        ekf.state.position = np.array([0.0, 0.0, 0.0])
        ekf.state.orientation = np.array([0.0, 0.0, 0.0])  # Facing x-direction
        
        # Odometry measurement: moved forward 1m
        odometry_measurement = {
            'dx': 1.0,
            'dy': 0.0,
            'dtheta': 0.0
        }
        odometry_noise = {
            'position': np.eye(2) * 0.1,  # x, y uncertainty
            'orientation': 0.05  # theta uncertainty
        }
        
        initial_position = ekf.state.position.copy()
        ekf.update_odometry(odometry_measurement, odometry_noise)
        
        # Position should be updated based on odometry
        position_change = ekf.state.position - initial_position
        assert abs(position_change[0] - 1.0) < 0.2  # Should move ~1m in x
        assert abs(position_change[1]) < 0.1  # Little change in y
        
    def test_ekf_redundant_sensor_fusion(self):
        """Test EKF handles multiple sensors of same type"""
        ekf = ExtendedKalmanFilter()
        
        # Multiple GPS measurements
        gps_measurements = [
            np.array([100.0, 200.0, 50.0]),
            np.array([101.0, 201.0, 49.0])
        ]
        gps_noises = [
            np.eye(3) * 2.0,
            np.eye(3) * 3.0
        ]
        
        initial_uncertainty = np.trace(ekf.covariance.matrix)
        
        # Apply both GPS updates
        for gps_meas, gps_noise in zip(gps_measurements, gps_noises):
            ekf.update_gps(gps_meas, gps_noise)
            
        # Final position should be influenced by both measurements
        final_position = ekf.state.position
        
        # Should be somewhere between the two measurements
        avg_measurement = np.mean(gps_measurements, axis=0)
        assert np.linalg.norm(final_position - avg_measurement) < 5.0
        
        # Uncertainty should be significantly reduced
        assert np.trace(ekf.covariance.matrix) < initial_uncertainty * 0.5
        
    def test_ekf_outlier_detection(self):
        """Test EKF can detect and reject outlier measurements"""
        ekf = ExtendedKalmanFilter()
        
        # Set known position
        ekf.state.position = np.array([100.0, 200.0, 50.0])
        
        # Normal GPS measurement
        normal_gps = np.array([101.0, 199.0, 51.0])
        gps_noise = np.eye(3) * 2.0
        
        ekf.update_gps(normal_gps, gps_noise)
        position_after_normal = ekf.state.position.copy()
        
        # Outlier GPS measurement (very far from current estimate)
        outlier_gps = np.array([500.0, 600.0, 200.0])
        
        # EKF should detect this as outlier and reject or heavily discount it
        ekf.update_gps(outlier_gps, gps_noise)
        position_after_outlier = ekf.state.position
        
        # Position shouldn't change dramatically due to outlier
        change = np.linalg.norm(position_after_outlier - position_after_normal)
        assert change < 50.0  # Should not jump to outlier position
        
    def test_ekf_convergence_properties(self):
        """Test EKF converges to true state with sufficient measurements"""
        ekf = ExtendedKalmanFilter()
        
        # True position
        true_position = np.array([150.0, 250.0, 75.0])
        
        # Start with incorrect initial estimate
        ekf.state.position = np.array([0.0, 0.0, 0.0])
        
        # Apply many GPS measurements with noise around true position
        gps_noise = np.eye(3) * 1.0
        
        for _ in range(50):
            # Noisy measurement around true position
            noise = np.random.normal(0, 1.0, 3)
            gps_measurement = true_position + noise
            ekf.update_gps(gps_measurement, gps_noise)
            
        # Should converge close to true position
        final_position = ekf.state.position
        error = np.linalg.norm(final_position - true_position)
        assert error < 5.0  # Should be within 5m of true position
        
    def test_ekf_motion_model_consistency(self):
        """Test EKF motion model is mathematically consistent"""
        ekf = ExtendedKalmanFilter()
        
        # Set initial conditions
        ekf.state.position = np.array([0.0, 0.0, 0.0])
        ekf.state.velocity = np.array([2.0, 1.0, 0.0])
        ekf.state.orientation = np.array([0.0, 0.0, 0.5])  # 0.5 rad rotation
        ekf.state.angular_velocity = np.array([0.0, 0.0, 0.1])
        
        # Predict forward in time
        dt = 1.0
        initial_state = ekf.state.get_full_state().copy()
        
        ekf.predict(dt)
        
        new_state = ekf.state.get_full_state()
        
        # Check that position updated correctly
        position_change = new_state[:3] - initial_state[:3]
        expected_position_change = initial_state[3:6] * dt  # velocity * dt
        
        np.testing.assert_allclose(position_change, expected_position_change, atol=0.01)
        
        # Check that orientation updated correctly
        orientation_change = new_state[6:9] - initial_state[6:9]
        expected_orientation_change = initial_state[9:12] * dt  # angular_velocity * dt
        
        np.testing.assert_allclose(orientation_change, expected_orientation_change, atol=0.01)
        
    def test_ekf_sensor_dropout_handling(self):
        """Test EKF handles sensor dropouts gracefully"""
        ekf = ExtendedKalmanFilter()
        
        # Normal operation with measurements
        gps_measurement = np.array([100.0, 200.0, 50.0])
        gps_noise = np.eye(3) * 2.0
        
        # Apply several normal updates
        for _ in range(10):
            ekf.predict(0.1)
            ekf.update_gps(gps_measurement, gps_noise)
            
        uncertainty_with_measurements = np.trace(ekf.covariance.matrix)
        
        # Now simulate sensor dropout (only prediction steps)
        for _ in range(20):
            ekf.predict(0.1)
            
        uncertainty_without_measurements = np.trace(ekf.covariance.matrix)
        
        # Uncertainty should increase during dropout
        assert uncertainty_without_measurements > uncertainty_with_measurements
        
        # But filter should still be stable (no infinite values)
        assert np.all(np.isfinite(ekf.state.get_full_state()))
        assert np.all(np.isfinite(ekf.covariance.matrix))
        
    def test_ekf_jacobian_accuracy(self):
        """Test EKF Jacobian matrices are computed correctly"""
        ekf = ExtendedKalmanFilter()
        
        # Set a specific state
        state_array = np.array([10, 20, 5, 1, 0.5, 0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        ekf.state.update_from_array(state_array)
        
        # Compute Jacobian numerically and analytically
        dt = 0.1
        epsilon = 1e-6
        
        # Get analytical Jacobian
        F_analytical = ekf.compute_motion_jacobian(dt)
        
        # Compute numerical Jacobian
        F_numerical = np.zeros((12, 12))
        
        for i in range(12):
            # Perturb state in direction i
            state_plus = state_array.copy()
            state_plus[i] += epsilon
            
            state_minus = state_array.copy()
            state_minus[i] -= epsilon
            
            # Compute motion model for perturbed states
            ekf.state.update_from_array(state_plus)
            state_plus_predicted = ekf.motion_model(dt)
            
            ekf.state.update_from_array(state_minus)
            state_minus_predicted = ekf.motion_model(dt)
            
            # Numerical derivative
            F_numerical[:, i] = (state_plus_predicted - state_minus_predicted) / (2 * epsilon)
            
        # Reset original state
        ekf.state.update_from_array(state_array)
        
        # Compare analytical and numerical Jacobians
        np.testing.assert_allclose(F_analytical, F_numerical, atol=1e-4)


class TestExtendedKalmanFilterIntegration:
    """Integration tests for EKF with multiple sensor types"""
    
    def test_full_sensor_fusion_scenario(self):
        """Test complete sensor fusion scenario"""
        ekf = ExtendedKalmanFilter()
        
        # Simulate rover moving in a straight line
        true_positions = []
        estimated_positions = []
        
        dt = 0.1
        for t in np.arange(0, 5.0, dt):
            # True motion: constant velocity
            true_pos = np.array([t * 2.0, t * 1.0, 0.0])  # 2 m/s in x, 1 m/s in y
            true_positions.append(true_pos)
            
            # Prediction step
            ekf.predict(dt)
            
            # GPS update (every 10 steps)
            if int(t / dt) % 10 == 0:
                gps_noise_vec = np.random.normal(0, 1.0, 3)
                gps_measurement = true_pos + gps_noise_vec
                gps_noise_matrix = np.eye(3) * 1.0
                ekf.update_gps(gps_measurement, gps_noise_matrix)
                
            # IMU update (every step)
            imu_measurement = {
                'acceleration': np.array([0.0, 0.0, -9.81]) + np.random.normal(0, 0.1, 3),
                'angular_velocity': np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.05, 3)
            }
            imu_noise = {
                'acceleration': np.eye(3) * 0.1,
                'angular_velocity': np.eye(3) * 0.05
            }
            ekf.update_imu(imu_measurement, imu_noise)
            
            estimated_positions.append(ekf.state.position.copy())
            
        # Check final accuracy
        final_error = np.linalg.norm(estimated_positions[-1] - true_positions[-1])
        assert final_error < 5.0  # Should be within 5m after 5 seconds
        
        # Check that estimates generally improve over time
        errors = [np.linalg.norm(est - true) for est, true in zip(estimated_positions, true_positions)]
        
        # Later errors should generally be smaller than early errors
        early_error = np.mean(errors[:10])
        late_error = np.mean(errors[-10:])
        assert late_error < early_error * 2  # Should not get much worse


if __name__ == "__main__":
    pytest.main([__file__])
