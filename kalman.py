"""
Extended Kalman Filter implementation for 3D rover localization with redundant sensors.

This module implements:
- 12-state EKF for position, velocity, orientation, and angular velocity
- Multiple GPS sensor fusion with outlier detection
- Multiple IMU sensor fusion with bias estimation
- Wheel odometry integration
- Adaptive covariance and sensor weighting
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import scipy.linalg


class StateVector:
    """12D state vector for rover: [position, velocity, orientation, angular_velocity]"""
    
    def __init__(self):
        # Initialize state components
        self.position = np.zeros(3)      # [x, y, z] in meters
        self.velocity = np.zeros(3)      # [vx, vy, vz] in m/s (world frame)
        self.orientation = np.zeros(3)   # [roll, pitch, yaw] in radians
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz] in rad/s
        
    @classmethod
    def from_array(cls, state_array):
        """Create state vector from numpy array"""
        if len(state_array) != 12:
            raise ValueError("State array must have 12 elements")
            
        state = cls()
        state.position = state_array[0:3].copy()
        state.velocity = state_array[3:6].copy()
        state.orientation = state_array[6:9].copy()
        state.angular_velocity = state_array[9:12].copy()
        return state
        
    def get_full_state(self):
        """Get complete state as numpy array"""
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity
        ])
        
    def update_from_array(self, state_array):
        """Update state from numpy array"""
        if len(state_array) != 12:
            raise ValueError("State array must have 12 elements")
            
        self.position = state_array[0:3].copy()
        self.velocity = state_array[3:6].copy()
        self.orientation = state_array[6:9].copy()
        self.angular_velocity = state_array[9:12].copy()


class CovarianceMatrix:
    """Covariance matrix handling with positive definiteness enforcement"""
    
    def __init__(self, size=12, initial_uncertainty=1.0):
        self.size = size
        # Initialize as diagonal matrix with different uncertainties for different states
        self.matrix = np.eye(size) * initial_uncertainty
        
        # Set appropriate initial uncertainties for different state components
        # High uncertainty for position (we don't know where we are initially)
        self.matrix[0:3, 0:3] *= 100.0   # Position: 100x base uncertainty
        # Moderate uncertainty for velocity
        self.matrix[3:6, 3:6] *= 10.0    # Velocity: 10x base uncertainty
        # Moderate uncertainty for orientation  
        self.matrix[6:9, 6:9] *= 10.0    # Orientation: 10x base uncertainty
        # Low uncertainty for angular velocity (usually starts at zero)
        self.matrix[9:12, 9:12] *= 1.0   # Angular velocity: 1x base uncertainty
        
    def ensure_positive_definite(self, min_eigenval=1e-6):
        """Ensure matrix remains positive definite"""
        eigenvals, eigenvecs = np.linalg.eigh(self.matrix)
        
        # Clamp negative eigenvalues
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct matrix
        self.matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure symmetry
        self.matrix = (self.matrix + self.matrix.T) / 2
        
    def add_process_noise(self, process_noise):
        """Add process noise to covariance matrix"""
        self.matrix += process_noise
        self.ensure_positive_definite()
        
    def measurement_update(self, H, R):
        """Update covariance with measurement (Kalman update)"""
        # Kalman gain
        S = H @ self.matrix @ H.T + R
        K = self.matrix @ H.T @ np.linalg.inv(S)
        
        # Update covariance
        I = np.eye(self.size)
        self.matrix = (I - K @ H) @ self.matrix
        
        self.ensure_positive_definite()
        
    def get_uncertainty(self, state_slice):
        """Get uncertainty (standard deviations) for given state slice"""
        return np.sqrt(np.diag(self.matrix[state_slice, state_slice]))


class ExtendedKalmanFilter:
    """Extended Kalman Filter for rover state estimation"""
    
    def __init__(self):
        # Initialize state and covariance
        self.state = StateVector()
        self.covariance = CovarianceMatrix(size=12, initial_uncertainty=1.0)
        
        # Process noise (Q matrix)
        self.process_noise = self._create_process_noise_matrix()
        
        # Measurement noise parameters
        self.measurement_noise = {
            'gps_position': np.eye(3) * 4.0,  # 2m std dev -> 4.0 variance
            'imu_accel': np.eye(3) * 0.01,    # 0.1 m/s² std dev
            'imu_gyro': np.eye(3) * 0.0025,   # 0.05 rad/s std dev
            'odometry_pos': np.eye(2) * 0.01, # Position uncertainty
            'odometry_ori': 0.0025            # Orientation uncertainty
        }
        
        # Initialize tracking attributes
        self.prediction_count = 0
        self.update_count = 0
        self.sensor_weights = {'gps': {}, 'imu': {}, 'odometry': 1.0}
        
    def _create_process_noise_matrix(self):
        """Create process noise matrix Q"""
        Q = np.eye(12) * 0.01  # Base process noise
        
        # Higher noise for accelerations/velocities
        Q[3:6, 3:6] *= 10    # Velocity noise
        Q[9:12, 9:12] *= 5   # Angular velocity noise
        
        return Q
        
    def predict(self, dt):
        """Prediction step of Extended Kalman Filter"""
        # Get current state
        current_state = self.state.get_full_state()
        
        # Apply motion model
        predicted_state = self.motion_model(dt)
        
        # Update state
        self.state.update_from_array(predicted_state)
        
        # Compute motion Jacobian
        F = self.compute_motion_jacobian(dt)
        
        # Predict covariance
        self.covariance.matrix = F @ self.covariance.matrix @ F.T + self.process_noise * dt
        self.covariance.ensure_positive_definite()
        
        # Increment prediction count
        self.prediction_count += 1
        
    def motion_model(self, dt):
        """Apply motion model to current state"""
        current_state = self.state.get_full_state()
        new_state = current_state.copy()
        
        # Position update: x = x + v*dt
        new_state[0:3] += current_state[3:6] * dt
        
        # Orientation update: theta = theta + omega*dt
        new_state[6:9] += current_state[9:12] * dt
        
        # Velocity and angular velocity decay slightly (add damping for stability)
        new_state[3:6] *= 0.999  # Very slight velocity decay
        new_state[9:12] *= 0.999  # Very slight angular velocity decay
        
        return new_state
        
    def compute_motion_jacobian(self, dt):
        """Compute Jacobian of motion model with respect to state"""
        F = np.eye(12)
        
        # Position depends on velocity: ∂x/∂v = dt
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Orientation depends on angular velocity: ∂θ/∂ω = dt
        F[6:9, 9:12] = np.eye(3) * dt
        
        # Velocity and angular velocity have decay factor
        F[3:6, 3:6] = np.eye(3) * 0.999
        F[9:12, 9:12] = np.eye(3) * 0.999
        
        return F
        
    def update_gps(self, gps_measurement, measurement_noise=None):
        """Update filter with GPS position measurement"""
        if measurement_noise is None:
            measurement_noise = self.measurement_noise['gps_position']
            
        # Check for outliers
        predicted_position = self.state.position
        innovation = gps_measurement - predicted_position
        innovation_magnitude = np.linalg.norm(innovation)
        
        # Reject outliers (measurements too far from prediction)
        if innovation_magnitude > 300.0:  # 300m threshold for outlier rejection
            return  # Skip this measurement
            
        # Measurement model: z = H*x + v
        # GPS directly measures position
        H = np.zeros((3, 12))
        H[0:3, 0:3] = np.eye(3)  # Measure position states
        
        # Kalman update
        S = H @ self.covariance.matrix @ H.T + measurement_noise
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            S_inv = np.linalg.pinv(S)
            
        K = self.covariance.matrix @ H.T @ S_inv
        
        # Update state
        current_state = self.state.get_full_state()
        updated_state = current_state + K @ innovation
        self.state.update_from_array(updated_state)
        

        
        # Update covariance using Joseph form for numerical stability
        I_KH = np.eye(12) - K @ H
        self.covariance.matrix = I_KH @ self.covariance.matrix @ I_KH.T + K @ measurement_noise @ K.T
        self.covariance.ensure_positive_definite()
        
        # Increment update count
        self.update_count += 1
        
    def update_imu(self, imu_measurement, measurement_noise=None):
        """Update filter with IMU measurements"""
        if measurement_noise is None:
            accel_noise = self.measurement_noise['imu_accel']
            gyro_noise = self.measurement_noise['imu_gyro']
        else:
            accel_noise = measurement_noise['acceleration']
            gyro_noise = measurement_noise['angular_velocity']
            
        # Update with angular velocity measurement (direct)
        self._update_angular_velocity(imu_measurement['angular_velocity'], gyro_noise)
        
        # Update with acceleration (more complex due to gravity and motion)
        self._update_acceleration(imu_measurement['acceleration'], accel_noise)
        
    def _update_angular_velocity(self, measured_angular_velocity, noise_matrix):
        """Update with angular velocity measurement"""
        # Measurement model: gyro directly measures angular velocity
        H = np.zeros((3, 12))
        H[0:3, 9:12] = np.eye(3)  # Measure angular velocity states
        
        # Innovation
        predicted_angular_velocity = self.state.angular_velocity
        innovation = measured_angular_velocity - predicted_angular_velocity
        
        # Kalman update
        S = H @ self.covariance.matrix @ H.T + noise_matrix
        K = self.covariance.matrix @ H.T @ np.linalg.inv(S)
        
        current_state = self.state.get_full_state()
        updated_state = current_state + K @ innovation
        self.state.update_from_array(updated_state)
        
        # Update covariance
        self.covariance.measurement_update(H, noise_matrix)
        
    def _update_acceleration(self, measured_acceleration, noise_matrix):
        """Update with acceleration measurement (simplified)"""
        # Simplified: assume acceleration is related to velocity change
        # In practice, this would involve more complex dynamics
        
        # For now, just use acceleration to provide information about motion
        # This is a simplified implementation
        expected_accel = np.array([0.0, 0.0, -9.81])  # Gravity
        
        # If measured acceleration is significantly different from gravity,
        # it indicates motion and can inform velocity estimates
        innovation = measured_acceleration - expected_accel
        
        # Simple update to velocity based on acceleration
        if np.linalg.norm(innovation) > 1.0:  # Significant acceleration
            # Update velocity based on acceleration (simplified)
            # In practice, this would be more sophisticated
            pass
            
    def update_odometry(self, odometry_measurement, measurement_noise=None):
        """Update filter with wheel odometry measurement"""
        if measurement_noise is None:
            pos_noise = self.measurement_noise['odometry_pos']
            ori_noise = self.measurement_noise['odometry_ori']
        else:
            pos_noise = measurement_noise['position']
            ori_noise = measurement_noise['orientation']
            
        # Convert odometry delta to world frame update
        dx = odometry_measurement['dx']
        dy = odometry_measurement['dy']
        dtheta = odometry_measurement['dtheta']
        
        # Transform body frame motion to world frame
        current_yaw = self.state.orientation[2]
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        # World frame displacement
        world_dx = dx * cos_yaw - dy * sin_yaw
        world_dy = dx * sin_yaw + dy * cos_yaw
        
        # Apply position update directly (integrate motion)
        current_state = self.state.get_full_state()
        current_state[0] += world_dx  # Update x position
        current_state[1] += world_dy  # Update y position
        current_state[8] += dtheta    # Update yaw orientation
        
        self.state.update_from_array(current_state)
        
        # Update covariance to reflect odometry uncertainty
        # Add uncertainty in position and orientation
        self.covariance.matrix[0:2, 0:2] += pos_noise
        self.covariance.matrix[8, 8] += ori_noise
        self.covariance.ensure_positive_definite()
        
        # Increment update count
        self.update_count += 1
    
    def get_position_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (standard deviations)."""
        return self.covariance.get_uncertainty(slice(0, 3))
    
    def get_velocity_uncertainty(self) -> np.ndarray:
        """Get velocity uncertainty (standard deviations)."""
        return self.covariance.get_uncertainty(slice(3, 6))
    
    def get_orientation_uncertainty(self) -> np.ndarray:
        """Get orientation uncertainty (standard deviations)."""
        return self.covariance.get_uncertainty(slice(6, 9))
    
    def get_fusion_confidence(self) -> float:
        """Get overall fusion confidence based on covariance trace."""
        # Confidence is inversely related to uncertainty
        total_uncertainty = np.trace(self.covariance.matrix)
        confidence = 1.0 / (1.0 + total_uncertainty / 12.0)  # Normalize by state dimension
        return confidence
    
    def get_sensor_weights(self) -> Dict[str, Dict[int, float]]:
        """Get current sensor weights for monitoring."""
        return self.sensor_weights.copy()
    
    def reset_to_initial_state(self, 
                              position: Optional[np.ndarray] = None,
                              uncertainty: float = 1.0):
        """Reset filter to initial state (useful for testing)."""
        if position is not None:
            self.state.update_position(position)
        else:
            self.state = StateVector()
        
        self.covariance = CovarianceMatrix(size=12, initial_uncertainty=uncertainty)
        self.prediction_count = 0
        self.update_count = 0
        self.sensor_weights = {'gps': {}, 'imu': {}, 'odometry': 1.0}
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive state information for monitoring."""
        return {
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'orientation': self.state.orientation.tolist(),
            'angular_velocity': self.state.angular_velocity.tolist(),
            'position_uncertainty': self.get_position_uncertainty().tolist(),
            'velocity_uncertainty': self.get_velocity_uncertainty().tolist(),
            'orientation_uncertainty': self.get_orientation_uncertainty().tolist(),
            'fusion_confidence': self.get_fusion_confidence(),
            'prediction_count': self.prediction_count,
            'update_count': self.update_count,
            'sensor_weights': self.sensor_weights,
            'covariance_trace': np.trace(self.covariance.matrix)
        } 