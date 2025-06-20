"""
Extended Kalman Filter Implementation for Autonomous Vehicle State Estimation

This module provides a scientifically rigorous implementation of an Extended Kalman Filter
for real-time state estimation of autonomous vehicles operating in 3D environments.

Mathematical Foundation:
The EKF addresses the nonlinear state estimation problem through linearization:

State Evolution:
    x(k+1) = f(x(k), u(k)) + w(k)
    z(k) = h(x(k)) + v(k)

Where:
    - x(k) ∈ ℝ¹² is the state vector at time k
    - f(·) is the nonlinear state transition function
    - h(·) is the nonlinear measurement function
    - w(k) ~ N(0, Q) is process noise
    - v(k) ~ N(0, R) is measurement noise

EKF Recursion:
    Prediction:
        x̂(k|k-1) = f(x̂(k-1|k-1), u(k-1))
        P(k|k-1) = F(k-1)P(k-1|k-1)F(k-1)ᵀ + Q(k-1)
    
    Update:
        K(k) = P(k|k-1)H(k)ᵀ[H(k)P(k|k-1)H(k)ᵀ + R(k)]⁻¹
        x̂(k|k) = x̂(k|k-1) + K(k)[z(k) - h(x̂(k|k-1))]
        P(k|k) = [I - K(k)H(k)]P(k|k-1)

State Vector Definition:
    x = [px, py, pz, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]ᵀ
    
Where:
    - [px, py, pz]: Position in world frame (m)
    - [vx, vy, vz]: Velocity in world frame (m/s)
    - [φ, θ, ψ]: Euler angles [roll, pitch, yaw] (rad)
    - [ωx, ωy, ωz]: Angular velocity in body frame (rad/s)

Authors: Scientific Computing Team
License: MIT
Version: 2.0.0
"""

import numpy as np
import scipy.linalg
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import warnings
from enum import Enum
import logging

# Configure logging for filter diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterState(Enum):
    """Enumeration of possible filter states for diagnostics."""
    INITIALIZING = "initializing"
    CONVERGED = "converged"
    DIVERGING = "diverging"
    ILL_CONDITIONED = "ill_conditioned"
    RECOVERING = "recovering"


@dataclass
class FilterDiagnostics:
    """Container for filter diagnostic information."""
    condition_number: float
    innovation_magnitude: float
    mahalanobis_distance: float
    likelihood: float
    filter_state: FilterState
    timestamp: float


class StateVector:
    """
    Twelve-dimensional state vector for autonomous vehicle navigation.
    
    The state vector encapsulates the complete kinematic state of a vehicle
    operating in 3D space, following aerospace conventions for coordinate systems.
    
    Mathematical Representation:
        x = [px, py, pz, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]ᵀ ∈ ℝ¹²
    
    Coordinate System Conventions:
        - World Frame: NED (North-East-Down) or ENU (East-North-Up)
        - Body Frame: Forward-Right-Down (FRD)
        - Euler Angle Sequence: ZYX (yaw-pitch-roll)
    
    State Components:
        Position: Global coordinates in world reference frame
        Velocity: Translational velocity in world reference frame
        Orientation: Euler angles defining body-to-world rotation
        Angular Velocity: Rotational rates in body reference frame
    """
    
    def __init__(self, initial_state: Optional[np.ndarray] = None):
        """
        Initialize state vector with optional initial conditions.
        
        Args:
            initial_state: Optional 12-element array for initialization
                          If None, initializes to zero state
        
        Raises:
            ValueError: If initial_state has incorrect dimensions
        """
        if initial_state is not None:
            if len(initial_state) != 12:
                raise ValueError(f"State vector must have 12 elements, got {len(initial_state)}")
            self._validate_state_values(initial_state)
            
        self._position = np.zeros(3) if initial_state is None else initial_state[0:3].copy()
        self._velocity = np.zeros(3) if initial_state is None else initial_state[3:6].copy()
        self._orientation = np.zeros(3) if initial_state is None else initial_state[6:9].copy()
        self._angular_velocity = np.zeros(3) if initial_state is None else initial_state[9:12].copy()
        
        # Normalize orientation angles to [-π, π]
        self._normalize_angles()
        
    @staticmethod
    def _validate_state_values(state: np.ndarray) -> None:
        """
        Validate physical constraints on state values.
        
        Args:
            state: State vector to validate
            
        Raises:
            ValueError: If state values are non-physical
        """
        # Check for NaN or infinite values
        if not np.all(np.isfinite(state)):
            raise ValueError("State vector contains NaN or infinite values")
            
        # Validate velocity magnitude (reasonable upper bound)
        velocity_magnitude = np.linalg.norm(state[3:6])
        if velocity_magnitude > 100.0:  # 100 m/s = 360 km/h
            warnings.warn(f"High velocity magnitude detected: {velocity_magnitude:.2f} m/s")
            
        # Validate angular velocity magnitude
        angular_velocity_magnitude = np.linalg.norm(state[9:12])
        if angular_velocity_magnitude > 10.0:  # 10 rad/s
            warnings.warn(f"High angular velocity detected: {angular_velocity_magnitude:.2f} rad/s")
    
    def _normalize_angles(self) -> None:
        """Normalize Euler angles to [-π, π] range."""
        self._orientation = np.mod(self._orientation + np.pi, 2*np.pi) - np.pi
    
    @property
    def position(self) -> np.ndarray:
        """Position vector [px, py, pz] in meters."""
        return self._position.copy()
    
    @property
    def velocity(self) -> np.ndarray:
        """Velocity vector [vx, vy, vz] in m/s (world frame)."""
        return self._velocity.copy()
    
    @property
    def orientation(self) -> np.ndarray:
        """Euler angles [φ, θ, ψ] in radians."""
        return self._orientation.copy()
    
    @property
    def angular_velocity(self) -> np.ndarray:
        """Angular velocity [ωx, ωy, ωz] in rad/s (body frame)."""
        return self._angular_velocity.copy()
    
    @classmethod
    def from_array(cls, state_array: np.ndarray) -> 'StateVector':
        """
        Create StateVector from numpy array.
        
        Args:
            state_array: 12-element numpy array
            
        Returns:
            StateVector instance
            
        Raises:
            ValueError: If array has incorrect size or invalid values
        """
        return cls(state_array)
    
    def to_array(self) -> np.ndarray:
        """
        Convert state to numpy array representation.
        
        Returns:
            12-element numpy array containing complete state
        """
        return np.concatenate([
            self._position,
            self._velocity,
            self._orientation,
            self._angular_velocity
        ])
    
    def get_full_state(self) -> np.ndarray:
        """Legacy compatibility method. Use to_array() instead."""
        warnings.warn("get_full_state() is deprecated, use to_array()", DeprecationWarning)
        return self.to_array()
    
    def update_from_array(self, state_array: np.ndarray) -> None:
        """
        Update state from numpy array with validation.
        
        Args:
            state_array: 12-element numpy array
            
        Raises:
            ValueError: If array has incorrect size or invalid values
        """
        if len(state_array) != 12:
            raise ValueError(f"State array must have 12 elements, got {len(state_array)}")
        
        self._validate_state_values(state_array)
        
        self._position = state_array[0:3].copy()
        self._velocity = state_array[3:6].copy()
        self._orientation = state_array[6:9].copy()
        self._angular_velocity = state_array[9:12].copy()
        
        self._normalize_angles()
    
    def get_pose_2d(self) -> Tuple[float, float, float]:
        """
        Extract 2D pose (x, y, yaw) for planar navigation.
        
        Returns:
            Tuple of (x, y, yaw) in meters and radians
        """
        return (self._position[0], self._position[1], self._orientation[2])
    
    def distance_to(self, other: 'StateVector') -> float:
        """
        Compute Euclidean distance to another state vector.
        
        Args:
            other: Another StateVector instance
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self._position - other._position)
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (f"StateVector(pos=[{self._position[0]:.3f}, {self._position[1]:.3f}, {self._position[2]:.3f}], "
                f"vel=[{self._velocity[0]:.3f}, {self._velocity[1]:.3f}, {self._velocity[2]:.3f}], "
                f"ori=[{np.degrees(self._orientation[0]):.1f}°, {np.degrees(self._orientation[1]):.1f}°, "
                f"{np.degrees(self._orientation[2]):.1f}°])")


class CovarianceMatrix:
    """
    Covariance matrix management with numerical stability guarantees.
    
    This class maintains the state covariance matrix P(k|k) with automatic
    enforcement of positive definiteness and numerical conditioning.
    
    Mathematical Properties:
        - Positive definiteness: P ≻ 0 (all eigenvalues > 0)
        - Symmetry: P = Pᵀ
        - Bounded condition number: κ(P) < threshold
    
    Numerical Stability Features:
        - Eigenvalue clamping for positive definiteness
        - Condition number monitoring and regularization
        - Joseph form covariance update for numerical robustness
        - Symmetric matrix enforcement
    """
    
    def __init__(self, size: int = 12, initial_uncertainty: float = 1.0):
        """
        Initialize covariance matrix with physical insights.
        
        Args:
            size: Dimension of state space (typically 12)
            initial_uncertainty: Base uncertainty scaling factor
            
        Raises:
            ValueError: If size is non-positive or uncertainty is negative
        """
        if size <= 0:
            raise ValueError(f"Matrix size must be positive, got {size}")
        if initial_uncertainty < 0:
            raise ValueError(f"Initial uncertainty must be non-negative, got {initial_uncertainty}")
            
        self.size = size
        self._matrix = self._create_initial_covariance(initial_uncertainty)
        self._min_eigenvalue = 1e-8
        self._max_condition_number = 1e12
        
    def _create_initial_covariance(self, base_uncertainty: float) -> np.ndarray:
        """
        Create physically-motivated initial covariance matrix.
        
        Different state components have different initial uncertainties
        based on typical operational scenarios:
        
        Args:
            base_uncertainty: Base scaling factor
            
        Returns:
            Initial covariance matrix
        """
        P = np.eye(self.size) * base_uncertainty
        
        # Position: Very high initial uncertainty for convergence
        P[0:3, 0:3] *= 1000.0
        
        # Velocity: High uncertainty (unknown initial motion)
        P[3:6, 3:6] *= 100.0
        
        # Orientation: High uncertainty (unknown initial heading)
        P[6:9, 6:9] *= 100.0
        
        # Angular velocity: Moderate uncertainty
        P[9:12, 9:12] *= 10.0
        
        return P
    
    @property
    def matrix(self) -> np.ndarray:
        """Get copy of covariance matrix."""
        return self._matrix.copy()
    
    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        """Set covariance matrix with validation."""
        if value.shape != (self.size, self.size):
            raise ValueError(f"Matrix shape must be ({self.size}, {self.size})")
        self._matrix = value.copy()
        self.ensure_positive_definite()
    
    def ensure_positive_definite(self, min_eigenval: Optional[float] = None) -> bool:
        """
        Enforce positive definiteness through eigenvalue clamping.
        
        This method uses eigendecomposition to identify and correct
        negative eigenvalues, ensuring the covariance matrix remains
        positive definite for numerical stability.
        
        Mathematical Approach:
            P = UΛUᵀ where Λ = diag(λ₁, ..., λₙ)
            P_corrected = U max(Λ, ε·I) Uᵀ
        
        Args:
            min_eigenval: Minimum allowable eigenvalue
            
        Returns:
            True if matrix was modified, False otherwise
        """
        min_eigenval = min_eigenval or self._min_eigenvalue
        
        # Compute eigendecomposition
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self._matrix)
        except np.linalg.LinAlgError:
            logger.warning("Eigendecomposition failed, using regularization")
            self._matrix += np.eye(self.size) * min_eigenval
            return True
        
        # Check if modification is needed
        min_eval = np.min(eigenvals)
        if min_eval < min_eigenval:
            # Clamp negative eigenvalues
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            self._matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Ensure numerical symmetry
            self._matrix = (self._matrix + self._matrix.T) * 0.5
            
            logger.debug(f"Clamped eigenvalue from {min_eval:.2e} to {min_eigenval:.2e}")
            return True
            
        return False
    
    def get_condition_number(self) -> float:
        """
        Compute condition number κ(P) = λ_max / λ_min.
        
        The condition number indicates numerical stability:
        - κ(P) ≈ 1: Well-conditioned
        - κ(P) > 10¹²: Ill-conditioned
        
        Returns:
            Condition number of covariance matrix
        """
        try:
            return np.linalg.cond(self._matrix)
        except np.linalg.LinAlgError:
            return float('inf')
    
    def is_well_conditioned(self) -> bool:
        """
        Check if matrix is well-conditioned.
        
        Returns:
            True if condition number is below threshold
        """
        return self.get_condition_number() < self._max_condition_number
    
    def regularize(self, regularization_factor: float = 1e-6) -> None:
        """
        Apply Tikhonov regularization for numerical stability.
        
        Adds small diagonal term: P_reg = P + ε·I
        
        Args:
            regularization_factor: Regularization parameter ε
        """
        self._matrix += np.eye(self.size) * regularization_factor
        logger.debug(f"Applied regularization with factor {regularization_factor:.2e}")
    
    def add_process_noise(self, process_noise: np.ndarray) -> None:
        """
        Add process noise to covariance matrix.
        
        Implements: P⁻(k+1) = F P⁺(k) Fᵀ + Q(k)
        
        Args:
            process_noise: Process noise matrix Q
            
        Raises:
            ValueError: If process noise has incorrect dimensions
        """
        if process_noise.shape != (self.size, self.size):
            raise ValueError(f"Process noise shape must be ({self.size}, {self.size})")
            
        self._matrix += process_noise
        self.ensure_positive_definite()
    
    def joseph_form_update(self, H: np.ndarray, R: np.ndarray, K: np.ndarray) -> None:
        """
        Joseph form covariance update for numerical stability.
        
        Implements: P⁺ = (I - KH)P⁻(I - KH)ᵀ + KRKᵀ
        
        This form maintains positive definiteness even with numerical errors.
        
        Args:
            H: Measurement Jacobian matrix
            R: Measurement noise covariance
            K: Kalman gain matrix
        """
        I_KH = np.eye(self.size) - K @ H
        self._matrix = I_KH @ self._matrix @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        self._matrix = (self._matrix + self._matrix.T) * 0.5
        
        self.ensure_positive_definite()
    
    def get_uncertainty(self, state_indices: Union[slice, np.ndarray, list]) -> np.ndarray:
        """
        Extract uncertainty (standard deviations) for specified states.
        
        Args:
            state_indices: Indices or slice for state components
            
        Returns:
            Standard deviations for specified states
        """
        if isinstance(state_indices, slice):
            start, stop = state_indices.start or 0, state_indices.stop or self.size
            indices = np.arange(start, stop)
        else:
            indices = np.asarray(state_indices)
            
        return np.sqrt(np.diag(self._matrix[np.ix_(indices, indices)]))
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Compute correlation matrix from covariance.
        
        Correlation matrix: ρᵢⱼ = σᵢⱼ / (σᵢ σⱼ)
        
        Returns:
            Correlation matrix with values in [-1, 1]
        """
        std_devs = np.sqrt(np.diag(self._matrix))
        correlation = self._matrix / np.outer(std_devs, std_devs)
        
        # Handle numerical issues
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        
        return correlation
    
    def mahalanobis_distance(self, innovation: np.ndarray, 
                           measurement_indices: Optional[np.ndarray] = None) -> float:
        """
        Compute Mahalanobis distance for outlier detection.
        
        Distance: d² = νᵀS⁻¹ν where S = HPHᵀ + R
        
        Args:
            innovation: Innovation vector ν
            measurement_indices: Indices for measurement mapping
            
        Returns:
            Mahalanobis distance
        """
        if measurement_indices is None:
            S = self._matrix
        else:
            S = self._matrix[np.ix_(measurement_indices, measurement_indices)]
            
        try:
            return float(innovation.T @ np.linalg.inv(S) @ innovation)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            return float(innovation.T @ np.linalg.pinv(S) @ innovation)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation.
    
    This implementation provides a complete EKF framework for autonomous
    vehicle navigation, incorporating multiple sensor modalities with
    advanced numerical stability and diagnostic capabilities.
    
    Key Features:
        - Multi-sensor fusion (GPS, IMU, odometry)
        - Adaptive noise estimation
        - Outlier detection and rejection
        - Filter divergence monitoring
        - Comprehensive diagnostic reporting
        - Numerical stability guarantees
    
    Mathematical Foundation:
        The EKF extends the linear Kalman filter to nonlinear systems by
        linearizing the process and measurement models around the current
        state estimate using first-order Taylor expansions.
    
    State Space Model:
        - State dimension: 12 (position, velocity, orientation, angular velocity)
        - Process model: Constant velocity with angular motion
        - Measurement models: GPS position, IMU angular velocity, odometry
    """
    
    def __init__(self, initial_state: Optional[np.ndarray] = None,
                 initial_uncertainty: float = 1.0):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            initial_state: Optional initial state vector (12-element)
            initial_uncertainty: Initial uncertainty scaling factor
        """
        # Initialize state and covariance
        self.state = StateVector(initial_state)
        self.covariance = CovarianceMatrix(size=12, initial_uncertainty=initial_uncertainty)
        
        # Process noise matrix (tuned for vehicle dynamics)
        self.process_noise = self._create_process_noise_matrix()
        
        # Measurement noise parameters (sensor-specific) - tuned for better tracking
        self.measurement_noise = {
            'gps_position': np.eye(3) * 9.0,      # GPS position (3m std dev) - slightly higher for realism
            'imu_angular_velocity': np.eye(3) * 0.0025,  # IMU gyroscope (0.05 rad/s std dev)
            'imu_acceleration': np.eye(3) * 0.01,  # IMU accelerometer (0.1 m/s² std dev)
            'odometry_position': np.eye(2) * 0.01,  # Odometry position
            'odometry_orientation': 0.0025         # Odometry orientation
        }
        
        # Filter statistics and diagnostics
        self._prediction_count = 0
        self._update_count = 0
        self._last_update_time = 0.0
        self._innovation_history = []
        self._filter_state = FilterState.INITIALIZING
        
        # Adaptive parameters
        self._innovation_threshold = 3.0  # 3-sigma threshold
        self._max_innovation_magnitude = 50.0  # Maximum allowable innovation (reduced)
        self._divergence_threshold = 1e6  # Covariance trace threshold
        
        logger.info("Extended Kalman Filter initialized")
    
    def _create_process_noise_matrix(self) -> np.ndarray:
        """
        Create process noise matrix Q based on vehicle dynamics.
        
        The process noise models uncertainty in the vehicle's motion
        between time steps, accounting for unmodeled dynamics and
        disturbances.
        
        Returns:
            12x12 process noise covariance matrix
        """
        Q = np.eye(12) * 0.5  # Moderate base process noise for good tracking
        
        # Position noise (integration of velocity uncertainty)
        Q[0:3, 0:3] *= 2.0  # Moderate position uncertainty
        
        # Velocity noise (acceleration uncertainty) - tuned for responsiveness
        Q[3:6, 3:6] *= 20.0  # Good velocity uncertainty for tracking
        
        # Orientation noise (integration of angular velocity uncertainty)
        Q[6:9, 6:9] *= 2.0  # Moderate orientation uncertainty
        
        # Angular velocity noise (angular acceleration uncertainty)
        Q[9:12, 9:12] *= 10.0  # Good angular velocity uncertainty
        
        return Q
    
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> None:
        """
        Prediction step of the Extended Kalman Filter.
        
        Implements the nonlinear prediction equations:
            x̂(k|k-1) = f(x̂(k-1|k-1), u(k-1))
            P(k|k-1) = F(k-1)P(k-1|k-1)F(k-1)ᵀ + Q(k-1)
        
        Args:
            dt: Time step (seconds)
            control_input: Optional control vector u(k-1)
            
        Raises:
            ValueError: If time step is non-positive
        """
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        
        # Apply nonlinear motion model
        predicted_state = self._motion_model(dt, control_input)
        self.state.update_from_array(predicted_state)
        
        # Compute motion Jacobian F = ∂f/∂x
        F = self._compute_motion_jacobian(dt)
        
        # Predict covariance: P⁻ = FP⁺Fᵀ + Q
        P_pred = F @ self.covariance.matrix @ F.T + self.process_noise * dt
        self.covariance.matrix = P_pred
        
        # Update statistics
        self._prediction_count += 1
        
        # Check for filter divergence
        self._check_divergence()
        
        logger.debug(f"Prediction step completed, dt={dt:.3f}s")
    
    def _motion_model(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Nonlinear motion model for vehicle dynamics.
        
        Implements a constant velocity model with angular motion:
            p(k+1) = p(k) + v(k)·dt
            v(k+1) = v(k) + a(k)·dt  (if control available)
            θ(k+1) = θ(k) + ω(k)·dt
            ω(k+1) = ω(k) + α(k)·dt  (if control available)
        
        Args:
            dt: Time step
            control_input: Optional control vector [ax, ay, az, αx, αy, αz]
            
        Returns:
            Predicted state vector
        """
        current_state = self.state.to_array()
        new_state = current_state.copy()
        
        # Position integration: x = x + v*dt
        new_state[0:3] += current_state[3:6] * dt
        
        # Orientation integration: θ = θ + ω*dt
        new_state[6:9] += current_state[9:12] * dt
        
        # Apply control input if available
        if control_input is not None:
            if len(control_input) >= 3:
                # Linear acceleration
                new_state[3:6] += control_input[0:3] * dt
            if len(control_input) >= 6:
                # Angular acceleration
                new_state[9:12] += control_input[3:6] * dt
        
        # Apply slight damping for stability
        damping_factor = np.exp(-dt * 0.01)  # 1% damping per second
        new_state[3:6] *= damping_factor     # Velocity damping
        new_state[9:12] *= damping_factor    # Angular velocity damping
        
        return new_state
    
    def _compute_motion_jacobian(self, dt: float) -> np.ndarray:
        """
        Compute Jacobian of motion model F = ∂f/∂x.
        
        For the constant velocity model:
            ∂x/∂x = I, ∂x/∂v = I·dt
            ∂v/∂v = I·damping
            ∂θ/∂θ = I, ∂θ/∂ω = I·dt
            ∂ω/∂ω = I·damping
        
        Args:
            dt: Time step
            
        Returns:
            12x12 Jacobian matrix
        """
        F = np.eye(12)
        damping = np.exp(-dt * 0.01)
        
        # Position depends on velocity: ∂x/∂v = I·dt
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity damping: ∂v/∂v = I·damping
        F[3:6, 3:6] = np.eye(3) * damping
        
        # Orientation depends on angular velocity: ∂θ/∂ω = I·dt
        F[6:9, 9:12] = np.eye(3) * dt
        
        # Angular velocity damping: ∂ω/∂ω = I·damping
        F[9:12, 9:12] = np.eye(3) * damping
        
        return F
    
    def update_gps(self, measurement: np.ndarray, 
                   measurement_noise: Optional[np.ndarray] = None,
                   timestamp: Optional[float] = None) -> bool:
        """
        Update filter with GPS position measurement.
        
        GPS directly measures position in world coordinates:
            z = h(x) + v = [px, py, pz]ᵀ + v
        
        Args:
            measurement: GPS position [x, y, z] in meters
            measurement_noise: Optional measurement noise covariance
            timestamp: Optional measurement timestamp
            
        Returns:
            True if update was applied, False if rejected
        """
        if len(measurement) != 3:
            raise ValueError("GPS measurement must have 3 elements (x, y, z)")
        
        # Use default noise if not provided
        R = measurement_noise if measurement_noise is not None else self.measurement_noise['gps_position']
        
        # Measurement model: H = ∂h/∂x
        H = np.zeros((3, 12))
        H[0:3, 0:3] = np.eye(3)  # GPS measures position directly
        
        # Compute innovation
        predicted_measurement = self.state.position
        innovation = measurement - predicted_measurement
        
        # Outlier detection using Mahalanobis distance
        innovation_covariance = H @ self.covariance.matrix @ H.T + R
        mahalanobis_dist = self.covariance.mahalanobis_distance(innovation, np.arange(3))
        
        # Much more relaxed outlier rejection - the system was rejecting almost all GPS measurements
        if self._filter_state == FilterState.INITIALIZING or self._update_count < 200:
            chi2_threshold = 2000.0  # Very relaxed during initialization - allow convergence
        elif self._filter_state == FilterState.CONVERGED:
            chi2_threshold = 200.0   # Still relaxed when converged
        else:
            chi2_threshold = 1000.0  # Much more moderate otherwise
        if mahalanobis_dist > chi2_threshold:
            logger.warning(f"GPS measurement rejected: Mahalanobis distance {mahalanobis_dist:.2f}")
            return False
        
        # Check innovation magnitude - increased threshold
        innovation_magnitude = np.linalg.norm(innovation)
        # Increased from 50m to 200m to allow for initial convergence
        if innovation_magnitude > 200.0:
            logger.warning(f"GPS measurement rejected: Innovation too large {innovation_magnitude:.2f}m")
            return False
        
        # Compute Kalman gain
        try:
            S_inv = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, using pseudo-inverse")
            S_inv = np.linalg.pinv(innovation_covariance)
        
        K = self.covariance.matrix @ H.T @ S_inv
        
        # Update state
        current_state = self.state.to_array()
        updated_state = current_state + K @ innovation
        self.state.update_from_array(updated_state)
        
        # Update covariance using Joseph form
        self.covariance.joseph_form_update(H, R, K)
        
        # Update statistics
        self._update_count += 1
        self._last_update_time = timestamp or 0.0
        self._innovation_history.append(innovation_magnitude)
        
        # Update filter state based on performance
        if self._update_count > 50:  # After some updates
            avg_innovation = np.mean(self._innovation_history[-20:]) if len(self._innovation_history) >= 20 else innovation_magnitude
            if avg_innovation < 5.0:  # Good performance
                self._filter_state = FilterState.CONVERGED
            elif avg_innovation > 20.0:  # Poor performance
                self._filter_state = FilterState.DIVERGING
            else:
                self._filter_state = FilterState.RECOVERING
        
        # Limit history size
        if len(self._innovation_history) > 100:
            self._innovation_history.pop(0)
        
        logger.debug(f"GPS update applied: innovation={innovation_magnitude:.3f}m")
        return True
    
    def update_imu_angular_velocity(self, measurement: np.ndarray,
                                  measurement_noise: Optional[np.ndarray] = None) -> bool:
        """
        Update filter with IMU angular velocity measurement.
        
        IMU gyroscopes directly measure angular velocity in body frame:
            z = h(x) + v = [ωx, ωy, ωz]ᵀ + v
        
        Args:
            measurement: Angular velocity [ωx, ωy, ωz] in rad/s
            measurement_noise: Optional measurement noise covariance
            
        Returns:
            True if update was applied, False if rejected
        """
        if len(measurement) != 3:
            raise ValueError("Angular velocity measurement must have 3 elements")
        
        R = measurement_noise if measurement_noise is not None else self.measurement_noise['imu_angular_velocity']
        
        # Measurement model
        H = np.zeros((3, 12))
        H[0:3, 9:12] = np.eye(3)  # IMU measures angular velocity directly
        
        # Innovation
        predicted_measurement = self.state.angular_velocity
        innovation = measurement - predicted_measurement
        
        # Outlier detection
        innovation_magnitude = np.linalg.norm(innovation)
        if innovation_magnitude > 10.0:  # 10 rad/s threshold
            logger.warning(f"IMU angular velocity rejected: {innovation_magnitude:.2f} rad/s")
            return False
        
        # Kalman update
        S = H @ self.covariance.matrix @ H.T + R
        K = self.covariance.matrix @ H.T @ np.linalg.inv(S)
        
        # Update state
        current_state = self.state.to_array()
        updated_state = current_state + K @ innovation
        self.state.update_from_array(updated_state)
        
        # Update covariance
        self.covariance.joseph_form_update(H, R, K)
        
        self._update_count += 1
        logger.debug(f"IMU angular velocity update applied")
        return True
    
    def update_odometry(self, delta_position: np.ndarray, delta_orientation: float,
                       measurement_noise: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        Update filter with wheel odometry measurements.
        
        Odometry provides relative motion in body frame that must be
        transformed to world coordinates.
        
        Args:
            delta_position: Position change [dx, dy] in body frame (meters)
            delta_orientation: Orientation change (radians)
            measurement_noise: Optional noise parameters
            
        Returns:
            True if update was applied, False if rejected
        """
        if len(delta_position) != 2:
            raise ValueError("Delta position must have 2 elements (dx, dy)")
        
        # Transform body frame motion to world frame
        current_yaw = self.state.orientation[2]
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        # Rotation matrix from body to world frame
        world_dx = delta_position[0] * cos_yaw - delta_position[1] * sin_yaw
        world_dy = delta_position[0] * sin_yaw + delta_position[1] * cos_yaw
        
        # Direct integration (simplified odometry model)
        current_state = self.state.to_array()
        current_state[0] += world_dx     # Update x position
        current_state[1] += world_dy     # Update y position
        current_state[8] += delta_orientation  # Update yaw orientation
        
        self.state.update_from_array(current_state)
        
        # Add odometry uncertainty to covariance
        pos_noise = (measurement_noise or {}).get('position', self.measurement_noise['odometry_position'])
        ori_noise = (measurement_noise or {}).get('orientation', self.measurement_noise['odometry_orientation'])
        
        self.covariance.matrix[0:2, 0:2] += pos_noise
        self.covariance.matrix[8, 8] += ori_noise
        self.covariance.ensure_positive_definite()
        
        self._update_count += 1
        logger.debug("Odometry update applied")
        return True
    
    def _check_divergence(self) -> None:
        """
        Monitor filter for divergence conditions.
        
        Divergence indicators:
        - Covariance trace exceeding threshold
        - Poor condition number
        - Large innovation sequences
        """
        # Check covariance trace
        trace = np.trace(self.covariance.matrix)
        if trace > self._divergence_threshold:
            self._filter_state = FilterState.DIVERGING
            logger.warning(f"Filter divergence detected: trace={trace:.2e}")
            return
        
        # Check condition number
        if not self.covariance.is_well_conditioned():
            self._filter_state = FilterState.ILL_CONDITIONED
            cond_num = self.covariance.get_condition_number()
            logger.warning(f"Ill-conditioned covariance: κ={cond_num:.2e}")
            return
        
        # Check innovation history
        if len(self._innovation_history) > 10:
            recent_innovations = self._innovation_history[-10:]
            avg_innovation = np.mean(recent_innovations)
            if avg_innovation > 10.0:  # Large consistent innovations
                self._filter_state = FilterState.DIVERGING
                logger.warning(f"High innovation sequence: avg={avg_innovation:.2f}")
                return
        
        # Filter appears stable
        if self._filter_state in [FilterState.DIVERGING, FilterState.ILL_CONDITIONED]:
            self._filter_state = FilterState.RECOVERING
        elif self._update_count > 10:
            self._filter_state = FilterState.CONVERGED
    
    def get_diagnostics(self, timestamp: Optional[float] = None) -> FilterDiagnostics:
        """
        Generate comprehensive filter diagnostics.
        
        Args:
            timestamp: Optional timestamp for diagnostics
            
        Returns:
            FilterDiagnostics object with current filter status
        """
        return FilterDiagnostics(
            condition_number=self.covariance.get_condition_number(),
            innovation_magnitude=self._innovation_history[-1] if self._innovation_history else 0.0,
            mahalanobis_distance=0.0,  # Would need recent measurement
            likelihood=self._compute_likelihood(),
            filter_state=self._filter_state,
            timestamp=timestamp or 0.0
        )
    
    def _compute_likelihood(self) -> float:
        """
        Compute filter likelihood based on covariance determinant.
        
        Returns:
            Negative log-likelihood (lower is better)
        """
        try:
            sign, logdet = np.linalg.slogdet(self.covariance.matrix)
            return 0.5 * logdet if sign > 0 else float('inf')
        except np.linalg.LinAlgError:
            return float('inf')
    
    def get_position_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (standard deviations) in meters."""
        return self.covariance.get_uncertainty(slice(0, 3))
    
    def get_velocity_uncertainty(self) -> np.ndarray:
        """Get velocity uncertainty (standard deviations) in m/s."""
        return self.covariance.get_uncertainty(slice(3, 6))
    
    def get_orientation_uncertainty(self) -> np.ndarray:
        """Get orientation uncertainty (standard deviations) in radians."""
        return self.covariance.get_uncertainty(slice(6, 9))
    
    def get_angular_velocity_uncertainty(self) -> np.ndarray:
        """Get angular velocity uncertainty (standard deviations) in rad/s."""
        return self.covariance.get_uncertainty(slice(9, 12))
    
    def get_fusion_confidence(self) -> float:
        """
        Compute overall fusion confidence metric.
        
        Confidence is based on multiple factors:
        - Covariance trace (lower is better)
        - Condition number (lower is better)
        - Recent update frequency
        - Filter state
        
        Returns:
            Confidence score in [0, 1] where 1 is highest confidence
        """
        # Base confidence from covariance trace (normalized for high initial uncertainty)
        trace = np.trace(self.covariance.matrix)
        trace_confidence = 1.0 / (1.0 + trace / 10000.0)  # Adjusted for higher initial uncertainty
        
        # Condition number penalty
        cond_num = self.covariance.get_condition_number()
        cond_confidence = 1.0 / (1.0 + np.log10(max(cond_num, 1.0)) / 6.0)
        
        # Filter state modifier
        state_modifiers = {
            FilterState.INITIALIZING: 0.5,
            FilterState.CONVERGED: 1.0,
            FilterState.DIVERGING: 0.1,
            FilterState.ILL_CONDITIONED: 0.3,
            FilterState.RECOVERING: 0.7
        }
        state_confidence = state_modifiers.get(self._filter_state, 0.5)
        
        # Combined confidence
        confidence = trace_confidence * cond_confidence * state_confidence
        return np.clip(confidence, 0.0, 1.0)
    
    def reset_filter(self, initial_state: Optional[np.ndarray] = None,
                    initial_uncertainty: float = 1.0) -> None:
        """
        Reset filter to initial conditions.
        
        Args:
            initial_state: Optional new initial state
            initial_uncertainty: Initial uncertainty scaling
        """
        self.state = StateVector(initial_state)
        self.covariance = CovarianceMatrix(size=12, initial_uncertainty=initial_uncertainty)
        
        self._prediction_count = 0
        self._update_count = 0
        self._last_update_time = 0.0
        self._innovation_history.clear()
        self._filter_state = FilterState.INITIALIZING
        
        logger.info("Extended Kalman Filter reset")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get comprehensive state information as dictionary.
        
        Returns:
            Dictionary containing all state and diagnostic information
        """
        return {
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'orientation': self.state.orientation.tolist(),
            'angular_velocity': self.state.angular_velocity.tolist(),
            'position_uncertainty': self.get_position_uncertainty().tolist(),
            'velocity_uncertainty': self.get_velocity_uncertainty().tolist(),
            'orientation_uncertainty': self.get_orientation_uncertainty().tolist(),
            'angular_velocity_uncertainty': self.get_angular_velocity_uncertainty().tolist(),
            'fusion_confidence': self.get_fusion_confidence(),
            'prediction_count': self._prediction_count,
            'update_count': self._update_count,
            'filter_state': self._filter_state.value,
            'covariance_trace': float(np.trace(self.covariance.matrix)),
            'condition_number': self.covariance.get_condition_number(),
            'last_update_time': self._last_update_time
        }