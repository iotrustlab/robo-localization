"""
Sensor fusion algorithms for robo localization.

This module implements Extended Kalman Filter and related fusion algorithms
for combining multiple sensor measurements into accurate state estimates.
"""

from .kalman import ExtendedKalmanFilter, StateVector, CovarianceMatrix

__all__ = [
    "ExtendedKalmanFilter",
    "StateVector", 
    "CovarianceMatrix"
]