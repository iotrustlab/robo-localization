# 3D Rover Localization

## Goal

Build a working demo of redundant sensor fusion for 3D rover localization showing improved reliability and accuracy.

## What to Build

A Python simulation demonstrating that multiple GPS + multiple IMU + wheel odometry fusion provides robust localization with sensor redundancy.

## Requirements

### Core Functionality

- **Rover simulation**: Moves along a simple 3D path (figure-8 with elevation changes)
- **Redundant sensors**:
  - **2x GPS units**: Different noise characteristics, independent dropouts
  - **2x IMUs**: Different bias drift patterns, independent failures
  - **Wheel odometry**: Speed/direction from encoders with slip modeling
- **Sensor fusion**: Extended Kalman filter with redundancy handling
- **Visualization**: Real-time 3D plot showing ground truth vs estimated trajectory

### Test-Driven Development

- Write tests FIRST for each sensor and fusion component
- Tests should verify mathematical correctness, not curve-fit to outputs
- Focus on: sensor noise models, coordinate transformations, filter convergence
- **Critical**: Logic must be mathematically sound - don't adjust tests to pass, fix the implementation

### Deliverables

- `test_*.py` - Unit tests for all components (write these first)
- `main.py` - simulation runner
- `sensors.py` - Multi-GPS, multi-IMU, odometry simulators
- `kalman.py` - EKF with redundant sensor handling
- `viz.py` - 3D trajectory plotting
- `README.md` - installation and usage instructions

### Demo Requirements

- Show individual sensor failures and redundancy benefits
- Display sensor health monitoring
- Include simultaneous GPS dropout + IMU failure scenario
- Run 30-60 seconds with realistic sensor degradation
- Metrics: position error, sensor availability, fusion confidence

## Dependencies

Use: numpy, matplotlib/plotly, scipy, pytest etc.
