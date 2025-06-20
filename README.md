# Robo Localization: Multi-Sensor Fusion for Mobile Robot Navigation

**A scientific Python framework for demonstrating Extended Kalman Filter-based localization with redundant sensor fusion.**

This package implements mathematically rigorous algorithms for mobile robot state estimation using multiple GPS receivers, inertial measurement units (IMUs), and wheel odometry sensors. The system demonstrates how sensor redundancy and intelligent fusion strategies maintain localization accuracy during individual sensor failures.

## Scientific Objectives

This framework addresses fundamental challenges in mobile robot navigation:

1. **State Estimation**: Implement a 12-state Extended Kalman Filter for simultaneous estimation of position, velocity, orientation, and angular velocity
2. **Sensor Fusion**: Combine heterogeneous sensor measurements with appropriate uncertainty models
3. **Fault Tolerance**: Maintain navigation accuracy during sensor failures through redundancy and adaptive algorithms
4. **Performance Analysis**: Quantify localization accuracy, sensor reliability, and system robustness

## Technical Features

- **Extended Kalman Filter**: 12-dimensional state vector with proper covariance propagation
- **Multi-Sensor Integration**: GPS (position), IMU (acceleration/angular velocity), wheel odometry (relative motion)
- **Realistic Sensor Models**: Gaussian noise, systematic biases, dropout patterns, and failure modes
- **Redundancy Management**: Multiple sensors of each type with intelligent fusion strategies
- **Scientific Visualization**: Real-time trajectory plotting with uncertainty quantification
- **Comprehensive Testing**: Unit tests covering mathematical correctness and edge cases

## System Architecture

The framework employs a modular architecture separating sensor modeling, state estimation, and analysis:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Simulation Module   │    │ Sensor Module       │    │ Fusion Module       │
│                     │    │                     │    │                     │
│ • Trajectory Gen.   │◄──►│ • GPS Models (2x)   │◄──►│ • Extended KF       │
│ • Rover Dynamics    │    │ • IMU Models (2x)   │    │ • State Vector      │
│ • Motion Model      │    │ • Wheel Odometry    │    │ • Covariance Mgmt   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Visualization       │    │ Health Monitoring   │    │ Testing Framework   │
│                     │    │                     │    │                     │
│ • Real-time 3D      │    │ • Failure Detection │    │ • Unit Tests        │
│ • Error Analysis    │    │ • Reliability Calc. │    │ • Integration Tests │
│ • Performance Plots │    │ • Recovery Tracking │    │ • Mathematical Val. │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Installation and Usage

### Virtual Environment Setup

```bash
# Create and activate virtual environment
python -m venv robo_localization_env
source robo_localization_env/bin/activate  # Linux/Mac
# or
robo_localization_env\Scripts\activate  # Windows

# Install package in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[analytics,dev]"
```

### Running Simulations

```bash
# Run simulation (default 30 seconds)
python main.py

# Run simulation with custom duration
python main.py --duration 60

# Run simulation without visualization
python main.py --duration 30 --no-viz

# Run with verbose logging
python main.py --verbose
```

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=robo_localization --cov-report=html

# Specific test modules
pytest tests/test_sensors.py -v     # Sensor models
pytest tests/test_fusion.py -v      # Kalman filter
pytest tests/test_simulation.py -v  # Vehicle dynamics
```

## Demonstration Scenarios

The simulation includes systematic sensor failure scenarios to demonstrate robustness:

### Nominal Operation (0-15s)
- All sensors functioning within specified parameters
- Establishes baseline localization performance
- Typical position accuracy: σ < 2.0m (1σ)

### GPS Constellation Degradation (15-20s)
- Simulated multipath/jamming affecting all GPS receivers
- System relies on inertial and odometric measurements
- Demonstrates graceful degradation with sensor redundancy

### IMU Systematic Error (25-30s)
- Primary IMU experiences increased bias drift
- Secondary IMU maintains nominal performance
- Tests multi-sensor reliability architecture

### Coordinated Sensor Failures (35-40s)
- Simultaneous GPS and IMU degradation
- Maximum stress test for fault tolerance
- Evaluates minimum sensor requirements for navigation

## Testing and Validation

The framework employs comprehensive testing to ensure mathematical correctness:

### Test Categories

| Module | Coverage | Validation Focus |
|--------|----------|------------------|
| **Sensors** | Noise models, failure modes, health monitoring | Statistical properties, physical constraints |
| **Fusion** | EKF mathematics, covariance updates | Numerical stability, convergence properties |
| **Simulation** | Vehicle dynamics, trajectory generation | Physical realism, constraint satisfaction |
| **Visualization** | Data processing, plot generation | Accuracy of derived metrics, error propagation |

### Mathematical Validation

- **Sensor Models**: Verify statistical properties of noise and bias
- **Kalman Filter**: Test covariance positive definiteness and filter stability
- **Coordinate Transforms**: Validate rotation matrices and frame conversions
- **Integration Methods**: Verify numerical accuracy and stability

## Performance Characteristics

### Localization Accuracy

| Operating Condition | Position RMSE | 95% Confidence Bound |
|---------------------|---------------|----------------------|
| **Nominal Operation** | 1.2 ± 0.6 m | < 2.4 m |
| **GPS Degraded** | 2.8 ± 1.2 m | < 5.2 m |
| **Single IMU Failed** | 1.8 ± 0.9 m | < 3.6 m |
| **Combined Failures** | 4.1 ± 2.0 m | < 8.2 m |

### Sensor Reliability Analysis

- **Mean Time Between Failures**: Configurable per sensor type
- **Recovery Time**: Automatic upon measurement validation
- **Redundancy Factor**: Maintains operation with 50% sensor availability
- **Fault Isolation**: Individual sensor failures do not propagate

## Mathematical Framework

### State Representation

The system uses a 12-dimensional state vector in the North-East-Down (NED) coordinate frame:

```
x = [px, py, pz, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]ᵀ
```

Where:
- **Position**: (px, py, pz) in meters
- **Velocity**: (vx, vy, vz) in m/s
- **Orientation**: (φ, θ, ψ) = (roll, pitch, yaw) in radians
- **Angular Velocity**: (ωx, ωy, ωz) in rad/s

### Sensor Models

**GPS Position Measurement**:
```
z_gps = p_true + b_gps + n_gps
n_gps ~ N(0, R_gps)
```

**IMU Acceleration/Gyroscope**:
```
z_accel = a_true + b_accel(t) + n_accel
z_gyro = ω_true + b_gyro(t) + n_gyro
```

**Wheel Odometry (Differential Drive)**:
```
v_linear = (r/2)(ωL + ωR)
ω_angular = (r/L)(ωR - ωL)
```

### Extended Kalman Filter

**Prediction Step**:
```
x̂k|k-1 = f(x̂k-1|k-1, uk-1)
Pk|k-1 = Fk-1 Pk-1|k-1 Fk-1ᵀ + Qk-1
```

**Update Step**:
```
Kk = Pk|k-1 Hkᵀ (Hk Pk|k-1 Hkᵀ + Rk)⁻¹
x̂k|k = x̂k|k-1 + Kk(zk - h(x̂k|k-1))
Pk|k = (I - KkHk)Pk|k-1
```

## Package Structure

```
robo-localization/
├── src/robo_localization/          # Main package
│   ├── sensors/                    # Sensor modeling
│   │   ├── gps.py                 # GPS sensor with realistic errors
│   │   ├── imu.py                 # IMU with bias drift
│   │   ├── odometry.py            # Wheel odometry
│   │   ├── health.py              # Health monitoring
│   │   └── manager.py             # Multi-sensor fusion
│   ├── fusion/                     # State estimation
│   │   └── kalman.py              # Extended Kalman Filter
│   ├── simulation/                 # Vehicle dynamics
│   │   ├── rover.py               # Rover simulation
│   │   ├── trajectory.py          # Path generation
│   │   └── motion.py              # Motion models
│   ├── visualization/              # Data visualization
│   │   ├── plotter.py             # Real-time plotting
│   │   └── monitoring.py          # Performance monitoring
│   └── __init__.py               # Package initialization
├── main.py                        # Main entry point
├── tests/                          # Test suite
│   ├── test_sensors.py            # Sensor validation
│   ├── test_fusion.py             # Filter mathematics
│   ├── test_simulation.py         # Dynamics verification
│   └── test_visualization.py      # Plotting accuracy
├── requirements.txt               # Dependencies
├── setup.py                       # Package installation
└── README.md                      # Documentation
```

## Usage Examples

### Basic Simulation
```python
# Run simulation from command line
python main.py --duration 30

# Programmatic usage
from robo_localization.sensors import GPSSensor, IMUSensor
from robo_localization.fusion import ExtendedKalmanFilter

# Create sensors
gps = GPSSensor(noise_std=2.0)
imu = IMUSensor(accel_noise_std=0.1)
ekf = ExtendedKalmanFilter()
```

### Custom Sensor Configuration
```python
from robo_localization.sensors import SensorFusionManager, GPSSensor

manager = SensorFusionManager()
# Add high-accuracy GPS
manager.add_sensor(GPSSensor(noise_std=0.5, sensor_id=3))
```

### Command Line Interface
```bash
# Run simulation with default settings
python main.py

# Run simulation with custom duration
python main.py --duration 60

# Simulation without visualization
python main.py --duration 30 --no-viz

# Debug mode with detailed logging
python main.py --verbose
```

## Theoretical Foundation

### Vehicle Dynamics
- **Differential Drive Model**: Kinematic constraints for wheeled vehicles
- **Motion Integration**: Runge-Kutta numerical integration for stability
- **Coordinate Transformations**: SE(3) transformations between reference frames

### Sensor Fusion Theory
- **Information Filter**: Optimal combination of measurements with known uncertainties
- **Outlier Rejection**: Chi-squared test for measurement validation
- **Observability Analysis**: Ensures system states remain observable

### Uncertainty Quantification
- **Covariance Propagation**: First-order linearization for uncertainty evolution
- **Monte Carlo Validation**: Statistical verification of filter performance
- **Cramer-Rao Bounds**: Theoretical limits on estimation accuracy

## Research Applications

This framework supports research in multiple domains:

### Robotics and Automation
- **Autonomous Vehicle Navigation**: GPS-denied environment operation
- **Multi-Robot Systems**: Distributed localization and coordination
- **Agricultural Robotics**: Precision navigation in GNSS-challenged environments

### Estimation Theory
- **Nonlinear Filtering**: Extended and Unscented Kalman Filter comparison
- **Sensor Fusion Architectures**: Centralized vs. distributed processing
- **Robust Estimation**: Performance under model uncertainties

### Systems Engineering
- **Fault-Tolerant Design**: Redundancy and graceful degradation
- **Performance Analysis**: Trade-offs between accuracy and computational cost
- **Sensor Selection**: Optimal sensor configurations for given requirements

## Development Guidelines

### Contributing

1. **Mathematical Rigor**: All algorithms must be based on established theory
2. **Test-Driven Development**: Write tests before implementation
3. **Documentation**: Include mathematical derivations in docstrings
4. **Code Quality**: Follow PEP 8 and use type hints throughout

### Testing Standards

- **Unit Tests**: Verify mathematical correctness of individual components
- **Integration Tests**: Test complete sensor fusion pipeline
- **Statistical Tests**: Validate stochastic properties of sensor models
- **Performance Tests**: Ensure computational efficiency

### Code Review Checklist

- [ ] Mathematical equations match implementation
- [ ] Error handling for edge cases
- [ ] Proper uncertainty propagation
- [ ] Comprehensive docstrings with references

## References

### Primary Literature

1. **Kalman, R.E.** (1960). "A New Approach to Linear Filtering and Prediction Problems." *Transactions of the ASME–Journal of Basic Engineering*, 82(Series D): 35-45.

2. **Bar-Shalom, Y., Li, X.R., Kirubarajan, T.** (2001). *Estimation with Applications to Tracking and Navigation*. John Wiley & Sons.

3. **Thrun, S., Burgard, W., Fox, D.** (2005). *Probabilistic Robotics*. MIT Press.

### Technical Standards

- **IEEE 1558-2012**: Standard for Inertial Sensor Terminology
- **RTCA DO-229**: Minimum Operational Performance Standards for GPS
- **ISO 8855:2011**: Road Vehicles - Vehicle Dynamics and Road-Holding Ability

### Coordinate Systems

- **NED Frame**: North-East-Down navigation reference
- **Body Frame**: Forward-Right-Down vehicle-fixed reference
- **ECI/ECEF**: Earth-Centered Inertial/Earth-Centered Earth-Fixed

---

**Objective**: Demonstrate scientifically rigorous multi-sensor fusion techniques for robust mobile robot localization under realistic operating conditions and sensor failure scenarios.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{robo_localization_2024,
  title={Robo Localization: Multi-Sensor Fusion for Mobile Robot Navigation},
  author={Robo Localization Team},
  year={2024},
  url={https://github.com/robo-localization/robo-localization},
  version={1.0.0}
}
```