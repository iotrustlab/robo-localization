#!/usr/bin/env python3
"""
3D Rover Localization Demo with Redundant Sensor Fusion

This demo shows how multiple GPS + IMU + wheel odometry sensors provide robust
3D localization with failure detection and redundancy benefits.

Run with: python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse

from sensors import SensorFusionManager
from kalman import ExtendedKalmanFilter
from rover import TrajectoryGenerator, RoverSimulation
from viz import RealTimeVisualizer

def simulate_sensor_failures(manager, simulation_time):
    """Simulate various sensor failure scenarios"""
    
    # Track when failures start to avoid spam
    if not hasattr(simulate_sensor_failures, 'failures_announced'):
        simulate_sensor_failures.failures_announced = set()
    
    # GPS constellation failure (15-20 seconds)
    if 15.0 <= simulation_time <= 20.0:
        manager.simulate_gps_constellation_failure()
        if 'gps' not in simulate_sensor_failures.failures_announced:
            print(f"[{simulation_time:.1f}s] GPS constellation failure - all GPS sensors degraded")
            simulate_sensor_failures.failures_announced.add('gps')
        
    # IMU failure (25-30 seconds)
    elif 25.0 <= simulation_time <= 30.0:
        manager.imu_sensors[0].simulate_failure('noisy')
        if 'imu1' not in simulate_sensor_failures.failures_announced:
            print(f"[{simulation_time:.1f}s] IMU 1 noise failure")
            simulate_sensor_failures.failures_announced.add('imu1')
        
    # Combined sensor failures (35-40 seconds)
    elif 35.0 <= simulation_time <= 40.0:
        manager.simulate_gps_constellation_failure()
        manager.imu_sensors[1].simulate_failure('stuck')
        if 'combined' not in simulate_sensor_failures.failures_announced:
            print(f"[{simulation_time:.1f}s] Combined GPS + IMU failures")
            simulate_sensor_failures.failures_announced.add('combined')

def run_simulation(duration=60.0, real_time=True, visualize=True):
    """Run the main 3D rover localization simulation"""
    
    print("=== 3D Rover Localization with Redundant Sensor Fusion ===")
    print(f"Simulation duration: {duration} seconds")
    print(f"Real-time mode: {real_time}")
    print(f"Visualization: {visualize}")
    print()
    
    # Initialize components
    print("Initializing simulation components...")
    
    # Trajectory: Figure-8 with elevation changes
    trajectory = TrajectoryGenerator(radius=15.0, height=8.0, period=45.0)
    
    # Rover simulation
    rover = RoverSimulation(wheel_base=1.2, wheel_radius=0.25, drag_coefficient=0.05)
    rover.trajectory = trajectory
    rover.position = trajectory.get_position(0.0)
    rover.orientation = np.array([0.0, 0.0, 0.0])
    
    # Sensor fusion manager with redundant sensors
    sensor_manager = SensorFusionManager()
    
    # Configure sensors with desired parameters (override defaults)
    sensor_manager.gps_sensors[0].noise_std = 2.0
    sensor_manager.gps_sensors[0].dropout_prob = 0.02
    sensor_manager.gps_sensors[1].noise_std = 3.5
    sensor_manager.gps_sensors[1].dropout_prob = 0.05
    
    sensor_manager.imu_sensors[0].accel_noise_std = 0.1
    sensor_manager.imu_sensors[0].gyro_noise_std = 0.05
    sensor_manager.imu_sensors[1].accel_noise_std = 0.2
    sensor_manager.imu_sensors[1].gyro_noise_std = 0.08
    
    # Update wheel odometry parameters
    sensor_manager.wheel_odometry.wheel_base = 1.2
    sensor_manager.wheel_odometry.wheel_radius = 0.25
    sensor_manager.wheel_odometry.encoder_noise_std = 0.02
    sensor_manager.wheel_odometry.slip_factor = 0.03
    
    # Extended Kalman Filter
    ekf = ExtendedKalmanFilter()
    
    # Visualization (if enabled)
    visualizer = None
    if visualize:
        visualizer = RealTimeVisualizer(update_rate=20.0, trail_length=200)
        print("3D visualization initialized")
    
    # Simulation parameters
    dt = 0.05  # 20 Hz simulation
    simulation_time = 0.0
    step = 0
    
    # Data collection for analysis
    true_positions = []
    estimated_positions = []
    position_errors = []
    sensor_health_log = []
    
    print(f"Starting simulation...")
    print("Watch for sensor failures and recovery demonstrations!")
    print()
    
    start_real_time = time.time()
    
    try:
        while simulation_time < duration:
            # === SIMULATION STEP ===
            
            # Update rover (true motion)
            rover.update(dt)
            true_position = rover.position.copy()
            true_velocity = rover.velocity.copy()
            true_orientation = rover.orientation.copy()
            
            # Simulate sensor failures (demonstration scenarios)
            simulate_sensor_failures(sensor_manager, simulation_time)
            
            # === SENSOR MEASUREMENTS ===
            
            # Get all measurements using the manager
            true_acceleration = np.array([0.0, 0.0, -9.81])  # Simplified: just gravity
            true_angular_velocity = np.array([0.0, 0.0, rover.angular_velocity])
            wheel_speeds = (rover.left_wheel_speed, rover.right_wheel_speed)
            
            measurements = sensor_manager.get_all_measurements(
                true_position, true_acceleration, true_angular_velocity, wheel_speeds
            )
            
            # === KALMAN FILTER UPDATE ===
            
            # Prediction step
            ekf.predict(dt)
            
            # GPS updates (redundant)
            gps_measurements = measurements['gps']
            for gps_data in gps_measurements:
                gps_position = gps_data['position']
                gps_noise = np.eye(3) * (2.0 if gps_data['sensor_id'] == 1 else 3.0)**2
                ekf.update_gps(gps_position, gps_noise)
            
            # IMU updates (redundant)
            imu_measurements = measurements['imu']
            for imu_data in imu_measurements:
                imu_measurement = imu_data['measurement']
                imu_noise = {
                    'acceleration': np.eye(3) * 0.1**2,
                    'angular_velocity': np.eye(3) * 0.05**2
                }
                ekf.update_imu(imu_measurement, imu_noise)
            
            # Odometry update
            if measurements['odometry'] is not None:
                odometry_noise = {
                    'position': np.eye(2) * 0.1**2,
                    'orientation': 0.05**2
                }
                ekf.update_odometry(measurements['odometry'], odometry_noise)
            
            # === DATA COLLECTION ===
            
            estimated_position = ekf.state.position.copy()
            position_error = np.linalg.norm(estimated_position - true_position)
            
            true_positions.append(true_position)
            estimated_positions.append(estimated_position)
            position_errors.append(position_error)
            
            # Sensor health monitoring
            health_status = sensor_manager.get_sensor_health()
            sensor_health_log.append({
                'time': simulation_time,
                'gps_health': [gps.health.reliability for gps in sensor_manager.gps_sensors],
                'imu_health': [imu.health.reliability for imu in sensor_manager.imu_sensors],
                'odometry_health': sensor_manager.wheel_odometry.health.reliability,
                'position_error': position_error,
                'fusion_confidence': ekf.get_fusion_confidence()
            })
            
            # === VISUALIZATION UPDATE ===
            
            if visualizer and step % 2 == 0:  # Update at 10 Hz for smooth visualization
                visualizer.update_position(estimated_position, estimated=True)
                visualizer.update_position(true_position, estimated=False)
                
                # Update sensor status for display
                sensor_status = {
                    'gps': [{'operational': gps.health.is_operational, 'reliability': gps.health.reliability} 
                           for gps in sensor_manager.gps_sensors],
                    'imu': [{'operational': imu.health.is_operational, 'reliability': imu.health.reliability} 
                           for imu in sensor_manager.imu_sensors]
                }
                visualizer.update_sensor_status(sensor_status)
                visualizer.update_plot()
            
            # === REAL-TIME CONTROL ===
            
            if real_time:
                # Maintain real-time pace
                elapsed = time.time() - start_real_time
                target_time = simulation_time
                if elapsed < target_time:
                    time.sleep(min(dt, target_time - elapsed))
            
            # === PROGRESS REPORTING ===
            
            if step % 100 == 0:  # Every 5 seconds
                avg_error = np.mean(position_errors[-100:]) if len(position_errors) >= 100 else np.mean(position_errors)
                gps_health = np.mean([gps.health.reliability for gps in sensor_manager.gps_sensors])
                imu_health = np.mean([imu.health.reliability for imu in sensor_manager.imu_sensors])
                
                print(f"[{simulation_time:5.1f}s] Error: {avg_error:5.2f}m | "
                      f"GPS: {gps_health:4.2f} | IMU: {imu_health:4.2f} | "
                      f"Confidence: {ekf.get_fusion_confidence():4.2f}")
            
            # Increment simulation
            simulation_time += dt
            step += 1
    
    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user")
    
    print(f"\\nSimulation completed at {simulation_time:.1f} seconds")
    
    # === FINAL ANALYSIS ===
    
    print("\\n=== SIMULATION RESULTS ===")
    
    final_position_error = np.linalg.norm(estimated_positions[-1] - true_positions[-1])
    mean_position_error = np.mean(position_errors)
    max_position_error = np.max(position_errors)
    
    print(f"Final position error: {final_position_error:.2f} m")
    print(f"Mean position error:  {mean_position_error:.2f} m")
    print(f"Max position error:   {max_position_error:.2f} m")
    
    # Sensor availability statistics
    gps_availability = []
    imu_availability = []
    
    for health in sensor_health_log:
        gps_availability.append(np.mean([h > 0.5 for h in health['gps_health']]))
        imu_availability.append(np.mean([h > 0.5 for h in health['imu_health']]))
    
    print(f"GPS availability:     {np.mean(gps_availability)*100:.1f}%")
    print(f"IMU availability:     {np.mean(imu_availability)*100:.1f}%")
    
    # Demonstrate redundancy benefits
    print("\\n=== REDUNDANCY DEMONSTRATION ===")
    failure_periods = []
    for i, health in enumerate(sensor_health_log):
        if health['gps_health'][0] < 0.3 or health['imu_health'][0] < 0.3:  # Primary sensor failure
            failure_periods.append(i)
    
    if failure_periods:
        failure_errors = [position_errors[i] for i in failure_periods]
        normal_errors = [position_errors[i] for i in range(len(position_errors)) if i not in failure_periods]
        
        print(f"Error during sensor failures: {np.mean(failure_errors):.2f} ± {np.std(failure_errors):.2f} m")
        print(f"Error during normal operation: {np.mean(normal_errors):.2f} ± {np.std(normal_errors):.2f} m")
        print(f"Redundancy maintained accuracy within {np.mean(failure_errors)/np.mean(normal_errors):.1f}x of normal")
    else:
        print("No significant sensor failures occurred during simulation")
    
    # === POST-SIMULATION VISUALIZATION ===
    
    if visualize:
        print("\\nGenerating trajectory comparison plot...")
        
        # Create trajectory comparison plot
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        true_pos_array = np.array(true_positions)
        est_pos_array = np.array(estimated_positions)
        
        ax1.plot(true_pos_array[:, 0], true_pos_array[:, 1], true_pos_array[:, 2], 
                'g-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(est_pos_array[:, 0], est_pos_array[:, 1], est_pos_array[:, 2], 
                'r-', linewidth=2, label='EKF Estimate', alpha=0.8)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend()
        
        # Position error over time
        ax2 = fig.add_subplot(2, 2, 2)
        time_axis = np.arange(len(position_errors)) * dt
        ax2.plot(time_axis, position_errors, 'b-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error Over Time')
        ax2.grid(True)
        
        # Sensor health over time
        ax3 = fig.add_subplot(2, 2, 3)
        times = [h['time'] for h in sensor_health_log]
        gps1_health = [h['gps_health'][0] for h in sensor_health_log]
        gps2_health = [h['gps_health'][1] for h in sensor_health_log]
        imu1_health = [h['imu_health'][0] for h in sensor_health_log]
        imu2_health = [h['imu_health'][1] for h in sensor_health_log]
        
        ax3.plot(times, gps1_health, 'g-', label='GPS 1', linewidth=2)
        ax3.plot(times, gps2_health, 'g--', label='GPS 2', linewidth=2)
        ax3.plot(times, imu1_health, 'b-', label='IMU 1', linewidth=2)
        ax3.plot(times, imu2_health, 'b--', label='IMU 2', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Sensor Reliability')
        ax3.set_title('Sensor Health Over Time')
        ax3.legend()
        ax3.grid(True)
        
        # Fusion confidence
        ax4 = fig.add_subplot(2, 2, 4)
        confidences = [h['fusion_confidence'] for h in sensor_health_log]
        ax4.plot(times, confidences, 'purple', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Fusion Confidence')
        ax4.set_title('Kalman Filter Confidence')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Keep the real-time 3D plot open
        input("\\nPress Enter to close visualization and exit...")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='3D Rover Localization Demo')
    parser.add_argument('--duration', type=float, default=60.0, 
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--no-real-time', action='store_true',
                       help='Run simulation as fast as possible')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable real-time visualization')
    
    args = parser.parse_args()
    
    try:
        run_simulation(
            duration=args.duration,
            real_time=not args.no_real_time,
            visualize=not args.no_viz
        )
    except Exception as e:
        print(f"\\nSimulation error: {e}")
        raise

if __name__ == "__main__":
    main() 