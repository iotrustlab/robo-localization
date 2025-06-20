#!/usr/bin/env python3
"""
Robo Localization: Multi-Sensor Fusion System

Main entry point for the robo-localization package demonstration.
Runs a complete system simulation with multi-sensor fusion.

Usage:
    python main.py --help
    python main.py --duration 30
    python main.py --duration 60 --no-viz
"""

import sys
import os
import argparse
import time
import numpy as np
import logging

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from robo_localization.sensors import GPSSensor, IMUSensor, SensorFusionManager
from robo_localization.fusion import ExtendedKalmanFilter
from robo_localization.simulation import TrajectoryGenerator, TrajectoryParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedRover:
    """Simplified rover simulation for full system demonstration."""
    
    def __init__(self):
        """Initialize the rover simulation."""
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.angular_velocity = 0.0
        self.time = 0.0
        
        # Control inputs
        self.target_linear_velocity = 1.0  # m/s
        self.target_angular_velocity = 0.0  # rad/s
        
    def set_trajectory_following(self, trajectory, current_time):
        """Improved trajectory following control with better tracking."""
        # Get target position and velocity from trajectory for better tracking
        target_position = trajectory.get_position(current_time)
        target_velocity = trajectory.get_velocity(current_time)
        
        # Compute position error
        position_error = target_position - self.position
        error_magnitude = np.linalg.norm(position_error[:2])  # Only use x,y for control
        
        # Use feedforward + feedback control for better tracking
        if error_magnitude > 0.2:  # Smaller deadband for tighter control
            # Feedforward from trajectory velocity + feedback from position error
            target_speed = np.linalg.norm(target_velocity[:2])  # Get speed from trajectory
            feedback_speed = error_magnitude * 0.8  # Feedback gain
            self.target_linear_velocity = min(3.0, target_speed + feedback_speed)  # Higher max speed
            
            # Compute desired heading from position error
            desired_heading = np.arctan2(position_error[1], position_error[0])
            current_heading = self.orientation[2]  # yaw
            
            # Angular control with better gains
            heading_error = desired_heading - current_heading
            # Normalize to [-pi, pi]
            heading_error = ((heading_error + np.pi) % (2 * np.pi)) - np.pi
            self.target_angular_velocity = np.clip(heading_error * 1.5, -1.0, 1.0)  # Better gains
        else:
            # Gradual slowdown instead of sudden stop
            self.target_linear_velocity *= 0.95
            self.target_angular_velocity *= 0.95
        
    def update(self, dt):
        """Update rover state using responsive realistic kinematics."""
        # Apply more responsive dynamics for better tracking
        alpha = 0.6  # More responsive for better trajectory following
        
        # Update velocities toward targets with better control
        current_linear_vel = np.linalg.norm(self.velocity[:2])  # Only x,y components
        vel_error = self.target_linear_velocity - current_linear_vel
        new_linear_vel = current_linear_vel + alpha * vel_error * dt
        
        angular_vel_error = self.target_angular_velocity - self.angular_velocity
        self.angular_velocity += alpha * angular_vel_error * dt
        
        # Update orientation first
        self.orientation[2] += self.angular_velocity * dt  # Update yaw
        
        # Update velocity in world frame based on current heading
        yaw = self.orientation[2]
        speed = new_linear_vel
        self.velocity[0] = speed * np.cos(yaw)
        self.velocity[1] = speed * np.sin(yaw)
        self.velocity[2] = 0.0  # No vertical motion
        
        # Update position using current velocity
        self.position += self.velocity * dt
        
        # Realistic process noise for sensor modeling
        self.position += np.random.normal(0, 0.005, 3)  # Small realistic position noise
        self.velocity += np.random.normal(0, 0.01, 3)   # Small realistic velocity noise
        
        # Update time
        self.time += dt


def run_full_simulation(duration=30.0, visualize=True, real_time=True):
    """
    Run a complete system simulation with all components.
    
    Args:
        duration: Simulation duration in seconds
        visualize: Whether to show real-time plots (requires matplotlib)
        real_time: Whether to run in real-time or as fast as possible
        
    Returns:
        Dictionary with simulation results and statistics
    """
    logger.info("Robo Localization: Full System Simulation")
    logger.info(f"Duration: {duration}s, Visualization: {visualize}, Real-time: {real_time}")
    
    # ===== INITIALIZATION =====
    
    # Create larger trajectory (figure-8 with elevation) for better visualization
    traj_params = TrajectoryParameters(radius=50.0, height=5.0, period=90.0)  # Larger radius, longer period
    trajectory = TrajectoryGenerator(traj_params)
    logger.info(f"Trajectory created: {traj_params.trajectory_type}")
    
    # Create simplified rover
    rover = SimplifiedRover()
    rover.position = trajectory.get_position(0.0)
    logger.info(f"Rover initialized at: {rover.position}")
    
    # Create sensors with realistic characteristics
    gps1 = GPSSensor(noise_std=2.0, dropout_prob=0.05, sensor_id=1)
    gps2 = GPSSensor(noise_std=3.0, dropout_prob=0.08, sensor_id=2)
    imu1 = IMUSensor(accel_noise_std=0.1, gyro_noise_std=0.05, sensor_id=1)
    imu2 = IMUSensor(accel_noise_std=0.15, gyro_noise_std=0.08, sensor_id=2)
    logger.info("Sensors created: 2x GPS, 2x IMU")
    
    # Create Extended Kalman Filter with better initialization
    initial_position = rover.position.copy()
    initial_state = np.zeros(12)
    initial_state[0:3] = initial_position  # Initialize position to match rover
    # Use moderate initial uncertainty for better convergence
    ekf = ExtendedKalmanFilter(initial_state=initial_state, initial_uncertainty=50.0)
    logger.info(f"Extended Kalman Filter initialized at position: {initial_position}")
    
    # Initialize visualization if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            plt.ion()  # Interactive mode
            fig = plt.figure(figsize=(15, 10))
            
            # 3D trajectory plot
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory: Ground Truth vs Estimated')
            
            # Position error plot
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position Error (m)')
            ax2.set_title('Position Error Over Time')
            ax2.grid(True)
            
            # Sensor reliability plot
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Reliability')
            ax3.set_title('Sensor Health Over Time')
            ax3.grid(True)
            
            # Filter confidence plot
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Confidence')
            ax4.set_title('Filter Confidence')
            ax4.grid(True)
            
            plt.tight_layout()
            logger.info("Visualization initialized")
            
        except ImportError:
            logger.warning("Matplotlib not available, disabling visualization")
            visualize = False
        except Exception as e:
            logger.warning(f"Visualization setup failed: {e}")
            visualize = False
    
    # ===== SIMULATION LOOP =====
    
    dt = 0.1  # 10 Hz simulation
    simulation_time = 0.0
    step_count = 0
    
    # Data collection
    results = {
        'times': [],
        'true_positions': [],
        'estimated_positions': [],
        'position_errors': [],
        'sensor_health': {'gps1': [], 'gps2': [], 'imu1': [], 'imu2': []},
        'filter_confidence': []
    }
    
    logger.info("Starting simulation loop...")
    start_time = time.time()
    
    # Simulation scenarios
    def simulate_failures(t):
        """Simulate sensor failures at specific times."""
        # GPS constellation failure (10-15s)
        if 10.0 <= t <= 15.0:
            return {'gps_degraded': True, 'imu_failure': False}
        # IMU failure (20-25s)
        elif 20.0 <= t <= 25.0:
            return {'gps_degraded': False, 'imu_failure': True}
        # Combined failures (35-40s)
        elif 35.0 <= t <= 40.0:
            return {'gps_degraded': True, 'imu_failure': True}
        else:
            return {'gps_degraded': False, 'imu_failure': False}
    
    try:
        while simulation_time < duration:
            # ===== ROVER SIMULATION =====
            rover.set_trajectory_following(trajectory, simulation_time)
            rover.update(dt)
            
            true_position = rover.position.copy()
            true_velocity = rover.velocity.copy()
            true_acceleration = np.array([0.0, 0.0, -9.81])  # Simplified gravity
            true_angular_velocity = np.array([0.0, 0.0, rover.angular_velocity])
            
            # ===== SENSOR MEASUREMENTS =====
            
            failure_scenario = simulate_failures(simulation_time)
            
            # GPS measurements
            gps_measurements = []
            
            # GPS 1 - always attempt measurement, record failure if degraded
            if failure_scenario['gps_degraded']:
                gps1.health.record_failure()  # Record failure during degraded periods
            else:
                gps1_meas = gps1.get_measurement(true_position)
                if gps1_meas is not None:
                    gps_measurements.append(('gps1', gps1_meas, gps1.noise_std))
            
            # GPS 2 - always attempt measurement, record failure if degraded
            if failure_scenario['gps_degraded']:
                gps2.health.record_failure()  # Record failure during degraded periods
            else:
                gps2_meas = gps2.get_measurement(true_position)
                if gps2_meas is not None:
                    gps_measurements.append(('gps2', gps2_meas, gps2.noise_std))
            
            # IMU measurements
            imu_measurements = []
            
            # IMU 1 - always attempt measurement, record failure if degraded
            if failure_scenario['imu_failure']:
                imu1.health.record_failure()  # Record failure during degraded periods
            else:
                imu1_meas = imu1.get_measurement(true_acceleration, true_angular_velocity)
                if imu1_meas is not None:
                    imu_measurements.append(('imu1', imu1_meas))
            
            # IMU 2 - always active as backup
            imu2_meas = imu2.get_measurement(true_acceleration, true_angular_velocity)
            if imu2_meas is not None:
                imu_measurements.append(('imu2', imu2_meas))
            
            # ===== KALMAN FILTER UPDATES =====
            
            # Prediction step with control input
            control_input = np.zeros(6)
            control_input[5] = rover.angular_velocity  # Use rover's angular velocity
            ekf.predict(dt, control_input)
            
            # GPS updates
            for sensor_name, measurement, noise_std in gps_measurements:
                noise_matrix = np.eye(3) * (noise_std ** 2)
                ekf.update_gps(measurement, noise_matrix)
            
            # IMU updates (angular velocity only for simplicity)
            for sensor_name, measurement in imu_measurements:
                # Use proper noise matrix based on sensor
                if sensor_name == 'imu1':
                    noise_std = imu1.gyro_noise_std
                else:
                    noise_std = imu2.gyro_noise_std
                noise_matrix = np.eye(3) * (noise_std ** 2)
                ekf.update_imu_angular_velocity(measurement['angular_velocity'], noise_matrix)
            
            # ===== DATA COLLECTION =====
            
            estimated_position = ekf.state.position.copy()
            position_error = np.linalg.norm(estimated_position - true_position)
            
            results['times'].append(simulation_time)
            results['true_positions'].append(true_position)
            results['estimated_positions'].append(estimated_position)
            results['position_errors'].append(position_error)
            
            # Sensor health
            results['sensor_health']['gps1'].append(gps1.health.reliability)
            results['sensor_health']['gps2'].append(gps2.health.reliability)
            results['sensor_health']['imu1'].append(imu1.health.reliability)
            results['sensor_health']['imu2'].append(imu2.health.reliability)
            
            results['filter_confidence'].append(ekf.get_fusion_confidence())
            
            # ===== VISUALIZATION UPDATE =====
            
            if visualize and step_count % 5 == 0:  # Update every 0.5 seconds
                try:
                    # Clear and update 3D plot
                    ax1.clear()
                    ax1.set_xlabel('X (m)')
                    ax1.set_ylabel('Y (m)')
                    ax1.set_zlabel('Z (m)')
                    ax1.set_title('3D Trajectory: Ground Truth vs Estimated')
                    
                    if len(results['true_positions']) > 1:
                        true_pos_array = np.array(results['true_positions'])
                        est_pos_array = np.array(results['estimated_positions'])
                        
                        ax1.plot(true_pos_array[:, 0], true_pos_array[:, 1], true_pos_array[:, 2], 
                                'g-', linewidth=2, label='Ground Truth', alpha=0.8)
                        ax1.plot(est_pos_array[:, 0], est_pos_array[:, 1], est_pos_array[:, 2], 
                                'r-', linewidth=2, label='EKF Estimate', alpha=0.8)
                        ax1.legend()
                    
                    # Update error plot
                    ax2.clear()
                    ax2.plot(results['times'], results['position_errors'], 'b-', linewidth=1)
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Position Error (m)')
                    ax2.set_title('Position Error Over Time')
                    ax2.grid(True)
                    
                    # Update sensor health
                    ax3.clear()
                    ax3.plot(results['times'], results['sensor_health']['gps1'], 'g-', label='GPS 1', linewidth=2)
                    ax3.plot(results['times'], results['sensor_health']['gps2'], 'g--', label='GPS 2', linewidth=2)
                    ax3.plot(results['times'], results['sensor_health']['imu1'], 'b-', label='IMU 1', linewidth=2)
                    ax3.plot(results['times'], results['sensor_health']['imu2'], 'b--', label='IMU 2', linewidth=2)
                    ax3.set_xlabel('Time (s)')
                    ax3.set_ylabel('Reliability')
                    ax3.set_title('Sensor Health Over Time')
                    ax3.legend()
                    ax3.grid(True)
                    
                    # Update filter confidence
                    ax4.clear()
                    ax4.plot(results['times'], results['filter_confidence'], 'purple', linewidth=2)
                    ax4.set_xlabel('Time (s)')
                    ax4.set_ylabel('Confidence')
                    ax4.set_title('Filter Confidence')
                    ax4.grid(True)
                    
                    plt.draw()
                    plt.pause(0.01)
                    
                except Exception as e:
                    logger.warning(f"Visualization update failed: {e}")
            
            # ===== REAL-TIME CONTROL =====
            
            if real_time:
                elapsed_real_time = time.time() - start_time
                if elapsed_real_time < simulation_time:
                    time.sleep(min(dt, simulation_time - elapsed_real_time))
            
            # ===== PROGRESS REPORTING =====
            
            if step_count % 50 == 0:  # Every 5 seconds
                avg_error = np.mean(results['position_errors'][-10:]) if len(results['position_errors']) >= 10 else position_error
                gps_health = (gps1.health.reliability + gps2.health.reliability) / 2
                imu_health = (imu1.health.reliability + imu2.health.reliability) / 2
                
                logger.info(f"[{simulation_time:5.1f}s] Error: {avg_error:5.2f}m | "
                           f"GPS: {gps_health:4.2f} | IMU: {imu_health:4.2f} | "
                           f"Confidence: {ekf.get_fusion_confidence():4.2f}")
            
            # Increment counters
            simulation_time += dt
            step_count += 1
            
        logger.info("Simulation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    
    # ===== FINAL ANALYSIS =====
    
    logger.info("\n" + "="*50)
    logger.info("SIMULATION RESULTS")
    logger.info("="*50)
    
    if results['position_errors']:
        final_error = results['position_errors'][-1]
        mean_error = np.mean(results['position_errors'])
        max_error = np.max(results['position_errors'])
        rms_error = np.sqrt(np.mean(np.array(results['position_errors'])**2))
        
        logger.info(f"Final position error:  {final_error:.2f} m")
        logger.info(f"Mean position error:   {mean_error:.2f} m")
        logger.info(f"RMS position error:    {rms_error:.2f} m")
        logger.info(f"Maximum position error: {max_error:.2f} m")
        
        # Sensor availability
        gps1_avail = np.mean([r > 0.5 for r in results['sensor_health']['gps1']]) * 100
        gps2_avail = np.mean([r > 0.5 for r in results['sensor_health']['gps2']]) * 100
        imu1_avail = np.mean([r > 0.5 for r in results['sensor_health']['imu1']]) * 100
        imu2_avail = np.mean([r > 0.5 for r in results['sensor_health']['imu2']]) * 100
        
        logger.info(f"GPS 1 availability:    {gps1_avail:.1f}%")
        logger.info(f"GPS 2 availability:    {gps2_avail:.1f}%")
        logger.info(f"IMU 1 availability:    {imu1_avail:.1f}%")
        logger.info(f"IMU 2 availability:    {imu2_avail:.1f}%")
        
        # Filter performance
        mean_confidence = np.mean(results['filter_confidence'])
        logger.info(f"Mean filter confidence: {mean_confidence:.3f}")
        
        results.update({
            'final_position_error': final_error,
            'mean_position_error': mean_error,
            'rms_position_error': rms_error,
            'max_position_error': max_error,
            'mean_filter_confidence': mean_confidence
        })
    
    if visualize:
        logger.info("\nVisualization is still open. Close the plot window to continue.")
        try:
            plt.show()
            input("Press Enter to continue...")
        except:
            pass
    
    logger.info("Full system simulation completed!")
    return results

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Robo Localization: Multi-Sensor Fusion Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Simulation duration in seconds')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable real-time visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("Running robo localization simulation...")
        results = run_full_simulation(
            duration=args.duration,
            visualize=not args.no_viz,
            real_time=True
        )
        
        print(f"\nSimulation successful!")
        print(f"Final position error: {results.get('final_position_error', 'N/A'):.2f} m")
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\nDemo failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())