import pytest
import numpy as np
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rover import TrajectoryGenerator, MotionModel, RoverSimulation


class TestTrajectoryGenerator:
    """Test 3D figure-8 trajectory generation with elevation changes"""
    
    def test_trajectory_initialization(self):
        """Test trajectory generator initializes correctly"""
        traj = TrajectoryGenerator(radius=10.0, height=5.0, period=30.0)
        
        assert traj.radius == 10.0
        assert traj.height == 5.0
        assert traj.period == 30.0
        
    def test_trajectory_figure_8_shape(self):
        """Test trajectory follows figure-8 pattern"""
        traj = TrajectoryGenerator(radius=10.0, height=0.0, period=30.0)
        
        # Sample trajectory over one complete period
        times = np.linspace(0, 30.0, 100)
        positions = []
        
        for t in times:
            pos = traj.get_position(t)
            positions.append(pos)
            
        positions = np.array(positions)
        
        # Check figure-8 properties
        # X coordinate should oscillate twice per period (figure-8)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Should return to starting position after one period
        start_pos = positions[0]
        end_pos = positions[-1]
        np.testing.assert_allclose(start_pos, end_pos, atol=0.1)
        
        # X should have maximum displacement around radius
        assert np.max(np.abs(x_coords)) <= traj.radius * 1.1
        assert np.max(np.abs(y_coords)) <= traj.radius * 1.1
        
    def test_trajectory_elevation_changes(self):
        """Test trajectory includes elevation changes"""
        traj = TrajectoryGenerator(radius=10.0, height=5.0, period=30.0)
        
        # Sample trajectory
        times = np.linspace(0, 30.0, 100)
        elevations = []
        
        for t in times:
            pos = traj.get_position(t)
            elevations.append(pos[2])
            
        elevations = np.array(elevations)
        
        # Check elevation properties
        assert np.max(elevations) <= traj.height * 1.1
        assert np.min(elevations) >= 0.0
        
        # Should have elevation variation
        elevation_range = np.max(elevations) - np.min(elevations)
        assert elevation_range > traj.height * 0.8
        
    def test_trajectory_velocity_consistency(self):
        """Test velocity is consistent with position derivatives"""
        traj = TrajectoryGenerator(radius=10.0, height=5.0, period=30.0)
        
        t = 15.0  # Mid-trajectory
        dt = 0.01
        
        # Get analytical velocity
        velocity = traj.get_velocity(t)
        
        # Compute numerical derivative
        pos_before = traj.get_position(t - dt/2)
        pos_after = traj.get_position(t + dt/2)
        numerical_velocity = (pos_after - pos_before) / dt
        
        # Should match within reasonable tolerance
        np.testing.assert_allclose(velocity, numerical_velocity, atol=0.1)
        
    def test_trajectory_acceleration_consistency(self):
        """Test acceleration is consistent with velocity derivatives"""
        traj = TrajectoryGenerator(radius=10.0, height=5.0, period=30.0)
        
        t = 10.0
        dt = 0.01
        
        # Get analytical acceleration
        acceleration = traj.get_acceleration(t)
        
        # Compute numerical derivative of velocity
        vel_before = traj.get_velocity(t - dt/2)
        vel_after = traj.get_velocity(t + dt/2)
        numerical_acceleration = (vel_after - vel_before) / dt
        
        # Should match within reasonable tolerance
        np.testing.assert_allclose(acceleration, numerical_acceleration, atol=0.5)
        
    def test_trajectory_periodic_properties(self):
        """Test trajectory is periodic with correct period"""
        traj = TrajectoryGenerator(radius=8.0, height=3.0, period=20.0)
        
        # Test multiple periods
        test_times = [5.0, 10.0, 15.0]
        
        for t in test_times:
            pos_t = traj.get_position(t)
            vel_t = traj.get_velocity(t)
            
            # Position and velocity should repeat after one period
            pos_t_plus_period = traj.get_position(t + traj.period)
            vel_t_plus_period = traj.get_velocity(t + traj.period)
            
            np.testing.assert_allclose(pos_t, pos_t_plus_period, atol=0.01)
            np.testing.assert_allclose(vel_t, vel_t_plus_period, atol=0.01)


class TestMotionModel:
    """Test differential drive rover motion model"""
    
    def test_motion_model_initialization(self):
        """Test motion model initializes correctly"""
        model = MotionModel(wheel_base=0.5, drag_coefficient=0.1)
        
        assert model.wheel_base == 0.5
        assert model.drag_coefficient == 0.1
        
    def test_differential_drive_kinematics(self):
        """Test differential drive kinematics are correct"""
        model = MotionModel(wheel_base=1.0, drag_coefficient=0.0)
        
        # Test straight motion
        left_wheel_speed = 2.0  # rad/s
        right_wheel_speed = 2.0  # rad/s
        
        linear_vel, angular_vel = model.compute_velocities(left_wheel_speed, right_wheel_speed)
        
        # For equal wheel speeds, should move straight
        assert abs(angular_vel) < 1e-6
        assert linear_vel > 0
        
        # Test pure rotation
        left_wheel_speed = 1.0
        right_wheel_speed = -1.0
        
        linear_vel, angular_vel = model.compute_velocities(left_wheel_speed, right_wheel_speed)
        
        # For opposite wheel speeds, should rotate in place
        assert abs(linear_vel) < 1e-6
        assert abs(angular_vel) > 0
        
    def test_coordinate_transformations(self):
        """Test coordinate frame transformations"""
        model = MotionModel(wheel_base=1.0)
        
        # Test body frame to world frame transformation
        body_velocity = np.array([1.0, 0.0])  # Forward motion in body frame
        orientation = np.array([0.0, 0.0, np.pi/2])  # 90 degree rotation
        
        world_velocity = model.body_to_world_velocity(body_velocity, orientation)
        
        # Should transform forward motion to sideways motion in world frame
        expected_world_velocity = np.array([0.0, 1.0])  # Sideways in world frame
        np.testing.assert_allclose(world_velocity, expected_world_velocity, atol=0.01)
        
    def test_motion_dynamics_with_drag(self):
        """Test motion dynamics including drag effects"""
        model = MotionModel(wheel_base=1.0, drag_coefficient=0.1)
        
        # Test that drag reduces velocity
        initial_velocity = np.array([5.0, 0.0])  # High initial velocity
        dt = 1.0
        
        final_velocity = model.apply_drag(initial_velocity, dt)
        
        # Velocity should be reduced by drag
        assert np.linalg.norm(final_velocity) < np.linalg.norm(initial_velocity)
        
        # Direction should be preserved
        if np.linalg.norm(initial_velocity) > 0 and np.linalg.norm(final_velocity) > 0:
            initial_direction = initial_velocity / np.linalg.norm(initial_velocity)
            final_direction = final_velocity / np.linalg.norm(final_velocity)
            np.testing.assert_allclose(initial_direction, final_direction, atol=0.1)
            
    def test_motion_integration(self):
        """Test motion integration over time"""
        model = MotionModel(wheel_base=1.0, drag_coefficient=0.0)
        
        # Initial state
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.array([0.0, 0.0, 0.0])
        velocity = np.array([1.0, 0.0])  # 1 m/s forward
        angular_velocity = 0.0
        
        dt = 1.0
        
        new_position, new_orientation = model.integrate_motion(
            position, orientation, velocity, angular_velocity, dt
        )
        
        # Should move forward 1 meter in X direction
        expected_position = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(new_position, expected_position, atol=0.01)
        
        # Orientation should remain unchanged
        np.testing.assert_allclose(new_orientation, orientation, atol=0.01)


class TestRoverSimulation:
    """Test complete rover simulation"""
    
    def test_rover_simulation_initialization(self):
        """Test rover simulation initializes correctly"""
        sim = RoverSimulation(wheel_base=0.6, wheel_radius=0.15)
        
        assert sim.wheel_base == 0.6
        assert sim.wheel_radius == 0.15
        assert hasattr(sim, 'trajectory')
        assert hasattr(sim, 'motion_model')
        
    def test_trajectory_following_control(self):
        """Test rover follows generated trajectory"""
        sim = RoverSimulation(wheel_base=0.5, wheel_radius=0.1)
        
        # Set up simple trajectory
        sim.trajectory = TrajectoryGenerator(radius=5.0, height=0.0, period=20.0)
        
        # Initialize rover at trajectory start
        start_pos = sim.trajectory.get_position(0.0)
        sim.position = start_pos.copy()
        sim.orientation = np.array([0.0, 0.0, 0.0])
        
        dt = 0.1
        simulation_time = 0.0
        
        # Run simulation for short time
        for _ in range(50):  # 5 seconds
            sim.update(dt)
            simulation_time += dt
            
        # Check that rover is tracking trajectory reasonably well
        target_pos = sim.trajectory.get_position(simulation_time)
        actual_pos = sim.position
        
        tracking_error = np.linalg.norm(actual_pos - target_pos)
        assert tracking_error < 3.0  # Should track within 3 meters
        
    def test_wheel_speed_control(self):
        """Test wheel speed control system"""
        sim = RoverSimulation(wheel_base=1.0, wheel_radius=0.1)
        
        # Test direct wheel speed setting
        left_speed = 2.0  # rad/s
        right_speed = 3.0  # rad/s
        
        sim.set_wheel_speeds(left_speed, right_speed)
        
        # Check that commanded speeds are applied
        assert abs(sim.left_wheel_speed - left_speed) < 0.1
        assert abs(sim.right_wheel_speed - right_speed) < 0.1
        
    def test_rover_state_consistency(self):
        """Test rover state remains physically consistent"""
        sim = RoverSimulation(wheel_base=0.8, wheel_radius=0.12)
        
        # Set initial state
        sim.position = np.array([10.0, 20.0, 5.0])
        sim.orientation = np.array([0.1, 0.2, 0.3])
        sim.velocity = np.array([1.0, 0.5])
        sim.angular_velocity = 0.2
        
        initial_energy = sim.compute_kinetic_energy()
        
        # Run simulation without external forces
        dt = 0.1
        for _ in range(10):
            sim.update(dt)
            
        # Energy should be conserved or decrease (due to drag)
        final_energy = sim.compute_kinetic_energy()
        assert final_energy <= initial_energy * 1.1  # Allow small numerical errors
        
    def test_rover_manual_override(self):
        """Test rover can be manually controlled"""
        sim = RoverSimulation(wheel_base=0.5, wheel_radius=0.1)
        
        # Disable trajectory following
        sim.enable_trajectory_following = False
        
        # Set manual wheel speeds
        sim.set_wheel_speeds(1.5, 1.0)  # Turn right
        
        initial_position = sim.position.copy()
        initial_orientation = sim.orientation.copy()
        
        dt = 0.1
        for _ in range(20):  # 2 seconds
            sim.update(dt)
            
        # Rover should have moved and turned
        position_change = np.linalg.norm(sim.position - initial_position)
        orientation_change = np.linalg.norm(sim.orientation - initial_orientation)
        
        assert position_change > 0.1  # Should have moved
        assert orientation_change > 0.01  # Should have turned
        
    def test_simulation_information_tracking(self):
        """Test simulation tracks useful information"""
        sim = RoverSimulation(wheel_base=0.6, wheel_radius=0.15)
        
        # Run simulation briefly
        dt = 0.1
        for _ in range(10):
            sim.update(dt)
            
        # Check that simulation provides useful information
        info = sim.get_simulation_info()
        
        assert 'position' in info
        assert 'velocity' in info
        assert 'orientation' in info
        assert 'wheel_speeds' in info
        assert 'trajectory_error' in info
        
        # Check data types and shapes
        assert info['position'].shape == (3,)
        assert info['velocity'].shape == (2,)
        assert info['orientation'].shape == (3,)
        assert len(info['wheel_speeds']) == 2
        assert isinstance(info['trajectory_error'], (int, float))


class TestRoverPhysics:
    """Test rover physics and dynamics"""
    
    def test_wheel_radius_scaling(self):
        """Test wheel radius affects linear velocity correctly"""
        # Two rovers with different wheel radii (no drag for pure kinematics)
        sim_small = RoverSimulation(wheel_base=1.0, wheel_radius=0.05, drag_coefficient=0.0)
        sim_large = RoverSimulation(wheel_base=1.0, wheel_radius=0.20, drag_coefficient=0.0)
        
        # Disable trajectory following for pure kinematics test
        sim_small.enable_trajectory_following = False
        sim_large.enable_trajectory_following = False
        
        # Same wheel speeds
        wheel_speed = 2.0  # rad/s
        sim_small.set_wheel_speeds(wheel_speed, wheel_speed)
        sim_large.set_wheel_speeds(wheel_speed, wheel_speed)
        
        # Update once to compute velocities
        sim_small.update(0.1)
        sim_large.update(0.1)
        
        # Large wheels should produce higher linear velocity
        small_velocity = np.linalg.norm(sim_small.velocity)
        large_velocity = np.linalg.norm(sim_large.velocity)
        
        assert large_velocity > small_velocity
        
        # Velocity should be proportional to wheel radius
        expected_ratio = sim_large.wheel_radius / sim_small.wheel_radius
        actual_ratio = large_velocity / small_velocity if small_velocity > 0 else 0
        
        assert abs(actual_ratio - expected_ratio) < 0.1
        
    def test_wheelbase_affects_turning(self):
        """Test wheelbase affects turning radius correctly"""
        # Two rovers with different wheelbases
        sim_narrow = RoverSimulation(wheel_base=0.3, wheel_radius=0.1)
        sim_wide = RoverSimulation(wheel_base=1.2, wheel_radius=0.1)
        
        # Same differential wheel speeds (turning)
        left_speed = 2.0
        right_speed = 1.0
        
        sim_narrow.set_wheel_speeds(left_speed, right_speed)
        sim_wide.set_wheel_speeds(left_speed, right_speed)
        
        # Update to compute angular velocities
        sim_narrow.update(0.1)
        sim_wide.update(0.1)
        
        # Narrower wheelbase should turn faster
        narrow_angular_vel = abs(sim_narrow.angular_velocity)
        wide_angular_vel = abs(sim_wide.angular_velocity)
        
        assert narrow_angular_vel > wide_angular_vel
        
    def test_energy_conservation(self):
        """Test energy conservation in rover dynamics"""
        sim = RoverSimulation(wheel_base=0.8, wheel_radius=0.1, drag_coefficient=0.0)
        
        # Disable trajectory following for pure physics test
        sim.enable_trajectory_following = False
        
        # Set high initial velocity
        sim.velocity = np.array([3.0, 2.0])
        sim.angular_velocity = 0.5
        
        initial_energy = sim.compute_kinetic_energy()
        
        # Coast without wheel input (no drag)
        sim.set_wheel_speeds(0.0, 0.0)
        
        dt = 0.01
        for _ in range(100):  # 1 second
            sim.update(dt)
            
        final_energy = sim.compute_kinetic_energy()
        
        # Energy should be approximately conserved
        energy_change = abs(final_energy - initial_energy) / initial_energy
        assert energy_change < 0.05  # Less than 5% change
        
    def test_drag_energy_dissipation(self):
        """Test drag dissipates energy correctly"""
        sim = RoverSimulation(wheel_base=0.8, wheel_radius=0.1, drag_coefficient=0.2)
        
        # Set high initial velocity
        sim.velocity = np.array([5.0, 0.0])
        sim.angular_velocity = 0.0
        
        initial_energy = sim.compute_kinetic_energy()
        
        # Coast with drag
        sim.set_wheel_speeds(0.0, 0.0)
        
        dt = 0.1
        for _ in range(50):  # 5 seconds
            sim.update(dt)
            
        final_energy = sim.compute_kinetic_energy()
        
        # Energy should decrease due to drag
        assert final_energy < initial_energy * 0.9  # At least 10% reduction
        
        # Final velocity should be significantly reduced
        final_speed = np.linalg.norm(sim.velocity)
        assert final_speed < 2.0  # Should slow down significantly


class TestRoverIntegration:
    """Integration tests for complete rover system"""
    
    def test_complete_trajectory_mission(self):
        """Test rover can complete a full trajectory mission"""
        sim = RoverSimulation(wheel_base=0.6, wheel_radius=0.15)
        
        # Set up trajectory
        sim.trajectory = TrajectoryGenerator(radius=8.0, height=4.0, period=30.0)
        
        # Initialize at trajectory start
        start_pos = sim.trajectory.get_position(0.0)
        sim.position = start_pos.copy()
        sim.orientation = np.array([0.0, 0.0, 0.0])
        
        # Run complete mission
        dt = 0.1
        simulation_time = 0.0
        max_errors = []
        
        # Run for one complete trajectory period
        while simulation_time < sim.trajectory.period:
            sim.update(dt)
            simulation_time += dt
            
            # Track trajectory following error
            target_pos = sim.trajectory.get_position(simulation_time)
            tracking_error = np.linalg.norm(sim.position - target_pos)
            max_errors.append(tracking_error)
            
        # Check overall performance
        max_tracking_error = max(max_errors)
        average_tracking_error = np.mean(max_errors)
        
        assert max_tracking_error < 20.0  # Maximum error should be reasonable for long mission
        assert average_tracking_error < 10.0  # Average error should be good for complex trajectory
        
        # Should return close to starting position (periodic trajectory)
        final_error = np.linalg.norm(sim.position - start_pos)
        assert final_error < 15.0  # Should complete the loop with reasonable drift
        
    def test_rover_robustness_to_disturbances(self):
        """Test rover handles external disturbances"""
        sim = RoverSimulation(wheel_base=0.5, wheel_radius=0.1)
        
        # Set up simple straight-line trajectory
        sim.trajectory = TrajectoryGenerator(radius=0.1, height=0.0, period=100.0)  # Nearly straight
        
        # Initialize
        sim.position = np.array([0.0, 0.0, 0.0])
        sim.orientation = np.array([0.0, 0.0, 0.0])
        
        dt = 0.1
        simulation_time = 0.0
        
        # Run with periodic disturbances
        for i in range(100):  # 10 seconds
            if i % 10 == 0:  # Apply disturbance every second
                # Add random position disturbance
                disturbance = np.random.normal(0, 0.5, 3)
                sim.position += disturbance
                
            sim.update(dt)
            simulation_time += dt
            
        # Despite disturbances, should still track reasonably well
        target_pos = sim.trajectory.get_position(simulation_time)
        tracking_error = np.linalg.norm(sim.position - target_pos)
        
        # Should recover from disturbances
        assert tracking_error < 5.0  # Should maintain reasonable tracking


if __name__ == "__main__":
    pytest.main([__file__]) 