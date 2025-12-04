from controller import Robot
import numpy as np
import math

# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())  # Typically 64ms for youBot

# Updated robot physical parameters from PROTO definition
WHEEL_RADIUS = 0.05  # meters (from Cylinder radius)
MAX_WHEEL_VELOCITY = 14.81  # rad/s (from RotationalMotor maxVelocity)
HALF_WHEELBASE_LENGTH = 0.235  # (0.47m total length / 2) - from youBot specs
HALF_WHEELBASE_WIDTH = 0.15    # (0.30m total width / 2) - from youBot specs

# Initialize all 4 wheels
wheel_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
wheels = [robot.getDevice(name) for name in wheel_names]
for wheel in wheels:
    wheel.setPosition(float('inf'))  # Velocity control mode
    wheel.setVelocity(0.0)

def clamp_wheel_velocity(velocity):
    """Ensure wheel velocity doesn't exceed maximum"""
    return max(-MAX_WHEEL_VELOCITY, min(velocity, MAX_WHEEL_VELOCITY))

def compute_wheel_velocities(vx, vy, omega):
    """Inverse kinematics for mecanum wheels with velocity limiting"""
    l = HALF_WHEELBASE_LENGTH
    w = HALF_WHEELBASE_WIDTH
    r = WHEEL_RADIUS
    
    # Compute raw wheel velocities
    velocities = [
        (vx - vy - (l + w)*omega)/r,  # Front left
        (vx + vy + (l + w)*omega)/r,  # Front right
        (vx + vy - (l + w)*omega)/r,  # Rear left
        (vx - vy + (l + w)*omega)/r   # Rear right
    ]
    
    # Apply velocity limits
    return [clamp_wheel_velocity(v) for v in velocities]

def move(vx, vy, omega):
    """Command base movement using inverse kinematics with safety checks"""
    wheel_vel = compute_wheel_velocities(vx, vy, omega)
    for i in range(4):
        wheels[i].setVelocity(wheel_vel[i])

def stop():
    """Stop all wheel motion"""
    for wheel in wheels:
        wheel.setVelocity(0.0)

def execute_movement(vx, vy, omega, duration):
    """Execute movement for specified duration"""
    start_time = robot.getTime()
    while robot.step(timestep) != -1:
        current_time = robot.getTime()
        if current_time - start_time < duration:
            move(vx, vy, omega)
        else:
            stop()
            break
    robot.step(timestep)  # Ensure one final step

def run_complete_sequence():
    """Execute the full movement sequence with updated parameters"""
    print("Starting movement sequence...")
    
    # Convert linear speeds to stay within wheel limits
    max_linear_speed = MAX_WHEEL_VELOCITY * WHEEL_RADIUS * 0.8  # 80% of max
    
    # 1. Move forward (2 seconds)
    print("Phase 1: Moving forward")
    execute_movement(min(0.5, max_linear_speed), 0.0, 0.0, 2.0)
    
    # 2. Turn right 90 degrees (π/2 rad)
    print("Phase 2: Turning right")
    turn_rate = min(0.5, MAX_WHEEL_VELOCITY * 0.8)  # 80% of max angular speed
    execute_movement(0.0, 0.0, -turn_rate, math.pi/2 / turn_rate)
    
    # 3. Move backward (2 seconds)
    print("Phase 3: Moving backward")
    execute_movement(max(-0.3, -max_linear_speed), 0.0, 0.0, 2.0)
    
    # 4. Rotate clockwise 180 degrees (π rad)
    print("Phase 4: Rotating clockwise")
    execute_movement(0.0, 0.0, turn_rate, math.pi / turn_rate)
    
    # 5. Turn left while rotating (3 seconds)
    print("Phase 5: Turning left while rotating")
    execute_movement(
        min(0.2, max_linear_speed),
        min(0.2, max_linear_speed),
        min(0.3, turn_rate),
        3.0
    )
    
    # 6. Final forward movement (5 seconds)
    print("Phase 6: Moving forward for 5 seconds")
    execute_movement(min(0.4, max_linear_speed), 0.0, 0.0, 5.0)
    
    print("Movement sequence completed!")

# Main execution
if __name__ == "__main__":
    run_complete_sequence()
    
    # Keep controller running after movement to prevent immediate exit
    while robot.step(timestep) != -1:
        pass
