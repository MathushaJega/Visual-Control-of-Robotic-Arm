from controller import Robot

# Create Robot instance and set time step
robot = Robot()
TIME_STEP = 64

# Motor names for 4 wheels
wheel_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
wheels = []

# Initialize motors
for name in wheel_names:
    motor = robot.getDevice(name)
    if motor is None:
        print(f"[ERROR] Could not find motor '{name}'")
    else:
        motor.setVelocity(2.0)  # Safe speed
        wheels.append(motor)

# Movement parameters
forward_position = 10.0
backward_position = -10.0
step_counter = 0
phase_duration = 100  # number of steps for each movement phase

# Initial movement: move forward
for motor in wheels:
    motor.setPosition(forward_position)

# Simulation loop
while robot.step(TIME_STEP) != -1:
    step_counter += 1

    # After moving forward, switch to backward
    if step_counter == phase_duration:
        for motor in wheels:
            motor.setPosition(backward_position)

    # After moving backward, stop
    elif step_counter == 2 * phase_duration:
        for motor in wheels:
            motor.setVelocity(0.0)
        break  # end simulation
