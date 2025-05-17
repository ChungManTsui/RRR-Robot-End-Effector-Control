import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize

#Configuration
L = 1.0  
n_links = 3
link_lengths = [L, L, L]
target_update_rate = 30    
control_loop_rate = 1000   
dt_target = 1.0 / target_update_rate
dt_control = 1.0 / control_loop_rate
T = 3.0  

#Obstacle Parameters
obs_center = np.array([L, 0.5*L])
obs_radius = L / 8 

#Joint Limits
q_min = np.array([-np.pi, -np.pi, -np.pi])
q_max = np.array([ np.pi,  np.pi,  np.pi])

#Target Trajectory Generator
def target_position(t, L=L, f=2):
    x = 2 * L
    y = L * np.sin(2 * np.pi * f * t)
    return np.array([x, y])

#Kinematics
def forward_kinematics(theta, link_lengths=link_lengths):
    x = y = 0
    angle = 0
    for i in range(len(theta)):
        angle += theta[i]
        x += link_lengths[i] * np.cos(angle)
        y += link_lengths[i] * np.sin(angle)
    return np.array([x, y])

def distance_to_obstacle(point):
    return np.linalg.norm(point - obs_center) - obs_radius

def inverse_kinematics(target, initial_guess, link_lengths=link_lengths, tol=0.1):
    """
    Numerically solve for joint angles to reach target,
    penalizing entering the obstacle and enforcing joint limits.
    """
    penalty_weight = 1000 

    def cost(theta):
        pos = forward_kinematics(theta, link_lengths)
        dist = distance_to_obstacle(pos)
        penalty = 0.0
        if dist < 0:
            penalty = penalty_weight * abs(dist)
        return np.sum((pos - target)**2) + penalty

    bounds = list(zip(q_min, q_max))
    res = minimize(cost, initial_guess, bounds=bounds, method='L-BFGS-B')
    ee_pos = forward_kinematics(res.x, link_lengths)
    is_reset = np.allclose(res.x, np.zeros_like(res.x))
    reset_allowed = np.allclose(target, np.array([3*L, 0]), atol=tol)
    if (
        not res.success
        or np.linalg.norm(ee_pos - target) > tol
        or (is_reset and not reset_allowed)
    ):
        return initial_guess
    return res.x

#Initial Joint Angles
initial_target = target_position(0)
q = inverse_kinematics(initial_target, np.zeros(n_links))

#Simulation Setup
n_steps = int(T / dt_control)
time_array = np.linspace(0, T, n_steps)
q_history = []
ee_history = []
target_history = []

Kp = 15.0 

# Target Sampling Buffe
target_buffer = []
last_target_update = 0

#MAIN SIMULATION 
for i, t in enumerate(time_array):
    if t - last_target_update >= dt_target or i == 0:
        curr_target = target_position(t)
        target_buffer.append((t, curr_target))
        last_target_update = t

    target_xy = target_buffer[-1][1]
    ee_xy = forward_kinematics(q)
    error = target_xy - ee_xy
    desired_xy = ee_xy + Kp * error * dt_control

    q_new = inverse_kinematics(desired_xy, q)
    q = q_new
    q_history.append(q.copy())
    ee_history.append(ee_xy.copy())
    target_history.append(target_xy.copy())

q_history = np.array(q_history)
ee_history = np.array(ee_history)
target_history = np.array(target_history)


#Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(0, 3*L)
ax.set_ylim(-1.5*L, 1.5*L)
ax.set_title("3-DOF RRR Robot: Obstacle Avoidance & Joint Limits")

# Plot the full target trajectory
ax.plot(target_history[:,0], target_history[:,1], 'g--', label='Target Trajectory')

# Draw the obstacle
circle = plt.Circle(obs_center, obs_radius, color='red', fill=True, alpha=0.5, label='Obstacle')
ax.add_patch(circle)

(robot_line,) = ax.plot([], [], 'o-', lw=4, color='blue', label='Robot Arm')
(trace_line,) = ax.plot([], [], '-', color='cyan', lw=2, alpha=0.7, label='EE Trace')
(target_dot,) = ax.plot([], [], 'go', markersize=12, label='Target')

def get_joint_positions(q):
    positions = [np.array([0, 0])]
    angle = 0
    x, y = 0, 0
    for i in range(n_links):
        angle += q[i]
        x += link_lengths[i] * np.cos(angle)
        y += link_lengths[i] * np.sin(angle)
        positions.append(np.array([x, y]))
    return np.array(positions)

def animate(i):
    skip = 1
    idx = i * skip
    if idx >= len(q_history):
        idx = len(q_history) - 1

    q = q_history[idx]
    ee_trace = ee_history[:idx+1]
    target = target_history[idx]

    joint_pos = get_joint_positions(q)
    robot_line.set_data(joint_pos[:,0], joint_pos[:,1])
    trace_line.set_data(ee_trace[:,0], ee_trace[:,1])
    target_dot.set_data([target[0]], [target[1]])
    return robot_line, trace_line, target_dot

animation_duration_sec = T
num_frames = len(q_history)
interval = int(animation_duration_sec * 1000 / num_frames)

ani = animation.FuncAnimation(
    fig, animate, frames=num_frames, interval=interval, blit=True
)

ax.legend()
plt.show()







