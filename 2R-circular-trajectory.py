import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ------------------------------
# 2R Robot Parameters
# ------------------------------
L1 = 1.0   # length of link 1
L2 = 1.0   # length of link 2

# ------------------------------
# Forward kinematics & Jacobian
# ------------------------------
def fk(theta):
    """Return end-effector (x, y) for given joint angles theta = [th1, th2]."""
    th1, th2 = theta
    x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
    y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
    return np.array([x, y])

def jacobian(theta):
    """Compute 2×2 Jacobian for 2R planar arm."""
    th1, th2 = theta
    J = np.array([
        [ -L1*np.sin(th1) - L2*np.sin(th1 + th2),  -L2*np.sin(th1 + th2) ],
        [  L1*np.cos(th1) + L2*np.cos(th1 + th2),   L2*np.cos(th1 + th2) ]
    ])
    return J

# ------------------------------
# Circle (trajectory) parameters
# ------------------------------
center = np.array([0.5, 0.5])  # center of circle
R = 0.5                        # radius of circle
w = 2 * np.pi / 5             # angular speed → full circle in 5 seconds
dt = 0.01                     # simulation timestep
T = 5.0                       # total time (seconds)
N = int(T / dt)               # number of simulation steps

# ------------------------------
# Initialize joint angles
# ------------------------------
theta = np.array([0.1, 1.3])  # some valid starting pose for the 2R arm

# storage
traj_joint = []
traj_ee = []

# ------------------------------
# Simulation loop: differential IK
# ------------------------------
for k in range(N):
    t = k * dt
    # desired end-effector velocity (circle)
    xd_dot = R * w * np.array([-np.sin(w*t), np.cos(w*t)])
    # compute Jacobian at current theta
    J = jacobian(theta)
    # compute joint velocities
    theta_dot = np.linalg.inv(J) @ xd_dot
    # integrate
    theta = theta + theta_dot * dt
    # store
    traj_joint.append(theta.copy())
    traj_ee.append(fk(theta))

traj_joint = np.array(traj_joint)
traj_ee = np.array(traj_ee)

# ------------------------------
# Setup visualization & animation
# ------------------------------
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(bottom=0.2)

# line for the robot (links)
line, = ax.plot([], [], 'o-', lw=3)
# trace of end-effector
ee_trace, = ax.plot([], [], 'r--', lw=1)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("2R Planar Arm — Circular Trajectory via Differential IK")

def init():
    line.set_data([], [])
    ee_trace.set_data([], [])
    return line, ee_trace

def update(frame):
    th1, th2 = traj_joint[frame]
    x1 = L1 * np.cos(th1)
    y1 = L1 * np.sin(th1)
    x2, y2 = fk([th1, th2])
    # draw links
    line.set_data([0, x1, x2], [0, y1, y2])
    # update end-effector trace
    ee_trace.set_data(traj_ee[:frame, 0], traj_ee[:frame, 1])
    return line, ee_trace

ani = FuncAnimation(fig, update, frames=N, init_func=init,
                    interval=dt*1000, blit=True)

# Button to start animation
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, "Animate")

def on_click(event):
    ani.event_source.start()

button.on_clicked(on_click)

plt.show()
