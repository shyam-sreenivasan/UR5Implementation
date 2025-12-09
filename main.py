import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# Utility functions
# -------------------------
def skew(w):
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ], dtype=float)

def dh_transform(alpha, a, d, theta):
    """Standard DH transformation matrix."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,    0.0,  1.0]
    ], dtype=float)
    return T

# -------------------------
# DH forward kinematics
# -------------------------
def forward_kinematics_dh(dh_params, joint_angles):
    """Compute FK using DH parameters."""
    T = np.eye(4, dtype=float)
    transforms = []
    for i, params in enumerate(dh_params):
        alpha_i, a_i, d_i, theta_offset = params
        theta_i = joint_angles[i] + theta_offset
        T_i = dh_transform(alpha_i, a_i, d_i, theta_i)
        T = T @ T_i
        transforms.append(T.copy())
    return T, transforms

# -------------------------
# PoE (screw) utilities
# -------------------------
def matrix_exp6(screw, theta):
    """Matrix exponential for SE(3)."""
    omega = np.array(screw[0:3], dtype=float)
    v = np.array(screw[3:6], dtype=float)
    omega_norm = np.linalg.norm(omega)
    T = np.eye(4, dtype=float)

    if omega_norm < 1e-12:
        T[0:3, 3] = v * theta
        return T

    omega_unit = omega / omega_norm
    angle = theta
    omega_hat = skew(omega_unit)
    
    R = np.eye(3) + np.sin(angle) * omega_hat + (1 - np.cos(angle)) * (omega_hat @ omega_hat)
    G = np.eye(3) * angle + (1 - np.cos(angle)) * omega_hat + (angle - np.sin(angle)) * (omega_hat @ omega_hat)
    p = G @ v

    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T

def matrix_log6(T):
    """Matrix logarithm for SE(3)."""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    trace_R = np.trace(R)
    acos_input = np.clip((trace_R - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(acos_input)

    if abs(theta) < 1e-8:
        return np.concatenate((np.zeros(3), p))

    lnR = (theta / (2.0*np.sin(theta))) * (R - R.T)
    omega_unit = np.array([lnR[2,1], lnR[0,2], lnR[1,0]])
    omega_hat = skew(omega_unit)
    G = np.eye(3) * theta + (1 - np.cos(theta)) * omega_hat + (theta - np.sin(theta)) * (omega_hat @ omega_hat)
    v = np.linalg.solve(G, p)
    return np.concatenate((omega_unit*theta, v))

def adjoint(T):
    """Adjoint representation of SE(3)."""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    p_hat = skew(p)
    Ad = np.zeros((6,6), dtype=float)
    Ad[0:3, 0:3] = R
    Ad[0:3, 3:6] = np.zeros((3,3))
    Ad[3:6, 0:3] = p_hat @ R
    Ad[3:6, 3:6] = R
    return Ad

def forward_kinematics_poe(screw_axes, M, joint_angles):
    """Forward kinematics using Product of Exponentials."""
    T = np.eye(4, dtype=float)
    for S, th in zip(screw_axes, joint_angles):
        T = T @ matrix_exp6(S, th)
    return T @ M

def compute_space_jacobian(screw_axes, joint_angles):
    """Compute space Jacobian."""
    n = len(screw_axes)
    J_s = np.zeros((6, n), dtype=float)
    T_accum = np.eye(4, dtype=float)
    
    for i in range(n):
        if i == 0:
            J_s[:, 0] = screw_axes[0]
        else:
            J_s[:, i] = adjoint(T_accum) @ screw_axes[i]
        
        T_accum = T_accum @ matrix_exp6(screw_axes[i], joint_angles[i])
    
    return J_s

def screw_axes_from_dh(dh_params):
    """Derive spatial screw axes from DH parameters."""
    n = dh_params.shape[0]
    T = np.eye(4, dtype=float)
    frames = [T.copy()]
    
    for i in range(n):
        alpha_i, a_i, d_i, theta_offset = dh_params[i]
        theta_i = 0.0 + theta_offset
        T = T @ dh_transform(alpha_i, a_i, d_i, theta_i)
        frames.append(T.copy())

    screw_axes = []
    for i in range(n):
        T_prev = frames[i]
        z_axis = T_prev[0:3, 2]
        q = T_prev[0:3, 3]
        omega_unit = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 1e-12 else np.zeros(3)
        v = -np.cross(omega_unit, q)
        S = np.concatenate((omega_unit, v))
        screw_axes.append(S)
    
    M = frames[-1].copy()
    return screw_axes, M

# -------------------------
# Workspace utilities
# -------------------------
def is_pose_reachable(T):
    """Check if pose is within UR5 workspace (max reach: 850mm)."""
    pos = T[0:3, 3]
    distance_from_base = np.linalg.norm(pos)
    max_reach = 850.0
    min_reach = 150.0
    return min_reach < distance_from_base < max_reach

def generate_reachable_test_target(screw_axes, M, start_config, max_distance=200):
    """
    Generate a random target position that's guaranteed to be reachable.
    Samples from workspace by perturbing a known reachable configuration.
    """
    # Generate random joint configuration
    q_random = np.random.uniform(-np.pi, np.pi, 6)
    T_random = forward_kinematics_poe(screw_axes, M, q_random)
    
    # Verify it's reachable
    if is_pose_reachable(T_random):
        # Optionally constrain to nearby positions
        T_start = forward_kinematics_poe(screw_axes, M, start_config)
        distance = np.linalg.norm(T_random[0:3, 3] - T_start[0:3, 3])
        
        if distance < max_distance:
            return T_random[0:3, 3]
    
    # Fallback: small perturbation from start
    T_start = forward_kinematics_poe(screw_axes, M, start_config)
    offset = np.random.uniform(-100, 100, 3)
    return T_start[0:3, 3] + offset

# -------------------------
# IK Solvers
# -------------------------
def _inverse_kinematics_space_single(T_desired, screw_axes, M, initial_guess, 
                                     tol=3.0, max_iters=2000):
    """Single attempt IK with line search."""
    theta = np.array(initial_guess, dtype=float)
    
    for iteration in range(max_iters):
        T_current = forward_kinematics_poe(screw_axes, M, theta)
        T_err = T_desired @ np.linalg.inv(T_current)
        se3_err = matrix_log6(T_err)
        err_norm = np.linalg.norm(se3_err)
        pos_error = np.linalg.norm(T_desired[0:3, 3] - T_current[0:3, 3])
        
        if pos_error < tol:
            return theta, True, iteration, pos_error
        
        J_s = compute_space_jacobian(screw_axes, theta)
        
        if err_norm > 200:
            damping = 0.00005
        elif err_norm > 100:
            damping = 0.0001
        elif err_norm > 50:
            damping = 0.0005
        elif err_norm > 10:
            damping = 0.001
        elif err_norm > 1:
            damping = 0.01
        else:
            damping = 0.05
        
        JJt = J_s @ J_s.T
        inv_term = np.linalg.inv(JJt + (damping**2) * np.eye(6))
        J_pinv = J_s.T @ inv_term
        delta = J_pinv @ se3_err

        if err_norm > 200:
            step_limit = 3.0
        elif err_norm > 100:
            step_limit = 2.5
        elif err_norm > 50:
            step_limit = 2.0
        elif err_norm > 10:
            step_limit = 1.5
        elif err_norm > 1:
            step_limit = 1.0
        else:
            step_limit = 0.5
        
        step_norm = np.linalg.norm(delta)
        if step_norm > step_limit:
            delta = delta * (step_limit / step_norm)
        
        if pos_error < 20:
            alphas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        else:
            alphas = [1.0, 0.5, 0.25]
        
        best_new_error = np.inf
        best_new_theta = theta + delta
        
        for alpha in alphas:
            theta_test = theta + alpha * delta
            T_test = forward_kinematics_poe(screw_axes, M, theta_test)
            T_err_test = T_desired @ np.linalg.inv(T_test)
            err_test = np.linalg.norm(matrix_log6(T_err_test))
            
            if err_test < best_new_error:
                best_new_error = err_test
                best_new_theta = theta_test
        
        theta = best_new_theta
    
    T_final = forward_kinematics_poe(screw_axes, M, theta)
    final_pos_error = np.linalg.norm(T_desired[0:3, 3] - T_final[0:3, 3])
    return theta, False, max_iters, final_pos_error

def inverse_kinematics_space_multistart(T_desired, screw_axes, M, initial_guess=None,
                                       tol=3.0, max_iters=1000, num_attempts=30, verbose=False):
    """IK solver with multi-start - terminates on first success."""
    if initial_guess is None:
        initial_guess = np.zeros(6)
    
    best_theta = initial_guess.copy()
    best_pos_error = np.inf
    total_iters = 0
    
    theta, success, iters, pos_error = _inverse_kinematics_space_single(
        T_desired, screw_axes, M, initial_guess, tol, 2000
    )
    
    total_iters += iters
    best_theta = theta.copy()
    best_pos_error = pos_error
    
    if pos_error < tol:
        if verbose:
            print(f"  ✓ Initial guess converged! Error: {pos_error:.2f} mm")
        return best_theta, True, total_iters, pos_error
    
    smart_guesses = [
        np.array([0, -np.pi/6, 0, 0, 0, 0]),
        np.array([0, -np.pi/4, np.pi/6, 0, 0, 0]),
        np.array([0, -np.pi/3, np.pi/4, 0, 0, 0]),
        np.array([np.pi/4, -np.pi/4, np.pi/4, 0, np.pi/4, 0]),
        np.array([-np.pi/4, -np.pi/3, np.pi/3, 0, 0, 0]),
    ]
    
    for q_init in smart_guesses:
        theta, success, iters, pos_error = _inverse_kinematics_space_single(
            T_desired, screw_axes, M, q_init, tol, 1000
        )
        
        total_iters += iters
        
        if pos_error < best_pos_error:
            best_theta = theta.copy()
            best_pos_error = pos_error
        
        if pos_error < tol:
            return best_theta, True, total_iters, best_pos_error
    
    return best_theta, best_pos_error < tol, total_iters, best_pos_error

def inverse_kinematics_space(T_desired, screw_axes, M, initial_guess=None, tol=3.0, verbose=False):
    """IK with multi-start and 3mm tolerance."""
    if initial_guess is None:
        initial_guess = np.zeros(6)
    
    return inverse_kinematics_space_multistart(
        T_desired, screw_axes, M, initial_guess, 
        tol=tol, num_attempts=10, verbose=False
    )

def plan_straight_line_trajectory(screw_axes, M, start_config, direction, step_size, num_steps, verbose=True):
    """
    Plan a straight-line trajectory in Cartesian space.
    Only proceeds if target positions are within workspace.
    """
    trajectory = []
    current_config = np.array(start_config, dtype=float)
    
    T_current = forward_kinematics_poe(screw_axes, M, current_config)
    start_position = T_current[0:3, 3].copy()
    
    if verbose:
        print(f"\nStarting position: {np.round(start_position, 2)}")
        print(f"Direction: {direction}")
        print(f"Step size: {step_size} mm")
        print(f"Total steps: {num_steps}")
        print(f"Total distance: {step_size * num_steps} mm\n")
    
    # Normalize direction
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    
    all_success = True
    
    for step in range(num_steps):
        # Compute target position
        target_position = start_position + direction * step_size * (step + 1)
        
        # Check if target is reachable
        T_target_check = np.eye(4)
        T_target_check[0:3, 3] = target_position
        
        if not is_pose_reachable(T_target_check):
            if verbose:
                print(f"Step {step+1}: ✗ Target position outside workspace! Stopping trajectory.")
            break
        
        # Create target transformation
        T_target = T_current.copy()
        T_target[0:3, 3] = target_position
        
        # Solve IK using current config as initial guess
        theta_new, success, iters, error = inverse_kinematics_space(
            T_target, screw_axes, M, current_config, tol=3.0, verbose=False
        )
        
        # Verify actual position achieved
        T_achieved = forward_kinematics_poe(screw_axes, M, theta_new)
        actual_position = T_achieved[0:3, 3]
        position_error = np.linalg.norm(target_position - actual_position)
        
        # Store in trajectory
        trajectory.append((theta_new.copy(), T_achieved.copy(), position_error))
        
        if verbose:
            status = "✓" if success else "✗"
            print(f"Step {step+1:2d}/{num_steps}: {status} "
                  f"target={np.round(target_position, 1)}, "
                  f"error={position_error:.2f}mm")
        
        if not success:
            all_success = False
        
        # Update for next iteration
        current_config = theta_new
        T_current = T_achieved
    
    if verbose and len(trajectory) > 0:
        errors = [err for _, _, err in trajectory]
        print(f"\nTrajectory Summary:")
        print(f"  Completed steps: {len(trajectory)}/{num_steps}")
        print(f"  Successful (< 3mm): {sum(1 for _, _, e in trajectory if e < 3.0)}/{len(trajectory)}")
        print(f"  Average error: {np.mean(errors):.2f} mm")
        print(f"  Max error: {np.max(errors):.2f} mm")
    
    return trajectory, all_success

# -------------------------
# Visualization Functions
# -------------------------
def draw_frame_3d(ax, T, scale=50, alpha=1.0, show_label=False):
    """Draw RGB coordinate frame in 3D."""
    origin = T[0:3, 3]
    x_axis = origin + scale * T[0:3, 0]
    y_axis = origin + scale * T[0:3, 1]
    z_axis = origin + scale * T[0:3, 2]
    
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
            'r-', linewidth=2.5, alpha=alpha, label='X-axis (red)' if show_label else '')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
            'g-', linewidth=2.5, alpha=alpha, label='Y-axis (green)' if show_label else '')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
            'b-', linewidth=2.5, alpha=alpha, label='Z-axis (blue)' if show_label else '')

def visualize_robot_home(dh_params, screw_axes, M):
    """1) Visualize UR5 at home position with frames."""
    print("\n=== Visualization 1: Home Position ===")
    
    q_home = np.zeros(6)
    T_final, transforms = forward_kinematics_dh(dh_params, q_home)
    
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw base frame with label
    draw_frame_3d(ax, np.eye(4), scale=80, show_label=True)
    
    # Draw robot links and frames
    pts = np.zeros((len(transforms) + 1, 3))
    pts[0] = np.zeros(3)
    
    for i, T in enumerate(transforms):
        pts[i + 1] = T[0:3, 3]
        draw_frame_3d(ax, T, scale=50, alpha=0.7)
    
    # Draw robot structure
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko-', linewidth=5, markersize=12, label='Robot Links')
    
    # Styling
    ax.set_xlabel('X (mm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=13, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=13, fontweight='bold')
    ax.set_title('UR5 Robot - Home Configuration', fontsize=16, fontweight='bold')
    
    lim = 900
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Home position visualization complete")

def visualize_fk_interactive(dh_params, screw_axes, M):
    """2) Interactive FK with joint angle sliders."""
    print("\n=== Visualization 2: Interactive Forward Kinematics ===")
    print("Use sliders to control joint angles (in degrees)")
    
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.35)
    
    # Current joint angles
    current_q = [0.0] * 6
    
    def plot_robot(q):
        ax.clear()
        T_final, transforms = forward_kinematics_dh(dh_params, q)
        
        # Draw base frame
        draw_frame_3d(ax, np.eye(4), scale=60)
        
        # Draw robot
        pts = np.zeros((len(transforms) + 1, 3))
        pts[0] = np.zeros(3)
        
        for i, T in enumerate(transforms):
            pts[i + 1] = T[0:3, 3]
            draw_frame_3d(ax, T, scale=40, alpha=0.6)
        
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko-', linewidth=5, markersize=12)
        ax.plot([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], 'ro', markersize=18, label='End-Effector')
        
        # Styling
        ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
        ax.set_title(f'UR5 FK - End-Effector: [{pts[-1,0]:.1f}, {pts[-1,1]:.1f}, {pts[-1,2]:.1f}] mm', 
                     fontsize=13, fontweight='bold')
        
        lim = 900
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        fig.canvas.draw_idle()
    
    plot_robot(np.zeros(6))
    
    # Create sliders with individual update functions
    sliders = []
    
    def make_update(idx):
        def update(val):
            current_q[idx] = np.deg2rad(val)
            plot_robot(np.array(current_q))
        return update
    
    for i in range(6):
        ax_slider = plt.axes([0.15, 0.28 - i * 0.04, 0.7, 0.025])
        slider = Slider(ax_slider, f'θ{i+1} (deg)', -180, 180, valinit=0, valstep=1)
        slider.on_changed(make_update(i))
        sliders.append(slider)
    
    plt.show()
    print("✓ Interactive FK visualization complete")

def visualize_ik_path(dh_params, screw_axes, M, start_config, target_position):
    """3) Visualize IK solution path from start to target."""
    print("\n=== Visualization 3: IK Path Visualization ===")
    
    T_start = forward_kinematics_poe(screw_axes, M, start_config)
    start_pos = T_start[0:3, 3]
    
    # Check if target is reachable
    T_target = T_start.copy()
    T_target[0:3, 3] = target_position
    
    if not is_pose_reachable(T_target):
        print(f"✗ Warning: Target position may be outside workspace!")
        print(f"  Distance from base: {np.linalg.norm(target_position):.1f} mm (max: 850mm)")
    
    print(f"Start position: {np.round(start_pos, 2)}")
    print(f"Target position: {np.round(target_position, 2)}")
    print(f"Distance: {np.linalg.norm(target_position - start_pos):.2f} mm")
    
    # Solve IK with verbose output
    print("\nSolving IK with multi-start strategy...")
    theta_solution, success, iters, error = inverse_kinematics_space(
        T_target, screw_axes, M, start_config, tol=3.0, verbose=True
    )
    
    print(f"\nIK Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"Final error: {error:.2f} mm, Total iterations: {iters}")
    
    # Plot
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw start configuration
    _, transforms_start = forward_kinematics_dh(dh_params, start_config)
    pts_start = np.zeros((len(transforms_start) + 1, 3))
    pts_start[0] = np.zeros(3)
    for i, T in enumerate(transforms_start):
        pts_start[i + 1] = T[0:3, 3]
    
    ax.plot(pts_start[:, 0], pts_start[:, 1], pts_start[:, 2], 
            'b-o', linewidth=4, markersize=10, alpha=0.5, label='Start Config')
    
    # Draw solution configuration
    _, transforms_sol = forward_kinematics_dh(dh_params, theta_solution)
    pts_sol = np.zeros((len(transforms_sol) + 1, 3))
    pts_sol[0] = np.zeros(3)
    for i, T in enumerate(transforms_sol):
        pts_sol[i + 1] = T[0:3, 3]
    
    ax.plot(pts_sol[:, 0], pts_sol[:, 1], pts_sol[:, 2], 
            'r-o', linewidth=4, markersize=10, label='IK Solution')
    
    # Draw desired path
    ax.plot([start_pos[0], target_position[0]], 
            [start_pos[1], target_position[1]], 
            [start_pos[2], target_position[2]], 
            'g--', linewidth=3, label='Desired Path')
    
    # Mark points
    ax.plot([start_pos[0]], [start_pos[1]], [start_pos[2]], 
            'go', markersize=18, label='Start Point')
    ax.plot([target_position[0]], [target_position[1]], [target_position[2]], 
            'mo', markersize=18, label='Target Point')
    
    actual_end = pts_sol[-1]
    ax.plot([actual_end[0]], [actual_end[1]], [actual_end[2]], 
            'r*', markersize=25, label=f'Achieved (err:{error:.1f}mm)')
    
    ax.set_xlabel('X (mm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=13, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=13, fontweight='bold')
    ax.set_title('IK Path Visualization', fontsize=16, fontweight='bold')
    
    lim = 900
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    print("✓ IK path visualization complete")

def visualize_trajectory_interactive(dh_params, screw_axes, M, start_config):
    """4) Interactive trajectory control with sliders and direction buttons."""
    print(f"\n=== Visualization 4: Interactive Trajectory Control ===")
    print("Use sliders to set joint angles, then use direction buttons")
    print("Step size: 10mm per click")
    
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.40, top=0.95)
    
    # State variables
    home_config = np.array(start_config, dtype=float)
    current_config = np.array(start_config, dtype=float)
    current_q = [0.0] * 6
    trajectory_history = []
    
    # Get initial position
    T_current = forward_kinematics_poe(screw_axes, M, current_config)
    current_position = T_current[0:3, 3].copy()
    trajectory_history.append(current_position.copy())
    
    def plot_robot():
        ax.clear()
        
        # Get current transforms
        _, transforms = forward_kinematics_dh(dh_params, current_config)
        
        # Draw robot
        pts = np.zeros((len(transforms) + 1, 3))
        pts[0] = np.zeros(3)
        for i, T in enumerate(transforms):
            pts[i + 1] = T[0:3, 3]
            if i == len(transforms) - 1:
                draw_frame_3d(ax, T, scale=60, alpha=0.8)
        
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko-', linewidth=5, markersize=12, label='Robot')
        
        # Draw trajectory history
        if len(trajectory_history) > 1:
            traj = np.array(trajectory_history)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   'r-', linewidth=3, alpha=0.7, label='Path Traced')
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   'r.', markersize=8)
        
        # Current end-effector
        current_ee = pts[-1]
        ax.plot([current_ee[0]], [current_ee[1]], [current_ee[2]], 
               'ro', markersize=18, label='End-Effector')
        
        # Styling
        ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
        
        distance = np.linalg.norm(current_ee)
        ax.set_title(f'Interactive Trajectory Control | Position: [{current_ee[0]:.1f}, {current_ee[1]:.1f}, {current_ee[2]:.1f}] mm | Reach: {distance:.1f}mm', 
                    fontsize=13, fontweight='bold')
        
        lim = 900
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        fig.canvas.draw_idle()
    
    def update_from_sliders():
        """Update robot from slider values."""
        nonlocal current_config, current_position, trajectory_history
        current_config = np.array(current_q, dtype=float)
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        current_position = T_current[0:3, 3].copy()
        trajectory_history = [current_position.copy()]
        plot_robot()
    
    def move_direction(direction, step_size=10.0):
        """Move end-effector in specified direction."""
        nonlocal current_config, current_position, current_q
        
        target_position = current_position + np.array(direction) * step_size
        
        T_check = np.eye(4)
        T_check[0:3, 3] = target_position
        if not is_pose_reachable(T_check):
            print(f"✗ Target outside workspace! Distance: {np.linalg.norm(target_position):.1f}mm")
            return
        
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        T_target = T_current.copy()
        T_target[0:3, 3] = target_position
        
        print(f"\nMoving {direction} by {step_size}mm...")
        theta_new, success, iters, error = inverse_kinematics_space(
            T_target, screw_axes, M, current_config, tol=3.0, verbose=False
        )
        
        T_achieved = forward_kinematics_poe(screw_axes, M, theta_new)
        actual_position = T_achieved[0:3, 3]
        pos_error = np.linalg.norm(target_position - actual_position)
        
        status = "✓" if success else "✗"
        print(f"{status} Target: {np.round(target_position, 1)}, Achieved: {np.round(actual_position, 1)}, Error: {pos_error:.2f}mm")
        
        current_config = theta_new
        current_position = actual_position.copy()
        trajectory_history.append(current_position.copy())
        
        # Update sliders
        for i in range(6):
            current_q[i] = theta_new[i]
            sliders[i].set_val(np.rad2deg(theta_new[i]))
        
        plot_robot()
    
    # Create sliders
    sliders = []
    
    def make_slider_update(idx):
        def update(val):
            current_q[idx] = np.deg2rad(val)
            update_from_sliders()
        return update
    
    for i in range(6):
        ax_slider = plt.axes([0.15, 0.33 - i * 0.035, 0.7, 0.02])
        slider = Slider(ax_slider, f'θ{i+1}', -180, 180, valinit=0, valstep=1)
        slider.on_changed(make_slider_update(i))
        sliders.append(slider)
    
    # Create buttons
    button_width = 0.08
    button_height = 0.035
    
    ax_px = plt.axes([0.15, 0.08, button_width, button_height])
    ax_nx = plt.axes([0.15, 0.03, button_width, button_height])
    ax_py = plt.axes([0.28, 0.08, button_width, button_height])
    ax_ny = plt.axes([0.28, 0.03, button_width, button_height])
    ax_pz = plt.axes([0.41, 0.08, button_width, button_height])
    ax_nz = plt.axes([0.41, 0.03, button_width, button_height])
    ax_reset = plt.axes([0.56, 0.055, button_width, button_height])
    ax_home = plt.axes([0.69, 0.055, button_width, button_height])
    ax_random = plt.axes([0.82, 0.055, button_width, button_height])
    
    btn_px = Button(ax_px, '+X', color='lightcoral', hovercolor='coral')
    btn_nx = Button(ax_nx, '-X', color='lightcoral', hovercolor='coral')
    btn_py = Button(ax_py, '+Y', color='lightgreen', hovercolor='green')
    btn_ny = Button(ax_ny, '-Y', color='lightgreen', hovercolor='green')
    btn_pz = Button(ax_pz, '+Z', color='lightblue', hovercolor='blue')
    btn_nz = Button(ax_nz, '-Z', color='lightblue', hovercolor='blue')
    btn_reset = Button(ax_reset, 'Reset Path', color='lightyellow', hovercolor='yellow')
    btn_home = Button(ax_home, 'Home', color='lightcyan', hovercolor='cyan')
    btn_random = Button(ax_random, 'Random', color='plum', hovercolor='violet')
    
    btn_px.on_clicked(lambda e: move_direction([1, 0, 0]))
    btn_nx.on_clicked(lambda e: move_direction([-1, 0, 0]))
    btn_py.on_clicked(lambda e: move_direction([0, 1, 0]))
    btn_ny.on_clicked(lambda e: move_direction([0, -1, 0]))
    btn_pz.on_clicked(lambda e: move_direction([0, 0, 1]))
    btn_nz.on_clicked(lambda e: move_direction([0, 0, -1]))
    
    def reset(e):
        nonlocal trajectory_history
        trajectory_history = [current_position.copy()]
        print("\n🔄 Trajectory cleared")
        plot_robot()
    
    def go_home(e):
        nonlocal current_config, current_position, trajectory_history, current_q
        current_config = home_config.copy()
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        current_position = T_current[0:3, 3].copy()
        trajectory_history = [current_position.copy()]
        for i in range(6):
            current_q[i] = 0.0
            sliders[i].set_val(0)
        print("\n🏠 Home")
        plot_robot()
    
    def random_pose(e):
        nonlocal current_config, current_position, trajectory_history, current_q
        for _ in range(50):
            q_random = np.random.uniform(-np.pi, np.pi, 6)
            T_random = forward_kinematics_poe(screw_axes, M, q_random)
            if is_pose_reachable(T_random):
                current_config = q_random
                current_position = T_random[0:3, 3].copy()
                trajectory_history = [current_position.copy()]
                for i in range(6):
                    current_q[i] = q_random[i]
                    sliders[i].set_val(np.rad2deg(q_random[i]))
                print(f"\n🎲 Random pose: {np.round(current_position, 1)}")
                plot_robot()
                return
    
    btn_reset.on_clicked(reset)
    btn_home.on_clicked(go_home)
    btn_random.on_clicked(random_pose)
    
    plot_robot()
    plt.show()
    print("✓ Interactive trajectory control complete")

def visualize_circular_motion(dh_params, screw_axes, M, start_config):
    """5) Circular motion with sliders and circle controls."""
    print(f"\n=== Visualization 5: Circular Motion Control ===")
    print("Use sliders to set joint angles, then trace circular paths")
    
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.35, top=0.95)
    
    # State variables
    home_config = np.array(start_config, dtype=float)
    current_config = np.array(start_config, dtype=float)
    current_q = [0.0] * 6
    trajectory_history = []
    
    circle_axis = [0, 0, 1]
    circle_radius = [50.0]
    current_angle = 0.0
    circle_center = None
    circle_initialized = False
    
    T_current = forward_kinematics_poe(screw_axes, M, current_config)
    current_position = T_current[0:3, 3].copy()
    trajectory_history.append(current_position.copy())
    
    def initialize_circle():
        nonlocal circle_center, circle_initialized, current_angle, current_position
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        current_pos = T_current[0:3, 3].copy()
        
        axis = np.array(circle_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        
        if not np.allclose(axis, [1, 0, 0]):
            v1 = np.cross(axis, [1, 0, 0])
        else:
            v1 = np.cross(axis, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)
        
        circle_center = current_pos - circle_radius[0] * v1
        circle_initialized = True
        current_angle = 0.0
        current_position = current_pos.copy()
        
        print(f"\n⭕ Circle initialized at 0°, radius: {circle_radius[0]:.0f}mm")
    
    def compute_circle_point(angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        axis = np.array(circle_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        
        if not np.allclose(axis, [1, 0, 0]):
            v1 = np.cross(axis, [1, 0, 0])
        else:
            v1 = np.cross(axis, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(axis, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        point = circle_center + circle_radius[0] * (np.cos(angle_rad) * v1 + np.sin(angle_rad) * v2)
        return point
    
    def plot_robot():
        ax.clear()
        _, transforms = forward_kinematics_dh(dh_params, current_config)
        
        pts = np.zeros((len(transforms) + 1, 3))
        pts[0] = np.zeros(3)
        for i, T in enumerate(transforms):
            pts[i + 1] = T[0:3, 3]
            if i == len(transforms) - 1:
                draw_frame_3d(ax, T, scale=60, alpha=0.8)
        
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko-', linewidth=5, markersize=12, label='Robot')
        
        if len(trajectory_history) > 1:
            traj = np.array(trajectory_history)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=3, alpha=0.7, label='Path')
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r.', markersize=8)
        
        if circle_initialized:
            ax.plot([circle_center[0]], [circle_center[1]], [circle_center[2]], 
                   'g*', markersize=20, label='Center')
            preview_angles = np.linspace(0, 360, 72)
            preview_pts = np.array([compute_circle_point(a) for a in preview_angles])
            ax.plot(preview_pts[:, 0], preview_pts[:, 1], preview_pts[:, 2], 
                   'g--', linewidth=2, alpha=0.4, label='Circle')
        
        current_ee = pts[-1]
        ax.plot([current_ee[0]], [current_ee[1]], [current_ee[2]], 'ro', markersize=18, label='End-Effector')
        
        ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
        
        axis_name = 'Z' if np.allclose(circle_axis, [0,0,1]) else 'X' if np.allclose(circle_axis, [1,0,0]) else 'Y'
        ax.set_title(f'Circle | Axis: {axis_name} | R: {circle_radius[0]:.0f}mm | Angle: {current_angle:.0f}°', 
                    fontsize=13, fontweight='bold')
        
        lim = 900
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        fig.canvas.draw_idle()
    
    def update_from_sliders():
        nonlocal current_config, current_position, trajectory_history, circle_initialized
        current_config = np.array(current_q, dtype=float)
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        current_position = T_current[0:3, 3].copy()
        trajectory_history = [current_position.copy()]
        circle_initialized = False  # Only reset when manually moved
        plot_robot()
    
    # Track if we're in the middle of automated movement (step_circle)
    in_automated_move = [False]
    
    def step_circle(e):
        nonlocal current_config, current_position, current_angle, trajectory_history, current_q
        if not circle_initialized:
            print("⚠️ Initialize circle first!")
            return
        
        current_angle += 10.0
        if current_angle >= 360.0:
            current_angle = 0.0
        
        target_position = compute_circle_point(current_angle)
        
        T_check = np.eye(4)
        T_check[0:3, 3] = target_position
        if not is_pose_reachable(T_check):
            print(f"✗ Outside workspace!")
            return
        
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        T_target = T_current.copy()
        T_target[0:3, 3] = target_position
        
        theta_new, success, iters, error = inverse_kinematics_space(
            T_target, screw_axes, M, current_config, tol=3.0, verbose=False
        )
        
        T_achieved = forward_kinematics_poe(screw_axes, M, theta_new)
        actual_position = T_achieved[0:3, 3]
        
        current_config = theta_new
        current_position = actual_position.copy()
        trajectory_history.append(current_position.copy())
        
        # Update sliders without triggering circle reset
        in_automated_move[0] = True
        for i in range(6):
            current_q[i] = theta_new[i]
            sliders[i].set_val(np.rad2deg(theta_new[i]))
        in_automated_move[0] = False
        
        status = "✓" if success else "✗"
        print(f"{status} {current_angle:.0f}°, Error: {error:.2f}mm")
        plot_robot()
    
    def set_axis_x(e):
        nonlocal circle_axis, circle_initialized
        circle_axis = [1, 0, 0]
        circle_initialized = False
        print("\n🔴 Axis: X")
    
    def set_axis_y(e):
        nonlocal circle_axis, circle_initialized
        circle_axis = [0, 1, 0]
        circle_initialized = False
        print("\n🟢 Axis: Y")
    
    def set_axis_z(e):
        nonlocal circle_axis, circle_initialized
        circle_axis = [0, 0, 1]
        circle_initialized = False
        print("\n🔵 Axis: Z")
    
    def increase_radius(e):
        nonlocal circle_initialized
        circle_radius[0] += 10.0
        circle_initialized = False
        print(f"\n📏 Radius: {circle_radius[0]:.0f}mm")
        plot_robot()
    
    def decrease_radius(e):
        nonlocal circle_initialized
        if circle_radius[0] > 50.0:
            circle_radius[0] -= 10.0
            circle_initialized = False
            print(f"\n📏 Radius: {circle_radius[0]:.0f}mm")
        else:
            print("\n⚠️ Min radius: 50mm")
        plot_robot()
    
    def init_circle_btn(e):
        initialize_circle()
        plot_robot()
    
    def reset(e):
        nonlocal trajectory_history, circle_initialized, current_angle
        trajectory_history = [current_position.copy()]
        circle_initialized = False
        current_angle = 0.0
        print("\n🔄 Reset")
        plot_robot()
    
    def go_home(e):
        nonlocal current_config, current_position, trajectory_history, circle_initialized, current_angle, current_q
        current_config = home_config.copy()
        T_current = forward_kinematics_poe(screw_axes, M, current_config)
        current_position = T_current[0:3, 3].copy()
        trajectory_history = [current_position.copy()]
        circle_initialized = False
        current_angle = 0.0
        for i in range(6):
            current_q[i] = 0.0
            sliders[i].set_val(0)
        print("\n🏠 Home")
        plot_robot()
    
    def random_pose(e):
        nonlocal current_config, current_position, trajectory_history, circle_initialized, current_angle, current_q
        for _ in range(50):
            q_random = np.random.uniform(-np.pi, np.pi, 6)
            T_random = forward_kinematics_poe(screw_axes, M, q_random)
            if is_pose_reachable(T_random):
                current_config = q_random
                current_position = T_random[0:3, 3].copy()
                trajectory_history = [current_position.copy()]
                circle_initialized = False
                current_angle = 0.0
                for i in range(6):
                    current_q[i] = q_random[i]
                    sliders[i].set_val(np.rad2deg(q_random[i]))
                print(f"\n🎲 Random: {np.round(current_position, 1)}")
                plot_robot()
                return
    
    # Create sliders
    sliders = []
    def make_slider_update(idx):
        def update(val):
            if not in_automated_move[0]:  # Only update from manual slider movement
                current_q[idx] = np.deg2rad(val)
                update_from_sliders()
            else:  # Just update the value during automated movement
                current_q[idx] = np.deg2rad(val)
        return update
    
    for i in range(6):
        ax_slider = plt.axes([0.15, 0.33 - i * 0.035, 0.7, 0.02])
        slider = Slider(ax_slider, f'θ{i+1}', -180, 180, valinit=0, valstep=1)
        slider.on_changed(make_slider_update(i))
        sliders.append(slider)
    
    # Create buttons
    button_width = 0.08
    button_height = 0.035
    
    ax_axis_x = plt.axes([0.15, 0.11, button_width, button_height])
    ax_axis_y = plt.axes([0.28, 0.11, button_width, button_height])
    ax_axis_z = plt.axes([0.41, 0.11, button_width, button_height])
    ax_rad_minus = plt.axes([0.15, 0.06, button_width, button_height])
    ax_rad_plus = plt.axes([0.28, 0.06, button_width, button_height])
    ax_init = plt.axes([0.54, 0.085, button_width, button_height])
    ax_step = plt.axes([0.62, 0.085, button_width, button_height])
    ax_home = plt.axes([0.70, 0.085, button_width, button_height])
    ax_random = plt.axes([0.78, 0.085, button_width, button_height])
    ax_reset = plt.axes([0.86, 0.085, button_width, button_height])
    
    btn_axis_x = Button(ax_axis_x, 'Axis: X', color='lightcoral', hovercolor='coral')
    btn_axis_y = Button(ax_axis_y, 'Axis: Y', color='lightgreen', hovercolor='green')
    btn_axis_z = Button(ax_axis_z, 'Axis: Z', color='lightblue', hovercolor='blue')
    btn_rad_minus = Button(ax_rad_minus, 'R -10mm', color='mistyrose', hovercolor='lightpink')
    btn_rad_plus = Button(ax_rad_plus, 'R +10mm', color='mistyrose', hovercolor='lightpink')
    btn_init = Button(ax_init, 'Init Circle', color='wheat', hovercolor='orange')
    btn_step = Button(ax_step, 'Step (10°)', color='lightsteelblue', hovercolor='steelblue')
    btn_home = Button(ax_home, 'Home', color='lightcyan', hovercolor='cyan')
    btn_random = Button(ax_random, 'Random', color='plum', hovercolor='violet')
    btn_reset = Button(ax_reset, 'Reset', color='lightyellow', hovercolor='yellow')
    
    btn_axis_x.on_clicked(set_axis_x)
    btn_axis_y.on_clicked(set_axis_y)
    btn_axis_z.on_clicked(set_axis_z)
    btn_rad_minus.on_clicked(decrease_radius)
    btn_rad_plus.on_clicked(increase_radius)
    btn_init.on_clicked(init_circle_btn)
    btn_step.on_clicked(step_circle)
    btn_reset.on_clicked(reset)
    btn_home.on_clicked(go_home)
    btn_random.on_clicked(random_pose)
    
    plot_robot()
    plt.show()
    print("✓ Circular motion control complete")

# -------------------------
# Main Program
# -------------------------
if __name__ == "__main__":
    # UR5 DH parameters
    ur5_dh_params = np.array([
        [np.pi/2, 0.0, 89.2, 0.0],
        [0.0, 425.0, 0.0, 0.0],
        [0.0, 392.0, 0.0, 0.0],
        [np.pi/2, 0.0, 109.3, 0.0],
        [-np.pi/2, 0.0, 94.75, 0.0],
        [0.0, 0.0, 82.5, 0.0]
    ], dtype=float)
    
    # Derive screw axes
    screw_axes, M = screw_axes_from_dh(ur5_dh_params)
    
    print("="*70)
    print("UR5 VISUALIZATION SUITE")
    print("="*70)
    
    # 1) Home position visualization
    visualize_robot_home(ur5_dh_params, screw_axes, M)
    
    # 2) Interactive FK
    visualize_fk_interactive(ur5_dh_params, screw_axes, M)
    
    # 3) IK path visualization with workspace-aware target
    print("\nGenerating reachable target for IK test...")
    start_config = np.zeros(6)
    
    # Generate a guaranteed reachable target
    target_pos = generate_reachable_test_target(screw_axes, M, start_config, max_distance=250)
    
    # Verify it's reachable
    T_check = np.eye(4)
    T_check[0:3, 3] = target_pos
    if is_pose_reachable(T_check):
        print(f"✓ Target is within workspace (distance: {np.linalg.norm(target_pos):.1f} mm)")
    else:
        print(f"✗ Warning: Target may be marginal")
    
    visualize_ik_path(ur5_dh_params, screw_axes, M, start_config, target_pos)
    
    # 4) Interactive trajectory control with sliders
    print("\nStarting interactive trajectory control...")
    visualize_trajectory_interactive(ur5_dh_params, screw_axes, M, start_config=np.zeros(6))
    
    # 5) Circular motion control with sliders
    print("\nStarting circular motion control...")
    visualize_circular_motion(ur5_dh_params, screw_axes, M, start_config=np.zeros(6))
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)