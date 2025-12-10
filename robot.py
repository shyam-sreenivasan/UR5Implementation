"""
Combined Robot Visualization Module
Contains BaseRobotView, FKInteractiveView classes, and IK solvers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
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

def forward_kinematics_poe(screw_axes, M, joint_angles):
    """Forward kinematics using Product of Exponentials."""
    T = np.eye(4, dtype=float)
    for S, th in zip(screw_axes, joint_angles):
        T = T @ matrix_exp6(S, th)
    return T @ M

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
# Jacobian Computation
# -------------------------
def compute_space_jacobian(screw_axes, joint_angles):
    """
    Compute space Jacobian.
    
    Args:
        screw_axes: List of screw axes in space frame
        joint_angles: Current joint configuration
        
    Returns:
        J_s: 6xn space Jacobian matrix
    """
    n = len(screw_axes)
    J_s = np.zeros((6, n), dtype=float)
    T_accum = np.eye(4, dtype=float)
    
    for i in range(n):
        if i == 0:
            J_s[:, 0] = screw_axes[0]
        else:
            # Compute adjoint transformation
            R = T_accum[0:3, 0:3]
            p = T_accum[0:3, 3]
            p_hat = skew(p)
            Ad = np.zeros((6, 6), dtype=float)
            Ad[0:3, 0:3] = R
            Ad[3:6, 0:3] = p_hat @ R
            Ad[3:6, 3:6] = R
            
            J_s[:, i] = Ad @ screw_axes[i]
        
        T_accum = T_accum @ matrix_exp6(screw_axes[i], joint_angles[i])
    
    return J_s

def matrix_log6(T):
    """
    Matrix logarithm for SE(3).
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        6-vector representing [omega*theta, v] in se(3)
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    trace_R = np.trace(R)
    acos_input = np.clip((trace_R - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(acos_input)

    if abs(theta) < 1e-8:
        return np.concatenate((np.zeros(3), p))

    lnR = (theta / (2.0 * np.sin(theta))) * (R - R.T)
    omega_unit = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])
    omega_hat = skew(omega_unit)
    
    # Compute G inverse
    G = np.eye(3) * theta + (1 - np.cos(theta)) * omega_hat + (theta - np.sin(theta)) * (omega_hat @ omega_hat)
    v = np.linalg.solve(G, p)
    
    return np.concatenate((omega_unit * theta, v))

# -------------------------
# IK Solvers
# -------------------------
def newton_raphson_ik(T_desired, screw_axes, M, initial_guess=None, 
                      tol=1e-3, max_iters=100, verbose=False):
    """
    Newton-Raphson Inverse Kinematics solver.
    
    Uses the Jacobian pseudoinverse to iteratively solve for joint angles
    that achieve the desired end-effector pose.
    
    Args:
        T_desired: 4x4 desired transformation matrix
        screw_axes: List of screw axes in space frame
        M: Home configuration matrix
        initial_guess: Initial joint configuration (defaults to zeros)
        tol: Convergence tolerance for twist magnitude
        max_iters: Maximum number of iterations
        verbose: Print iteration details
        
    Returns:
        theta: Joint angles solution
        success: True if converged within tolerance
        iterations: Number of iterations used
        error: Final error magnitude
    """
    if initial_guess is None:
        theta = np.zeros(len(screw_axes))
    else:
        theta = np.array(initial_guess, dtype=float)
    
    for iteration in range(max_iters):
        # Compute current end-effector pose
        T_current = forward_kinematics_poe(screw_axes, M, theta)
        
        # Compute error transformation
        T_error = T_desired @ np.linalg.inv(T_current)
        
        # Convert to twist (exponential coordinates)
        twist_error = matrix_log6(T_error)
        
        # Check convergence
        error_magnitude = np.linalg.norm(twist_error)
        
        if verbose:
            print(f"Iteration {iteration + 1}: Error = {error_magnitude:.6f}")
        
        if error_magnitude < tol:
            if verbose:
                print(f"✓ Converged in {iteration + 1} iterations")
            return theta, True, iteration + 1, error_magnitude
        
        # Compute space Jacobian
        J_s = compute_space_jacobian(screw_axes, theta)
        
        # Compute pseudoinverse using SVD for numerical stability
        J_pinv = np.linalg.pinv(J_s)
        
        # Newton-Raphson update
        delta_theta = J_pinv @ twist_error
        theta = theta + delta_theta
    
    # Did not converge
    T_final = forward_kinematics_poe(screw_axes, M, theta)
    T_error = T_desired @ np.linalg.inv(T_final)
    final_error = np.linalg.norm(matrix_log6(T_error))
    
    if verbose:
        print(f"✗ Did not converge after {max_iters} iterations. Final error: {final_error:.6f}")
    
    return theta, False, max_iters, final_error

def simple_IK_solver(T_desired, screw_axes, M, initial_guess, tol=3.0, max_iters=500):
    """
    Simple IK solver optimized for trajectory planning.
    Uses damped least squares with adaptive damping.
    
    Args:
        T_desired: Target SE(3) transformation
        screw_axes: List of screw axes
        M: Home configuration
        initial_guess: Initial joint configuration
        tol: Position error tolerance in mm
        max_iters: Maximum iterations
        
    Returns:
        theta: Joint angles solution
        success: True if converged within tolerance
        iterations: Number of iterations used
        pos_error: Final position error in mm
    """
    theta = np.array(initial_guess, dtype=float)
    
    for iteration in range(max_iters):
        # Compute current pose
        T_current = forward_kinematics_poe(screw_axes, M, theta)
        
        # Compute error in SE(3)
        T_err = T_desired @ np.linalg.inv(T_current)
        se3_err = matrix_log6(T_err)
        
        # Position error for convergence check
        pos_error = np.linalg.norm(T_desired[0:3, 3] - T_current[0:3, 3])
        
        # Check convergence
        if pos_error < tol:
            return theta, True, iteration, pos_error
        
        # Compute Jacobian
        J_s = compute_space_jacobian(screw_axes, theta)
        
        # Adaptive damping based on error magnitude
        err_norm = np.linalg.norm(se3_err)
        if err_norm > 100:
            damping = 0.001
        elif err_norm > 50:
            damping = 0.005
        elif err_norm > 10:
            damping = 0.01
        else:
            damping = 0.05
        
        # Damped least squares
        JJt = J_s @ J_s.T
        inv_term = np.linalg.inv(JJt + (damping**2) * np.eye(6))
        J_pinv = J_s.T @ inv_term
        
        # Compute update
        delta = J_pinv @ se3_err
        
        # Adaptive step size
        if err_norm > 50:
            step_limit = 0.5
        elif err_norm > 10:
            step_limit = 0.3
        else:
            step_limit = 0.2
        
        step_norm = np.linalg.norm(delta)
        if step_norm > step_limit:
            delta = delta * (step_limit / step_norm)
        
        # Update joint angles
        theta = theta + delta
    
    # Final error
    T_final = forward_kinematics_poe(screw_axes, M, theta)
    final_pos_error = np.linalg.norm(T_desired[0:3, 3] - T_final[0:3, 3])
    
    return theta, final_pos_error < tol, max_iters, final_pos_error

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
        tol=tol, num_attempts=10, verbose=verbose
    )

# -------------------------
# Base Robot View Class
# -------------------------
class BaseRobotView:
    """
    Base class for all robot visualizations.
    Handles the 3D robot scene setup and rendering.
    """
    
    def __init__(self, dh_params, screw_axes, M, figsize=(16, 12), workspace_limit=900):
        """
        Initialize base robot view.
        
        Args:
            dh_params: DH parameters for the robot
            screw_axes: Screw axes for PoE formulation
            M: Home configuration matrix
            figsize: Figure size (width, height)
            workspace_limit: Workspace visualization limit in mm
        """
        self.dh_params = dh_params
        self.screw_axes = screw_axes
        self.M = M
        self.workspace_limit = workspace_limit
        
        # State
        self.current_config = np.zeros(len(dh_params))
        
        # Store view state
        self.view_state = {'elev': 20, 'azim': 45, 'dist': 6}
        
        # Create figure with grid layout
        self.fig = plt.figure(figsize=figsize)
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the grid layout: 75% robot view, 25% control panel."""
        # Create grid: 4 rows, robot takes top 3 rows (75%)
        self.gs = gridspec.GridSpec(4, 12, figure=self.fig, 
                                    height_ratios=[3, 3, 3, 1],
                                    hspace=0.15, wspace=0.3)
        
        # Robot view (top 75%, left 75%)
        self.ax_robot = self.fig.add_subplot(self.gs[0:3, 0:9], projection='3d')
        self._setup_3d_axis()
        
        # Info panel (top 75%, right 25%)
        self.ax_info = self.fig.add_subplot(self.gs[0:3, 9:12])
        self.ax_info.axis('off')
        
        # Control panel (bottom 25%) - will be customized by subclasses
        self.ax_controls = self.fig.add_subplot(self.gs[3, :])
        self.ax_controls.axis('off')
        
    def _setup_3d_axis(self):
        """Configure 3D axis properties."""
        lim = self.workspace_limit
        self.ax_robot.set_xlim(-lim, lim)
        self.ax_robot.set_ylim(-lim, lim)
        self.ax_robot.set_zlim(-lim, lim)
        self.ax_robot.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        self.ax_robot.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        self.ax_robot.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
        self.ax_robot.set_box_aspect([1, 1, 1])
        self.ax_robot.grid(True, alpha=0.3)
        
        # Disable auto-scaling to prevent zoom changes
        self.ax_robot.set_autoscale_on(False)
        
        self.ax_robot.view_init(elev=20, azim=45)
        self.ax_robot.dist = 6  # Initial zoom level
    
    def _save_camera(self):
        """Save current camera state."""
        self.view_state = dict(
            elev=self.ax_robot.elev,
            azim=self.ax_robot.azim,
            dist=self.ax_robot.dist,
            xlim=self.ax_robot.get_xlim(),
            ylim=self.ax_robot.get_ylim(),
            zlim=self.ax_robot.get_zlim(),
        )
    
    def _restore_camera(self):
        """Restore saved camera state."""
        self.ax_robot.view_init(
            elev=self.view_state['elev'],
            azim=self.view_state['azim']
        )
        self.ax_robot.dist = self.view_state['dist']
        self.ax_robot.set_xlim(self.view_state['xlim'])
        self.ax_robot.set_ylim(self.view_state['ylim'])
        self.ax_robot.set_zlim(self.view_state['zlim'])
        
    def draw_frame(self, T, scale=50, alpha=1.0, show_label=False):
        """
        Draw a coordinate frame (RGB = XYZ).
        
        Args:
            T: 4x4 transformation matrix
            scale: Length of axis arrows
            alpha: Transparency
            show_label: Whether to show axis labels in legend
        """
        origin = T[0:3, 3]
        x_axis = origin + scale * T[0:3, 0]
        y_axis = origin + scale * T[0:3, 1]
        z_axis = origin + scale * T[0:3, 2]
        
        self.ax_robot.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
                          'r-', linewidth=2.5, alpha=alpha, label='X-axis' if show_label else '')
        self.ax_robot.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
                          'g-', linewidth=2.5, alpha=alpha, label='Y-axis' if show_label else '')
        self.ax_robot.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
                          'b-', linewidth=2.5, alpha=alpha, label='Z-axis' if show_label else '')
    
    def draw_robot(self, joint_angles=None, show_frames=True, show_end_effector=True):
        """
        Draw the robot at the specified configuration.
        
        Args:
            joint_angles: Joint configuration (uses current_config if None)
            show_frames: Whether to show coordinate frames at each joint
            show_end_effector: Whether to highlight the end-effector
        """
        if joint_angles is None:
            joint_angles = self.current_config
        
        # Save camera state before clearing
        self._save_camera()
        
        # Use cla() instead of clear() to preserve camera state better
        self.ax_robot.cla()
        
        # Re-setup axis properties (labels, grid, etc.)
        self.ax_robot.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        self.ax_robot.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        self.ax_robot.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
        self.ax_robot.set_box_aspect([1, 1, 1])
        self.ax_robot.grid(True, alpha=0.3)
        self.ax_robot.set_autoscale_on(False)
        
        # Draw base frame
        self.draw_frame(np.eye(4), scale=80, show_label=True)
        
        # Compute forward kinematics
        T_final, transforms = forward_kinematics_dh(self.dh_params, joint_angles)
        
        # Extract joint positions
        pts = np.zeros((len(transforms) + 1, 3))
        pts[0] = np.zeros(3)
        for i, T in enumerate(transforms):
            pts[i + 1] = T[0:3, 3]
            if show_frames:
                self.draw_frame(T, scale=50, alpha=0.7)
        
        # Draw robot structure (links)
        self.ax_robot.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                          'ko-', linewidth=5, markersize=12, label='Robot Links')
        
        # Highlight end-effector
        if show_end_effector:
            end_effector = pts[-1]
            self.ax_robot.plot([end_effector[0]], [end_effector[1]], [end_effector[2]], 
                              'ro', markersize=18, label='End-Effector', zorder=10)
        
        # Update title with end-effector position
        ee_pos = pts[-1]
        distance = np.linalg.norm(ee_pos)
        self.ax_robot.set_title(
            f'UR5 Robot | End-Effector: [{ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f}] mm | Reach: {distance:.1f}mm',
            fontsize=13, fontweight='bold'
        )
        
        self.ax_robot.legend(loc='upper right', fontsize=10)
        
        # Update info panel with T_06 matrix
        self._update_info_panel(T_final, joint_angles)
        
        # Restore camera state after drawing
        self._restore_camera()
    
    def _update_info_panel(self, T_final, joint_angles):
        """Update the info panel with transformation matrix and joint angles."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Format transformation matrix
        info_text = "End-Effector Transform (T_06)\n"
        info_text += "=" * 35 + "\n\n"
        
        # Rotation and position
        info_text += "Rotation & Position:\n"
        for i in range(4):
            row_str = "["
            for j in range(4):
                if j == 3 and i < 3:
                    # Position values (in mm)
                    row_str += f"{T_final[i, j]:8.2f}"
                else:
                    # Rotation values and last row
                    row_str += f"{T_final[i, j]:7.4f}"
                if j < 3:
                    row_str += "  "
            row_str += "]"
            info_text += row_str + "\n"
        
        info_text += "\n" + "-" * 35 + "\n\n"
        
        # Position vector
        info_text += "Position (mm):\n"
        info_text += f"  X: {T_final[0, 3]:8.2f}\n"
        info_text += f"  Y: {T_final[1, 3]:8.2f}\n"
        info_text += f"  Z: {T_final[2, 3]:8.2f}\n"
        
        info_text += "\n" + "-" * 35 + "\n\n"
        
        # Joint angles
        info_text += "Joint Angles (deg):\n"
        for i, angle in enumerate(joint_angles):
            info_text += f"  θ{i+1}: {np.rad2deg(angle):7.2f}°\n"
        
        # Display text
        self.ax_info.text(0.05, 0.95, info_text, 
                         transform=self.ax_info.transAxes,
                         fontsize=9,
                         verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
    def set_configuration(self, joint_angles):
        """
        Update robot configuration and redraw.
        
        Args:
            joint_angles: New joint configuration
        """
        self.current_config = np.array(joint_angles, dtype=float)
        self.draw_robot()
        
    def update_display(self):
        """Refresh the display."""
        self.fig.canvas.draw_idle()
        
    def show(self):
        """Display the figure."""
        plt.tight_layout()
        plt.show()


# -------------------------
# FK Interactive View
# -------------------------
class FKInteractiveView(BaseRobotView):
    """
    Interactive Forward Kinematics view with joint angle sliders.
    Inherits robot drawing from BaseRobotView and adds control sliders.
    """
    
    def __init__(self, dh_params, screw_axes, M, figsize=(18, 12), workspace_limit=900):
        super().__init__(dh_params, screw_axes, M, figsize, workspace_limit)
        
        # Storage for slider widgets
        self.sliders = []
        
        # Storage for button widgets and axes (initialized once)
        self.btn_reset = None
        self.btn_random = None
        self.ax_reset = None
        self.ax_random = None
        
        # Setup buttons ONCE during initialization
        self._setup_buttons()
        
        # Setup controls
        self._setup_controls()
        
        # Initial draw
        self.draw_robot()
        
    def _setup_buttons(self):
        """Setup Reset and Random buttons ONCE during initialization."""
        button_y = 0.02
        button_height = 0.035
        button_width = 0.08
        button_spacing = 0.02
        
        total_width = 2 * button_width + button_spacing
        center_x = 0.5 - total_width / 2
        
        # Reset button (left)
        reset_x = center_x
        self.ax_reset = self.fig.add_axes([reset_x, button_y, button_width, button_height])
        self.btn_reset = Button(self.ax_reset, 'Reset', color='lightcoral', hovercolor='coral')
        self.btn_reset.on_clicked(self._on_reset)
        
        # Random button (right)
        random_x = center_x + button_width + button_spacing
        self.ax_random = self.fig.add_axes([random_x, button_y, button_width, button_height])
        self.btn_random = Button(self.ax_random, 'Random', color='lightgreen', hovercolor='green')
        self.btn_random.on_clicked(self._on_random)
        
    def _setup_controls(self):
        """Setup joint angle sliders in the control panel area."""
        gs_controls = gridspec.GridSpecFromSubplotSpec(
            3, 4, subplot_spec=self.gs[3, :],
            hspace=0.8, wspace=0.5
        )
        
        slider_positions = [
            (0, 0, 2), (1, 0, 2), (2, 0, 2),
            (0, 2, 4), (1, 2, 4), (2, 2, 4),
        ]
        
        for i, (row, col_start, col_end) in enumerate(slider_positions):
            ax_slider = self.fig.add_subplot(gs_controls[row, col_start:col_end])
            slider = Slider(
                ax_slider, 
                f'θ{i+1} (deg)', 
                -180, 180, 
                valinit=0, 
                valstep=1,
                color='steelblue'
            )
            slider.on_changed(self._make_slider_callback(i))
            self.sliders.append(slider)
    
    def _make_slider_callback(self, joint_index):
        """Create callback function for a specific slider."""
        def callback(val):
            self.current_config[joint_index] = np.deg2rad(val)
            self.draw_robot()
            self.update_display()
        return callback
    
    def _on_reset(self, event):
        """Reset all joints to home position (zero)."""
        print("🏠 Resetting to home position...")
        self.current_config = np.zeros(len(self.dh_params))
        
        for slider in self.sliders:
            slider.set_val(0)
        
        self.draw_robot()
        self.update_display()
    
    def _on_random(self, event):
        """
        Override random to clear trajectory and start fresh.
        """
        print("🎲 Generating random pose and resetting trajectory...")
        random_config = np.random.uniform(-np.pi, np.pi, len(self.dh_params))
        self.current_config = random_config
        
        # Update all sliders
        for i, slider in enumerate(self.sliders):
            slider.set_val(np.rad2deg(random_config[i]))
        
        # Clear and reinitialize trajectory from new position
        T_current = forward_kinematics_poe(self.screw_axes, self.M, self.current_config)
        self.trajectory_history = [T_current[0:3, 3].copy()]
        
        self.draw_robot()
        self.update_display()


# -------------------------
# Line Trajectory View
# -------------------------
class LineTrajectoryView(FKInteractiveView):
    """
    Cartesian straight-line trajectory control view.
    Extends FKInteractiveView with Cartesian motion buttons.
    
    Allows user to move the end-effector in straight lines along
    X, Y, Z axes using inverse kinematics.
    """
    
    def __init__(self, dh_params, screw_axes, M, figsize=(18, 12), 
                 workspace_limit=900, step_size=10.0):
        """
        Initialize line trajectory view.
        
        Args:
            dh_params: DH parameters for the robot
            screw_axes: Screw axes for PoE formulation
            M: Home configuration matrix
            figsize: Figure size (width, height)
            workspace_limit: Workspace visualization limit in mm
            step_size: Cartesian step size in mm (default: 10mm)
        """
        # Initialize trajectory BEFORE calling parent __init__
        # (parent calls draw_robot which needs trajectory_history to exist)
        self.trajectory_history = []  # List of end-effector positions
        self.step_size = step_size
        self._updating_from_trajectory = False  # Flag to prevent trajectory clearing
        
        # Initialize parent class (FK view with sliders)
        # This will call draw_robot() which now safely checks trajectory_history
        super().__init__(dh_params, screw_axes, M, figsize, workspace_limit)
        
        # Store initial end-effector position
        T_current = forward_kinematics_poe(self.screw_axes, self.M, self.current_config)
        self.trajectory_history.append(T_current[0:3, 3].copy())
        
        # Add Cartesian motion controls
        self._setup_cartesian_controls()
        
        # Redraw to show initial trajectory point
        self.draw_robot()
    
    def _make_slider_callback(self, joint_index):
        """
        Override slider callback to clear trajectory when manually adjusting joints.
        Only clears trajectory when user manually moves sliders, not when
        trajectory buttons programmatically update sliders.
        """
        def callback(val):
            # Update joint angle
            self.current_config[joint_index] = np.deg2rad(val)
            
            # Only clear trajectory if this is a manual user change
            # (not from trajectory button programmatically updating sliders)
            if not self._updating_from_trajectory:
                # Clear trajectory and start fresh from new position
                T_current = forward_kinematics_poe(self.screw_axes, self.M, self.current_config)
                self.trajectory_history = [T_current[0:3, 3].copy()]
            
            # Redraw robot
            self.draw_robot()
            self.update_display()
        return callback
        
    def _setup_cartesian_controls(self):
        """
        Add Cartesian motion buttons (+X, -X, +Y, -Y, +Z, -Z).
        Positioned below the existing Reset/Random buttons.
        """
        button_width = 0.06
        button_height = 0.035
        button_spacing = 0.01
        
        # Y position: below the Reset/Random buttons
        buttons_y = 0.07
        
        # Calculate center position for 6 buttons
        total_width = 6 * button_width + 5 * button_spacing
        start_x = 0.5 - total_width / 2
        
        # Create button axes and buttons
        button_specs = [
            ('+X', 'lightcoral', 'coral', self._on_plus_x),
            ('-X', 'lightcoral', 'coral', self._on_minus_x),
            ('+Y', 'lightgreen', 'green', self._on_plus_y),
            ('-Y', 'lightgreen', 'green', self._on_minus_y),
            ('+Z', 'lightblue', 'blue', self._on_plus_z),
            ('-Z', 'lightblue', 'blue', self._on_minus_z),
        ]
        
        self.cartesian_buttons = []
        
        for i, (label, color, hovercolor, callback) in enumerate(button_specs):
            button_x = start_x + i * (button_width + button_spacing)
            ax_button = self.fig.add_axes([button_x, buttons_y, button_width, button_height])
            button = Button(ax_button, label, color=color, hovercolor=hovercolor)
            button.on_clicked(callback)
            self.cartesian_buttons.append(button)
    
    def _cartesian_step(self, direction):
        """
        Perform a single Cartesian step in the specified direction.
        Automatically reduces step size if IK fails to converge.
        
        Args:
            direction: 3-vector specifying direction (e.g., [1, 0, 0] for +X)
        
        Process:
            1. Get current end-effector pose from FK
            2. Compute target pose (translate position, keep orientation)
            3. Solve IK for target pose
            4. If IK fails, retry with half step size (up to 3 attempts)
            5. Update joint configuration if IK succeeds
            6. Redraw robot and update trajectory visualization
        """
        # Step 1: Get current end-effector pose
        T_current = forward_kinematics_poe(self.screw_axes, self.M, self.current_config)
        current_position = T_current[0:3, 3].copy()
        
        # Try with current step size, then progressively smaller steps
        current_step_size = self.step_size
        max_attempts = 3
        
        for attempt in range(max_attempts):
            # Step 2: Compute target pose
            direction = np.array(direction, dtype=float)
            target_position = current_position + direction * current_step_size
            
            # Create target transformation (same orientation, new position)
            T_target = T_current.copy()
            T_target[0:3, 3] = target_position
            
            # Check if target is within workspace (simple sphere check)
            distance_from_base = np.linalg.norm(target_position)
            if distance_from_base > self.workspace_limit:
                if attempt == 0:
                    print(f"⚠️  Target position outside workspace!")
                    print(f"   Distance: {distance_from_base:.1f} mm (limit: {self.workspace_limit} mm)")
                
                # Try with smaller step
                if attempt < max_attempts - 1:
                    current_step_size /= 2.0
                    print(f"   Retrying with step size: {current_step_size:.1f} mm...")
                    continue
                else:
                    print(f"   Failed after {max_attempts} attempts. No motion applied.")
                    return
            
            # Step 3: Solve IK using current configuration as initial guess
            if attempt == 0:
                print(f"\n→ Moving {direction} by {current_step_size:.1f} mm...")
                print(f"   Current pos: {np.round(current_position, 1)}")
                print(f"   Target pos:  {np.round(target_position, 1)}")
            else:
                print(f"   Attempt {attempt + 1}: Trying step size {current_step_size:.1f} mm...")
            
            # Use simple_IK_solver for trajectory continuity (warm start from current config)
            theta_new, success, iters, pos_error = simple_IK_solver(
                T_target, 
                self.screw_axes, 
                self.M, 
                self.current_config,  # Use current config as initial guess
                tol=3.0,              # 3mm tolerance
                max_iters=500
            )
            
            # Step 4: Check if IK succeeded
            if not success:
                print(f"   IK did not converge (error: {pos_error:.2f} mm)")
                
                # Try with smaller step size
                if attempt < max_attempts - 1:
                    current_step_size /= 2.0
                    print(f"   Reducing step size to {current_step_size:.1f} mm and retrying...")
                    continue
                else:
                    print(f"⚠️  IK failed after {max_attempts} attempts with progressively smaller steps")
                    print(f"   Final step size tried: {current_step_size:.1f} mm")
                    print(f"   Final error: {pos_error:.2f} mm")
                    print(f"   No motion applied. Try different direction or reset pose.")
                    return
            
            # Success! Verify achieved position
            T_achieved = forward_kinematics_poe(self.screw_axes, self.M, theta_new)
            achieved_position = T_achieved[0:3, 3]
            actual_error = np.linalg.norm(target_position - achieved_position)
            
            print(f"✓ IK converged in {iters} iterations")
            print(f"   Achieved pos: {np.round(achieved_position, 1)}")
            print(f"   Error: {actual_error:.2f} mm")
            if attempt > 0:
                print(f"   (Used reduced step size: {current_step_size:.1f} mm)")
            
            # Step 5: Update configuration and trajectory history
            self.current_config = theta_new
            self.trajectory_history.append(achieved_position.copy())
            
            # Update sliders to reflect new joint angles
            # Set flag to prevent slider callback from clearing trajectory
            self._updating_from_trajectory = True
            for i, slider in enumerate(self.sliders):
                slider.set_val(np.rad2deg(theta_new[i]))
            self._updating_from_trajectory = False
            
            # Update step size for next time if we had to reduce it
            if attempt > 0:
                self.step_size = current_step_size
                print(f"   Step size updated to {self.step_size:.1f} mm for subsequent moves")
            
            # Redraw robot with trajectory
            self.draw_robot()
            self.update_display()
            return
        
        # Should not reach here, but just in case
        print(f"⚠️  Motion failed after all attempts")
    
    def draw_robot(self, joint_angles=None, show_frames=True, show_end_effector=True):
        """
        Override draw_robot to include trajectory visualization.
        Calls parent draw_robot, then adds trajectory line.
        """
        # Call parent's draw_robot method
        super().draw_robot(joint_angles, show_frames, show_end_effector)
        
        # Draw trajectory history if we have more than one point
        if len(self.trajectory_history) > 1:
            trajectory = np.array(self.trajectory_history)
            
            # Draw trajectory line (red dashed line)
            self.ax_robot.plot(
                trajectory[:, 0], 
                trajectory[:, 1], 
                trajectory[:, 2],
                'r--', 
                linewidth=2, 
                alpha=0.6, 
                label='Trajectory Path'
            )
            
            # Draw trajectory points (red dots)
            self.ax_robot.plot(
                trajectory[:, 0], 
                trajectory[:, 1], 
                trajectory[:, 2],
                'r.', 
                markersize=8, 
                alpha=0.7
            )
            
            # Highlight start point (green)
            self.ax_robot.plot(
                [trajectory[0, 0]], 
                [trajectory[0, 1]], 
                [trajectory[0, 2]],
                'go', 
                markersize=12, 
                label='Start Point',
                zorder=5
            )
            
            # Update legend to include trajectory
            self.ax_robot.legend(loc='upper right', fontsize=9)
        
        # Update title to include trajectory info
        ee_pos = self.trajectory_history[-1] if self.trajectory_history else [0, 0, 0]
        distance = np.linalg.norm(ee_pos)
        num_steps = len(self.trajectory_history) - 1
        
        self.ax_robot.set_title(
            f'UR5 Line Trajectory | Steps: {num_steps} | '
            f'End-Effector: [{ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f}] mm | '
            f'Reach: {distance:.1f}mm',
            fontsize=12, fontweight='bold'
        )
    
    def _on_plus_x(self, event):
        """Move end-effector +X direction."""
        self._cartesian_step([1, 0, 0])
    
    def _on_minus_x(self, event):
        """Move end-effector -X direction."""
        self._cartesian_step([-1, 0, 0])
    
    def _on_plus_y(self, event):
        """Move end-effector +Y direction."""
        self._cartesian_step([0, 1, 0])
    
    def _on_minus_y(self, event):
        """Move end-effector -Y direction."""
        self._cartesian_step([0, -1, 0])
    
    def _on_plus_z(self, event):
        """Move end-effector +Z direction."""
        self._cartesian_step([0, 0, 1])
    
    def _on_minus_z(self, event):
        """Move end-effector -Z direction."""
        self._cartesian_step([0, 0, -1])
    
    def _on_reset(self, event):
        """
        Override reset to also clear trajectory history.
        """
        print("🏠 Resetting to home position and clearing trajectory...")
        
        # Call parent reset
        super()._on_reset(event)
        
        # Clear and reinitialize trajectory
        T_home = forward_kinematics_poe(self.screw_axes, self.M, self.current_config)
        self.trajectory_history = [T_home[0:3, 3].copy()]
        
        # Redraw to show cleared trajectory
        self.draw_robot()
        self.update_display()


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
    print("ROBOT VISUALIZATION WITH IK SOLVERS")
    print("="*70)
    
    # # Test Newton-Raphson IK
    # print("\n--- Testing Newton-Raphson IK ---")
    # T_test = forward_kinematics_poe(screw_axes, M, [0.5, -0.5, 0.3, 0.0, 0.5, 0.0])
    # theta_sol, success, iters, error = newton_raphson_ik(
    #     T_test, screw_axes, M, verbose=True
    # )
    # print(f"Solution: {np.round(np.rad2deg(theta_sol), 2)} degrees")
    
    # # Test multi-start IK
    # print("\n--- Testing Multi-Start IK ---")
    # theta_sol2, success2, iters2, error2 = inverse_kinematics_space(
    #     T_test, screw_axes, M, verbose=True
    # )
    # print(f"Solution: {np.round(np.rad2deg(theta_sol2), 2)} degrees")
   
    # print("\n--- Demo 1: Interactive FK View ---")
    # print("Use sliders to control joint angles")
    # fk_view = FKInteractiveView(ur5_dh_params, screw_axes, M)
    # fk_view.show()
    
    print("\n--- Demo 2: Line Trajectory View ---")
    print("Use sliders to set initial pose, then use +X/-X/+Y/-Y/+Z/-Z buttons")
    print("to move the end-effector in straight lines")
    line_view = LineTrajectoryView(ur5_dh_params, screw_axes, M, step_size=10.0)
    line_view.show()