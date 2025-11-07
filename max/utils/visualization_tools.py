import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle


def animate_drone(
    traj_state_true,
    traj_control=None,
    dt=0.05,
    save_path="drone_animation.gif",
    figsize=(12, 10),
    fps=20,
    dpi=100,
    track_length=50,
    show_trajectory=True,
    reference_state=None,
    reference_traj=None,
):
    """
    Create an animation of the 2D drone trajectory.

    Parameters:
    -----------
    traj_state_true : jnp.ndarray
        Trajectory of true states, shape (T, 6) where states are [x, y, yaw, vx, vy, omega]
    traj_control : jnp.ndarray, optional
        Control inputs, shape (T-1, 2) where controls are [thrust_left, thrust_right]
    dt : float
        Time step between frames
    save_path : str
        Path to save the GIF
    figsize : tuple
        Figure size (width, height)
    fps : int
        Frames per second for the GIF
    dpi : int
        Dots per inch for the GIF
    track_length : int
        Number of past positions to show as trajectory
    show_trajectory : bool
        Whether to show the past trajectory
    reference_state : jnp.ndarray, optional
        Reference state for a single goal position.
    reference_traj : jnp.ndarray, optional
        Full reference trajectory, shape (T, 6). Takes precedence over reference_state.
    """
    # Convert to numpy for matplotlib
    states = np.array(traj_state_true)
    if traj_control is not None:
        controls = np.array(traj_control)
    if reference_traj is not None:
        reference_traj = np.array(reference_traj)

    # Extract state components
    x = states[:, 0]
    y = states[:, 1]
    yaw = states[:, 2]
    vx = states[:, 3]
    vy = states[:, 4]

    # 2D Drone parameters
    DRONE_SIZE = 0.3  # m
    ARM_LENGTH = DRONE_SIZE / 2
    PROP_RADIUS = 0.08

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],
        hspace=0.3,
        left=0.08,
        right=0.92,
        top=0.95,
        bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    # Calculate plot limits, including reference trajectory
    margin = 1.0
    all_x = x.copy()
    all_y = y.copy()
    if reference_traj is not None:
        all_x = np.concatenate([all_x, reference_traj[:, 0]])
        all_y = np.concatenate([all_y, reference_traj[:, 1]])

    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    # Calculate ranges for equal aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    # Center the plot
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Set symmetric limits
    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    y_min = y_center - max_range / 2
    y_max = y_center + max_range / 2

    # Set up main plot
    ax_main.set_aspect("equal")
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.grid(True, alpha=0.3, zorder=0)
    ax_main.set_xlabel("X Position (m)")
    ax_main.set_ylabel("Y Position (m)")
    ax_main.set_title("2D Drone Trajectory")

    # Plot reference trajectory and goal
    if reference_traj is not None:
        ax_main.plot(
            reference_traj[:, 0],
            reference_traj[:, 1],
            "g--",
            linewidth=2,
            label="Reference Path",
            zorder=1,
        )
        ax_main.plot(
            reference_traj[-1, 0],
            reference_traj[-1, 1],
            "g*",
            markersize=20,
            label="Goal",
            zorder=5,
        )
    elif reference_state is not None:
        ax_main.plot(
            reference_state[0],
            reference_state[1],
            "g*",
            markersize=20,
            label="Goal",
            zorder=5,
        )

    # Initialize trajectory line
    if show_trajectory:
        (trajectory_line,) = ax_main.plot(
            [], [], "b-", alpha=0.6, linewidth=2, zorder=4
        )
        (trajectory_points,) = ax_main.plot(
            [], [], "bo", markersize=3, alpha=0.4, zorder=4
        )

    # Initialize 2D drone components
    drone_body = Rectangle(
        (0, 0),
        ARM_LENGTH * 2,
        0.03,
        facecolor="black",
        edgecolor="black",
        zorder=6,
    )
    ax_main.add_patch(drone_body)

    propellers = []
    prop_positions = [(-ARM_LENGTH, 0), (ARM_LENGTH, 0)]
    prop_colors = ["red", "blue"]

    for i, pos in enumerate(prop_positions):
        prop = Circle(
            (0, 0),
            PROP_RADIUS,
            facecolor=prop_colors[i],
            edgecolor="darkgray",
            alpha=0.8,
            zorder=7,
        )
        ax_main.add_patch(prop)
        propellers.append(prop)

    center_dot = Circle(
        (0, 0), 0.03, facecolor="yellow", edgecolor="orange", zorder=8
    )
    ax_main.add_patch(center_dot)
    velocity_quiver = ax_main.quiver(
        [0], [0], [0], [0], scale=5, color="blue", alpha=0.7, zorder=5
    )

    thrust_visuals = []
    if traj_control is not None:
        thrust_left_line = Rectangle(
            (0, 0),
            0.04,
            0,
            facecolor="red",
            edgecolor="darkred",
            alpha=0.8,
            zorder=6,
        )
        thrust_right_line = Rectangle(
            (0, 0),
            0.04,
            0,
            facecolor="blue",
            edgecolor="darkblue",
            alpha=0.8,
            zorder=6,
        )
        ax_main.add_patch(thrust_left_line)
        ax_main.add_patch(thrust_right_line)
        thrust_visuals = [thrust_left_line, thrust_right_line]

    ax_info.set_xlim(0, len(states))
    if traj_control is not None:
        ctrl_min = controls.min() - 0.1
        ctrl_max = controls.max() + 0.1
        ax_info.set_ylim(ctrl_min, ctrl_max)
    else:
        ax_info.set_ylim(-1, 1)
    ax_info.set_xlabel("Time Step")
    ax_info.set_ylabel("Thrust (N)")
    ax_info.grid(True, alpha=0.3)

    control_lines = []
    if traj_control is not None:
        (control_line1,) = ax_info.plot(
            [], [], "r-", label="Thrust Left", linewidth=2
        )
        (control_line2,) = ax_info.plot(
            [], [], "b-", label="Thrust Right", linewidth=2
        )
        control_lines = [control_line1, control_line2]
        ax_info.legend(loc="upper right")

    time_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        zorder=10,
    )
    ax_main.legend(loc="upper right")

    def init():
        if show_trajectory:
            trajectory_line.set_data([], [])
            trajectory_points.set_data([], [])
        for line in control_lines:
            line.set_data([], [])
        time_text.set_text("")
        return []

    def animate(frame):
        curr_x, curr_y, curr_yaw = x[frame], y[frame], yaw[frame]
        cos_yaw, sin_yaw = np.cos(curr_yaw), np.sin(curr_yaw)

        # Update drone body
        transform = (
            plt.matplotlib.transforms.Affine2D()
            .rotate_around(0, 0, curr_yaw)
            .translate(curr_x, curr_y)
            + ax_main.transData
        )
        drone_body.set_transform(transform)
        drone_body.set_xy((-ARM_LENGTH, -0.015))

        # Update propellers
        for i, (dx, dy) in enumerate(prop_positions):
            rot_x = dx * cos_yaw - dy * sin_yaw
            rot_y = dx * sin_yaw + dy * cos_yaw
            propellers[i].center = (curr_x + rot_x, curr_y + rot_y)

        center_dot.center = (curr_x, curr_y)

        if traj_control is not None and frame < len(controls):
            thrust_scale = 0.3
            left_thrust, right_thrust = (
                controls[frame, 0] * thrust_scale,
                controls[frame, 1] * thrust_scale,
            )

            # Left thrust
            left_x, left_y = (
                curr_x - ARM_LENGTH * cos_yaw,
                curr_y - ARM_LENGTH * sin_yaw,
            )
            thrust_visuals[0].set_xy(
                (left_x - 0.02 * cos_yaw, left_y - 0.02 * sin_yaw)
            )
            thrust_visuals[0].set_height(max(0, left_thrust))
            thrust_visuals[0].set_angle(np.degrees(curr_yaw - np.pi / 2))

            # Right thrust
            right_x, right_y = (
                curr_x + ARM_LENGTH * cos_yaw,
                curr_y + ARM_LENGTH * sin_yaw,
            )
            thrust_visuals[1].set_xy(
                (right_x - 0.02 * cos_yaw, right_y - 0.02 * sin_yaw)
            )
            thrust_visuals[1].set_height(max(0, right_thrust))
            thrust_visuals[1].set_angle(np.degrees(curr_yaw - np.pi / 2))

        v_mag = np.sqrt(vx[frame] ** 2 + vy[frame] ** 2)
        if v_mag > 0.01:
            velocity_quiver.set_offsets([[curr_x, curr_y]])
            velocity_quiver.set_UVC([vx[frame]], [vy[frame]])

        if show_trajectory:
            start_idx = max(0, frame - track_length)
            trajectory_line.set_data(
                x[start_idx : frame + 1], y[start_idx : frame + 1]
            )
            trajectory_points.set_data(
                x[start_idx : frame + 1], y[start_idx : frame + 1]
            )

        if traj_control is not None and frame < len(controls):
            control_lines[0].set_data(
                range(frame + 1), controls[: frame + 1, 0]
            )
            control_lines[1].set_data(
                range(frame + 1), controls[: frame + 1, 1]
            )

        time_text.set_text(
            f"Time: {frame*dt:.2f}s\n"
            f"Pos: ({curr_x:.2f}, {curr_y:.2f})\n"
            f"Vel: {np.sqrt(vx[frame]**2 + vy[frame]**2):.2f} m/s"
        )

        artists = (
            [drone_body, center_dot, velocity_quiver, time_text]
            + propellers
            + thrust_visuals
        )
        if show_trajectory:
            artists.extend([trajectory_line, trajectory_points])
        artists.extend(control_lines)
        return artists

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(states),
        interval=1000 / fps,
        blit=True,
    )
    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {save_path}")
    return ani
