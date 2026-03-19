"""Cheetah visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_cheetah_xy_animation(states, max_frames=100, save_path=None, ghost_alpha=0.3):
    """
    Creates an animated GIF showing the cheetah as a stick-figure mesh.

    The camera follows the cheetah's forward progress. If multiple trajectories
    are provided, the first trajectory is rendered in full color and subsequent
    trajectories are rendered as semi-transparent "ghosts".

    Args:
        states: Array of 18D states [qpos (9), qvel (9)].
                Single episode: shape (T, 18)
                Multi-episode: shape (N, T, 18) where N is number of episodes
        max_frames: Maximum number of frames (subsampled if needed)
        save_path: Optional path to save GIF. If None, uses temp file.
        ghost_alpha: Alpha transparency for ghost cheetahs (0.0-1.0)

    Returns:
        Path to the saved GIF file.
    """
    import tempfile

    states = np.array(states)
    dt = 0.02

    # Detect if multi-episode: shape (N, T, 18) vs single (T, 18)
    if states.ndim == 3:
        num_episodes = states.shape[0]
        primary_states = states[0]
        ghost_states = states[1:] if num_episodes > 1 else None
    else:
        primary_states = states
        ghost_states = None
        num_episodes = 1

    # Subsample if too many frames
    orig_len = len(states[0]) if states.ndim == 3 else len(states)
    if len(primary_states) > max_frames:
        indices = np.linspace(0, len(primary_states) - 1, max_frames, dtype=int)
        primary_states = primary_states[indices]
        if ghost_states is not None:
            ghost_states = ghost_states[:, indices, :]
        effective_dt = dt * (orig_len / max_frames)
    else:
        effective_dt = dt

    torso_length = 1.0
    thigh_length = 0.23
    shin_length = 0.23
    foot_length = 0.14
    TORSO_INITIAL_Z = 0.7

    GHOST_TORSO_COLOR = (0.6, 0.6, 0.9)
    GHOST_BACK_COLOR = (0.9, 0.6, 0.6)
    GHOST_FRONT_COLOR = (0.6, 0.9, 0.6)

    def get_cheetah_points(state):
        rootx = state[0]
        rootz = TORSO_INITIAL_Z + state[1]
        rooty = state[2]

        bthigh_angle = state[3]
        bshin_angle = state[4]
        bfoot_angle = state[5]
        fthigh_angle = state[6]
        fshin_angle = state[7]
        ffoot_angle = state[8]

        torso_front = np.array([
            rootx + torso_length / 2 * np.cos(rooty),
            rootz + torso_length / 2 * np.sin(rooty)
        ])
        torso_back = np.array([
            rootx - torso_length / 2 * np.cos(rooty),
            rootz - torso_length / 2 * np.sin(rooty)
        ])

        back_hip = torso_back.copy()
        angle = rooty - np.pi / 2 + bthigh_angle
        back_knee = back_hip + thigh_length * np.array([np.cos(angle), np.sin(angle)])
        angle += bshin_angle
        back_ankle = back_knee + shin_length * np.array([np.cos(angle), np.sin(angle)])
        angle += bfoot_angle
        back_toe = back_ankle + foot_length * np.array([np.cos(angle), np.sin(angle)])

        front_hip = torso_front.copy()
        angle = rooty - np.pi / 2 + fthigh_angle
        front_knee = front_hip + thigh_length * np.array([np.cos(angle), np.sin(angle)])
        angle += fshin_angle
        front_ankle = front_knee + shin_length * np.array([np.cos(angle), np.sin(angle)])
        angle += ffoot_angle
        front_toe = front_ankle + foot_length * np.array([np.cos(angle), np.sin(angle)])

        return {
            'torso': (torso_back, torso_front),
            'back_leg': [back_hip, back_knee, back_ankle, back_toe],
            'front_leg': [front_hip, front_knee, front_ankle, front_toe],
            'rootx': rootx,
            'rootz': rootz,
        }

    fig, ax = plt.subplots(figsize=(10, 5))

    torso_line, = ax.plot([], [], 'b-', linewidth=6, solid_capstyle='round')
    back_thigh, = ax.plot([], [], 'r-', linewidth=4, solid_capstyle='round')
    back_shin, = ax.plot([], [], 'r-', linewidth=3, solid_capstyle='round')
    back_foot, = ax.plot([], [], 'r-', linewidth=2, solid_capstyle='round')
    front_thigh, = ax.plot([], [], 'g-', linewidth=4, solid_capstyle='round')
    front_shin, = ax.plot([], [], 'g-', linewidth=3, solid_capstyle='round')
    front_foot, = ax.plot([], [], 'g-', linewidth=2, solid_capstyle='round')
    joints, = ax.plot([], [], 'ko', markersize=4, zorder=5)
    ground_line, = ax.plot([], [], 'k-', linewidth=2)
    ax.axhspan(-0.5, 0, color='#8B4513', alpha=0.3)
    trail_line, = ax.plot([], [], 'b--', linewidth=1, alpha=0.3)

    ghost_lines = []
    if ghost_states is not None:
        for _ in range(len(ghost_states)):
            ghost_set = {
                'torso': ax.plot([], [], color=GHOST_TORSO_COLOR, linewidth=4, alpha=ghost_alpha)[0],
                'back_thigh': ax.plot([], [], color=GHOST_BACK_COLOR, linewidth=3, alpha=ghost_alpha)[0],
                'back_shin': ax.plot([], [], color=GHOST_BACK_COLOR, linewidth=2, alpha=ghost_alpha)[0],
                'back_foot': ax.plot([], [], color=GHOST_BACK_COLOR, linewidth=1.5, alpha=ghost_alpha)[0],
                'front_thigh': ax.plot([], [], color=GHOST_FRONT_COLOR, linewidth=3, alpha=ghost_alpha)[0],
                'front_shin': ax.plot([], [], color=GHOST_FRONT_COLOR, linewidth=2, alpha=ghost_alpha)[0],
                'front_foot': ax.plot([], [], color=GHOST_FRONT_COLOR, linewidth=1.5, alpha=ghost_alpha)[0],
            }
            ghost_lines.append(ghost_set)

    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Z Position (m)')
    ax.grid(True, alpha=0.3)

    trail_x = []
    trail_z = []

    def init():
        for line in [torso_line, back_thigh, back_shin, back_foot,
                     front_thigh, front_shin, front_foot, joints,
                     ground_line, trail_line]:
            line.set_data([], [])
        for ghost_set in ghost_lines:
            for line in ghost_set.values():
                line.set_data([], [])
        artists = [torso_line, back_thigh, back_shin, back_foot,
                   front_thigh, front_shin, front_foot, joints,
                   ground_line, trail_line]
        for ghost_set in ghost_lines:
            artists.extend(ghost_set.values())
        return tuple(artists)

    def animate(frame):
        points = get_cheetah_points(primary_states[frame])
        rootx = points['rootx']
        rootz = points['rootz']

        trail_x.append(rootx)
        trail_z.append(rootz)
        trail_line.set_data(trail_x, trail_z)

        torso = points['torso']
        torso_line.set_data([torso[0][0], torso[1][0]], [torso[0][1], torso[1][1]])

        bl = points['back_leg']
        back_thigh.set_data([bl[0][0], bl[1][0]], [bl[0][1], bl[1][1]])
        back_shin.set_data([bl[1][0], bl[2][0]], [bl[1][1], bl[2][1]])
        back_foot.set_data([bl[2][0], bl[3][0]], [bl[2][1], bl[3][1]])

        fl = points['front_leg']
        front_thigh.set_data([fl[0][0], fl[1][0]], [fl[0][1], fl[1][1]])
        front_shin.set_data([fl[1][0], fl[2][0]], [fl[1][1], fl[2][1]])
        front_foot.set_data([fl[2][0], fl[3][0]], [fl[2][1], fl[3][1]])

        all_joints_x = [bl[0][0], bl[1][0], bl[2][0], bl[3][0],
                        fl[0][0], fl[1][0], fl[2][0], fl[3][0]]
        all_joints_z = [bl[0][1], bl[1][1], bl[2][1], bl[3][1],
                        fl[0][1], fl[1][1], fl[2][1], fl[3][1]]
        joints.set_data(all_joints_x, all_joints_z)

        if ghost_states is not None:
            for ghost_idx, ghost_set in enumerate(ghost_lines):
                gp = get_cheetah_points(ghost_states[ghost_idx, frame])
                gt = gp['torso']
                ghost_set['torso'].set_data([gt[0][0], gt[1][0]], [gt[0][1], gt[1][1]])
                gbl = gp['back_leg']
                ghost_set['back_thigh'].set_data([gbl[0][0], gbl[1][0]], [gbl[0][1], gbl[1][1]])
                ghost_set['back_shin'].set_data([gbl[1][0], gbl[2][0]], [gbl[1][1], gbl[2][1]])
                ghost_set['back_foot'].set_data([gbl[2][0], gbl[3][0]], [gbl[2][1], gbl[3][1]])
                gfl = gp['front_leg']
                ghost_set['front_thigh'].set_data([gfl[0][0], gfl[1][0]], [gfl[0][1], gfl[1][1]])
                ghost_set['front_shin'].set_data([gfl[1][0], gfl[2][0]], [gfl[1][1], gfl[2][1]])
                ghost_set['front_foot'].set_data([gfl[2][0], gfl[3][0]], [gfl[2][1], gfl[3][1]])

        view_width = 4.0
        ax.set_xlim(rootx - view_width / 2, rootx + view_width / 2)
        ax.set_ylim(-0.3, 1.5)
        ground_line.set_data([rootx - view_width, rootx + view_width], [0, 0])

        forward_vel = primary_states[frame, 9]
        time_val = frame * effective_dt
        ax.set_title(f'Cheetah Animation | t={time_val:.2f}s | vel={forward_vel:.2f} m/s')

        artists = [torso_line, back_thigh, back_shin, back_foot,
                   front_thigh, front_shin, front_foot, joints,
                   ground_line, trail_line]
        for ghost_set in ghost_lines:
            artists.extend(ghost_set.values())
        return tuple(artists)

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(primary_states), interval=effective_dt * 300, blit=False
    )

    if save_path is None:
        save_path = tempfile.mktemp(suffix='.gif')

    anim.save(save_path, writer='pillow')
    plt.close(fig)

    return save_path
