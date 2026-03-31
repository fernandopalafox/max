"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, NamedTuple

_TORSO_LENGTH = 1.0
_THIGH_LENGTH = 0.23
_SHIN_LENGTH  = 0.23
_FOOT_LENGTH  = 0.14
_TORSO_Z      = 0.7


class Visualizer(NamedTuple):
    visualize: Callable  # trajectory (mjx.Data) -> str (video path)


def init_visualizer(config) -> Visualizer:
    """Initialize a visualizer based on config["visualizer"]["type"]."""
    vis_type = config["visualizer"]["type"]
    if vis_type == "cheetah":
        return _init_cheetah_visualizer(config)
    raise ValueError(f"Unknown visualizer type: {vis_type!r}")


def _init_cheetah_visualizer(config) -> Visualizer:
    def visualize(trajectory):
        full_states = np.concatenate([trajectory.qpos, trajectory.qvel], axis=-1)
        return _create_cheetah_video(full_states)
    return Visualizer(visualize=visualize)


def _batch_kinematics(states):
    """Vectorized forward kinematics for all frames at once.

    Args:
        states: (T, 9) qpos array
    Returns:
        pts: (T, 10, 2) joint [x, z] positions
            indices: 0=torso_back, 1=torso_front,
                     2=back_hip, 3=back_knee, 4=back_ankle, 5=back_toe,
                     6=front_hip, 7=front_knee, 8=front_ankle, 9=front_toe
        rootx: (T,)
        rootz: (T,)
    """
    rootx  = states[:, 0]
    rootz  = _TORSO_Z + states[:, 1]
    rooty  = states[:, 2]
    cx, sx = np.cos(rooty), np.sin(rooty)

    tbx = rootx - _TORSO_LENGTH / 2 * cx;  tbz = rootz - _TORSO_LENGTH / 2 * sx
    tfx = rootx + _TORSO_LENGTH / 2 * cx;  tfz = rootz + _TORSO_LENGTH / 2 * sx

    a   = rooty - np.pi / 2 + states[:, 3]
    bkx = tbx + _THIGH_LENGTH * np.cos(a); bkz = tbz + _THIGH_LENGTH * np.sin(a)
    a   = a + states[:, 4]
    bax = bkx + _SHIN_LENGTH  * np.cos(a); baz = bkz + _SHIN_LENGTH  * np.sin(a)
    a   = a + states[:, 5]
    btx = bax + _FOOT_LENGTH  * np.cos(a); btz = baz + _FOOT_LENGTH  * np.sin(a)

    a   = rooty - np.pi / 2 + states[:, 6]
    fkx = tfx + _THIGH_LENGTH * np.cos(a); fkz = tfz + _THIGH_LENGTH * np.sin(a)
    a   = a + states[:, 7]
    fax = fkx + _SHIN_LENGTH  * np.cos(a); faz = fkz + _SHIN_LENGTH  * np.sin(a)
    a   = a + states[:, 8]
    ftx = fax + _FOOT_LENGTH  * np.cos(a); ftz = faz + _FOOT_LENGTH  * np.sin(a)

    def p(x, z): return np.stack([x, z], axis=1)

    pts = np.stack([
        p(tbx, tbz), p(tfx, tfz),          # torso back, front
        p(tbx, tbz), p(bkx, bkz), p(bax, baz), p(btx, btz),  # back leg
        p(tfx, tfz), p(fkx, fkz), p(fax, faz), p(ftx, ftz),  # front leg
    ], axis=1)  # (T, 10, 2)
    return pts, rootx, rootz


def _create_cheetah_video(states, max_frames=300, save_path=None, fps=50, ghost_alpha=0.3):
    """
    Creates an MP4 video showing the cheetah as a stick figure.

    Uses vectorized kinematics, blit=True, and ffmpeg for fast encoding.
    The primary cheetah is kept centered (x-coords normalized per frame) so
    axis limits never change — this is what enables blit=True. Ghost cheetahs
    are drawn relative to the same origin.

    Args:
        states: (T, 18) or (N, T, 18) array of [qpos(9), qvel(9)]
        max_frames: subsample to at most this many frames
        save_path: path for the .mp4 file (temp file if None)
        fps: frames per second
        ghost_alpha: transparency for ghost cheetahs when N > 1
    Returns:
        Path to the saved MP4.
    """
    import tempfile

    states = np.array(states)
    if states.ndim == 3:
        primary_states = states[0]
        ghost_states_raw = states[1:] if states.shape[0] > 1 else None
    else:
        primary_states = states
        ghost_states_raw = None

    # Subsample
    if len(primary_states) > max_frames:
        idx = np.linspace(0, len(primary_states) - 1, max_frames, dtype=int)
        primary_states = primary_states[idx]
        if ghost_states_raw is not None:
            ghost_states_raw = ghost_states_raw[:, idx, :]

    # Pre-compute all joint positions at once
    pts, rootx, _ = _batch_kinematics(primary_states[:, :9])

    # Normalize x so the primary cheetah is always centered — enables blit=True
    pts_c = pts.copy()
    pts_c[:, :, 0] -= rootx[:, None]   # (T, 10, 2), x relative to primary torso

    # Pre-compute ghost kinematics (same x reference frame as primary)
    ghost_pts_c = []
    if ghost_states_raw is not None:
        for gs in ghost_states_raw:
            gpts, _, _ = _batch_kinematics(gs[:, :9])
            gpts_c = gpts.copy()
            gpts_c[:, :, 0] -= rootx[:, None]  # same origin as primary
            ghost_pts_c.append(gpts_c)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.3, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, cheetah-relative)')
    ax.set_ylabel('Z (m)')
    ax.grid(True, alpha=0.3)
    ax.axhspan(-0.5, 0, color='#8B4513', alpha=0.3)
    ax.plot([-2.5, 2.5], [0, 0], 'k-', linewidth=2)

    # Ghost artists
    ghost_lines = []
    for _ in ghost_pts_c:
        ghost_lines.append({
            'torso':  ax.plot([], [], color=(0.6, 0.6, 0.9), linewidth=4, alpha=ghost_alpha, solid_capstyle='round')[0],
            'bthigh': ax.plot([], [], color=(0.9, 0.6, 0.6), linewidth=3, alpha=ghost_alpha, solid_capstyle='round')[0],
            'bshin':  ax.plot([], [], color=(0.9, 0.6, 0.6), linewidth=2, alpha=ghost_alpha, solid_capstyle='round')[0],
            'bfoot':  ax.plot([], [], color=(0.9, 0.6, 0.6), linewidth=1.5, alpha=ghost_alpha, solid_capstyle='round')[0],
            'fthigh': ax.plot([], [], color=(0.6, 0.9, 0.6), linewidth=3, alpha=ghost_alpha, solid_capstyle='round')[0],
            'fshin':  ax.plot([], [], color=(0.6, 0.9, 0.6), linewidth=2, alpha=ghost_alpha, solid_capstyle='round')[0],
            'ffoot':  ax.plot([], [], color=(0.6, 0.9, 0.6), linewidth=1.5, alpha=ghost_alpha, solid_capstyle='round')[0],
        })

    torso_line,  = ax.plot([], [], 'b-',  linewidth=6, solid_capstyle='round')
    back_thigh,  = ax.plot([], [], 'r-',  linewidth=4, solid_capstyle='round')
    back_shin,   = ax.plot([], [], 'r-',  linewidth=3, solid_capstyle='round')
    back_foot,   = ax.plot([], [], 'r-',  linewidth=2, solid_capstyle='round')
    front_thigh, = ax.plot([], [], 'g-',  linewidth=4, solid_capstyle='round')
    front_shin,  = ax.plot([], [], 'g-',  linewidth=3, solid_capstyle='round')
    front_foot,  = ax.plot([], [], 'g-',  linewidth=2, solid_capstyle='round')
    joints,      = ax.plot([], [], 'ko',  markersize=4, zorder=5)
    info_text    = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=8)

    primary_artists = [torso_line, back_thigh, back_shin, back_foot,
                       front_thigh, front_shin, front_foot, joints, info_text]
    ghost_artist_list = [line for gs in ghost_lines for line in gs.values()]
    all_artists = ghost_artist_list + primary_artists

    def init():
        for a in all_artists:
            if hasattr(a, 'set_data'):
                a.set_data([], [])
        info_text.set_text('')
        return tuple(all_artists)

    def animate(frame):
        # Draw ghosts first (behind primary)
        for glines, gpts in zip(ghost_lines, ghost_pts_c):
            p = gpts[frame]
            glines['torso'].set_data( [p[0,0], p[1,0]], [p[0,1], p[1,1]])
            glines['bthigh'].set_data([p[2,0], p[3,0]], [p[2,1], p[3,1]])
            glines['bshin'].set_data( [p[3,0], p[4,0]], [p[3,1], p[4,1]])
            glines['bfoot'].set_data( [p[4,0], p[5,0]], [p[4,1], p[5,1]])
            glines['fthigh'].set_data([p[6,0], p[7,0]], [p[6,1], p[7,1]])
            glines['fshin'].set_data( [p[7,0], p[8,0]], [p[7,1], p[8,1]])
            glines['ffoot'].set_data( [p[8,0], p[9,0]], [p[8,1], p[9,1]])

        p = pts_c[frame]  # (10, 2) — already centered
        torso_line.set_data( [p[0,0], p[1,0]], [p[0,1], p[1,1]])
        back_thigh.set_data( [p[2,0], p[3,0]], [p[2,1], p[3,1]])
        back_shin.set_data(  [p[3,0], p[4,0]], [p[3,1], p[4,1]])
        back_foot.set_data(  [p[4,0], p[5,0]], [p[4,1], p[5,1]])
        front_thigh.set_data([p[6,0], p[7,0]], [p[6,1], p[7,1]])
        front_shin.set_data( [p[7,0], p[8,0]], [p[7,1], p[8,1]])
        front_foot.set_data( [p[8,0], p[9,0]], [p[8,1], p[9,1]])
        jx = p[[2,3,4,5,6,7,8,9], 0]
        jz = p[[2,3,4,5,6,7,8,9], 1]
        joints.set_data(jx, jz)
        vel = primary_states[frame, 9]
        info_text.set_text(f't={frame/fps:.2f}s  x={rootx[frame]:.1f}m  vel={vel:.2f}m/s')
        return tuple(all_artists)

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(primary_states), interval=1000/fps, blit=True)

    if save_path is None:
        save_path = tempfile.mktemp(suffix='.mp4')

    anim.save(save_path, writer='ffmpeg', fps=fps,
              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)
    return save_path
