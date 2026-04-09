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
    elif vis_type == "humanoid":
        return _init_humanoid_visualizer(config)
    elif vis_type == "quadruped":
        return _init_quadruped_visualizer(config)
    raise ValueError(f"Unknown visualizer type: {vis_type!r}")


def _init_cheetah_visualizer(config) -> Visualizer:
    def visualize(trajectory):
        full_states = np.concatenate([trajectory.qpos, trajectory.qvel], axis=-1)
        return _create_cheetah_video(full_states)
    return Visualizer(visualize=visualize)

def _init_humanoid_visualizer(config) -> Visualizer:
    def visualize(trajectory):
        full_states = np.concatenate([trajectory.qpos, trajectory.qvel], axis=-1)
        return _create_humanoid_video(full_states)
    return Visualizer(visualize=visualize)

def _init_quadruped_visualizer(config) -> Visualizer:
    def visualize(trajectory):
        # trajectory is expected to be a dict with 'observations' and 'body_positions'
        # If it's just observations, we'll create a basic visualization
        if isinstance(trajectory, dict):
            if 'body_positions' in trajectory:
                return _create_quadruped_video(trajectory['body_positions'], trajectory.get('observations'))
            else:
                # Fallback: just observations available
                return _create_quadruped_video_from_obs(trajectory.get('observations'))
        else:
            # Assume it's a trajectory object with observations
            return _create_quadruped_video_from_obs(trajectory)
    return Visualizer(visualize=visualize)


def _create_quadruped_video(body_positions=None, observations=None, max_frames=300, save_path=None, fps=25):
    """
    Creates an MP4 video of the quadruped as a stick figure.
    
    Args:
        body_positions: (T, num_bodies, 3) array of body xyz positions from dm_control
        observations: (T, obs_dim) observation array (optional, for info display)
        max_frames: subsample to at most this many frames
        save_path: path for the .mp4 file (temp file if None)
        fps: frames per second (dm_control typically runs at 25 fps)
    Returns:
        Path to the saved MP4.
    """
    import tempfile
    
    if body_positions is None:
        # No position data available
        print("Warning: No body position data in trajectory. Creating placeholder video...")
        return _create_placeholder_video(save_path, fps)
    
    body_positions = np.array(body_positions)
    if observations is not None:
        observations = np.array(observations)
    
    # Subsample if needed
    if len(body_positions) > max_frames:
        idx = np.linspace(0, len(body_positions) - 1, max_frames, dtype=int)
        body_positions = body_positions[idx]
        if observations is not None:
            observations = observations[idx]
    
    # For dog/run, typically:
    # Body 0: torso, Bodies 1-4: front legs (thigh, shin for L/R), 5-8: back legs
    # Extract x-z positions (side view), ignoring y
    T = len(body_positions)
    torso_x = body_positions[:, 0, 0]
    torso_z = body_positions[:, 0, 2]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, dog-relative)')
    ax.set_ylabel('Z (m)')
    ax.grid(True, alpha=0.3)
    ax.axhspan(-0.5, 0, color='#8B4513', alpha=0.3)
    ax.plot([-2, 2], [0, 0], 'k-', linewidth=2)
    
    torso_line, = ax.plot([], [], 'b-', linewidth=6, solid_capstyle='round')
    leg_lines = [ax.plot([], [], 'r-', linewidth=3, solid_capstyle='round')[0] for _ in range(8)]
    joints, = ax.plot([], [], 'ko', markersize=4, zorder=5)
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=8)
    
    all_artists = [torso_line] + leg_lines + [joints, info_text]
    
    def init():
        torso_line.set_data([], [])
        for line in leg_lines:
            line.set_data([], [])
        joints.set_data([], [])
        info_text.set_text('')
        return tuple(all_artists)
    
    def animate(frame):
        # Normalize x positions relative to torso
        x_offset = torso_x[frame]
        
        # Draw torso as a line segment
        if len(body_positions[frame]) > 0:
            torso_x_rel = body_positions[frame, 0, 0] - x_offset
            torso_z = body_positions[frame, 0, 2]
            # Draw torso with some length
            torso_line.set_data([torso_x_rel - 0.1, torso_x_rel + 0.1], 
                               [torso_z, torso_z])
            
            # Draw legs (pairs of segments)
            leg_idx = 0
            joint_xs = []
            joint_zs = []
            
            # Front left and right legs
            for leg_start in [1, 3]:
                if leg_start + 1 < len(body_positions[frame]):
                    x1 = body_positions[frame, leg_start, 0] - x_offset
                    z1 = body_positions[frame, leg_start, 2]
                    x2 = body_positions[frame, leg_start + 1, 0] - x_offset
                    z2 = body_positions[frame, leg_start + 1, 2]
                    leg_lines[leg_idx].set_data([x1, x2], [z1, z2])
                    joint_xs.extend([x1, x2])
                    joint_zs.extend([z1, z2])
                    leg_idx += 1
            
            # Back left and right legs
            for leg_start in [5, 7]:
                if leg_start + 1 < len(body_positions[frame]):
                    x1 = body_positions[frame, leg_start, 0] - x_offset
                    z1 = body_positions[frame, leg_start, 2]
                    x2 = body_positions[frame, leg_start + 1, 0] - x_offset
                    z2 = body_positions[frame, leg_start + 1, 2]
                    leg_lines[leg_idx].set_data([x1, x2], [z1, z2])
                    joint_xs.extend([x1, x2])
                    joint_zs.extend([z1, z2])
                    leg_idx += 1
            
            if joint_xs:
                joints.set_data(joint_xs[:8], joint_zs[:8])
            
            vel_info = ""
            if observations is not None:
                vel_info = f"  vel={observations[frame, 0]:.2f}m/s" if len(observations[frame]) > 0 else ""
            info_text.set_text(f't={frame/fps:.2f}s  x={torso_x[frame]:.1f}m{vel_info}')
        
        return tuple(all_artists)
    
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(body_positions), interval=1000/fps, blit=True)
    
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.mp4')
    
    anim.save(save_path, writer='ffmpeg', fps=fps,
              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)
    return save_path


def _create_quadruped_video_from_obs(observations, max_frames=300, save_path=None, fps=25):
    """
    Creates a placeholder video when only observations are available.
    Shows observation statistics over time.
    """
    import tempfile
    
    observations = np.array(observations)
    if len(observations) > max_frames:
        idx = np.linspace(0, len(observations) - 1, max_frames, dtype=int)
        observations = observations[idx]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
    
    # Plot 1: Forward velocity if it's in the observation
    ax1 = axes[0]
    ax1.set_title('Quadruped Training - Observation Stats')
    if observations.shape[1] > 0:
        ax1.plot(observations[:, 0], label='Obs[0] (likely forward_vel)', color='blue')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overall observation magnitude
    ax2 = axes[1]
    obs_mag = np.linalg.norm(observations, axis=1)
    ax2.plot(obs_mag, label='Observation magnitude', color='green')
    ax2.set_ylabel('||Obs||')
    ax2.set_xlabel('Frame')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.mp4')
    
    fig.savefig(save_path.replace('.mp4', '.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # Create a simple video from the plot
    import tempfile
    from PIL import Image
    img = Image.open(save_path.replace('.mp4', '.png'))
    # For now, return the PNG path or create a simple video
    return save_path.replace('.mp4', '.png')


def _create_placeholder_video(save_path=None, fps=25):
    """Creates a placeholder video when no trajectory data is available."""
    import tempfile
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.text(0.5, 0.5, 'Quadruped Visualization\n(Trajectory data not yet captured)',
            ha='center', va='center', fontsize=16, transform=ax.transAxes)
    ax.axis('off')
    
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.mp4')
    
    fig.savefig(save_path.replace('.mp4', '.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    return save_path.replace('.mp4', '.png')




def _create_humanoid_video(states, max_frames=300, save_path=None, fps=50):
    """
    Creates an MP4 video showing the humanoid as a stick figure.

    Displays a simplified 2D side-view stick figure with:
    - Head/Neck
    - Torso/Chest
    - Arms (shoulders, elbows)
    - Legs (hips, knees, ankles)

    Args:
        states: (T, N) array where N >= 46 (qpos + qvel for humanoid)
                Assumes structure: [qpos (23), qvel (23), ...] or similar
        max_frames: subsample to at most this many frames
        save_path: path for the .mp4 file (temp file if None)
        fps: frames per second
    Returns:
        Path to the saved MP4.
    """
    import tempfile
    
    states = np.array(states)
    
    # Subsample if needed
    if len(states) > max_frames:
        idx = np.linspace(0, len(states) - 1, max_frames, dtype=int)
        states = states[idx]
    
    # Extract root position and velocity from qpos and qvel
    # Humanoid qpos typically: [root_x, root_z, root_y, ...joints...]
    # qvel typically: [root_x_vel, root_y_vel, root_z_vel, root_y_omega, ...joint_vels...]
    T = len(states)
    
    # For visualization, we'll use a simple forward kinematics approximation
    # Root position (assuming first 3 qpos are x, z, y)
    root_x = states[:, 0]
    root_z = states[:, 1] + 1.0  # Offset torso height above ground
    
    # Forward velocity from qvel[0]
    if states.shape[1] > 23:
        forward_vel = states[:, 23]  # qvel[0] after qpos (23 DOF)
    else:
        forward_vel = np.zeros(T)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, humanoid-relative)')
    ax.set_ylabel('Z (m)')
    ax.grid(True, alpha=0.3)
    ax.axhspan(-0.5, 0, color='#8B4513', alpha=0.3)
    ax.plot([-3, 3], [0, 0], 'k-', linewidth=2)
    
    # Humanoid stick figure elements
    head_circle = plt.Circle((0, 0), 0.1, color='blue', fill=True)
    ax.add_patch(head_circle)
    
    torso_line,     = ax.plot([], [], 'b-', linewidth=5, solid_capstyle='round')
    left_arm,       = ax.plot([], [], 'g-', linewidth=3, solid_capstyle='round')
    right_arm,      = ax.plot([], [], 'g-', linewidth=3, solid_capstyle='round')
    left_leg,       = ax.plot([], [], 'r-', linewidth=4, solid_capstyle='round')
    right_leg,      = ax.plot([], [], 'r-', linewidth=4, solid_capstyle='round')
    
    joints,         = ax.plot([], [], 'ko', markersize=5, zorder=5)
    info_text       = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                              verticalalignment='top', fontsize=9)
    
    all_artists = [head_circle, torso_line, left_arm, right_arm, left_leg, right_leg, joints, info_text]
    
    def init():
        torso_line.set_data([], [])
        left_arm.set_data([], [])
        right_arm.set_data([], [])
        left_leg.set_data([], [])
        right_leg.set_data([], [])
        joints.set_data([], [])
        info_text.set_text('')
        return tuple(all_artists)
    
    def animate(frame):
        x_offset = root_x[frame]
        z_base = root_z[frame]
        
        # Normalized positions (relative to torso center)
        torso_x = -x_offset
        torso_z = z_base
        torso_height = 0.4
        
        # Head position
        head_x = torso_x
        head_z = torso_z + torso_height + 0.15
        head_circle.center = (head_x, head_z)
        
        # Torso
        torso_line.set_data([torso_x, torso_x], 
                           [torso_z + torso_height, torso_z])
        
        # Arms (simple representation: point outward from shoulders)
        shoulder_z = torso_z + torso_height - 0.05
        left_elbow_x = torso_x - 0.3
        left_hand_x = torso_x - 0.5
        left_arm.set_data([torso_x, left_elbow_x, left_hand_x],
                         [shoulder_z, shoulder_z - 0.1, shoulder_z - 0.25])
        
        right_elbow_x = torso_x + 0.3
        right_hand_x = torso_x + 0.5
        right_arm.set_data([torso_x, right_elbow_x, right_hand_x],
                          [shoulder_z, shoulder_z - 0.1, shoulder_z - 0.25])
        
        # Legs (simple knee model)
        hip_z = torso_z
        knee_offset = 0.2
        foot_z = 0.05
        
        # Left leg
        left_hip_x = torso_x - 0.1
        left_knee_x = torso_x - 0.15
        left_leg.set_data([left_hip_x, left_knee_x, left_hip_x],
                         [hip_z, hip_z - knee_offset, foot_z])
        
        # Right leg
        right_hip_x = torso_x + 0.1
        right_knee_x = torso_x + 0.15
        right_leg.set_data([right_hip_x, right_knee_x, right_hip_x],
                          [hip_z, hip_z - knee_offset, foot_z])
        
        # Draw joint markers
        jx = np.array([torso_x, head_x, left_elbow_x, left_hand_x, right_elbow_x, 
                       right_hand_x, left_hip_x, left_knee_x, right_hip_x, right_knee_x])
        jz = np.array([torso_z, head_z, shoulder_z - 0.1, shoulder_z - 0.25, 
                       shoulder_z - 0.1, shoulder_z - 0.25, hip_z - knee_offset, foot_z,
                       hip_z - knee_offset, foot_z])
        joints.set_data(jx, jz)
        
        info_text.set_text(f't={frame/fps:.2f}s  x={root_x[frame]:.1f}m  vel={forward_vel[frame]:.2f}m/s')
        
        return tuple(all_artists)
    
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=T, interval=1000/fps, blit=True)
    
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.mp4')
    
    anim.save(save_path, writer='ffmpeg', fps=fps,
              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)
    return save_path


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
