"""
Sim-to-sim gap diagnostic.

Takes a TDMPC2 episode (initial state + action sequence), sets mujoco_playground
CheetahRun to the same initial state, applies the same actions at the corrected
timestep (n_substeps=2 → 0.02s per step, matching TDMPC2's 2×0.01s), and
compares the resulting trajectories.

In the ideal world the two trajectories should be identical — any divergence
reveals a fundamental physics/XML mismatch that explains why the learned model
(trained on TDMPC2 data) does not transfer to mujoco_playground.

Usage:
    python scripts/debug_env_gap.py
    python scripts/debug_env_gap.py --episode 3 --steps 100
"""

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import argparse
import json
import pickle

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import mjx_env as _mjx_env

from max.environments import make_cheetah_env, EnvParams
from collect_data_cheetah import create_cheetah_xy_animation


def load_episode(data_path, episode_idx=0, max_steps=200):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"][0]   # (N, 17)
    actions = data["actions"][0] # (N, 6)
    dones = data["dones"]        # (N,)

    done_indices = np.where(dones == 1.0)[0]
    ep_starts = np.concatenate([[0], done_indices[:-1] + 1])

    if episode_idx >= len(ep_starts):
        episode_idx = 0

    start = ep_starts[episode_idx]
    end = done_indices[episode_idx]

    ep_states = states[start:end + 1]
    ep_actions = actions[start:end]

    if max_steps is not None:
        ep_states = ep_states[:max_steps + 1]
        ep_actions = ep_actions[:max_steps]

    print(f"Loaded episode {episode_idx}: {len(ep_states)} states, {len(ep_actions)} actions")
    print(f"  GT vel_x: mean={ep_states[:, 8].mean():.2f}, max={ep_states[:, 8].max():.2f} m/s")
    print(f"  Action  : mean_abs={np.abs(ep_actions).mean():.3f}")
    return ep_states, ep_actions


def set_state_from_obs(mjx_model, base_data, obs_17d):
    """
    Create mjx.Data with qpos/qvel set from a 17D observation.

    17D layout: [rootz(0), rooty(1), joints(2:8), vel_x(8), vel_z(9), vel_y(10), joint_vels(11:17)]
    qpos layout: [rootx(0), rootz(1), rooty(2), joints(3:9)]  — 9D
    qvel layout: [vel_x(0), vel_z(1), vel_y(2), joint_vels(3:9)] — 9D
    """
    qpos = jnp.concatenate([jnp.array([0.0]), jnp.array(obs_17d[:8])])   # rootx=0
    qvel = jnp.array(obs_17d[8:17])
    data = base_data.replace(qpos=qpos, qvel=qvel)
    data = mjx.forward(mjx_model, data)
    return data


def run_comparison(data_path, episode_idx=0, max_steps=200, n_substeps=2):
    print(f"\nSim-to-sim comparison: TDMPC2 vs mujoco_playground (n_substeps={n_substeps})")
    print("=" * 60)

    # Load TDMPC2 episode
    gt_states, gt_actions = load_episode(data_path, episode_idx, max_steps)
    T = len(gt_actions)

    # Load mujoco_playground env (same XML as used in finetuning)
    env = registry.load('CheetahRun')
    mjx_model = env.mjx_model

    # Build step function with corrected n_substeps (bypass make_cheetah_env
    # so we can also control initial state without the 200-step stabilisation)
    @jax.jit
    def step(data, action_1d):
        return _mjx_env.step(mjx_model, data, action_1d, n_substeps)

    # Get a valid base mjx.Data (structure only; we overwrite qpos/qvel below)
    # Use env.reset but then replace state — avoids dependency on make_cheetah_env
    import jax.random as jr
    base_state = env.reset(jr.key(0))
    base_data = base_state.data

    # Set playground initial state = TDMPC2 episode initial state
    init_data = jax.jit(lambda d: set_state_from_obs(mjx_model, d, gt_states[0]))(base_data)

    print(f"\nInitial state set from TDMPC2 episode {episode_idx}")
    print(f"  qpos: {np.array(init_data.qpos)}")
    print(f"  qvel: {np.array(init_data.qvel)}")
    print(f"  GT   qpos[1:9]: {gt_states[0, :8]}")
    print(f"  GT   qvel[0:9]: {gt_states[0, 8:]}")

    # Roll out playground with TDMPC2 actions
    print(f"\nRolling out {T} steps at n_substeps={n_substeps} (dt={n_substeps*0.01:.3f}s per step)...")
    data = init_data
    # Store full qpos+qvel (18D) so we can animate
    pg_full = [np.concatenate([np.array(data.qpos), np.array(data.qvel)])]  # 18D

    for t in range(T):
        action = jnp.array(gt_actions[t])  # (6,)
        data = step(data, action)
        pg_full.append(np.concatenate([np.array(data.qpos), np.array(data.qvel)]))

    pg_full = np.array(pg_full)        # (T+1, 18)
    playground_states = pg_full[:, 1:] # 17D: drop rootx for numerical comparison
    pg_vel = playground_states[:, 8]   # vel_x
    gt_vel = gt_states[:, 8]           # vel_x

    # Build 18D GT states by integrating rootx from vel_x
    dt_step = n_substeps * 0.01
    gt_rootx = np.zeros(len(gt_states))
    for t in range(1, len(gt_states)):
        gt_rootx[t] = gt_rootx[t-1] + gt_states[t-1, 8] * dt_step
    gt_full = np.concatenate([gt_rootx[:, None], gt_states], axis=1)  # (T+1, 18)

    # Report
    print(f"\nResults over {T} steps:")
    print(f"  TDMPC2 GT vel_x:       mean={gt_vel[1:].mean():.2f}, max={gt_vel[1:].max():.2f} m/s")
    print(f"  Playground vel_x:      mean={pg_vel[1:].mean():.2f}, max={pg_vel[1:].max():.2f} m/s")

    # Per-state MAE
    state_labels = ["rootz","rooty","bthigh","bshin","bfoot","fthigh","fshin","ffoot",
                    "vel_x","vel_z","vel_y","vel_bthigh","vel_bshin","vel_bfoot",
                    "vel_fthigh","vel_fshin","vel_ffoot"]
    T_cmp = min(len(gt_states), len(playground_states))
    mae = np.abs(gt_states[:T_cmp] - playground_states[:T_cmp]).mean(axis=0)
    print("\n  Per-dimension MAE (TDMPC2 vs playground):")
    for label, err in zip(state_labels, mae):
        print(f"    {label:15s}: {err:.4f}")

    overall_mae = mae.mean()
    if overall_mae < 0.05:
        print("\n  [OK] Trajectories match well — physics are equivalent.")
    elif overall_mae < 0.5:
        print("\n  [PARTIAL] Moderate divergence — small physics differences accumulate.")
    else:
        print("\n  [MISMATCH] Large divergence — fundamentally different physics.")
        print("     Possible causes: different XML, different solver settings,")
        print("     or mujoco_playground stabilisation changes initial state.")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t_axis = np.arange(T + 1) * (n_substeps * 0.01)

    ax = axes[0]
    ax.plot(t_axis, gt_vel, label="TDMPC2 GT", linewidth=2)
    ax.plot(t_axis, pg_vel, label="mujoco_playground", linewidth=2, linestyle="--")
    ax.set_ylabel("vel_x (m/s)")
    ax.set_title(f"Sim-to-sim comparison (episode {episode_idx}, n_substeps={n_substeps})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, label in enumerate(state_labels[:8]):
        if mae[i] > 0.01:  # only plot dimensions that differ noticeably
            ax.plot(t_axis[:T_cmp], gt_states[:T_cmp, i] - playground_states[:T_cmp, i],
                    label=f"{label} (mae={mae[i]:.3f})", alpha=0.8)
    ax.set_ylabel("GT - Playground (positions)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for i, label in enumerate(state_labels[8:], start=8):
        if mae[i] > 0.05:
            ax.plot(t_axis[:T_cmp], gt_states[:T_cmp, i] - playground_states[:T_cmp, i],
                    label=f"{label} (mae={mae[i]:.3f})", alpha=0.8)
    ax.set_ylabel("GT - Playground (velocities)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "/tmp/debug_sim2sim.png"
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Plot saved: {save_path}")

    # Animation: TDMPC2 GT (primary) vs playground (ghost)
    print("Generating side-by-side animation...")
    min_len = min(len(gt_full), len(pg_full))
    combined = np.stack([pg_full[:min_len], gt_full[:min_len]], axis=0)  # (2, T, 18)
    gif_path = create_cheetah_xy_animation(combined, save_path="/tmp/debug_sim2sim.gif")
    print(f"Animation saved: {gif_path}")
    print("  Solid = mujoco_playground,  ghost = TDMPC2 GT")

    return gt_states, playground_states


def main():
    parser = argparse.ArgumentParser(description="Sim-to-sim gap diagnostic")
    parser.add_argument("--config", type=str, default="cheetah.json")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-substeps", type=int, default=2,
                        help="n_substeps for mujoco_playground (2 = 0.02s matches TDMPC2)")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(cfg_path) as f:
        full_cfg = json.load(f)

    data_path = full_cfg["pretraining"]["data_path"]
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    run_comparison(data_path, args.episode, args.steps, args.n_substeps)


if __name__ == "__main__":
    main()
