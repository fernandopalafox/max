"""Inspect and clean a MAX-format pkl file.

Diagnoses NaN/inf/extreme issues in states and actions, drops the individual
corrupt rows, and saves a cleaned pkl.

Usage:
    # Dry-run diagnostics only
    python scripts/inspect_and_clean_pkl.py \
        --input data/.../foo.pkl --action-threshold 1.0 --dry-run

    # Full clean + save
    python scripts/inspect_and_clean_pkl.py \
        --input data/.../foo.pkl \
        --output data/.../foo_clean.pkl --action-threshold 1.0

    # With diagnostic plots
    python scripts/inspect_and_clean_pkl.py \
        --input data/.../foo.pkl --action-threshold 1.0 --dry-run --plot
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp


def diagnose(name, arr, threshold=None):
    """Print per-dimension NaN/inf counts and value ranges."""
    print(f"\n--- {name}: shape {arr.shape} ---")
    nan_total = np.isnan(arr).sum()
    inf_total = np.isinf(arr).sum()
    print(f"  NaN: {nan_total}  |  Inf: {inf_total}")
    out_col = f"out(>{threshold:.0f})" if threshold is not None else ""
    print(
        f"  {'dim':>4}  {'nan':>8}  {'inf':>8}  {out_col:>10}  "
        f"{'min':>12}  {'max':>12}  {'mean':>12}"
    )
    for d in range(arr.shape[1]):
        col = arr[:, d]
        nan_r = int(np.isnan(col).sum())
        inf_r = int(np.isinf(col).sum())
        with np.errstate(invalid="ignore", over="ignore"):
            mn = float(np.nanmin(col))
            mx = float(np.nanmax(col))
            mu = float(np.nanmean(col))
        if threshold is not None:
            extreme = int(np.sum(np.abs(col) > threshold))
            tag = " *** CORRUPT ***" if extreme > len(col) * 0.5 else ""
            print(
                f"  {d:>4}  {nan_r:>8}  {inf_r:>8}  {extreme:>10}  "
                f"{mn:>12.4g}  {mx:>12.4g}  {mu:>12.4g}{tag}"
            )
        else:
            print(
                f"  {d:>4}  {nan_r:>8}  {inf_r:>8}  {'':>10}  "
                f"{mn:>12.4g}  {mx:>12.4g}  {mu:>12.4g}"
            )


def find_bad_rows(states, actions, action_threshold):
    """Boolean mask: NaN/inf in states/actions, or action exceeds threshold."""
    bad = np.any(np.isnan(states) | np.isinf(states), axis=1) | np.any(
        np.isnan(actions) | np.isinf(actions), axis=1
    )
    if action_threshold is not None:
        with np.errstate(invalid="ignore", over="ignore"):
            bad |= np.any(np.abs(actions) > action_threshold, axis=1)
    return bad


def _parse_episodes(dones):
    """Return ep_starts, ep_ends (inclusive), ep_lens from dones."""
    done_at = np.where(dones == 1.0)[0]
    ep_starts = np.concatenate([[0], done_at[:-1] + 1])
    ep_ends = done_at
    ep_lens = ep_ends - ep_starts + 1
    return ep_starts, ep_ends, ep_lens


def _compute_first_bad(bad_rows, ep_starts, ep_ends, ep_lens):
    """first_bad[i] == ep_lens[i] means episode i is clean."""
    n_eps = len(ep_starts)
    first_bad = np.empty(n_eps, dtype=int)
    for i in range(n_eps):
        ep_bad = bad_rows[ep_starts[i]: ep_ends[i] + 1]
        first_bad[i] = int(ep_bad.argmax()) if ep_bad.any() else ep_lens[i]
    return first_bad


def _print_arr_stats(label, arr, action_threshold=None):
    """Print per-dim min/max/mean for a slice of states or actions."""
    print(f"\n  {label} (n={len(arr):,}):")
    for d in range(arr.shape[1]):
        col = arr[:, d]
        with np.errstate(invalid="ignore", over="ignore"):
            mn = float(np.nanmin(col))
            mx = float(np.nanmax(col))
            mu = float(np.nanmean(col))
        extra = ""
        if action_threshold is not None:
            n_bad = int(np.sum(np.abs(col) > action_threshold))
            extra = f"  out(>{action_threshold:.0f})={n_bad}"
        print(f"    dim {d}: min={mn:.4g}  max={mx:.4g}  mean={mu:.4g}{extra}")


def diagnose_step0_corruption(
    states, actions, bad_rows, ep_starts, ep_ends, ep_lens, first_bad,
    action_threshold=None,
):
    """For episodes corrupt at step 0, show stats for the bad step and step 1."""
    step0_idx = np.where(first_bad == 0)[0]
    n_step0 = len(step0_idx)
    if n_step0 == 0:
        return

    print(
        f"\n[STEP-0 DIAG] {n_step0:,} episodes are corrupt at step 0."
        f" Checking remainder (steps 1..end)..."
    )

    clean_rem = sum(
        1 for i in step0_idx
        if ep_lens[i] > 1
        and not bad_rows[ep_starts[i] + 1: ep_ends[i] + 1].any()
    )
    still_bad = n_step0 - clean_rem
    print(
        f"  Clean remainder (steps 1..end all fine) : "
        f"{clean_rem:,}  ({100.0 * clean_rem / n_step0:.1f}%)"
    )
    print(
        f"  Still corrupt in remainder              : "
        f"{still_bad:,}  ({100.0 * still_bad / n_step0:.1f}%)"
    )

    # Indices of the corrupt step-0 rows and their step-1 successors
    step0_rows = ep_starts[step0_idx]
    step1_rows = np.array(
        [ep_starts[i] + 1 for i in step0_idx if ep_lens[i] > 1]
    )

    _print_arr_stats(
        "Actions at step 0 (the corrupt rows)",
        actions[step0_rows],
        action_threshold,
    )
    _print_arr_stats("States  at step 0 (the corrupt rows)", states[step0_rows])

    if len(step1_rows):
        _print_arr_stats(
            "Actions at step 1 (first kept step after drop)",
            actions[step1_rows],
            action_threshold,
        )
        _print_arr_stats(
            "States  at step 1 (first kept step after drop)",
            states[step1_rows],
        )
    print()


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def plot_diagnostics(dones, bad_rows, save_prefix):
    """Histogram of first-corrupt-step per episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep_starts, ep_ends, ep_lens = _parse_episodes(dones)
    n_eps = len(ep_starts)
    first_bad = _compute_first_bad(bad_rows, ep_starts, ep_ends, ep_lens)

    has_bad = first_bad < ep_lens
    n_corrupt_eps = int(has_bad.sum())
    corrupt_steps = first_bad[has_bad]

    print("\n[DIAG] === Corruption Summary ===")
    print(f"[DIAG] Total episodes   : {n_eps:,}")
    print(
        f"[DIAG] Corrupt episodes : {n_corrupt_eps:,}"
        f"  ({100.0 * n_corrupt_eps / n_eps:.2f}%)"
    )
    print(
        f"[DIAG] Clean episodes   : {n_eps - n_corrupt_eps:,}"
        f"  ({100.0 * (n_eps - n_corrupt_eps) / n_eps:.2f}%)"
    )

    if n_corrupt_eps:
        q10, q25, q50, q75, q90 = (
            np.percentile(corrupt_steps, [10, 25, 50, 75, 90]).astype(int)
        )
        print(
            f"[DIAG] First-corrupt-step p10/p25/p50/p75/p90: "
            f"{q10} / {q25} / {q50} / {q75} / {q90}"
        )
        n_at_0 = int((corrupt_steps == 0).sum())
        n_last = int((corrupt_steps >= ep_lens.max() - 1).sum())
        print(
            f"[DIAG] Corruption at step 0     : {n_at_0:,}"
            f"  ({100.0 * n_at_0 / n_corrupt_eps:.1f}% of bad eps)"
        )
        print(
            f"[DIAG] Corruption at last step  : {n_last:,}"
            f"  ({100.0 * n_last / n_corrupt_eps:.1f}% of bad eps)"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Corruption Diagnostics", fontsize=14, fontweight="bold")

    if n_corrupt_eps:
        ax.hist(corrupt_steps, bins=50, color="crimson", alpha=0.8,
                edgecolor="black")
        ax.axvline(
            float(np.median(corrupt_steps)), color="gold", linestyle="--",
            linewidth=1.5, label=f"median = {int(np.median(corrupt_steps))}"
        )
        ax.legend()
    ax.set_title(
        "Corrupted-Step Distribution\n(first corrupt step per bad episode)"
    )
    ax.set_xlabel("Step index within episode")
    ax.set_ylabel(f"Bad episodes  (total = {n_corrupt_eps:,})")

    plt.tight_layout()
    out_path = f"{save_prefix}_diagnostics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[DIAG] PNG saved → {out_path}")

# ============================================================
# END DIAGNOSTIC PLOTS
# ============================================================


def generate_zero_action_data(num_transitions, dim_state, dataset_states, rollout_len=20):
    """Generate transitions with zero actions using MJX cheetah env.

    dim_state: 17 for TDMPC2 format (qpos[1:] + qvel), 18 for full (qpos + qvel)
    dataset_states: existing states to sample init conditions from (50% of rollouts)
    """
    from max.environments import make_cheetah_env, EnvParams
    from mujoco import mjx

    env_params = EnvParams(cheetah_n_substeps=2, max_episode_steps=5000)
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)

    # Get template data and mjx_model for state injection
    from mujoco_playground import registry
    env = registry.load('CheetahRun')
    mjx_model = env.mjx_model

    states_list, actions_list, rewards_list, dones_list = [], [], [], []
    rng = np.random.default_rng(42)
    key = jax.random.PRNGKey(0)
    collected = 0
    rollout_idx = 0

    while collected < num_transitions:
        key, reset_key = jax.random.split(key)
        data = reset_fn(reset_key)  # Always reset first to get valid template

        # 50% chance: init from random dataset state instead of reset
        if rollout_idx % 2 == 1:
            idx = rng.integers(len(dataset_states))
            src_state = dataset_states[idx]
            if dim_state == 17:
                qpos = jnp.concatenate([jnp.zeros(1), jnp.array(src_state[:8])])
                qvel = jnp.array(src_state[8:])
            else:
                qpos = jnp.array(src_state[:9])
                qvel = jnp.array(src_state[9:])
            data = data.replace(qpos=qpos, qvel=qvel)
            data = mjx.forward(mjx_model, data)

        zero_action = jnp.zeros((1, 6))
        rollout_idx += 1

        for step in range(rollout_len):
            if dim_state == 17:
                state = jnp.concatenate([data.qpos[1:], data.qvel])
            else:
                state = jnp.concatenate([data.qpos, data.qvel])

            data, obs, rewards, terminated, truncated, info = step_fn(data, step, zero_action)

            states_list.append(np.array(state))
            actions_list.append(np.zeros(6))
            rewards_list.append(float(rewards[0]))

            done = (step == rollout_len - 1)
            dones_list.append(1.0 if done else 0.0)
            collected += 1

            if collected >= num_transitions:
                break

    return (np.stack(states_list), np.stack(actions_list),
            np.array(rewards_list), np.array(dones_list))


def main():
    parser = argparse.ArgumentParser(description="Inspect and clean MAX pkl file")
    parser.add_argument("--input", required=True, help="Input pkl file")
    parser.add_argument("--output", default=None,
                        help="Output pkl path (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diagnostics only, do not save")
    parser.add_argument("--action-threshold", type=float, default=None,
                        help="Flag/drop rows where any action dim exceeds this. "
                             "E.g. 1.0 for cheetah (actions in [-1, 1]).")
    parser.add_argument("--state-threshold", type=float, default=None,
                        help="Threshold for state diagnostics display only.")
    parser.add_argument("--plot", action="store_true",
                        help="Save a diagnostic PNG alongside the input file.")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of clean episodes to keep (0-1). "
                             "Randomly samples episodes after cleaning.")
    parser.add_argument("--augment", action="store_true",
                        help="Augment with zero-action rollouts")
    parser.add_argument("--augment-ratio", type=float, default=0.2,
                        help="Fraction of transitions to add (e.g., 0.2 = add 20%%)")
    parser.add_argument("--augment-rollout-len", type=int, default=100,
                        help="Steps per zero-action rollout")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    states  = data["states"][0]
    actions = data["actions"][0]
    rewards = data["rewards"][0]
    dones   = data["dones"]
    N = states.shape[0]

    print(f"Loaded: {args.input}")
    print(f"  N={N:,}  dim_state={states.shape[1]}  dim_action={actions.shape[1]}")

    diagnose("states",  states,  args.state_threshold)
    diagnose("actions", actions, args.action_threshold)

    bad_rows = find_bad_rows(states, actions, args.action_threshold)
    n_bad = int(bad_rows.sum())
    reason = "NaN/inf"
    if args.action_threshold:
        reason += f" or |action|>{args.action_threshold}"
    print(f"\nFound {n_bad:,} corrupt transitions ({100.0 * n_bad / N:.4f}%) [{reason}]")

    # --- Episode-level step-0 diagnostic ---
    done_at = np.where(dones == 1.0)[0]
    if len(done_at) == 0:
        print("WARNING: no done flags found; skipping episode diagnostics.")
    else:
        ep_starts, ep_ends, ep_lens = _parse_episodes(dones)
        first_bad = _compute_first_bad(bad_rows, ep_starts, ep_ends, ep_lens)
        diagnose_step0_corruption(
            states, actions, bad_rows,
            ep_starts, ep_ends, ep_lens, first_bad,
            action_threshold=args.action_threshold,
        )

    # --- Diagnostic plots ---
    if args.plot:
        if len(done_at) == 0:
            print("WARNING: --plot skipped; no done flags found.")
        else:
            save_prefix = str(Path(args.input).with_suffix(""))
            plot_diagnostics(dones, bad_rows, save_prefix)

    # --- Drop bad rows ---
    if n_bad > 0:
        keep = ~bad_rows
        print(f"Dropping {n_bad:,} individual corrupt rows.")
        states  = states[keep]
        actions = actions[keep]
        rewards = rewards[keep]
        dones   = dones[keep]

    # --- Subsample episodes if requested ---
    if args.fraction < 1.0:
        ep_starts, ep_ends, _ = _parse_episodes(dones)
        n_eps = len(ep_starts)
        n_keep = max(1, int(n_eps * args.fraction))
        rng = np.random.default_rng(42)
        keep_eps = np.sort(rng.choice(n_eps, size=n_keep, replace=False))
        # Build mask for rows belonging to kept episodes
        keep_mask = np.zeros(len(states), dtype=bool)
        for i in keep_eps:
            keep_mask[ep_starts[i]:ep_ends[i] + 1] = True
        states = states[keep_mask]
        actions = actions[keep_mask]
        rewards = rewards[keep_mask]
        dones = dones[keep_mask]
        print(f"\nSubsampled {n_keep:,}/{n_eps:,} episodes ({args.fraction*100:.1f}%)")
        print(f"  Remaining transitions: {len(states):,}")

    # --- Augment with zero-action rollouts ---
    if args.augment:
        dim_state = states.shape[1]
        n_to_add = int(len(states) * args.augment_ratio)
        print(f"\n=== Augmenting with {n_to_add:,} zero-action transitions ({dim_state}D states) ===")
        print(f"  (50% from reset, 50% from random dataset states)")
        aug_states, aug_actions, aug_rewards, aug_dones = generate_zero_action_data(
            n_to_add, dim_state, states, args.augment_rollout_len)
        states = np.concatenate([states, aug_states])
        actions = np.concatenate([actions, aug_actions])
        rewards = np.concatenate([rewards, aug_rewards])
        dones = np.concatenate([dones, aug_dones])
        print(f"  Total transitions after augmentation: {len(states):,}")

    # --- Post-clean diagnostics ---
    print("\n=== After cleaning ===")
    diagnose("states",  states,  args.state_threshold)
    diagnose("actions", actions, args.action_threshold)

    # --- Normalization stats ---
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.diff(states, axis=0)
    def _fmt(vals):
        return "[" + ", ".join(f"{int(v)}" for v in vals) + "]"

    print("\nNormalization stats (copy into config normalization_params):")
    for name, arr in [("state", states), ("action", actions), ("delta", d)]:
        mn = np.floor(arr.min(axis=0))
        mx = np.ceil(arr.max(axis=0))
        print(f'  "{name}": {{')
        print(f'    "min": {_fmt(mn)},')
        print(f'    "max": {_fmt(mx)}')
        print(f'  }}')

    if args.dry_run:
        print("\n--dry-run: not saving.")
        return

    output_path = args.output or args.input
    cleaned = {
        "states":  states[np.newaxis].astype(np.float32),
        "actions": actions[np.newaxis].astype(np.float32),
        "rewards": rewards[np.newaxis].astype(np.float32),
        "dones":   dones.astype(np.float32),
        "num_transitions": states.shape[0],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cleaned, f)
    print(f"\nSaved to {output_path}")
    print(f"  states:  {cleaned['states'].shape}")
    print(f"  actions: {cleaned['actions'].shape}")
    print(f"  Total transitions: {cleaned['num_transitions']:,}")


if __name__ == "__main__":
    main()
