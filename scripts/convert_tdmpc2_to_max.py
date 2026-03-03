"""Convert TDMPC2 data to MAX format, filtering by task."""
import argparse
import numpy as np
import pickle
import torch
from pathlib import Path
import gc


def convert_tdmpc2_task(input_dir, output_path, task_id, episode_length=501):
    chunk_files = sorted(Path(input_dir).glob("chunk_*.pt"))

    all_states, all_actions, all_rewards, all_dones = [], [], [], []

    for chunk_path in chunk_files:
        print(f"Loading {chunk_path.name}...")
        data = torch.load(chunk_path, map_location='cpu', weights_only=False)

        # Filter episodes by task (task is constant per episode)
        task_per_episode = data['task'][:, 0].numpy()
        mask = task_per_episode == task_id

        if mask.sum() == 0:
            print(f"  No episodes for task {task_id}")
            del data
            gc.collect()
            continue

        obs = data['obs'][mask].numpy()
        action = data['action'][mask].numpy()
        reward = data['reward'][mask].numpy()

        n_eps = obs.shape[0]
        n_trans = n_eps * episode_length

        # Flatten episodes to transitions
        states = obs.reshape(n_trans, -1)
        actions = action.reshape(n_trans, -1)
        rewards = reward.reshape(n_trans)

        # Mark episode boundaries (1.0 at last step of each episode)
        dones = np.zeros(n_trans, dtype=np.float32)
        dones[episode_length-1::episode_length] = 1.0

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)

        print(f"  Found {n_eps} episodes, {n_trans} transitions")
        del data
        gc.collect()

    if not all_states:
        raise ValueError(f"No data found for task {task_id}")

    # Concatenate and add num_agents=1 dimension
    states = np.concatenate(all_states)[np.newaxis].astype(np.float32)
    actions = np.concatenate(all_actions)[np.newaxis].astype(np.float32)
    rewards = np.concatenate(all_rewards)[np.newaxis].astype(np.float32)
    dones = np.concatenate(all_dones).astype(np.float32)

    output = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'num_transitions': states.shape[1]
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nSaved to {output_path}")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  Total transitions: {output['num_transitions']:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TDMPC2 data to MAX pretraining format"
    )
    parser.add_argument(
        "--input-dir",
        default="data/tdmpc2",
        help="Directory containing TDMPC2 chunk_*.pt files"
    )
    parser.add_argument(
        "--output",
        default="data/cheetah_data/cheetah_run_tdmpc2.pkl",
        help="Output pickle file path"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=3,
        help="Task ID to extract (3=cheetah-run). See TDMPC2 TASK_SET for IDs."
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=501,
        help="Episode length in TDMPC2 data (default: 501 for mt30)"
    )
    args = parser.parse_args()
    convert_tdmpc2_task(args.input_dir, args.output, args.task_id, args.episode_length)
