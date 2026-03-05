# pretrain.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import os
import pickle
import json
from datetime import datetime
from max.normalizers import init_normalizer
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer

def main(config):
    # Create unique run directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["save_path"], run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    wandb.init(
        project=config.get("wandb_project", "pretraining"),
        config=config,
        name=f"pretrain_{config['dynamics']}",
        reinit=True,
    )

    key = jax.random.key(config["seed"])

    # Load data — keep as numpy on CPU, only transfer batches to GPU
    with open(config["data_path"], "rb") as f:
        raw_data = pickle.load(f)

    raw_states = np.array(raw_data["states"][0])   # (N, dim_state), CPU
    raw_actions = np.array(raw_data["actions"][0]) # (N, dim_action), CPU
    raw_dones = raw_data["dones"]
    del raw_data  # free raw buffer memory

    # Extract full episodes using dones boundaries
    episode_ends = np.where(raw_dones == 1)[0]
    episode_boundaries = np.concatenate([[0], episode_ends + 1])
    episodes = []
    for start, end in zip(episode_boundaries[:-1], episode_boundaries[1:]):
        if end - start > 1:  # skip degenerate episodes
            episodes.append({
                "states": raw_states[start:end],    # (T+1, dim_s)
                "actions": raw_actions[start:end-1], # (T, dim_a)
            })

    # Episode-level shuffle + split
    key, ep_key = jax.random.split(key)
    ep_perm = np.array(jax.random.permutation(ep_key, len(episodes)))
    n_train_ep = int(len(episodes) * config["train_split"])
    train_eps = [episodes[i] for i in ep_perm[:n_train_ep]]
    test_eps  = [episodes[i] for i in ep_perm[n_train_ep:]]

    # Flatten episodes into transitions
    def flatten_episodes(eps):
        s, a, ns = [], [], []
        for ep in eps:
            s.append(ep["states"][:-1])
            a.append(ep["actions"])
            ns.append(ep["states"][1:])
        return {"states": np.concatenate(s), "actions": np.concatenate(a), "next_states": np.concatenate(ns)}

    train_data = flatten_episodes(train_eps)
    test_data  = flatten_episodes(test_eps)

    # Pick visualization episodes (one from each split)
    key, vis_key = jax.random.split(key)
    vis_train_ep = train_eps[int(jax.random.randint(vis_key, (), 0, len(train_eps)))]
    key, vis_key = jax.random.split(key)
    vis_test_ep  = test_eps[int(jax.random.randint(vis_key, (), 0, len(test_eps)))]

    n_train = len(train_data["states"])
    n_samples = n_train + len(test_data["states"])

    # Init model and trainer
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Count and log trainable parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
    print(f"Trainable parameters: {num_params:,}")
    wandb.config.update({"num_params": num_params})

    # Autoregressive rollout
    max_rollout_steps = config.get("eval_rollout_steps", 200)

    def autoregressive_rollout(params, init_state, actions):
        """Runs model autoregressively. actions: (T, dim_a), returns (T+1, dim_s)."""
        states = [jnp.array(init_state)]
        for action in actions[:max_rollout_steps]:
            next_state = dynamics_model.pred_one_step(params, states[-1], jnp.array(action))
            states.append(next_state)
        return np.array(jax.device_get(jnp.stack(states)))

    state_norm = config.get("normalization_params", {}).get("state", {})
    state_ylims = list(zip(state_norm.get("min", []), state_norm.get("max", []))) or None

    def plot_episode_comparison(true_states, pred_states, title, state_labels=None):
        T, dim = true_states.shape
        ncols = 4
        nrows = (dim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.5))
        axes = axes.flatten()
        for d in range(dim):
            ax = axes[d]
            ax.plot(true_states[:, d], label="true", lw=1.5)
            ax.plot(pred_states[:, d], label="model", lw=1.5, linestyle="--")
            if state_ylims and d < len(state_ylims):
                ax.set_ylim(state_ylims[d])
            label = state_labels[d] if state_labels and d < len(state_labels) else f"dim {d}"
            ax.set_title(label, fontsize=8)
            ax.tick_params(labelsize=6)
            if d == 0:
                ax.legend(fontsize=7)
        for d in range(dim, len(axes)):
            axes[d].set_visible(False)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        return fig

    # JIT'd loss function over a single batch (GPU)
    # Compute loss in normalized delta space
    vmap_pred_norm_delta = jax.vmap(dynamics_model.pred_norm_delta, in_axes=(None, 0, 0))
    vmap_normalize = jax.vmap(normalizer.normalize, in_axes=(None, 0))

    @jax.jit
    def compute_loss(params, states, actions, next_states):
        pred_norm_delta = vmap_pred_norm_delta(params, states, actions)
        target_delta = next_states - states
        target_norm_delta = vmap_normalize(norm_params["delta"], target_delta)
        return jnp.mean((pred_norm_delta - target_norm_delta) ** 2)

    def compute_loss_batched(params, data, batch_size):
        """Compute loss over data in CPU numpy, transferring one batch at a time."""
        n = len(data["states"])
        losses = []
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            loss = compute_loss(
                params,
                jnp.array(data["states"][sl]),
                jnp.array(data["actions"][sl]),
                jnp.array(data["next_states"][sl]),
            )
            losses.append(float(loss))
        return np.mean(losses)

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    train_losses, test_losses = [], []

    # Checkpoint config
    checkpoint_enabled = config.get("checkpoint_enabled", False)
    checkpoint_freq = config.get("checkpoint_freq", 10)
    if checkpoint_enabled:
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpointing every {checkpoint_freq} epochs to {checkpoint_dir}")

    eval_plot_freq = config.get("eval_plot_freq", 20)
    print(f"Training: {n_train} samples, Testing: {n_samples - n_train} samples")

    def run_eval_plots(step):
        for ep, tag in [(vis_train_ep, "train_episode"), (vis_test_ep, "test_episode")]:
            T = min(len(ep["actions"]), max_rollout_steps)
            true_states = ep["states"][:T+1]
            pred_states = autoregressive_rollout(train_state.params, ep["states"][0], ep["actions"])
            pred_states = pred_states[:T+1]
            fig = plot_episode_comparison(true_states, pred_states,
                                          title=f"Epoch {step} — {tag}",
                                          state_labels=config.get("state_labels"))
            wandb.log({f"eval/{tag}": wandb.Image(fig)}, step=step)
            plt.close(fig)

    train_loss0 = compute_loss_batched(train_state.params, train_data, batch_size)
    test_loss0 = compute_loss_batched(train_state.params, test_data, batch_size)
    wandb.log({"losses/train": train_loss0, "losses/test": test_loss0}, step=0)
    run_eval_plots(step=0)

    try:
        for epoch in range(num_epochs):
            # Shuffle training data on CPU
            key, shuffle_key = jax.random.split(key)
            perm = np.array(jax.random.permutation(shuffle_key, n_train))
            shuffled = {k: v[perm] for k, v in train_data.items()}

            # Train over batches — only transfer each mini-batch to GPU
            epoch_losses = []
            for i in range(0, n_train, batch_size):
                sl = slice(i, i + batch_size)
                batch = {k: jnp.array(v[sl]) for k, v in shuffled.items()}
                train_state, loss = trainer.train(train_state, batch)
                epoch_losses.append(float(loss))

            train_loss = np.mean(epoch_losses)
            test_loss = compute_loss_batched(train_state.params, test_data, batch_size)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            epochs_so_far = list(range(len(train_losses)))
        wandb.log({
            "losses/train": train_loss,
            "losses/test": test_loss,
            "losses/combined": wandb.plot.line_series(
                xs=epochs_so_far,
                ys=[train_losses, test_losses],
                keys=["train", "test"],
                title="Train vs Test Loss",
                xname="epoch",
            ),
        }, step=epoch)

            if (epoch + 1) % eval_plot_freq == 0:
                run_eval_plots(step=epoch + 1)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: train={train_loss:.6f}, test={test_loss:.6f}")

            # Save checkpoint
            if checkpoint_enabled and (epoch + 1) % checkpoint_freq == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(jax.device_get(train_state.params), f)
                print(f"Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {epoch+1}. Saving model...")

    # Save model with descriptive name: {dynamics}_{data_source}.pkl
    params_np = jax.device_get(train_state.params)
    data_name = os.path.basename(config["data_path"]).replace("_buffer.pkl", "")
    model_name = f"{config['dynamics']}_{data_name}.pkl"
    model_path = os.path.join(run_dir, model_name)
    with open(model_path, "wb") as f:
        pickle.dump(params_np, f)
    print(f"Model saved to {model_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain dynamics model")
    parser.add_argument("--config", type=str, default="linear_with_nn.json")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    config = full_config["pretraining"]

    if args.data_path:
        config["data_path"] = args.data_path
    if args.save_dir:
        config["save_path"] = args.save_dir
    if args.seed:
        config["seed"] = args.seed
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project

    main(config)
