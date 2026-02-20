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
from max.normalizers import init_normalizer
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer


def main(config):
    wandb.init(
        project=config.get("wandb_project", "pretraining"),
        config=config,
        name=f"pretrain_{config['dynamics']}",
        reinit=True,
    )

    key = jax.random.key(config["seed"])

    # Load data
    with open(config["data_path"], "rb") as f:
        raw_data = pickle.load(f)

    states = jnp.array(raw_data["states"][0])  # (N, dim_state)
    actions = jnp.array(raw_data["actions"][0])  # (N, dim_action)
    dones = raw_data["dones"]

    # Create transitions (skip episode boundaries)
    valid_idx = np.where(dones[:-1] == 0)[0]
    states = states[valid_idx]
    actions = actions[valid_idx]
    next_states = jnp.array(raw_data["states"][0])[valid_idx + 1]
    n_samples = len(states)

    # Train/test split
    n_train = int(n_samples * config["train_split"])
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, n_samples)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_data = {
        "states": states[train_idx],
        "actions": actions[train_idx],
        "next_states": next_states[train_idx],
    }
    test_data = {
        "states": states[test_idx],
        "actions": actions[test_idx],
        "next_states": next_states[test_idx],
    }

    # Init model and trainer
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # JIT'd test loss function
    @jax.jit
    def compute_loss(params, data):
        vmap_pred = jax.vmap(dynamics_model.pred_norm_delta, in_axes=(None, 0, 0))
        pred = vmap_pred(params, data["states"], data["actions"])
        true_deltas = data["next_states"] - data["states"]
        vmap_norm = jax.vmap(normalizer.normalize, in_axes=(None, 0))
        true_norm = vmap_norm(norm_params["delta"], true_deltas)
        return jnp.mean((pred - true_norm) ** 2)

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    train_losses, test_losses = [], []

    print(f"Training: {n_train} samples, Testing: {n_samples - n_train} samples")

    for epoch in range(num_epochs):
        # Shuffle training data
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_train)
        shuffled = {k: v[perm] for k, v in train_data.items()}

        # Train over batches
        epoch_losses = []
        for i in range(0, n_train, batch_size):
            batch = {k: v[i : i + batch_size] for k, v in shuffled.items()}
            train_state, loss = trainer.train(train_state, batch)
            epoch_losses.append(float(loss))

        train_loss = np.mean(epoch_losses)
        test_loss = float(compute_loss(train_state.params, test_data))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        wandb.log({"train/loss": train_loss, "test/loss": test_loss}, step=epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: train={train_loss:.6f}, test={test_loss:.6f}")

    # Plot losses
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train")
    ax.plot(test_losses, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.set_title("Pretraining Loss")
    ax.grid(True, alpha=0.3)
    wandb.log({"loss_curves": wandb.Image(fig)})
    plt.close(fig)

    # Save model with descriptive name: {dynamics}_{data_source}.pkl
    save_path = config["save_path"]
    os.makedirs(save_path, exist_ok=True)
    params_np = jax.device_get(train_state.params)
    data_name = os.path.basename(config["data_path"]).replace("_buffer.pkl", "")
    model_name = f"{config['dynamics']}_{data_name}"
    with open(os.path.join(save_path, model_name), "wb") as f:
        pickle.dump(params_np, f)
    print(f"Model saved to {os.path.join(save_path, model_name)}")

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
