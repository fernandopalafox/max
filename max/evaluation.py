# evaluation.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional


class Evaluator:
    """A class for evaluating dynamics models."""

    def __init__(
        self,
        pred_one_step: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ):
        """
        Initializes the Evaluator.

        Args:
            pred_one_step: A function that predicts the next state
                                   given params, the current state, and action.
                                   It should have the signature:
                                   `predict(params, state, action) -> next_state`.
        """
        self.pred_one_step = pred_one_step

    def compute_one_step_loss(
        self, params: Any, trajectory_data: dict
    ) -> jnp.ndarray:
        """
        Computes the mean squared error for single-step predictions.

        This function compares the model's prediction for the next state (s_{t+1})
        with the true next state, given the current state (s_t) and action (a_t),
        averaged over an entire trajectory. It's fast and ideal for checking
        local model accuracy.

        Args:
            params: The parameters of the dynamics model.
            trajectory_data: A dictionary containing the ground truth 'trajectory'
                             and 'actions'.

        Returns:
            A single scalar value for the one-step prediction loss.
        """
        # Unpack data for one-step prediction
        states = trajectory_data["trajectory"][
            :-1
        ]  # All states except the last
        actions = trajectory_data["actions"]
        true_next_states = trajectory_data["trajectory"][
            1:
        ]  # All states except the first

        # Use jax.vmap to efficiently apply the prediction function across the batch
        # of (state, action) pairs from the trajectory.
        vmapped_predict = jax.vmap(self.pred_one_step, in_axes=(None, 0, 0))
        predicted_next_states = vmapped_predict(params, states, actions)

        # Calculate the mean squared error between predicted and true next states
        loss = jnp.mean((predicted_next_states - true_next_states) ** 2)
        return loss

    def compute_multi_step_loss(
        self, params: Any, trajectory_data: dict
    ) -> jnp.ndarray:
        """
        Computes the mean squared error over a multi-step rollout.

        This function simulates an entire trajectory starting from the initial state
        using the learned dynamics model and a sequence of actions. It then
        compares this predicted trajectory to the ground truth. This is a more
        challenging metric as errors can accumulate over time.

        Args:
            params: The parameters of the dynamics model.
            trajectory_data: A dictionary containing the ground truth 'trajectory'
                             and 'actions'.

        Returns:
            A single scalar value for the multi-step rollout loss.
        """
        # Unpack data for multi-step rollout
        true_trajectory = trajectory_data["trajectory"]
        actions = trajectory_data["actions"]
        initial_state = true_trajectory[0]

        # Define the function for a single step of the rollout.
        # This function takes the state (carry) and action (input) and
        # returns the next state as both the new carry and the output.
        def rollout_step(state, action):
            next_state = self.pred_one_step(params, state, action)
            return next_state, next_state

        # Use jax.lax.scan for an efficient, sequential rollout.
        _, predicted_trajectory_tail = jax.lax.scan(
            rollout_step, initial_state, actions
        )

        # The full predicted trajectory is the initial state plus the rolled-out states.
        predicted_trajectory = jnp.vstack(
            [initial_state, predicted_trajectory_tail]
        )

        # Calculate the mean squared error over the entire trajectory
        loss = jnp.mean((predicted_trajectory - true_trajectory) ** 2)
        return loss

    # --------------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------------

    def plot_trajectories(
        self,
        param_sets: list,
        labels: list,
        trajectory_data: dict,
        title: str,
        state_labels: Optional[list] = None,
    ) -> plt.Figure:
        """
        Plots ground truth vs. predicted trajectories for each state dimension.

        This function creates a separate subplot for each state dimension. Each
        subplot shows the state's value over time for the ground truth
        trajectory and for the predicted trajectories generated from each set
        of model parameters.

        Args:
            param_sets: A list of model parameter sets to evaluate.
            labels: A list of labels for the legend corresponding to each param_set.
            trajectory_data: A dictionary with the ground truth trajectory data.
            title: The main title for the entire figure.
            state_labels: An optional list of strings to label each state dimension.

        Returns:
            A matplotlib Figure object containing the subplots.
        """
        # 1. Unpack data and get dimensions
        true_trajectory = trajectory_data["trajectory"]
        initial_state = trajectory_data["trajectory"][0]
        actions = trajectory_data["actions"]
        num_dims = true_trajectory.shape[1]
        time_steps = jnp.arange(true_trajectory.shape[0])

        # Use provided labels or fall back to a default
        if state_labels is None or len(state_labels) != num_dims:
            state_labels = [f"State Dim {i}" for i in range(num_dims)]
            print(
                "Warning: Missing or incorrect state labels. Using defaults."
            )

        # 2. Generate all predicted trajectories first
        @jax.jit
        def generate_trajectory(p, s0, acts):
            def step(state, action):
                next_state = self.pred_one_step(p, state, action)
                return next_state, next_state

            _, tail = jax.lax.scan(step, s0, acts)
            return jnp.vstack([s0, tail])

        predicted_trajectories = [
            generate_trajectory(params, initial_state, actions)
            for params in param_sets
        ]

        # 3. Create subplots
        # Use squeeze=False to ensure `axes` is always a 2D array for easy indexing
        fig, axes = plt.subplots(
            nrows=num_dims,
            ncols=1,
            figsize=(12, 3 * num_dims),
            sharex=True,
            squeeze=False,
        )

        # 💥 New Change: Calculate y-axis limits based on ground truth
        y_min = jnp.min(true_trajectory, axis=0)
        y_max = jnp.max(true_trajectory, axis=0)
        y_range = y_max - y_min
        margin = 0.20  # 10% margin
        y_min_clipped = y_min - y_range * margin
        y_max_clipped = y_max + y_range * margin

        # 4. Plot data on each subplot
        for i in range(num_dims):
            ax = axes[i, 0]  # Access the subplot

            # Plot Ground Truth
            ax.plot(
                time_steps,
                true_trajectory[:, i],
                label="Ground Truth",
                color="black",
                linewidth=2,
                linestyle="--",
            )

            # Plot Predictions
            for pred_traj, label in zip(predicted_trajectories, labels):
                ax.plot(
                    time_steps,
                    pred_traj[:, i],
                    label=label,
                    linewidth=1.5,
                    alpha=0.9,
                )

            # 💥 New Change: Apply the calculated y-axis limits
            ax.set_ylim(y_min_clipped[i], y_max_clipped[i])

            # Key Change: Use the new `state_labels` here
            ax.set_ylabel(state_labels[i], fontsize=12)
            ax.grid(True, linestyle=":", alpha=0.6)

        # 5. Finalize plot details
        # Add a shared legend at the top of the figure
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),  # Position legend above the subplots
            ncol=len(labels) + 1,
            fontsize=10,
        )

        # Set the x-axis label only for the bottom-most plot
        axes[-1, 0].set_xlabel("Time Step", fontsize=12)

        # Add a main title for the entire figure
        fig.suptitle(title, fontsize=16, y=0.97)

        # Adjust layout to prevent titles/labels from overlapping
        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # rect=[left, bottom, right, top]

        return fig
