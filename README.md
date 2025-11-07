# ⚠️ MAX: A JAX-based **Research** Library for Online RL ⚠️

**MAX** is a highly **experimental and rapidly evolving** modular reinforcement learning library built on JAX. It is **primarily designed** to prioritize **online adaptation algorithms** and **information-gathering strategies** in reinforcement learning, with a focus on both model-based and model-free control, and first-class support for multi-agent systems.

## Features

- **Pure JAX Implementation**: Leverage JIT compilation, automatic differentiation, and GPU/TPU acceleration for fast iteration.
- **Emphasis on Online Adaptation**: Core design centers around algorithms and components for efficient **adaptation to changing or uncertain dynamics**.
- **Model-Based Algorithms with Parameter Belief**: Supports model-based control where the dynamics components **maintain a distribution or belief over uncertain parameters** (e.g., in a Bayesian context).
- **Multi-Agent RL**: Built-in support for IPPO (Independent PPO) and multi-agent environments.
- **Modular Design**: Mix and match components (environments, policies, trainers, normalizers) for rapid prototyping of novel online algorithms.

## Installation

### From source

```bash
git clone <repository-url>
cd max
pip install -e .
```

## Library Structure

### Core Modules

- **`environments`**: Multi-agent tracking and pursuit-evasion environments
- **`dynamics`**: Learned dynamics models (MLP-based, analytical models)
- **`policies`**: Actor-critic policies and model-based planners
- **`policy_trainers`**: PPO and IPPO training algorithms
- **`trainers`**: Dynamics model training (gradient descent, EKF, PETS)
- **`normalizers`**: State/action/reward normalization utilities
- **`buffers`**: JAX-based replay buffers for efficient data storage
- **`planners`**: Model-based planning algorithms (CEM, iCEM)
- **`policy_evaluators`**: Policy evaluation and rollout utilities
- **`evaluation`**: Dynamics model evaluation metrics

### Auxiliary Modules

- **`estimators`**: Extended Kalman Filter for online Bayesian optimization

## Examples

### Pursuit-Evasion

<div style="text-align: center;">
  <img src="figures/readme_pursuit_evasion.gif" alt="Multi-agent pursuit-evasion policy visualization" width=400 style="display: block; margin: 0 auto;"/>
  <p style="text-align: center; font-style: italic;">
    <strong>Figure 1:</strong> Multi-agent pursuit-evasion policy
  </p>
</div>

- **`scripts/ippo_pe.py`**: Train IPPO agents on pursuit-evasion task
  - Multi-agent coordination
  - Rolling return normalization
  - Periodic policy evaluation
  - WandB logging support

- **`scripts/visualize_pe.py`**: Visualize trained policies
  - Generate animated GIFs
  - Custom initial conditions
  - Multi-agent trajectory visualization

## Architecture Highlights

### Functional Design

All components follow JAX's functional programming paradigm:
- Immutable state containers (NamedTuples, PyTreeNodes)
- Pure functions for transformations
- JIT-compiled operations for performance

### Multi-Agent Support

The library is designed with multi-agent systems as a first-class citizen:
- Independent parameter sets per agent
- Shared or separate training
- Flexible observation/action spaces

### Composability

Mix and match components easily:
```python
# Use model-based planner as policy
policy = create_planner_policy(planner, dynamics_model)

# Or use model-free actor-critic
policy = create_actor_critic_policy(config)

# Same trainer interface for both!
trainer = init_policy_trainer(config, policy)
```

## License

MIT License