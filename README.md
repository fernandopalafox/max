# MAX: A JAX-based Reinforcement Learning Library

MAX is a modular reinforcement learning library built on JAX, designed for both model-based and model-free control with first-class support for multi-agent systems.

## Features

- **Pure JAX Implementation**: Leverage JIT compilation, automatic differentiation, and GPU/TPU acceleration
- **Multi-Agent RL**: Built-in support for IPPO (Independent PPO) and multi-agent environments
- **Model-Based & Model-Free**: Combine learned dynamics models with planning or use pure policy optimization
- **Modular Design**: Mix and match components (environments, policies, trainers, normalizers)
- **Production-Ready**: Includes state normalization, replay buffers, and evaluation utilities

## Installation

### From source

```bash
git clone <repository-url>
cd max
pip install -e .
```

### With optional dependencies

```bash
# For Weights & Biases logging
pip install -e ".[wandb]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Training IPPO on Pursuit-Evasion

```python
from max import (
    init_env,
    init_policy,
    init_policy_trainer,
    init_normalizer,
    init_jax_buffers,
)
import jax

# Initialize environment
key = jax.random.PRNGKey(0)
env = init_env("pursuit_evasion", config)

# Create policy and trainer
policy, policy_state = init_policy(config, key)
trainer, train_state = init_policy_trainer(config, policy, key)

# Initialize replay buffer
buffers = init_jax_buffers(config)

# Training loop (see scripts/ippo_pe.py for complete example)
```

### Visualizing Trained Policies

```bash
python scripts/visualize_pe.py --policy-path trained_policies/my_policy.pkl
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

- **`estimators`**: Extended Kalman Filter for state estimation
- **`solvers`**: LQR and other optimal control solvers
- **`utils`**: Visualization and utility functions

## Examples

### Training Scripts

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

## Configuration

All components use dictionary-based configuration for flexibility. See `scripts/ippo_pe.py` for a complete example configuration.

## Requirements

- Python >= 3.8, < 3.14
- JAX >= 0.4.0
- Flax >= 0.8.0
- Optax >= 0.2.0
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0

## Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black max/ scripts/
isort max/ scripts/
```

## Citation

If you use MAX in your research, please cite:

```bibtex
@software{max2024,
  title={MAX: A JAX-based Reinforcement Learning Library},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/max}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

This library builds on research in:
- Model-based reinforcement learning
- Multi-agent coordination
- JAX ecosystem tools (Flax, Optax)
