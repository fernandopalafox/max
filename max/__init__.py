"""
MAX: A JAX-based Reinforcement Learning Library
================================================

A modular library for model-based reinforcement learning
with first-class support for the HalfCheetah environment.

Core Modules:
- environments: Cheetah environment wrapper
- dynamics: Learned dynamics models (MLP, PETS)
- trainers: Dynamics model training (GD, EKF, PETS)
- normalizers: State/action/reward normalization
- buffers: JAX-based replay buffers
- planners: Model-based planning (CEM, iCEM)
- evaluation: Dynamics model evaluation
- estimators: EKF and state estimation
"""

__version__ = "0.1.0"

# Core components
from max.environments import init_env
from max.dynamics import DynamicsModel
from max.dynamics_trainers import (
    Trainer,
    TrainState,
    init_trainer,
    create_gradient_descent_trainer,
)
from max.normalizers import (
    Normalizer,
    init_normalizer,
    init_rolling_return_normalizer,
    NONE_NORMALIZER,
    STANDARD_NORMALIZER,
)
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.planners import Planner, PlannerState, init_planner
from max.dynamics_evaluators import init_evaluator

__all__ = [
    # Version
    "__version__",
    # Environments
    "init_env",
    # Dynamics
    "DynamicsModel",
    # Trainers
    "Trainer",
    "TrainState",
    "init_trainer",
    "create_gradient_descent_trainer",
    # Normalizers
    "Normalizer",
    "init_normalizer",
    "init_rolling_return_normalizer",
    "NONE_NORMALIZER",
    "STANDARD_NORMALIZER",
    # Buffers
    "init_jax_buffers",
    "update_buffer_dynamic",
    # Planners
    "Planner",
    "PlannerState",
    "init_planner",
    # Evaluation
    "init_evaluator",
]
