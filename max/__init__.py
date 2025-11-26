"""
MAX: A JAX-based Reinforcement Learning Library
================================================

A modular library for model-based and model-free reinforcement learning
with first-class support for multi-agent systems.

Core Modules:
- environments: Multi-agent tracking and pursuit-evasion environments
- dynamics: Learned dynamics models (MLP, PETS)
- policies: Actor-critic policies and model-based planners
- policy_trainers: PPO and IPPO training algorithms
- trainers: Dynamics model training (GD, EKF, PETS)
- normalizers: State/action/reward normalization
- buffers: JAX-based replay buffers
- planners: Model-based planning (CEM, iCEM)
- policy_evaluators: Policy evaluation utilities
- evaluation: Dynamics model evaluation
- estimators: EKF and state estimation
- solvers: LQR and other control solvers
"""

__version__ = "0.1.0"

# Core components
from max.environments import init_env, make_env, make_pursuit_evasion_env
from max.dynamics import (
    DynamicsModel,
    create_MLP_residual_dynamics,
    create_analytical_pendulum_dynamics,
)
from max.policies import (
    Policy,
    PolicyState,
    init_policy,
    create_actor_critic_policy,
)
from max.policy_trainers import (
    PolicyTrainer,
    PolicyTrainState,
    init_policy_trainer,
    create_ippo_policy_trainer,
)
from max.max.dynamics_trainers import (
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
from max.policy_evaluators import evaluate_policy
from max.evaluation import Evaluator

__all__ = [
    # Version
    "__version__",
    # Environments
    "init_env",
    "make_env",
    "make_pursuit_evasion_env",
    # Dynamics
    "DynamicsModel",
    "create_MLP_residual_dynamics",
    "create_analytical_pendulum_dynamics",
    # Policies
    "Policy",
    "PolicyState",
    "init_policy",
    "create_actor_critic_policy",
    # Policy Trainers
    "PolicyTrainer",
    "PolicyTrainState",
    "init_policy_trainer",
    "create_ppo_policy_trainer",
    "create_ippo_policy_trainer",
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
    "evaluate_policy",
    "Evaluator",
]
