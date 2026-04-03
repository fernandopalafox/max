"""
MAX: A JAX-based Reinforcement Learning Library
================================================

A modular library for model-based reinforcement learning
with first-class support for the HalfCheetah environment.

Core Modules:
- environments: Cheetah environment wrapper
- dynamics: Learned dynamics models (MLP, LoRA)
- encoders: Encoder abstraction (TDMPC2 path)
- critics: Q-function ensemble (TDMPC2 path)
- policies: Squashed Gaussian policy (TDMPC2 path)
- rewards: Reward models (learned + analytical)
- trainers: TDMPC2 trainer
- normalizers: State/action/reward normalization
- buffers: JAX-based replay buffers
- planners: Model-based planning (CEM, iCEM, MPPI)
- evaluators: Rollout evaluation
"""

__version__ = "0.1.0"

# Core components
from max.environments import init_env
from max.dynamics import Dynamics, init_dynamics
from max.encoders import Encoder, init_encoder
from max.critics import Critic, init_critic
from max.policies import Policy, init_policy
from max.rewards import Reward, init_reward_model
from max.trainers import Trainer, TrainState, init_trainer
from max.normalizers import (
    Normalizer,
    init_normalizer,
    init_rolling_return_normalizer,
    NONE_NORMALIZER,
    STANDARD_NORMALIZER,
)
from max.buffers import init_buffer, update_buffer
from max.planners import Planner, PlannerState, init_planner
from max.evaluators import init_evaluator

__all__ = [
    # Version
    "__version__",
    # Environments
    "init_env",
    # Dynamics
    "Dynamics",
    "init_dynamics",
    # Encoders (TDMPC2)
    "Encoder",
    "init_encoder",
    # Critics (TDMPC2)
    "Critic",
    "init_critic",
    # Policies (TDMPC2)
    "Policy",
    "init_policy",
    # Rewards
    "Reward",
    "init_reward_model",
    # TDMPC2 Trainer
    "Trainer",
    "TrainState",
    "init_trainer",
    # Normalizers
    "Normalizer",
    "init_normalizer",
    "init_rolling_return_normalizer",
    "NONE_NORMALIZER",
    "STANDARD_NORMALIZER",
    # Buffers
    "init_buffer",
    "update_buffer",
    # Planners
    "Planner",
    "PlannerState",
    "init_planner",
    # Evaluation
    "init_evaluator",
]
