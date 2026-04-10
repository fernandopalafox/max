"""
Progressive diagnostic: jacrev -> vmap(jacrev) -> full trajectory_value_fn_ig
Run with: python scripts/debug_jacrev.py
"""
import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache_debug"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json, pickle, copy
import jax
import jax.numpy as jnp
import jax.flatten_util

from max.dynamics import init_dynamics
from max.encoders import init_encoder

CONFIG_PATH = "configs/cheetah.json"
PRETRAINED_PATH = "data/models/cheetah/20260323_less_features/final.pkl"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)["training"]

def main():
    config = load_config()
    key = jax.random.key(0)

    # ---- Init encoder + dynamics ----
    pretrained = {}
    if PRETRAINED_PATH:
        with open(PRETRAINED_PATH, "rb") as f:
            pretrained = pickle.load(f)["mean"]

    key, enc_key, dyn_key = jax.random.split(key, 3)
    encoder, enc_params = init_encoder(enc_key, config, pretrained=pretrained.get("encoder"))
    dynamics, dyn_params = init_dynamics(dyn_key, config, pretrained=pretrained.get("dynamics"))

    flat_params, unflatten = jax.flatten_util.ravel_pytree(dyn_params)
    n_params = flat_params.shape[0]

    dummy_obs = jnp.zeros(config["dim_state"])
    z0 = encoder.encode(enc_params, dummy_obs)
    latent_dim = z0.shape[0]
    dim_a = config["dim_action"]
    a0 = jnp.zeros(dim_a)

    print(f"n_dyn_params={n_params}, latent_dim={latent_dim}, dim_a={dim_a}")
    print(f"J shape will be ({latent_dim}, {n_params}), size={latent_dim*n_params*4/1e6:.2f} MB per J")

    # ---- Test 1: jacrev alone ----
    print("\n[1] jacrev alone...")
    J = jax.jacrev(lambda fp: dynamics.predict(unflatten(fp), z0, a0))(flat_params)
    print(f"    OK — J.shape={J.shape}")

    # ---- Test 2: vmap(jacrev) at increasing batch sizes ----
    for batch in [8, 64]:  # 512 known to fail — XLA autotuning bug
        print(f"\n[2] vmap(jacrev) batch={batch}...")
        actions = jnp.zeros((batch, dim_a))
        def compute_J(a):
            return jax.jacrev(lambda fp: dynamics.predict(unflatten(fp), z0, a))(flat_params)
        Js = jax.vmap(compute_J)(actions)
        jax.block_until_ready(Js)
        print(f"    OK — Js.shape={Js.shape}, total={Js.nbytes/1e6:.1f} MB")

    # ---- Test 3: lax.map(jacrev) — sequential, avoids XLA batched kernel issue ----
    print(f"\n[3] lax.map(jacrev) batch=512 (sequential)...")
    actions_seq = jnp.zeros((512, 3, dim_a))

    def compute_info_gain(actions):
        J = jax.jacrev(lambda fp: dynamics.predict(unflatten(fp), z0, actions[0]))(flat_params)
        P = jnp.eye(n_params) * 0.01
        R = jnp.eye(latent_dim)
        S = J @ P @ J.T + R
        return 0.5 * (jnp.linalg.slogdet(S)[1] - jnp.linalg.slogdet(R)[1])

    info_gains = jax.lax.map(compute_info_gain, actions_seq)
    jax.block_until_ready(info_gains)
    print(f"    OK — info_gains.shape={info_gains.shape}")

    # ---- Test 4: full trajectory_value_fn_ig ----
    print("\n[4] Full trajectory_value_fn_ig (as used in MPPI)...")
    from max.critics import init_critic
    from max.policies import init_policy
    from max.rewards import init_reward_model
    import copy

    key, ck, pk = jax.random.split(key, 3)
    critic, critic_params = init_critic(ck, config, pretrained=pretrained.get("critic"))
    policy, policy_params = init_policy(pk, config, pretrained=pretrained.get("policy"))
    reward_model, reward_params = init_reward_model(config, pretrained=pretrained.get("reward"))

    parameters = {
        "mean": {
            "encoder": enc_params,
            "dynamics": dyn_params,
            "reward": reward_params,
            "critic": critic_params,
            "ema_critic": copy.deepcopy(critic_params),
            "policy": policy_params,
        },
        "normalizer": {"q_scale": jnp.array(1.0)},
    }

    from max.trainers import init_ekf_efficient_trainer
    _, _ = init_ekf_efficient_trainer(key, config, encoder, dynamics, parameters)
    # ^ this adds parameters["covariance"]

    from max.planners import make_tdmpc2_trajectory_value_fn_ig
    pp = config["planner"]
    traj_value_fn = make_tdmpc2_trajectory_value_fn_ig(
        dynamics, reward_model, critic, policy,
        horizon=pp["horizon"],
        discount_factor=pp["discount_factor"],
        meas_noise_scale=pp["meas_noise_scale"],
        info_weight=pp["info_weight"],
        info_horizon=pp["info_horizon"],
    )

    action_seqs = jnp.zeros((pp["batch_size"], pp["horizon"], dim_a))
    key, vk = jax.random.split(key)
    values = traj_value_fn(parameters, z0, action_seqs, vk)
    jax.block_until_ready(values)
    print(f"    OK — values.shape={values.shape}")

    print("\nAll tests passed.")

if __name__ == "__main__":
    main()
