
## OGD LR Sweep: ogd_lastlayer → project=ogd-lastlayer-lr
  ERROR lr1e-5: exit 1
    ~~~~^^^^^^^^^^^^
  File "/home/fp5275/repos/max/scripts/train.py", line 130, in main
    eval_results = evaluator.evaluate(parameters)
  File "/home/fp5275/repos/max/max/evaluators.py", line 28, in evaluate
    return self.evaluate_fn(params)
           ~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/fp5275/repos/max/max/evaluators.py", line 143, in evaluate_fn
    "eval/episode_reward": float(jnp.mean(all_rewards)),
                           ~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fp5275/miniconda3/envs/max/lib/python3.13/site-packages/jax/_src/array.py", line 299, in __float__
    return self._value.__float__()
           ^^^^^^^^^^^
  File "/home/fp5275/miniconda3/envs/max/lib/python3.13/site-packages/jax/_src/profiler.py", line 384, in wrapper
    return func(*args, **kwargs)
  File "/home/fp5275/miniconda3/envs/max/lib/python3.13/site-packages/jax/_src/array.py", line 639, in _value
    npy_value, did_copy = self._single_device_array_to_np_array_did_copy()
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 11.95MiB.

ERROR conda.cli.main_run:execute(127): `conda run python /home/fp5275/repos/max/scripts/train.py --config /tmp/tmpmwf4anlq.json --run-name lr1e-5 --num-seeds 3` failed. (See above for error)
  lr=1.0e-05: NO RESULTS
  lr=1.0e-04: mean=447.6 ±63.7 (n=3)
  lr=1.0e-03: mean=401.3 ±51.5 (n=3)
  lr=1.0e-02: mean=288.6 ±125.4 (n=3)
  → Best lr: 1.0e-04  (mean=447.6 ±63.7 (n=3))

## OGD LR Sweep: ogd_dense → project=ogd-dense-lr
  lr=1.0e-05: mean=431.8 ±48.7 (n=3)
  lr=1.0e-04: mean=644.5 ±69.6 (n=3)
  lr=1.0e-03: mean=180.2 ±98.6 (n=3)
  lr=1.0e-02: mean=123.1 ±57.6 (n=3)
  → Best lr: 1.0e-04  (mean=644.5 ±69.6 (n=3))

## OGD LR Sweep: ogd_loraxs → project=ogd-loraxs-lr
  lr=1.0e-05: mean=491.0 ±6.7 (n=3)
  lr=1.0e-04: mean=515.2 ±51.7 (n=3)
  lr=1.0e-03: mean=224.6 ±71.7 (n=3)
  lr=1.0e-02: mean=99.0 ±36.1 (n=3)
  → Best lr: 1.0e-04  (mean=515.2 ±51.7 (n=3))

## OGD LR Sweep: ogd_tinylora → project=ogd-tinylora-lr
  lr=1.0e-05: mean=341.0 ±32.6 (n=3)
  lr=1.0e-04: mean=429.4 ±39.0 (n=3)
  lr=1.0e-03: mean=473.1 ±33.9 (n=3)
  lr=1.0e-02: mean=455.2 ±17.8 (n=3)
  → Best lr: 1.0e-03  (mean=473.1 ±33.9 (n=3))

## OGD Best LRs Summary
  ogd_lastlayer: 1e-04
  ogd_dense: 1e-04
  ogd_loraxs: 1e-04
  ogd_tinylora: 1e-03

## Updating experiment_log.md Key Results table...
  OGD last-layer: best lr = 1e-04
  OGD full network: best lr = 1e-04
  OGD LoRA-XS: best lr = 1e-04
  OGD TinyLoRA: best lr = 1e-03

## Phase 3: Final comparison → streaming-adaptation-final
  Running ekf-loraxs (5 seeds)...
  ekf-loraxs: mean=609.7 ±40.7 (n=5)
  Running ekf-lastlayer (5 seeds)...
  ekf-lastlayer: mean=480.7 ±69.2 (n=5)
  Running ekf-tinylora (5 seeds)...
  ekf-tinylora: mean=502.4 ±31.8 (n=5)
  Running ogd-loraxs (5 seeds)...
  ogd-loraxs: mean=540.2 ±51.3 (n=5)
  Running ogd-tinylora (5 seeds)...
  ogd-tinylora: mean=474.2 ±26.5 (n=5)
  Running ogd-lastlayer (5 seeds)...
  ogd-lastlayer: mean=445.4 ±68.5 (n=5)
  Running ogd-dense (5 seeds)...
  ogd-dense: mean=589.5 ±89.1 (n=5)

---
## Final Results — streaming-adaptation-final — 2026-04-13

| Method | Trainer | Mean | Std | Seeds |
|---|---|---|---|---|
| LoRA-XS | EKF | 609.7 | ±40.7 | 5 |
| last-layer | EKF | 480.7 | ±69.2 | 5 |
| TinyLoRA | EKF | 502.4 | ±31.8 | 5 |
| LoRA-XS | OGD | 540.2 | ±51.3 | 5 |
| TinyLoRA | OGD | 474.2 | ±26.5 | 5 |
| last-layer | OGD | 445.4 | ±68.5 | 5 |
| full network | OGD | 589.5 | ±89.1 | 5 |

Completed: 2026-04-13 20:39
