> **Looking for the generalized information-gathering work?**
> If you arrived here from the paper [*Generalized Information Gathering Under Dynamics Uncertainty*](https://arxiv.org/abs/2601.21988), see the [`generalized`](../../tree/generalized) branch.

# MAX

**Experimental** JAX-based model-based RL library. Re-implements TDMPC2 with an emphasis on functional/pure-style code. Expect sharp edges.

## Installation

### From source

```bash
git clone <repository-url>
cd max
pip install -e .
```

## Dependencies

Core (installed via pip):
- `jax`, `jaxlib` - Core framework
- `flax` - Neural network definitions
- `optax` - Optimizers
- `numpy`, `scipy`, `matplotlib`

Optional:
- `wandb` - Experiment tracking (`pip install -e ".[wandb]"`)

## License

MIT License
