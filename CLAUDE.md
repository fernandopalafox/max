# MAX Codebase Guidelines

## Environment

Scripts require the `max` conda environment: `conda run -n max python scripts/train.py ...`

## General Principles

**Functional and pure.** Functions should be pure and side-effect-free so they can be JIT-compiled with JAX. Avoid hidden state.

**Simple and readable.** Code should be easy to review at a glance. Prefer straightforward logic over clever tricks. If something looks complicated, it probably should be simplified.

**Bake config into closures at construction time.** Use the config to set up structure when a component is built, not at call time. The returned function should be a clean, specialized callable with no config-parsing inside it. See `max/dynamics.py` and `max/environments.py` for examples.

**Factory functions for abstractions.** Components (dynamics, encoders, evaluators, planners, etc.) are created via `init_*` factory functions that return a struct/NamedTuple bundling the callable(s). Follow this pattern for any new abstraction.

**No defensive clutter.** Don't add asserts, input validation, or error checks unless explicitly asked for. Trust that inputs are correct.

**No config defaults.** Always access config keys directly (e.g. `config["key"]`), never with `.get("key", default)`. Missing keys should raise an error immediately so misconfigured runs fail loudly rather than silently using a fallback.

## Commits and PRs

Keep commit messages and PR descriptions plain and concise — no section headers, checklists, or structured templates. Describe the problem, what the change does, and any non-obvious tradeoffs. A good PR description reads like a short paragraph to a colleague, not a form. Don't hard-wrap at 88 columns; let each sentence flow on one line since messages are rendered on GitHub.
