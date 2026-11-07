# Soft-Bellman v1 (Rust)

Prototype of goal-conditioned finite-horizon differentiable graph planning. All model training and reverse-mode gradients are implemented in `src/main.rs`, using only Rust std. CPU-only f64; no external crates/GPU/LLM. Nine trainable metric/potential parameters. This crate is deliberately isolated from the repository's root Cargo workspace.

Read [`PLAN.md`](../docs/soft_bellman_v1/PLAN.md) before changing an experiment and [`RESULTADOS.md`](../docs/soft_bellman_v1/RESULTADOS.md) for measured results and limitations.

```sh
nice -n 10 cargo test --manifest-path soft_bellman/Cargo.toml -j 2
nice -n 10 cargo build --release --manifest-path soft_bellman/Cargo.toml -j 2
nice -n 10 soft_bellman/target/release/soft-bellman-v1 train 46080 CONFIRM /workspace/new-output
```

Training CLI accepts decimal seed, phase label, output directory; this label does not select hyperparameters. `run.py` orchestrates disjoint preregistered seed ranges, source hash verification and up to4 processes. `summarize.py` performs only paired bootstrap analysis. Avoid rerunning directly into stored evidence: individual training CLI writes its output files.

## Algorithm
1. Edge costs from local terrain features via positive metric heads and nonnegative potential.
2. V0=0 at goal, infinity elsewhere. For H=number of cells, update softmin(c+V_previous), clamping goal0. Only legal free-cell edges exist.
3. Oracle-action cross entropy at layerH. Its adjoints initialize edge-cost and V_(H-1) gradients; each earlier softmin distributes adjoint with its stored policy probabilities. Chain edge-cost derivatives into9 shared parameters, then Adam update.
4. Held-out routes greedily minimize Q with a decreasing horizon. Oracle costs are used only to score resulting routes; labels are used to train and measure NLL/accuracy, never route selection.

Runtime O(H*E), tape memory O(H*(E+N)) per map. Cost heads O(E*P). Extending to high dimensions increases both graph size and degree; no arbitrary-dimensional scaling claim. The 3D fixture ties axes>=1 to the second metric head. This is a research executable, not yet a stable integration library/API.
