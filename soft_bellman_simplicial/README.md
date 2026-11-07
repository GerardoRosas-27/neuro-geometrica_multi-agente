# Soft-Bellman simplicial3D v2 — Rust research executable

A genuine tetrahedral complex with executable straight segments through triangular/tetrahedral interiors, feasible full SPD3x3 learned geometry and positive face impedances. This is a finite Steiner discretization, NOT the exact continuous simplicial Bellman equation from the source.

- `src/main.rs`: Rust-only mesh/data generation, boundaries, SPD geometry, cost derivatives, reverse-mode unrolled Bellman, Adam, Dijkstra teacher, route rollout, Hodge/Regge/CM diagnostics and unit tests.
- `run.py`: orchestration only, four low-priority processes maximum; frozen-source SHA required for CONFIRM.
- `summarize.py`: paired seed-bootstrap statistics only.
- Isolated Cargo workspace, no dependencies. v1 and existing application unchanged.

```sh
nice -n 10 cargo test --manifest-path soft_bellman_simplicial/Cargo.toml -j 2 -- --test-threads=2
nice -n 10 cargo build --release --manifest-path soft_bellman_simplicial/Cargo.toml -j 2
nice -n 10 soft_bellman_simplicial/target/release/soft-bellman-simplicial-v2 train 46080 CONFIRM /workspace/new-simplicial-output
nice -n 10 python3 soft_bellman_simplicial/run.py DEV --workers 4 --output /workspace/simplicial-repro
nice -n 10 python3 soft_bellman_simplicial/run.py CONFIRM --workers 4 --output /workspace/simplicial-repro
nice -n 10 python3 soft_bellman_simplicial/summarize.py
```

Individual CLI accepts decimal seed, phase label and output directory; it overwrites that directory's experiment files, so always use a fresh path. `run.py` refuses overwriting a completed seed range. Aggregator reads stored docs evidence by default; change its DOC path in a scratch copy to analyze external reproduction.

Read `../docs/soft_bellman_simplicial_v2/PLAN.md`, `TRAZABILIDAD.md` and `RESULTADOS.md`. Geometry is global affine SPD, not independently curved Regge lengths. Hodge/Regge are diagnostic operators, not active training layers/regularizers. The executable currently generates meshes internally; external-mesh IO and stable public integration API are future work. No BQ10/LLM production integration.

Bellman cost O(H*(N+E)), tape memory O(H*(N+E)), H=N. Cost head O(E*9), geometry/topology construction separate. Refinement grows nodes/actions substantially. Dijkstra implementation favors transparent small-fixture correctness over large-mesh efficiency.
