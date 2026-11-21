# `aura-engine` — Arquitectura Cognitiva Neuromórfica y de Triple Escala Temporal (AURA + CorticalFabric)

Implementación modular en Rust puro (**cero dependencias externas**, cero tokens BPE, cero multiplicaciones de matrices densas en inferencia ternaria) basada en la propuesta archivada en [`docs/aura_v1/PROPUESTA.md`](../docs/aura_v1/PROPUESTA.md).

## Estructura del Workspace (7 crates)

```text
aura-engine/
├── Cargo.toml
├── run.py                  # Orquestación determinista DEV/CONFIRM con verificación SHA-256
├── summarize.py            # Bootstrap pareado por semilla (10,000 remuestreos, seed 12345)
└── crates/
    ├── aura-math/          # TernaryMatrix en bit-planes u64 (project_zero_mul), FastWeightMemory NLMS Q12, LUTs, effective_rank
    ├── aura-perception/    # Zero-Token Entropy Patcher causal + FixedPatcher control + delta-modulación de bytes
    ├── aura-cortex/        # DualCorticalCore (SSM selectivo + Fast Weights) + CorticalFabric (event-driven U(1), Laplace FE, tropical, 3-factores)
    ├── aura-world/         # Modelo de Mundo latente K=4 (Randomized Priors ternarios + NLMS + EMA objetivo JEPA)
    ├── aura-hippocampus/   # Grafo navegable L1 (estilo HNSW) + persistencia binaria + SleepConsolidator con histéresis
    ├── aura-metacog/       # Arbitraje Sistema 1 / Sistema 2 (EFE H=2) / AbstainAndAsk + bucle NE/ACh/DA
    └── aura-runtime/       # AuraAgent completo + CLI `aura-exp` (E1, E2, E3, E4, E5)
```

## Cómo compilar, probar y reproducir

Desde la raíz del repositorio:

```bash
# 1. Pruebas unitarias en debug y release (todas las invariantes matemáticas y de ingeniería)
nice -n 10 cargo test --workspace --manifest-path aura-engine/Cargo.toml -j 2
nice -n 10 cargo test --workspace --release --manifest-path aura-engine/Cargo.toml -j 2

# 2. Compilar binario release
nice -n 10 cargo build --release --manifest-path aura-engine/Cargo.toml -j 2

# 3. Microbenchmark de ingeniería Zero-Mul (E5)
nice -n 10 aura-engine/target/release/aura-exp bench 0xA400

# 4. Reproducir en un directorio separado (sin sobreescribir la evidencia archivada)
nice -n 10 python3 aura-engine/run.py DEV --workers 2 --output /tmp/aura-repro
nice -n 10 python3 aura-engine/run.py CONFIRM --workers 2 --output /tmp/aura-repro
nice -n 10 python3 aura-engine/summarize.py --raw /tmp/aura-repro --out /tmp/aura-repro/summary.json
```

Reportes y evidencia completa en [`docs/aura_v1/`](../docs/aura_v1/).
