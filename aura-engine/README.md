# `aura-engine` v2 — Arquitectura AURA + CorticalFabric desde Cero con Resolución de Cuellos de Botella de Entrenamiento

Implementación modular desde cero en Rust puro (**8 crates, 0 dependencias externas**, cero tokens BPE, inferencia de pesos lentos 100% en planos de bits ternarios 2-bit con AVX2 LUT-4 `vpshufb`) basada en [`docs/aura_v2/PROPUESTA.md`](../docs/aura_v2/PROPUESTA.md) y prerregistrada en [`docs/aura_v2/PLAN.md`](../docs/aura_v2/PLAN.md).

## Estructura del Workspace

```text
aura-engine/
├── Cargo.toml
├── run.py                  # Orquestador determinista DEV/CONFIRM con verificación SHA-256
├── summarize.py            # Bootstrap pareado (10,000 remuestreos, seed 12345, alpha=0.05)
├── freeze_and_verify.py    # Congelación FROZEN.json, replay bit-exacto y auditoría
└── crates/
    ├── aura-math/          # TernaryMatrix (LUT-4 AVX2 + tiling B=16), ScaledTernaryLayer (BitNet b1.58 STE + AdamW + histéresis), FastWeightMemory (NLMS zero-alloc AVX2), Jacobi effective_rank
    ├── aura-perception/    # EntropyPatcher v2 (corte pre-pico estilo BLT, fase/frontera, VICReg) + FixedPatcher + delta-modulación
    ├── aura-cortex/        # Blelloch O(log L) Parallel Scan SSM + DualCorticalCore + CorticalFabric v2 (CSR en baldosas, warm-start early-exit, NE/ACh con compuerta)
    ├── aura-world/         # Modelo de Mundo JEPA K=4 + EMA 0.996 + descomposición aleatoria heteroscedástica / progreso de aprendizaje epistémico
    ├── aura-hippocampus/   # HippocampusGraph L1 HNSW + persistencia binaria + ScaledTernaryMlp + SleepConsolidator v2
    ├── aura-metacog/       # Arbitraje Sistema 1 / Sistema 2 (EFE H=2) / AbstainAndAsk + compuerta Learning Progress
    ├── aura-decoder/       # Decodificador motor latente-a-byte (0 tokens, 256 bytes, BitNet 2-bit + adaptación online FastWeight + Hipocampo)
    └── aura-runtime/       # AuraAgent v2 + CLI `aura-v2` (B1–B6)
```

## Cómo compilar, probar y reproducir

Desde la raíz del repositorio:

```bash
# 1. Pruebas unitarias (debug y release)
nice -n 10 cargo test --workspace --manifest-path aura-engine/Cargo.toml -j 2
nice -n 10 cargo test --workspace --release --manifest-path aura-engine/Cargo.toml -j 2

# 2. Compilar binario release
nice -n 10 cargo build --release --manifest-path aura-engine/Cargo.toml -j 2

# 3. Microbenchmark B1 aislado (Silicon Mismatch / AVX2 LUT-4 / CSR en baldosas / NLMS)
nice -n 10 aura-engine/target/release/aura-v2 b1 0xB500

# 4. Reproducir barridos completos DEV y CONFIRM en directorio temporal (sin sobreescribir docs/aura_v2)
nice -n 10 python3 aura-engine/run.py DEV --workers 2 --output /tmp/aura_v2_repro
nice -n 10 python3 aura-engine/run.py CONFIRM --workers 2 --output /tmp/aura_v2_repro
nice -n 10 python3 aura-engine/summarize.py --raw /tmp/aura_v2_repro --out /tmp/aura_v2_repro/summary.json
```

Resultados, auditoría y evidencia cruda completa en [`docs/aura_v2/`](../docs/aura_v2/).
