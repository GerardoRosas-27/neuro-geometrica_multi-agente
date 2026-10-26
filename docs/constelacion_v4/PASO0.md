# Constelación v4 — Paso 0 (gates)

Comando: `cst4-paso0 --seed 0xC400`, ejecutado dos veces en procesos separados (`artifacts/constelacion_v4/paso0_run{1,2}.json`). **Todas las gates PASS.**

| gate | resultado |
|---|---|
| 0a determinismo / regla | mismatch 0 de 31040 → PASS |
| 0b reproducibilidad entre procesos | data 83cca53a3826e7e8, weights b416a6632742d1a9, metrics 708f457e946fa06e; run1 = run2: True → PASS (los hashes cambian frente a v3 porque los flujos y t_f son mayores) |
| 0c estabilidad | max‖·‖ free-run 23.3; max E 3.96; ρ_K 0.990; ‖Jψ‖ ≤ 1.107 → PASS |
| 0d ÓRBITA congelada | corr media 0.99999 → PASS |
| 0e coste | 10496 pesos; 239 µs/punto (train), 205 µs (eval) |
| 0f v3 apagado ≡ v2 bit a bit | weights 8e14c5880d854d46, metrics a139b204454bcfbc = referencia v2 → PASS |
| 0g cabeza cero ≡ persistencia | mismatch 0 de 3072 → PASS |
| **0h (nueva) `cst4-run --v3` ≡ binario v3 `cst3-run`** | 0xC400, brazo cst, N_A=768, N_B=48: hash de la red 9482fc2b40674f98 idéntico; held y E33 idénticos; además `e33_v3head` de v4 = E33 de v3 → PASS |
