# Constelación v3 — Paso 0 (gates)

Comando: `cst3-paso0`, ejecutado dos veces en procesos separados (`artifacts/constelacion_v3/paso0_run{1,2}.json`). **Todas las gates PASS.**

| gate | resultado |
|---|---|
| 0a determinismo / regla (eta_p=0) | mismatch 0 de 31040 → PASS |
| 0b reproducibilidad entre procesos | data a35386552788de67, weights 5df90d6f7ed0d7bd, metrics 9f3aa474190bfa2a, idénticos en run1 y run2 → PASS |
| 0c estabilidad | finito; free-run 1000 pasos max‖·‖ 23.3; inferencia 1000 pasos max E 3.96; ρ_K max 0.990; ‖Jψ‖ ≤ 1.107 → PASS |
| 0d ÓRBITA congelada | corr media 0.99999 → PASS |
| 0e coste | 10496 pesos, 76 parámetros fijos por neurona, 216 µs por punto (train), 185 µs (eval) |
| 0f v3 apagado ≡ v2 bit a bit | weights 8e14c5880d854d46, metrics a139b204454bcfbc, igual que la referencia v2 → PASS |
| 0g cabeza cero ≡ persistencia (análogo de γ=0 ≡ ÓRBITA) | mismatch 0 de 3072 → PASS |
