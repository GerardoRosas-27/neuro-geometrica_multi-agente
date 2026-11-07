# Proposed pull request (NOT submitted)
Repository: GerardoRosas-27/neuro-geometrica_multi-agente
Base: main
Head: exp/soft-bellman-v1
Title: experimento: Soft-Bellman Rust v1 con DEV+CONFIRM preregistrados

Body:
Añade un crate Rust aislado `soft_bellman/` con Soft-Bellman diferenciable de horizonte finito, métrica diagonal SPD y9 parámetros; no modifica la aplicación existente. Prerregistro3a1901f, congelación43fcd21. Entrenamiento real DEV16/16 y CONFIRM16/16;9/9 tests y replay determinista B400.

CONFIRM: éxito100%, acción óptima99.1430%, regret0.006907% frente a3.841467% del planificador uniforme; diferencia−3.834560pp, IC95%[−4.004103,−3.659101] (bootstrap pareado10000, seed12345). Se cumple el criterio primario únicamente en planificación inversa sintética de mapas pequeños; no demuestra lenguaje, geometría física ni IA general. Datos/logs/pesos y limitaciones en `docs/soft_bellman_v1/RESULTADOS.md`.

No ejecutar merge. Paso0/known_acc16/E33 no evaluados por ser un crate sin integración. Se solicita revisión, no promoción a producción.
