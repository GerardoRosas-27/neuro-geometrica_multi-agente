# Comparativa de decisión: v4 (`exp/field-autonomy-v4`) vs token-free (`exp/token-free-liquid-field`)

Fecha: 2026-10-10. Documento idéntico en ambas ramas. Ninguna rama se mergea a `main`.

Resultados completos:
- v4: [`docs/resultados_v4.md`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente/blob/exp/field-autonomy-v4/docs/resultados_v4.md) (E31–E45, seguimientos 1–4, §12 resumen final) · artefactos `artifacts/v4/`.
- token-free: [`resultados_token_free_e45_e46.md`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente/blob/exp/token-free-liquid-field/docs/token_free_liquid_field/resultados_token_free_e45_e46.md) (ciclo 1) y [`resultados_token_free_ciclo2.md`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente/blob/exp/token-free-liquid-field/docs/token_free_liquid_field/resultados_token_free_ciclo2.md) (E46b, E47, E48, H2H) · artefactos `artifacts/token_free/` (caché pesada fuera del repo, manifiesto `cache_manifest.sha256`).

## 1. Alcance de cada línea

| | v4 (field-autonomy) | token-free liquid field |
|---|---|---|
| Hipótesis | Experiencia episódica (CDT) consolidada fuera del hot path cambia de forma persistente la dinámica Dφ, que responde sin memoria | Un campo continuo Ψ (sin tokens) con dinámica "líquida" supera a otras dinámicas y puede servir de núcleo de razonamiento |
| Núcleo | Dφ residual por punto, pesos compartidos (962 params), Rust f64 | Ψ de 64-D desde encoder congelado + dinámica (mlp/liquid/liquid_ss/ssm/linear/v4res, ~33k params), JAX → core Rust f32 `tfl-v1` |
| Tareas | Generador sellado de transformaciones geométricas (8 familias), lenguaje (E43/E44) | `tfl-gen-v1` (permutaciones, lineal), `tfl-gen-v2` (Z₁₀³ no lineal, textos en español → Gemma) y la tarea sellada de v4 (H2H) |
| Protocolo | Pre-registro, DEV 16 + CONF 16 seeds, auditoría de leakage/queries | Pre-registro, DEV 3 + CONF 3 seeds, paridad JAX↔Rust |

## 2. Cara a cara

| Eje | v4 | token-free | Ganador |
|---|---|---|---|
| Tarea E35 (v4 sellada), acc h1 rel_err<5 % (DEV / CONF) | A 0.231 / 0.248; **B 0.339 / 0.377** (genérico: 0.360 / 0.389) | mejor (ssm B) 0.0005 / 0.0010; mlp y liquid_ss 0.000 | **v4** (Δ −0.338 [−0.382, −0.293] DEV; −0.377 [−0.421, −0.329] CONF) |
| Composición 2 pasos (H2H) | B 0.059 / 0.059 | 0.000 | v4 |
| Ranking de dinámicas E46 (lineal, eje 1 CONF) | — | lineal 0.862 ≈ ssm 0.859 > mlp 0.727 > liquid 0.407 | liquid último |
| Ranking E46b (Gemma, no lineal, eje 1 CONF) | v4res (análogo Dφ) 0.291 | mlp 0.298 ≈ v4res 0.291 ≈ ssm 0.279 > liquid 0.221 > liquid_ss 0.181 > linear 0.127 | empate mlp/v4res/ssm; liquid FAIL |
| Líquido vs consolidado (E45-v4) | T1 CDT 0.808 / 0.802 (rule), 0.482 / 0.504 (gen); líquidos ≤0.07; gates 0/16 | — | CDT/consolidación |
| Bottleneck 64-D (E45 token-free) | — | MLP 2304/512→64: 0.984 / 0.983 / OOD 0.979, fuga superficie R² 0.04 (encoder proxy); 64-D necesario y no esparsificable (E47) | resultado positivo de token-free |
| Horizonte largo | E33 FAIL (h4 ≈0); E33S FAIL | LONG h16 ≤0.13 en E46b; sólo ssm en E46 lineal (0.77 h16) | ninguno |
| Sparse / parada adaptativa | iteración adaptativa E45a FAIL | E47 FAIL (k<64 colapsa eje 1 a ≤0.03), E48 FAIL (adaptativo más lento y peor que K=4) | ninguno |
| Lenguaje real | E43/E44 FAIL (Dφ es 0.17–0.19, < OFFSET lineal) | Gemma → Ψ64 decodifica estados ~perfecto; dinámica sobre él ≤0.30 eje 1 | ninguno sólido |
| Paridad Rust | Rust nativo (sin port) | PASS ciclo 1 (24/24); ciclo 2 discreto 100 %, PORT_PARITY_FAIL a h16 en mlp/ssm/liquid_ss (deriva f32) | infra token-free |
| Latencia | Dφ ~9.2 µs/consulta (harness f64, CPU ocupada) | 0.4–2.1 µs/consulta H2H; 10–45 µs/paso E46b | token-free más rápido, pero con acc≈0 en la tarea v4 no vale |
| Tiempo de entrenamiento / tooling | E35 A+B 3.7 s/seed, todo Rust | 0.4–0.8 s/seed H2H; JAX `lax.scan`+`vmap` 1.67×; cuello real = extracción Gemma (2749 s) | comparable; JAX más cómodo para iterar arquitecturas |

## 3. Qué está sólidamente respaldado y qué no

Sólido (replicado DEV + CONF, pre-registrado):
- La consolidación transfiere información persistentemente a los pesos de Dφ sin rutas de memoria en TEST (E35–E38, leakage 0), también con teacher genérico.
- En la tarea de v4, Dφ por punto con pesos compartidos supera por ~0.34–0.38 a cualquier modelo token-free de igual tamaño y presupuesto.
- La regla consolidada (CDT/T1) domina a los sustratos líquidos (E45-v4); la dinámica líquida pierde en E46, E46b y H2H (y E47/E48 no la rescatan).
- Un Ψ de 64-D entrenado es un bottleneck relacional eficaz (E45 token-free) y no se puede reducir ni esparsificar sin pérdida (E45 <64, E47).

No respaldado:
- Que el campo descubra regularidades por sí mismo; ventaja de "dinámica" sobre MLP de igual capacidad (E34, E46b); horizontes largos; transferencia lingüística/swap de LLM; ventaja alguna de lo "líquido".

## 4. Salvedades

- La consolidación de v4 es **destilación de un teacher** de reglas (0.83) o genérico RFF (0.89–0.90); Dφ captura sólo ~40 % de lo que el teacher sabe.
- E33 (rollout largo) falla en v4; el cuello es el error de un paso (~8 %), no la amplificación.
- E45-v4: la CDT/consolidación gana a los líquidos, pero la comparación favorece estructuralmente a T1 (clase de hipótesis ≈ generador).
- Token-free: E45/E46 con encoder proxy aleatorio; E46b con Gemma real pero sólo 3+3 seeds; en H2H los modelos sobre el objeto completo ni siquiera ajustan TRAIN (sesgo inductivo, no "tokens", explica la diferencia).
- La línea líquida falla 3 pruebas pre-registradas (E46, E46b, H2H) más E47/E48.
- Latencias medidas en harnesses distintos (f64 vs f32, CPU compartida): no comparables con precisión.

## 5. Decisión y hoja de ruta (priorizada)

**Continuar en v4.** Token-free aporta infraestructura y el bottleneck de 64-D, no una dinámica mejor.

1. **v4: Dφ por punto + consolidación CDT** como núcleo. Objetivo inmediato: subir la captura del teacher consolidado (de ~40 % hacia >70 %): capacidad de Dφ, pérdida de destilación directa sobre el teacher, más replay; gate: B ≥ 0.50 en E35 manteniendo leakage 0.
2. **Reducir el error de un paso de Dφ por debajo del 5 %** y entonces re-proyección al manifold (V1, que ya funciona en v3: cos h64 0.45 → 0.75) para horizontes largos; repetir E33 pre-registrado.
3. **Ψ de 64-D de Gemma (E45 token-free) como entrada a Dφ de v4** para E43/E44, sustituyendo el pooling medio crudo; comparar contra OFFSET lineal.
4. **Conservar de token-free**: core Rust sin tokens (`tfl-v1`), runner de paridad JAX↔Rust, exportación de parámetros, puente H2H y pipeline JAX (`lax.scan`/`vmap`) como banco de prototipado.
5. **Abandonar la línea líquida** como candidata; mantener `liquid_ss` sólo como control negativo en futuras baterías. No volver a probar sparse top-k/parada adaptativa sobre Ψ64 sin reentrenamiento consciente de la máscara.
