# El «yo» y el LLM que lo observa: búsqueda del plan previo y propuesta (sin construir)

## 1. Búsqueda (read-only, 63 ramas remotas, 2026-10-17 23:3x CST)
`git grep -i` en todas las ramas (md/rs/py, incluidos docs/historico) de: primer/segundo orden, first/second/higher order,
self-model, automodelo, metacognición, «yo», self-awareness, introspección, observer/observador, SelfState.

- **No existe** ningún «plan de primer orden y segundo orden» ni las expresiones primer/segundo orden / first/second-order /
  higher-order en ninguna rama. No hay diseño previo de un «yo» en la pila ÓRBITA/encoder/LLM.
- Lo único relacionado:
  - `src/bin/cdt_rqm_self_awareness_validation.rs` (rama `feature/sustrato-termo-dinamico`, commit 1a3b6ab «se elimina SNGA»; rama en
    17efe0d, 2026-07-07), listado en el README de esa rama. Sobre el sustrato CDT-RQM pre-v4: un `SELF_OBSERVER` (ObserverId 190_999)
    refuerza relaciones de un «self cue» (derivado del estado del sustrato) a un «self target»; `self_score` = puntuación del colapso
    sobre esos targets. Decisión impresa: conservar como métrica si `self_score` sube sin degradar la tarea, si no
    «discard value=self_loop_adds_cost_or_no_metacognitive_signal». Incluye también Φ («integrated information») y Orch-OR. No hay doc
    de resultados ni lo usa la pila actual; es otro sustrato (RQM con observadores).
  - «observador» en `docs/paper.md` y `src/relational_field.rs` (`ObserverId`) = contexto relacional RQM, no un auto-modelo.
  - «introspección» aparece sólo en `docs/arquitectura_liquido_cdt_memoria.md` (soft-recall para tests), irrelevante.

## 2. Interpretación propuesta (a aprobar)
- **1er orden (mundo):** texto → Gemma capa 13 → D2 + numnorm → ĉ → ÓRBITA-v1 predice la trayectoria. (Existe y está validado: e2e 925015f.)
- **«Yo» (auto-modelo de ÓRBITA):** un vector de estado interno legible por episodio, sólo con señales que ÓRBITA/encoder ya producen:
  familia elegida y su margen (softmax D2), `parse_ok`/fuente de p (numnorm vs fallback D2), sorpresa contra observación
  r = rel_err(ÓRBITA(x0), y1) cuando hay observación, norma/estabilidad del rollout (σ_max del jacobiano, ratio de norma) y una
  **confianza predicha** p̂(acierto h) dada por un calibrador pequeño entrenado en TRAIN/DEV (no TEST).
  Evidencia previa que lo condiciona: el residuo teacher-ÓRBITA **no** predijo fallos (H-C2b inconclusa), mientras que la sorpresa contra
  observación sí separa fallos (AUROC de ORACLE_ORB 0,8649 y BASE 0,8852 en `artifacts/orbita/llm_cdt2/analyze_confirm.log`).
- **2do orden (LLM observa al yo):** Gemma recibe el estado del yo serializado (plantilla) y responde preguntas sobre ÓRBITA
  («¿qué crees que hará?», «¿estás seguro?», «¿entendiste el número?»), sin ver la respuesta correcta. Se mide con lector objetivo
  (numnorm + palabras clave), como en E4.

## 3. Hipótesis falsables (δ = 0,05; IC95 bootstrap pareado; 16 seeds DEV → CONFIRM; split por plantilla; 0 queries TEST)
- **H1 (yo calibrado):** AUROC(p̂ → acierto h16) − max(AUROC margen D2 solo, AUROC r solo) IC95 > 0; ECE del yo < ECE del margen D2.
- **H2 (el yo sirve):** abstenerse/pedir observación cuando p̂ < umbral mejora acc en lo respondido a cobertura fija frente a umbral sobre
  margen D2 y frente a abstención aleatoria con la misma cobertura.
- **H3 (2º orden fiel):** la frase de Gemma, interpretada por el lector, recupera familia/confianza del yo mejor que controles:
  yo barajado entre episodios (SHUF), yo nulo (NOYO) y Gemma sin el yo; criterio: fidelidad − max(controles) IC95 > 0.
- **H4 (2º orden aporta algo más que leer):** Gemma con el yo predice el fallo de ÓRBITA mejor que el propio p̂ (si no, el 2º orden
  sólo verbaliza; se declararía así).
- Riesgo/declaración: Gemma 2 2B en f32 necesita ≈10 GB de RAM; a las 23:4x CST había 5 GB disponibles en el box (free -g) → H3/H4 requieren verificar memoria
  o usar el binario `gemma-embed` en modo generación (no existe aún: habría que añadir decodificación).
