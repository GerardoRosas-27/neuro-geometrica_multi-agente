# Arquitectura METAFE-3S-Dial — Consola Conversacional Neuro-Geométrica de Tres Escalas con Periferia Lingüística Congelada

## 1. Principio Rector: La Arquitectura Piensa, el LLM Solo Habla (y Es Auditado)

En los sistemas convencionales basados en LLM, el modelo de lenguaje decide la lógica, el estado de memoria y la confianza — por lo que alucina sin control interno. En **METAFE-3S-Dial** (`src/neuro_dialogo/` + `src/bin/neuro_consola.rs`), la separación de responsabilidades es estricta:

```text
Entrada usuario (texto libre)
   │
   ▼
[ 1. Periferia Sensorial Congelada (`encoder.rs`) ]
   ├─ Reutiliza `prompt_spider` (ambigüedad y fundamentación deterministas)
   ├─ Extrae proposiciones, reglas condicionales, cuantificadores, aritmética y geometría
   ├─ Codifica vector semántico continuo x_t ∈ R^64
   └─ Genera contrafactual automático x_cf ∈ R^64 (invirtiendo polaridad/premisas)
   │
   ├──► [ 2. Grafo de Creencias y Verificador Lógico (`belief.rs`) ]
   │       ├─ Memoria multi-turno + revisión explícita (RS-A)
   │       ├─ Cierre deductivo (Modus Ponens, Modus Tollens, Silogismos)
   │       ├─ Rechazo formal de falacias (afirmar consecuente / negar antecedente)
   │       ├─ Contradicción directa e inter-turno + Sandbox contrafactual
   │       ├─ Aritmética determinista + geometría real (`v4_dataset::apply_rule`)
   │       └─ Auto-modelo conceptual + abstención ante lo inverificable
   │
   └──► [ 3. Núcleo Cognitivo 3-Escalas (`core3s.rs`) ]
           ├─ Núcleo Base Lento W_0 (CONGELADO en diálogo)
           ├─ Dos Ramas Rápidas Independientes y Ortogonales: F_t^A ⊥ F_t^B en Π_ρ
           │    • Rama A (proponente): genera hipótesis z_A y postura p_st
           │    • Rama B (refutadora adversarial): busca z_B ≠ z_A con bajo error de ciclo Φ_rev(z_B) ≈ x_t
           ├─ Monitor ACC m_t ∈ R^4 (sorpresa, desacuerdo A vs B, bucle dogmático, saturación)
           ├─ Controlador Neuromodulador MetaNet GRU (α_t, λ_t, β_t, g_t)
           ├─ Señal anti-dogmática δ_t^real + Plasticidad 3-factores (E_t)
           ├─ Calibrador metacognitivo "Yo-FE 3S" (Platt L2 monotónico)
           └─ [ Solo offline / Sueño ] Buffer Episódico filtrado ──► EWC proximal (Fisher exacto Ω_i) a W_0
   │
   ▼
[ DecisionPacket (tesis, evidencia, postura epistémica, notas de refutación, telemetría) ]
   │
   ▼
[ 4. Periferia Lingüística Congelada + Guardia de Fidelidad (`periphery.rs`) ]
   ├─ Modo nativo (0 LLM, determinista), o Gemma 2 GGUF local (`FrozenGemma2Probe`),
   │  o API OpenAI-compatible (`/usr/bin/curl`, ej. Ollama / LM Studio / vLLM)
   └─ `verify_periphery_fidelity`: bloquea alucinaciones y rechaza cualquier intento
      del LLM de contradecir la postura epistémica decidida por la arquitectura
```

---

## 2. Solución al Cuello de Botella del Razonamiento Abierto sin Feedback Externo

### El Problema (Colapso Dogmático por Entropía)

En entornos con verificación objetiva (aritmética, lógica formal, compilación o simulaciones físicas), la señal de evaluación $\delta_t$ está anclada en la realidad. El límite crítico aparece en **razonamiento abierto sin feedback externo**: si el sistema define

$$\delta_t^{\text{naive}} = -\Delta H(p_t) + c\,(1 - H(p_t))$$

y actualiza sus pesos rápidos siguiendo el gradiente analítico exacto $-\nabla H(p_t)$ (equivalente a auto-entrenamiento ciego / *Tent*), descubre rápidamente que la forma más barata de reducir su entropía es **volverse dogmático**, colapsando hacia una sola postura con alta confianza falsa, baja diversidad y pérdida de reconstrucción de premisas.

### La Solución en `Core3S` (`DeltaMode::AdversarialCrossRefute`)

Sustituimos la minimización ciega de entropía por una señal de cuatro pilares que mide **coherencia real y resistencia a refutación**, incluso cuando `verdict = None` (cero feedback externo):

1. **Refutación Cruzada Adversarial con Dos Ramas Rápidas Ortogonales ($F_t^A \perp F_t^B$):**
   - Cada rama tiene su propia matriz de pesos rápidos y sesgo rápido $(F_t, b_t^f)$ proyectados dentro de la bola Frobenius $\sqrt{\|F\|_F^2 + \|b^f\|_2^2} \le \rho$.
   - Tras cada paso, la **Rama B** se ortogonaliza estrictamente vía Gram-Schmidt contra la **Rama A** ($\langle (F_t^B, b_t^{f,B}), (F_t^A, b_t^{f,A}) \rangle = 0$, verificado empíricamente con `branch_orthogonality = 1.000`).
   - La Rama B ejecuta $K=3$ pasos buscando una **contra-hipótesis** $z_B$ que explique la premisa ($E_{\text{cyc}}(z_B, x_t)$ bajo) mientras se aleja de $z_A$. Si B encuentra una explicación competitiva que contradice a A, la vulnerabilidad a refutación $V_{\text{ref}} = d(z_A, z_B) \exp(-2.2 \max(0, E_{\text{cyc}}^B - E_{\text{cyc}}^A))$ penaliza a $\delta_t$ y el controlador neuromodulador reduce la ganancia $\beta_t$.
2. **Consistencia de Ciclo Inverso ($\Phi_{\text{rev}}(z_t) \approx x_t$):**
   - Exige que la representación latente de respuesta conserve información suficiente para reconstruir la premisa de entrada; una respuesta genérica o dogmática destruye la invertibilidad y dispara $E_{\text{cyc}}$.
3. **Sensibilidad Contrafactual Automática ($\Delta_{\text{cf}}(x_t, \tilde{x}_t)$):**
   - En cada turno, la periferia sensorial genera automáticamente $\tilde{x}_t$ invirtiendo polaridad y premisas. Una arquitectura que responda igual ante $x_t$ y $\tilde{x}_t$ es insensible a la evidencia y recibe penalización directa en $\delta_t$.
4. **Detector de Bucle Dogmático (ACC) y Consolidación Filtrada con EWC Proximal:**
   - Si el estado interno $h_t$ se repite ($\cos(h_t, h_{t-k})$ alto) mientras las entradas cambian ($\cos(x_t, x_{t-k})$ bajo), el índice de dogmatismo dispara una caída en $\lambda_t$ (reset de $F_t^A$), bloquea la escritura en pesos rápidos y rechaza el episodio del Buffer Episódico.
   - Durante el sueño offline (`sleep_consolidate_ewc`), `W_0` se actualiza mediante un **paso proximal semi-implícito de Online EWC** sobre los 8 bloques de pesos y sesgos (`w_enc`, `b_enc`, `w_dec`, `w_rev`, `w_intent`, `b_intent`, `w_stance`, `b_stance`), usando la **diagonal exacta de la matriz de Fisher / Gauss-Newton esperada** $\Omega_i = \mathbb{E}_{x}\mathbb{E}_{y \sim p_\theta}[(\partial \log p_\theta / \partial W_i)^2]$ (que, a diferencia del Fisher empírico $(p - y)^2$, nunca colapsa a cero al converger y es incondicionalmente estable en entrenamiento infinito).

$$\delta_t^{\text{real}} = 0.75\,\Delta E_{\text{cyc}} + 0.55\,(\Delta_{\text{cf}} - 0.25) - 0.85\,V_{\text{ref}}(A \text{ vs } B) - 1.10\,\text{dogma}_t + 0.55\,\delta_{\text{sym}}$$

---

## 3. Auditoría y Reutilización del Repositorio Existente

| Componente existente auditado | Falla real detectada | Acción en `METAFE-3S-Dial` |
|---|---|---|
| `src/field_v4.rs` (`chat_field`) | Stub de 8 palabras clave + hash FNV sin comprensión ni memoria | Reemplazado por `encoder.rs` + `belief.rs` + `core3s.rs`; se **reutiliza** `parse_family` y `v4_dataset::apply_rule` para transformaciones geométricas reales |
| `src/prompt_spider.rs` | Scorer determinista de ambigüedad y fundamentación sin usar en diálogo | **Reutilizado directamente** en `encoder.rs` (`tokenize`, `PromptContext::new`, `score_token`) como telemetría sensorial |
| `src/field_gemma_probe.rs` + `src/native_gemma2.rs` | Probe Gemma 2 congelado sin lazo metacognitivo ni guardia anti-alucinación | **Reutilizado** como backend `PeripheryBackend::LocalGemma` bajo `verify_periphery_fidelity` |
| `gemma_campo/` y `campo_turno/` | Sin metacognición, sin EWC y sin refutación en abierto ($\eta=0$ en turno) | Resuelto con 3 escalas (`W_0` + `F_t^A \perp F_t^B` + EWC proximal + ACC + MetaNet GRU) |
| `orbita/src/metafe3s/` | Arquitectura validada 32/32 semillas solo en trayectorias geométricas 2D | Patrón completo (5 módulos biológicos + Yo-FE 3S) trasladado al dominio conversacional y ampliado con refutación dual ortogonal |

---

## 4. Uso de la Aplicación de Consola (`neuro_consola`)

```bash
# Compilar en release
cargo build --release --bin neuro_consola

# 1. Hablar en consola interactiva (con entrenamiento continuo en segundo plano opcional)
./target/release/neuro_consola chat --ckpt artifacts/neuro_consola/ckpt.json --bg-train

# 2. Entrenamiento infinito en consola (--cycles 0 = infinito, reanudable vía checkpoint atómico)
./target/release/neuro_consola train --cycles 0 --batch 28 --eval-every 10 \
    --ckpt artifacts/neuro_consola/ckpt.json --log artifacts/neuro_consola/train_stream.jsonl

# 3. Conectar un LLM congelado como periferia lingüística auditada:
#    a) Modelo local Gemma 2 GGUF
./target/release/neuro_consola chat --ckpt artifacts/neuro_consola/ckpt.json \
    --gguf ruta/gemma-2-2b-it.gguf --tokenizer ruta/tokenizer.json
#    b) API OpenAI-compatible (Ollama, LM Studio, vLLM, etc.)
./target/release/neuro_consola chat --ckpt artifacts/neuro_consola/ckpt.json \
    --api-url http://127.0.0.1:11434/v1/chat/completions --api-model gemma2:2b

# 4. Ejecutar ablaciones multi-semilla y demostración multi-turno
./target/release/neuro_consola eval --seeds 8 --cycles 20 --out artifacts/neuro_consola/eval_v2.json
./target/release/neuro_consola demo --ckpt artifacts/neuro_consola/ckpt.json
```

Comandos interactivos dentro de `chat`: `/why` (traza glass-box completa del último turno), `/stats`, `/train start`, `/train stop`, `/sleep` (consolidación EWC manual), `/correct <texto>`, `/ablation`, `/llm local | gguf <path> <tok> | api <url> <model> [key]`, `/clear`, `/save`, `/quit`.
