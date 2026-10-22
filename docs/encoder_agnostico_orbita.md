# Encoder entrenable y agnóstico al lenguaje para ÓRBITA — análisis y diseño

**Fecha:** 2026-10-16 (CDMX). **Tipo:** sólo análisis y diseño; no se entrenó nada. **Rama:** `exp/orbita-llm` (sobre 5827f42). Todas las cifras citadas provienen de documentos/commits del repo indicados; no hay cifras nuevas.

## 0. Punto de partida: diagnóstico de Fase 0 (5827f42, `docs/resultados_orbita_llm.md`)
- ÓRBITA-v1 congelada + P ridge (Gemma capa 13/26, media de tokens) → **acc 0 en todos los horizontes**; mediana rel_err h1 ≈ 0,94 (DEV) / 1,00 (CONFIRM) frente a 0,0059 / 0,0055 con c verdadero. H-L1 refutada; H-L2/H-L3/H-C1 inconclusas.
- argmax de la parte de familia de ĉ acierta 0,7415 (DEV) / 0,7507 (CONFIRM). Es decir, el texto **sí** transporta la familia, pero ĉ continuo cae fuera de la variedad de contextos con que se entrenó ÓRBITA (one-hot exacto + p ∈ [−1,1]) y el núcleo amplifica el error desde h1.
- Conclusión de diseño: el problema principal no es "más capacidad" sino **(i) que la salida del encoder viva en la variedad de contextos** y **(ii) que se entrene contra lo que importa (la trayectoria), no contra c por MSE**.

## 1. Inventario de encoders en el repo (read-only)

| Componente | Ruta / rama / commit | Entra | Sale | Dim. | ¿Entrenado? |
|---|---|---|---|---|---|
| `FieldEncoder` (encoder-sonda) | `src/field_encoder.rs`, `exp/encoder-campo` 8685519 (también `exp/campo-sin-tokens` c85b815); doc `docs/historico/encoder_campo_sin_tokens.md` | texto → `features_from_text`: huella determinista de 26 "capas" (sin GGUF) + 16 n-gramas, con `LayerSkipMask` (T2.1 apaga 2,3,4,5,17) | 12 fasores (`Vec<Phasor>`) del campo | 26+16 → 12 | Sí, regla delta/Hebb/Equilibrium Propagation hacia atractores de cluster; sólo en tests (margen +0,526, 120 pasos). No usa Gemma real |
| `LinguisticPacket` / `LinguisticFieldCodec` / `GemmaShapedLexicon` | `src/field_linguistic_layer.rs`, `exp/encoder-linguistico` a0eff2a; doc `docs/historico/capa_linguistica_gemma.md` | texto → sonda (`FrozenLinguisticProbe`) | `into_field_features()` = [26 RMS por capa; hidden; stem-bag], normalizado, **tira los IDs** → proyector entrenable → 12 fasores | 26 + 64 (sonda) + stem | Proyector sí (regla delta); el resultado publicado (cluster 0,33→1,00) es con la sonda `gemma-shaped-lexicon`, **sin GGUF** |
| `FrozenGemma2Probe` | `src/field_gemma_probe.rs` (en `main` 49c00a9) | GGUF real | genera/choice_probs; arma `LinguisticPacket` con traza RMS + `sequence_hidden` | 2304 | Congelado |
| N-gramas | `FieldEncoder` (16-D) y baseline NGRAM de Fase 0 (`orbita/llm/pipeline.py`, hash char 1–3 → 2304) | texto | vector | 16 / 2304 | No (Fase 0: ridge encima, acc 0) |
| Embeddings v4 E43/E44 | `examples/v4_e43_e44.rs`, `artifacts/v4/lang/emb_{gemma,qwen}.json.gz`, `artifacts/v4/psi64/lang_task.json` (`exp/field-autonomy-v4`); resultados en `docs/resultados_v4.md` §11 | conceptos es/en/fr/ja | hidden final medio Gemma-2-2B Q3_K_L / Qwen2.5-1.5B | 2304 / 1536 | Congelado; Dφ encima. **E43/E44 FAIL** (2/16, 3/16); el doc anota "no se exploraron capas intermedias" |
| Ψ64 bottleneck (token-free E45) | `exp/token-free-liquid-field` b71130a, `docs/token_free_liquid_field/resultados_token_free_e45_e46.md`; extractor `examples/tfl_gemma_embed.rs` | hidden (512 proxy / 2304 Gemma en E46b) | Ψ ∈ ℝ⁶⁴ | 64 | Sí: encoder lineal/MLP entrenado 64-D 0,970–0,984, fuga de superficie R² 0,06; **con encoder proxy aleatorio**, no Gemma (limitación declarada); E47: <64 colapsa |
| "Encoder del campo Dφ de v4" | `src/field_autonomy_v4.rs` / `v4_controls::Dphi` | (punto q, contexto c ∈ ℝ¹⁰) | punto siguiente | — | No es un encoder de texto: Dφ consume c tal cual (one-hot + 2 parámetros). No hay módulo que traduzca texto→c en v4 fuera de E43/E44 |

## 2. Gemma "desestructurado" (capas accesibles por separado)
En `src/native_gemma2.rs` (`main`; heredado de `exp/encoder-linguistico`, `src/adaptive_gemma2.rs`, binarios `native_gemma2_*`, y `docs/historico/gemma2_runtime_optimization.md` con rutas de capas V8):
- `QuantizedGemma2::forward_with_mask(token_ids, position, mask: Option<&LayerExecutionMask>, exit_after: Option<usize>, capture_trace, capture_sequence_hidden)` → `Gemma2ForwardOutput { logits, trace: Gemma2ForwardTrace (RMS entrada/salida/delta por capa, ventana local/global), sequence_hidden: [batch, seq, 2304] }`.
- `LayerExecutionMask::{all, from_enabled}`: ejecutar/saltar capas individuales (26); `exit_after = k` corta tras la capa k (Fase 0 usó k = 12).
- `prefill_teacher_hiddens(&[u32]) -> Vec<Vec<f32>>`: estado oculto por token.
- `Gemma2Tokenizer::{from_gguf, encode, decode}`.
**Conectable directamente al latente de ÓRBITA:** sí, como periferia: `sequence_hidden` de una capa intermedia (por token, no sólo media) → adaptador entrenable → contexto en la variedad (o z). Lo que *no* existe: retropropagar a través de Gemma (candle cuantizado, sin gradientes en el repo) — Gemma queda congelado y sus salidas se cachean. Coste medido de extracción en este box (prerregistro Fase 0): 8,48 s/texto a 13 capas con la CPU ocupada por otro proceso; ~3–4 s/texto por proceso con 3 en paralelo y la CPU más libre.

## 3. ¿Sirve el encoder del campo como puente?
Parcialmente. `FieldEncoder`/`LinguisticFieldCodec` aportan el **patrón de frontera** (los tokens mueren en `into_field_features`) y un entrenamiento local sin next-token, pero su salida son 12 fasores de otro sustrato (`FieldState`), no c ni z de ÓRBITA, y nunca se validaron con Gemma real. Lo reutilizable: (a) la frontera de tipos (ÓRBITA no debe ver IDs), (b) el vector `layer_rms` de 26 capas como feature barata, (c) la lección de Ψ64: un bottleneck **entrenado** descarta superficie léxica (R² 0,06) mejor que proyecciones fijas. Lo no reutilizable: la dinámica de fasores/Hopfield y su regla Hebb (objetivo distinto).

## 4. Diseños propuestos (rankeados)

Notación: $h_\ell(t)\in\mathbb R^{T\times2304}$ estados de Gemma en la capa $\ell$ para el texto $t$; ÓRBITA $\Phi(x,c)$; $c=[\mathbf 1_f;\,p]$, $p\in[-1,1]^2$; trayectoria $x_k=\Phi^k(x,c)$.

### D1 (recomendado). Encoder entrenable con **cuello de botella en la variedad de contextos** + pérdida de trayectoria + invariancia entre idiomas, ÓRBITA congelada
$$\pi=\mathrm{softmax}(W_f\,u/\tau),\quad \hat p=\tanh(W_p\,u),\quad u=\mathrm{MLP}\big(\mathrm{pool}(h_{\ell_1},h_{\ell_2},h_{\ell_3})\big),\qquad \hat c=[\mathrm{ST}(\pi);\hat p]$$
con ST = *straight-through* (one-hot en el forward, gradiente de $\pi$ en el backward) o Gumbel-softmax con $\tau\to0$. Así $\hat c$ **siempre** es un contexto válido para ÓRBITA (corrige la causa diagnosticada).
Pérdida:
$$\mathcal L=\underbrace{\mathrm{CE}(\pi,f)+\lambda_p\|\hat p-p\|^2}_{\text{supervisión}}+\lambda_T\sum_{k\in\{1,2,4,8\}}w_k\frac{\|\Phi^k(x,\hat c)-\mathrm{sg}[\Phi^k(x,c)]\|^2}{\|\Phi^k(x,c)\|^2}+\lambda_I\,\mathrm{InfoNCE}\big(u(t^{es}),u(t^{en}),u(t^{fr})\big)$$
El término de trayectoria usa el ÓRBITA PyTorch existente (`orbita/train_orbita.py`, paridad Rust ~1e-15) con pesos congelados; el objetivo $\Phi^k(x,c)$ con c verdadero se detiene (sg), igual que el patrón JEPA del plan. Capas: reutilizar la caché de Fase 0 (capa 13, media) en el primer experimento; añadir capas {6, 13, 20} sólo si H-E1 falla por precisión de $\hat p$.
- **Implementación:** script nuevo `orbita/llm/train_encoder.py` (PyTorch, CPU) sobre embeddings ya cacheados en `/workspace/orbita-data/llm/` (pool 768 + DEV/CONFIRM 1536 c/u); el Rust `llm-eval` existente evalúa sin cambios (recibe ĉ por CSV).
- **Coste CPU:** sin extracción nueva para el experimento mínimo; el entrenamiento es de un MLP pequeño (~2304×128 + cabezas) más rollouts de ÓRBITA de 8 pasos por lote → del orden de minutos a pocas decenas de minutos por seed (estimado, no medido).
- **Riesgo:** medio. La precisión de $\hat p$ desde la media de tokens puede ser insuficiente (los números son pocos tokens en ~37); ÓRBITA puede ser muy sensible a p. Riesgo de que InfoNCE colapse $u$ si $\lambda_I$ domina.
- **Experimento mínimo:** (0) curva de sensibilidad de ÓRBITA-v1 a ruido en p con familia correcta (sin entrenar; con `llm-eval` y c verdadero + ruido gaussiano σ ∈ {0,005; 0,01; 0,02; 0,05}) → fija la precisión necesaria; (1) D1 sobre caché capa 13, DEV → CONFIRM.

### D2. Adaptador por **atención sobre tokens** de capas intermedias de Gemma (mismo cuello y pérdidas que D1)
$$u=\sum_{j}\alpha_j\,W_v h_{\ell}(t)_j,\qquad \alpha=\mathrm{softmax}_j\big(q^\top W_k h_\ell(t)_j\big)$$
(una o pocas *queries* entrenables por salida: familia, p₁, p₂), con $\ell$ intermedia vía `exit_after`. Ataca directamente el cuello probable de D1: la media diluye los tokens numéricos.
- **Implementación:** extender `orbita gemma-embed` para volcar `sequence_hidden` por token de 1–3 capas (actualmente sólo media); mismo `train_encoder.py` con módulo de atención.
- **Coste CPU:** extracción nueva de ~3840 textos a 3–8,5 s/texto según carga (medido en Fase 0) → varias horas; almacenamiento por token en f32 ≈ T × 2304 × 4 B por capa y texto (con T ≈ 37 tokens medidos en el benchmark: ~0,34 MB/capa/texto). Entrenamiento: minutos–decenas de minutos (estimado).
- **Riesgo:** medio-alto (más parámetros que datos: 768 textos de pool; sobreajuste a plantillas). Mitigación: más plantillas/parafraseos y split por plantilla como en Fase 0.

### D3. **Entrenamiento conjunto** encoder + ajuste fino ligero de ÓRBITA (cabeza de contexto)
Igual que D1/D2, pero desbloqueando sólo `head(c)` de ÓRBITA (que mapea c → ω, s, A, β, b_k, a_k) con LR pequeño y la pérdida multi-horizonte original sobre TRAIN más el término de trayectoria con $\hat c$. El núcleo espectral y el corrector siguen congelados.
- **Coste:** como D1 + un re-entrenamiento parcial de ÓRBITA por seed (el entrenamiento completo de ÓRBITA tarda ~1 h por ronda de 8 seeds sin carga; uno parcial debería ser menor, no medido).
- **Riesgo:** alto: rompe la comparabilidad con ORBITA-v1 prerregistrada y puede degradar el acierto con c verdadero; exige reportar acc(TRUE) antes/después. Sólo si D1 y D2 fallan por la sensibilidad de ÓRBITA (no por el encoder).

**No recomendado:** cuantización VQ libre (códigos aprendidos sin anclar a familias): la variedad de contextos ya es conocida (8 familias × [−1,1]²); un VQ genérico reintroduce el problema de salir de distribución. Fine-tune de Gemma: no hay gradientes en el loader candle del repo y es inviable en CPU.

## 5. Hipótesis falsables (a prerregistrar en un documento nuevo antes de correr; δ y umbrales se fijarán allí)
Protocolo como Fase 0: mismas 16+16 seeds, mismos 16 ejemplos TEST iterables × 6 textos (plantillas 2–3 no vistas), IC95 bootstrap pareado (10 000, semilla 12345), DEV → CONFIRM sin cambios, 0 queries.
- **H-E0 (sensibilidad, sin entrenar):** existe σ_p* tal que con familia correcta y ruido gaussiano σ ≤ σ_p* en p, acc16 no cae más de δ respecto a TRUE. Define el objetivo de precisión del encoder.
- **H-E1 (D1 vs Fase 0):** acc16(D1) − acc16(GEMMA Fase 0) con IC95 > 0 en DEV y CONFIRM.
- **H-E2 (D1 vs TRUE, el objetivo real):** acc16(D1) − acc16(TRUE) con IC95 inferior > −δ (repite H-L1 con el encoder nuevo).
- **H-E3 (agnóstico al lenguaje):** para los tres pares de idiomas, IC95 de Δacc16 dentro de ±δ **y** acc16(D1) > acc16(RAND) con IC95 > 0 (evita la aceptación degenerada de Fase 0, donde todo era 0); D_same16 − D_diff16 con IC95 < 0.
- **H-E4 (control de superficie):** acc16(D1 sobre Gemma) − acc16(D1 sobre NGRAM, mismo cuello y pérdidas) con IC95 > 0. Si falla, la "agnosticidad" la da la plantilla, no Gemma.
- **H-E5 (D2 vs D1):** sólo si H-E2 falla con error dominado por $\hat p$: |p̂ − p| mediano de D2 menor que el de D1 con IC95 < 0, y acc16 mayor.
- **H-C1 (repetición):** con acc16 > 0, AUROC de `teacher_div16` para fallo de ÓRBITA − 0,5 con IC95 > 0 (en Fase 0 no era evaluable por fallo del 100 %).

## 6. Recomendación
Orden: **H-E0 (minutos, sin entrenar) → D1 con la caché de Fase 0 → D2 sólo si la precisión de p es el cuello → D3 sólo si ÓRBITA es el cuello.** La pieza crítica de D1 no es el tamaño del encoder sino el **cuello en la variedad** (one-hot ST + tanh) y la **pérdida de trayectoria** contra ÓRBITA congelada, que atacan directamente la causa medida en Fase 0. Los encoders de campo existentes (`FieldEncoder`, `LinguisticFieldCodec`) no se conectan tal cual; se reutiliza su frontera "los tokens mueren en la periferia" y la evidencia de Ψ64 de que un bottleneck entrenado elimina superficie léxica.
