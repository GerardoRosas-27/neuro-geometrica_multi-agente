# Fase 0 — LLM → ÓRBITA → CDT-RQM-EPR (investigación; sin entrenar)

Fecha: 2026-10-15 (CDMX). Sin cifras nuevas: todo número viene del repo o de las tablas ya publicadas.

## 1. Qué existe
**LLM.** Gemma 2 2B instruct GGUF, congelado, en CPU vía candle (`Cargo.toml` de `main` 49c00a9, features `cuda`/`mkl`). Archivos en el box: `/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-{Q2_K,Q3_K_L,Q4_K_M}.gguf` (1.2–1.7 GB). Código: `src/field_gemma_probe.rs` (`FrozenGemma2Probe::try_open`, `forward_with_mask`, `sequence_hidden`) y `src/field_linguistic_layer.rs` (`LinguisticPacket` → `into_field_features()` descarta los IDs de token → proyector entrenable → z). Documentado en `docs/historico/capa_linguistica_gemma.md` (rama `exp/encoder-linguistico` a0eff2a): el resultado publicado ahí (cluster 0.33→1.00) es con la sonda *sin GGUF* (`gemma-shaped-lexicon`), no con Gemma real. Otros: Ollama `gemma2:2b` (sólo benchmarks, `docs/historico/gemma2_runtime_optimization.md`), proveedores OpenAI-compatibles (`LLM_API_BASE`, `docs/deploy_railway_agentic.md`), TinyLlama mencionado en `ollama-models/`. En el box hay además GGUF Qwen (2.5-1.5B, 3.5-0.8B/2B/9B en `/workspace/models`, `/workspace/vol-test`) que no usa el repo.
**CDT-RQM-EPR.** Es el propio crate `cdt_rqm_epr` (harness v4): `generate_and_seal` (`src/v4_dataset.rs:442`), `collect_experience` → `CdtStore` (`src/field_autonomy_v4.rs:298`), `consolidate` → `LearningSignal` (teacher por familia, :567), `family_fit_residual` (`src/v4_distill.rs:714`). "Lo consolidado" = el teacher por familia + máscara *trusted*. La "línea A CDT-RQM-EPR" histórica está en `docs/historico/`. Rama `feat/live-train-llm-cdt` (ea3ba78) añade dataset LLM + CDT por lotes.
**Box.** 8 CPU, 15 GB RAM (~8 GB libres hoy), sin GPU, 78 GB disco. `run_v6` y v5 de otro agente ocupan ~7 núcleos.

## 2. Diseño
Estado del mundo = el mismo de E33 (nube de puntos + contexto c de la familia). Una descripción textual t del estado/contexto (p. ej. "rotar 30° y reflejar sobre x") reemplaza a c.
1. **Encoder:** $e=\mathrm{pool}_\ell(\mathrm{Gemma}(t))\in\mathbb R^{2304}$ (hidden de la capa $\ell$, congelado, precomputado una vez) → $\hat c=P e$, $P$ lineal o MLP pequeño (~2304·10 parámetros).
2. **Acople:** ÓRBITA-v1 congelada (pesos publicados) recibe $\hat c$ en vez de $c$; sólo se entrena $P$.
3. **Pérdidas:** (a) JEPA/predictiva: free-run de ÓRBITA con $\hat c$ vs con $c$ verdadero, $\sum_k w_k\|z_k(\hat c)-\mathrm{sg}[z_k(c)]\|^2$; (b) contrastiva InfoNCE entre parafraseos/idiomas del mismo $c$ y entre $c$ distintos.
4. **CDT sobre ÓRBITA:** correr `collect_experience`/`consolidate` usando trayectorias de ÓRBITA (en z o decodificadas) como experiencia; preguntar qué familia/transformación infiere el teacher consolidado y si su residuo (`family_fit_residual`) detecta los estados donde ÓRBITA diverge. Requiere un adaptador ÓRBITA→`Example` (código nuevo en `orbita/`, sin tocar v4).

## 3. Datos
Plantillas de texto por familia × parámetros, ≥ 5 parafraseos × 2–3 idiomas (es/en/fr), generados por plantilla (determinista) y, opcional, reescritos por el LLM; split por plantilla para que TEST tenga redacciones nunca vistas. Embeddings precomputados en disco (archivo pequeño).

## 4. Experimento mínimo en CPU
Una seed DEV, ~200 descripciones; Gemma Q4_K_M sólo en forward (sin generación); entrenar $P$ (minutos). Coste dominante: el forward de Gemma; no hay una cifra medida para encoding en este box (las tasas tok/s del repo son de decode en otra máquina), así que se medirá antes de escalar. La GPU del laptop sólo es necesaria para escalar o para ajuste fino.

## 5. Métricas e hipótesis falsables (DEV→CONFIRM, IC95 bootstrap pareado, como v4)
- **H-L1 (predicción):** acc E33 h16 de ÓRBITA con $\hat c$ no es inferior a ÓRBITA con $c$ verdadero en más de un margen δ prerregistrado; refutada si IC95 inferior < −δ.
- **H-L2 (agnóstico al lenguaje):** divergencia de trayectorias $\|z_h(\hat c(t_1))-z_h(\hat c(t_2))\|/\|z_h\|$ entre parafraseos/idiomas del mismo estado < la de estados distintos (IC95 de la diferencia < 0) y < umbral prerregistrado.
- **H-L3 (invariancia al input):** acc estable al cambiar idioma/orden/sinónimos no vistos; línea base: encoder de bolsa de n-gramas (ya en el repo) y $c$ aleatorio.
- **H-C1 (CDT):** el teacher consolidado sobre experiencia de ÓRBITA recupera la familia correcta por encima del azar, y su residuo predice los fallos de ÓRBITA (AUROC > 0.5 con IC95).

## 6. Riesgos
Gemma puede codificar números/ángulos mal (cifras en texto); la invariancia puede ser trivial si las plantillas son demasiado parecidas; la acc de ÓRBITA ya es ≈0 desde h≈96, así que el texto no puede mejorar horizontes largos.
