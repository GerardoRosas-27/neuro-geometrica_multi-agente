# PLAN de implementación — ÓRBITA vs TimesFM

1. Rama `exp/orbita-vs-timesfm` desde 18ba4a4 (worktree `/workspace/ngm-timesfm`).
2. venv `/workspace/timesfm-venv`: torch CPU + `timesfm[torch]`; checkpoint `google/timesfm-2.5-200m-pytorch`.
3. `orbita/vstimesfm/latents.py`: reconstruye s_t exactamente como `dinlat/run.py` desde `texts.emb` cacheado
   (verificado: PERS/AR2 coinciden con dinlat a ~1e-14).
4. `orbita/vstimesfm/run_tfm.py`: por seed, TimesFM univariado (16 dims × episodio) con contexto creciente t=2..7 → 1 paso;
   h3 desde t=2; VAR2_OLS; también en C1/C2. Asserts de igualdad con dinlat.
5. `orbita/vstimesfm/analyze.py`: une con `res_*.json` de dinlat, G + IC95 bootstrap (código de dinlat), criterios, gate.
6. `orbita/vstimesfm/orchestrate.sh` (nohup): DEV → commit/push RESULTADOS → gate → CONFIRM (Gemma + ÓRBITA + TimesFM)
   → commit/push → `/workspace/orbita-data/timesfm/orchestrate.done`.
   Logs: `/workspace/orbita-data/timesfm_{dev,confirm}.log`, `/workspace/orbita-data/timesfm/<fase>/log_<seed>.txt`.
7. Coste estimado: DEV ≈ 1 h CPU (sin re-embeber). CONFIRM ≈ 13 h (embeddings Gemma, como en dinlat DEV) + ~1 h.
