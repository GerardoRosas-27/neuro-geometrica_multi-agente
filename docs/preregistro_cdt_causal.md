# Prerregistro — ¿CDT-RQM-epr aporta REALMENTE a ÓRBITA? Pruebas causales OBS / CORR con oráculo-justo

Rama `exp/orbita-llm`, tras e328ff2 (rasgos CDT sin observación: DIRECTO = SUEÑO = NOCDT = CORRUPT = SHUF). Publicado
**antes de lanzar nada** sobre DEV/CONFIRM. No hay extracción nueva: se reusan los contextos E2E_R
(`/workspace/orbita-data/llm_e2e/<fase>/ctx_<seed>.csv`: familia argmax D2 + p de numnorm sobre plantillas 6–7, split por
plantilla; D2 nunca vio esas plantillas). ÓRBITA-v1 congelada; nada se entrena (pesos fijados aquí). δ = 0,05; IC95
bootstrap pareado sobre 16 seeds (10 000, semilla 12345). DEV 0xA400–0xA40F y CONFIRM 0xB400–0xB40F, nada cambia entre fases.

## Señales (mecanismos v4 existentes, `orbita cdt-causal`, sin tocar v4)
DIR = `consolidate(Full)` de la CDT de vigilia sobre TRAIN. SLEEP = misma experiencia → máscara trusted
(`family_fit_residual`, `TRUST_TAU`) → poda de reglas no confiables → borrado de la CDT → `pseudo_experiences` →
re-consolidación offline (lo de `sleep_consolidate`, idéntico a e328ff2). CORRUPT = `ConsolidateMode::Corrupt`.

## Observación online (no es entrenamiento)
Para cada texto y su ejemplo TEST iterable se observan los **K_OBS = 2** primeros estados reales y₁, y₂ desde x₀ (la
misma trayectoria contra la que se mide acierto). Sólo se usan en inferencia; nada se ajusta con TEST entre ejemplos.
Las métricas acc1/acc8/acc16 se miden desde x₀ como siempre (acc1 queda trivialmente informada en las variantes con
observación; el criterio usa acc16). Contador de queries de auditoría: debe ser 0.

## Condiciones (p = p̂ de E2E en todas)
- **BASE** (= NOCDT): E2E actual, ÓRBITA con ĉ.
- **OBS_v** (v ∈ DIR, SLEEP): r_g = Σ_{t<K} rel_err(teacher_g(p̂, y_t), y_{t+1}); familia g* = argmin r_g; ÓRBITA con c_{g*}.
  score = r_{g*}.
- **CORR_v**: corrección por paso z_{t+1} = (1−α)·ÓRBITA(z_t, ĉ) + α·teacher_{ĝ}(p̂, z_t), **α = 0,5 fijado**.
- **OBSCORR_v**: g* de OBS_v y corrección CORR con g*.
- Controles: **CORRUPT** (cada variante con la señal Corrupt), **SHUF** (OBS: residuos de otra fila por permutación fija
  por seed — observaciones ajenas; CORR: teacher de una familia desarreglada fija g→(g+o) mod 8), **ORÁCULO-JUSTO sin CDT con
  la MISMA observación**: ORACLE_ORB = misma re-identificación pero con ÓRBITA como predictor (r_g = Σ rel_err(ÓRBITA(y_t,
  c_g), y_{t+1})); ORACLE_LS = mapa afín por mínimos cuadrados ajustado a los K·6 pares observados e iterado desde x₀.
  Para CORR (que no observa) el control equivalente sin CDT es BASE.

## Criterio prerregistrado «CDT aporta»
Para cada variante V ∈ {OBS, CORR, OBSCORR} × {DIR, SLEEP}: Δ_V = acc16(V) − max_seed(controles de V), donde los controles
son OBS/OBSCORR: {BASE, CORRUPT, SHUF, ORACLE_ORB, ORACLE_LS}; CORR: {BASE, CORRUPT, SHUF} (máximo por seed, pesimista).
**CDT aporta** si existe V con IC95(Δ_V) > 0 **en DEV y en CONFIRM** (H-V acc16). Igual con AUROC (score → fallo h16)
como criterio secundario. Refutada para V si IC95 superior < 0 en CONFIRM; si no, inconclusa. H-SD-{OBS,CORR,OBSCORR}:
acc16 SLEEP − DIR (IC95 > 0 sueño mejor, < 0 peor, ⊂ (−δ, δ) equivalentes). Descriptivo: acc1/8/16 y AUROC de todas,
diferencias contra cada control, tamaño de señal antes/después de la poda.

## Ejecución
`/workspace/orbita-data/llm_cdt2/chain.sh` (copia `orbita/llm/causal_chain.sh`), `setsid nohup nice -n 0`, reanudable,
`.done`: `dev.done`, `confirm.done`; log `chain.log`. Resultados: `docs/resultados_cdt_causal.md`.
