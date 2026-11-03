# ATLAS-FW — Preregistro de ablaciones leave-one-out

Rama: `exp/atlas-fw`. Base: `docs/atlas_fw/PREREG.md` (constantes, presupuestos y métricas congelados; sin cambios).
Escrito y publicado **antes** de cualquier corrida de ablación. 0 consultas TEST (se verifica con `v4_provenance` en cada semilla; aborta si ≠ 0).

## Motivación (resultados previos, commit d189de4, `artifacts/atlas_fw/RESULTADOS.md`)

- ALL (M1–M6) vs CONSTELACIÓN v4 en CONFIRM: V−v4 new_med_log −0.4082 [−0.4985, −0.3190]; known_acc16 +0.0225; E33 16/16.
- Solos: M1 dañino; M2 IC cruza 0; M3–M6 idénticos a v4.
- Pregunta: ¿de qué mecanismos proviene la ganancia de ALL? (interacciones).

## Brazos

`v4` (referencia), `ALL`, `ALL-M1`, `ALL-M2`, `ALL-M3`, `ALL-M4`, `ALL-M5`, `ALL-M6`
(`ALL-Mi` = todos los mecanismos encendidos salvo Mi; `Mech::all().without(i)` en `orbita/src/constelacion4/atlas.rs`).
v4 se entrena una vez por semilla y todos los brazos lo reutilizan (idéntico a la corrida original).

## Semillas (frescas, sin solapamiento)

- ABL_DEV: 0xA490–0xA49F (fase `abl_dev`).
- ABL_CONFIRM: 0xB490–0xB49F (fase `abl_confirm`).
- Verificado antes de publicar: ninguna aparece en el código, docs ni artifacts de ninguna rama (`git grep` en todas las refs + árbol de trabajo). Usadas previamente: 0xA480–0xA48F, 0xB480–0xB48F, 0xC4A0 (smoke/paso0), 0xA400–0xA40F / 0xB400–0xB40F (ÓRBITA-v1), 0xC410… (v3).
- Pesos ÓRBITA-v1 (sólo para E33/ÓRBITA): semilla 0x{A,B}49i ↔ `artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json` (mismo emparejamiento por índice que la corrida original).

## Métricas y estadística

- Bootstrap pareado por semilla, 10000 réplicas, semilla del RNG 12345, IC percentil. δ = 0.05.
- **Contraste primario**: Δ_i = (ALL−Mi) − ALL en new_med_log (held-out h16), i = 1..6. Corrección Bonferroni sobre 6 ablaciones: IC al 1 − 0.05/6 = 0.9917.
  - Δ_i IC inf > 0 ⇒ **Mi contribuye** (quitarlo empeora).
  - Δ_i IC sup < 0 ⇒ **Mi perjudica** dentro de ALL (quitarlo mejora).
  - En otro caso ⇒ sin efecto detectable.
- **No-daño** (cada brazo vs v4, igual que PREREG.md): known_acc16 V−v4 IC95 inf > −δ; known_acc16 final−pre IC95 inf > −δ; E33 pass ≥ 15/16; 0 NaN. Se reporta además (ALL−Mi) − ALL en known_acc16 (IC95) y V−v4 en new_med_log (IC95), como secundarios.

## Procedimiento

1. ABL_DEV: 16 semillas × {v4, ALL, ALL-M1..ALL-M6}.
2. Gate: ablaciones con Δ_i significativo (cualquier signo) en ABL_DEV.
3. ABL_CONFIRM sólo para las ablaciones significativas en DEV + ALL + v4. Si ninguna es significativa: se publica DEV y CHAIN_STOP (sin CONFIRM).
4. Veredicto por Mi: CONTRIBUYE / PERJUDICA si el signo significativo se replica en CONFIRM; si no, NO CONFIRMADO; más el estado de no-daño de ALL−Mi en CONFIRM.
5. Veredicto sólo si 16/16 semillas válidas (rc=0, sin panic, sin NaN, metadatos correctos, 0 TEST); si no, CHAIN_STOP_ERROR sin veredicto.

## Ejecución

- `orbita/scripts/atlas_fw_abl_chain.sh` (adaptación de `atlas_fw_chain.sh`) → `orbita/scripts/atlas_fw_abl.sh`, JOBS=7, nice 10.
- Análisis: `scripts/analyze_atlas_fw_abl.py`. Salida: `artifacts/atlas_fw/ablaciones/RESULTADOS_ABLACIONES.md`, log `artifacts/atlas_fw/ablaciones/chain.log`.
- Termina con CHAIN_DONE, CHAIN_STOP o CHAIN_STOP_ERROR; commit+push sólo a `exp/atlas-fw` (nunca main, sin force push).
