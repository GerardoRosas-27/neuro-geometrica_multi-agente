# Demo ÓRBITA (pila final)

```
texto (es/en/fr; pt/it sin entrenar) → Gemma 2 2B capa 13 (Q4_K_M) → D2 (3000 pasos) + numnorm → ÓRBITA-v1 congelada
   [+ re-observación de k = 1–2 estados reales y re-identificación con ÓRBITA como verificador] → yo (informa confianza/sorpresa, no decide)
   → explicación breve de Gemma Q4_K_M (greedy, prompt fijo) a partir del yo
```

Uso (desde `/workspace/ngm-orbita`, venv `/workspace/orbita-venv`, `PYTHONPATH=orbita`):

```
python orbita/llm/demo.py --text "Aplica una rotación con parámetros 0,5 y -0,2."                       # sin observar
python orbita/llm/demo.py --text "..." --obs 1 --observed "x1,y1,...(12)"                              # 1 estado real dado
python orbita/llm/demo.py --text "..." --obs 2 --simulate 9,0.6,-0.5                                   # simula 2 pasos de la familia 9 (espiral)
   opciones: --x0 "12 números" (por defecto hexágono r = 0,8)  --seed 0xA400 (DEV) / 0xB40x (CONFIRM)  --no-explain  --json
sh orbita/demo/smoke.sh       # test de humo (sin generación)
sh orbita/demo/examples.sh    # regenera artifacts/orbita/demo/*.json (salidas reales)
```
Familias de `--simulate`: 0–7 v4 (traslación, rotación, reflexión, escalado, cizalla, afín, rotación con escalado, deformación radial),
8 = giro ondulado, 9 = espiral. Requiere los pesos por seed en `/workspace/orbita-data/{dev,confirm}/` y los datos de calibración del yo en
`/workspace/orbita-data/yo/`. Detalles y límites: `docs/demo_orbita.md`.
