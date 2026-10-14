# ÓRBITA — release de pesos y datos (32 seeds: DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F)

Código: `orbita/` en `exp/orbita` (sin cambios de entrenamiento/evaluación desde el pre-registro `f5d7781`). Verificar integridad con `sha256sum -c SHA256SUMS`.

| Archivo | Contenido |
|---|---|
| `orbita_weights_32seeds.tar.xz` | por seed: `weights/<fase>/w_<seed>.bin` (todos los parámetros en float64 little-endian, sin pérdida, orden `w1,b1,w2,b2,w3,b3,w4,b4,wc`), `w_<seed>.json` (cabecera: arquitectura, formas, sha256 del JSON original) y `w_<seed>.original.json` (pesos exactos usados por la evaluación Rust). Sin checkpoints `.ckpt` intermedios. |
| `orbita_traindata_32seeds.tar.xz` | por seed: `<fase>/<seed>/train.bin` (TRAIN: 192 × [x(12), c(10), y(12)] f64 LE) y `meta.json` (residuos de consolidación, máscara/pesos de confianza, hits del auditor, sha256 del TEST). |
| `REPLAY_SHA256_regenerable.txt` | sha256 de cada `replay.bin` (pool de 100 000 muestras del teacher, 56.8 MB por seed, ≈1.8 GB en total): **no se sube** por tamaño; se regenera bit a bit. |
| `SHA256SUMS` | sumas de los archivos anteriores. |

## Descomprimir
```bash
tar -xJf orbita_weights_32seeds.tar.xz      # → weights/{dev,confirm}/
tar -xJf orbita_traindata_32seeds.tar.xz    # → {dev,confirm}/0x..../{train.bin,meta.json}
```

## Regenerar el pool de replay (determinista) y verificar
```bash
cd orbita && cargo build --release --locked   # usar rustup cargo (~/.cargo/bin), rustc ≥ 1.88
./target/release/orbita export --seed 0xA400 --out /tmp/data/dev/0xA400   # train.bin, replay.bin, meta.json
(cd /tmp/data && sha256sum dev/0xA400/replay.bin)   # comparar con REPLAY_SHA256_regenerable.txt
```

## Cargar pesos
- Rust (evaluación oficial): `orbita eval --seed 0xA400 --weights weights/dev/w_0xA400.original.json --meta dev/0xA400/meta.json --out /tmp/eval`.
- Python (binario):
```python
import json, numpy as np
h = json.load(open("weights/dev/w_0xA400.json"))
flat = np.fromfile("weights/dev/w_0xA400.bin", dtype="<f8"); W, i = {}, 0
for k in h["order"]:
    n = int(np.prod(h["shapes"][k])); W[k] = flat[i:i+n].reshape(h["shapes"][k]); i += n
```
o cargar `w_<seed>.original.json` en `orbita/train_orbita.py::Orbita` (mismos nombres de tensores).
