# Runtime Gemma 2 optimizado

## Ruta de inferencia

`QuantizedGemma2` carga directamente GGUF mediante Candle 0.11. Los pesos
lineales y el embedding permanecen cuantizados. `QMatMul::embedding` desquantiza
únicamente las filas solicitadas; si el LM head está ligado al embedding, ambos
comparten el mismo `Arc<QTensor>`.

Cada capa global usa `candle_nn::kv_cache::KvCache` con capacidad máxima
preasignada. Las capas de ventana Gemma 2 usan `RotatingKvCache`, por lo que
retienen únicamente la ventana local. Las máscaras causal y local se construyen
una vez por forward y se comparten entre capas equivalentes.

La atención agrupada es híbrida: el prefill materializa K/V para conservar un
matmul batched eficiente, mientras el decode token-a-token procesa cada query
head contra su cabeza KV sin copiar todo el contexto repetido.

## Runtime y sesiones

`native_gemma2_runtime` centraliza:

- prefill y decode;
- sampling reproducible;
- decoder incremental de `tokenizers`;
- métricas TTFT, prefill y tokens/s;
- reutilización KV cuando el prompt nuevo extiende exactamente el prefijo
  procesado;
- callback por token para subsistemas concurrentes.

Si el historial retokenizado no coincide con el prefijo almacenado, el runtime
limpia la cache y hace un prefill completo. Nunca reutiliza una cache ambigua.

## Acople fasorial concurrente

`gemma_phasor_coupling` crea un worker Rust propietario de
`NativePhasorThermodynamicEngine`. El hilo Candle sólo envía `(token, position)`
por canal. El worker proyecta cada token de forma determinista a nodos,
inyecta amplitud/fase y ejecuta pasos termodinámicos mientras Gemma calcula el
siguiente token.

Esta separación evita compartir tensores o bloquear el hot path del LLM. El
estado fasorial resultante sirve para memoria, routing o diagnóstico de la
siguiente interacción; no modifica logits durante la generación actual.
El preset interactivo acumula estímulos sobre 256 nodos y evoluciona cada
16 tokens para limitar la competencia por CPU; el config permite aumentar
resolución y frecuencia en entrenamiento offline.

## Benchmark

```powershell
cargo run --release --bin native_gemma2_benchmark -- \
  --prompt-lengths 32,256,1024,2048 \
  --generated 64 \
  --repetitions 3
```

El CSV informa:

- tiempo de carga;
- RSS antes/después;
- almacenamiento del embedding;
- TTFT;
- throughput de prefill y decode;
- hit de reutilización KV.

### Resultado CPU medido

GGUF local Gemma 2 2B, release, `target-cpu=native`, 30 de julio de
2026:

- carga: 5.94 s;
- RSS incremental tras cargar: 1,716 MiB;
- embedding cuantizado: 461 MiB;
- prefill: 18.65–19.53 tokens/s en prompts de 32 y 256 tokens;
- decode: 10.58–11.13 tokens/s;
- TTFT a 256 tokens: 13.73 s;
- TTFT al extender la misma sesión: 0.98 s, 14.0 veces menor.

La referencia publicada anterior era 2.04–2.68 tokens/s; la nueva ruta midió
entre 3.9 y 5.5 veces esa velocidad. Es una comparación con la evidencia
histórica del proyecto, no un benchmark pareado de commits.

Una prueba de calidad produjo una definición coherente de entropía con
30 tokens, 11.47 tokens/s sin worker fasorial y salida determinista para la misma
semilla.

Con el preset fasorial final (256 nodos, un paso cada 16 tokens), una comparación
inmediata del mismo binario midió 5.94 tokens/s con acople y 5.89 sin acople,
diferencia indistinguible del ruido de la máquina. Los valores absolutos bajaron
durante esa corrida por carga del sistema; se usa el par únicamente para evaluar
el overhead concurrente.

El modelo no se versiona. Las corridas deben registrar hash y cuantización del
GGUF, CPU/GPU, RAM y flags de compilación.

## Backends

El build por defecto es CPU portable dentro de la CPU objetivo configurada.

```powershell
cargo build --release --features mkl
cargo build --release --features cuda
```

CUDA es opt-in porque requiere toolkit y driver compatibles. MKL debe aceptarse
sólo si el benchmark demuestra mejora: gran parte del tiempo se consume en
`QMatMul` cuantizado, donde BLAS puede no ser el cuello dominante.

En la máquina de validación no se detectó `nvidia-smi`. El build MKL compiló,
pero el ejecutable no inició por una DLL de runtime ausente
(`STATUS_DLL_NOT_FOUND`); por ello CPU Candle sigue siendo el backend validado y
MKL/CUDA permanecen opt-in.

## Política de shards

No se generan shards para inferencia CPU de un solo dispositivo. Dividir el
GGUF no reduce FLOPs y puede aumentar fallos de página. Un formato por capas se
justifica únicamente para:

1. asignar rangos de capas a varios dispositivos;
2. cargar selectivamente modelos que no caben en RAM/VRAM;
3. distribuir artefactos grandes.

`native_gguf_paged_thermo` continúa siendo un almacén de aristas
termodinámicas; sus shards F32 no deben usarse como pesos del Transformer.

## Límites pendientes

- Los kernels QKV, GeGLU, RMSNorm/residual y softcap/softmax no están fusionados.
  Se implementarán sólo después de perfilar el runtime optimizado.
- El batching continuo y paged attention multiusuario requieren soporte
  adicional en Candle.
- La búsqueda adaptativa interactiva está limitada a un prefill candidato. La
  evidencia actual no demuestra que omitir capas preserve calidad.
