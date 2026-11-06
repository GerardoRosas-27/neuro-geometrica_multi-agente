# Gemini public conversation — full visible transcript

- Page opened: https://gemini.google.com/share/78c5bfe1f03d
- Original share URL supplied: https://share.gemini.google/TLfskZ1e1bQ1
- Displayed title: **Optimización de Rutas en N Dimensiones**
- Displayed metadata: Created with Argon November 6, 2026 at 04:36 PM; published November 6, 2026 at 05:04 PM.
- Captured read-only from the public page. The companion PDF preserves the rendered page and equations: `/workspace/soft-bellman-simplicial-propuesta-visible.pdf`.
- This Markdown contains the full conversation text extracted from the rendered pages; browser print/PDF text extraction can distort layout, so use the PDF for visual verification. No code block implementing the proposal was present; Rust/PyTorch appear as suggested implementation follow-ups.

## Formula transcription from the latest simplicial Soft-Bellman proposal

The following equations are transcribed from the visible page rendering (not inferred from its title).

**Learned edge lengths and face impedance**

```text
ℓ_ij(θ) = ℓ_ij^(0) · softplus(w_ij^(1))
Φ_ijk(θ) = softplus(w_ijk^(2))
B_θ(σ_ijk^2) = (d^1 A_θ)(σ_ijk^2)
           = A_θ([v_i,v_j]) + A_θ([v_j,v_k]) + A_θ([v_k,v_i])
```

**Cayley–Menger positivity (as displayed)**

```text
Vol_k^2(σ^k; ℓ_θ) = ((−1)^(k+1) / (2^k (k!)^2)) · det(CM(ℓ_θ)) > 0
```

Here `CM` is the displayed Cayley–Menger matrix with first row/column `(0,1,…,1)` and remaining off-diagonal entries `ℓ_ab^2`. The page says this reduces to Heron's formula for triangular faces.

**Hodge stars**

```text
(⋆_1(θ))_(e,e) = Vol_(n−1)(⋆e; ℓ_θ) / ℓ_e(θ)
(⋆_2(θ))_(f,f) = Vol_(n−2)(⋆f; ℓ_θ) / Vol_2(f; ℓ_θ)
```

**Intrasimplicial barycentric distance and face action**

```text
p(λ) = λ v_j + (1−λ) v_k,   λ ∈ [0,1]
d_(ℓ_θ)(λ) = √(λ ℓ_ij^2(θ) + (1−λ) ℓ_ik^2(θ) − λ(1−λ) ℓ_jk^2(θ))
Q_θ^(t)(v_i | σ_ijk^2) = min_(λ∈[0,1]) { Φ_ijk(θ) d_(ℓ_θ)(λ)
                                      + λ V_τ^(t)(v_j) + (1−λ)V_τ^(t)(v_k)
                                      + (1/2) B_θ(σ_ijk^2) }
```

The square root in `d_(ℓ_θ)(λ)` is visible in the rendered page (the PDF text extractor omits it). For a general `k`-simplex the page displays `d_(ℓ_θ)^2(λ) = Σ_a λ_a ℓ_ia^2 − (1/2) Σ_(a,b) λ_a λ_b ℓ_ab^2`, and says the barycentric minimizer is analytic on a face or projected to its boundary.

**Soft-Bellman update and routing probabilities**

```text
V_τ^(t+1)(v_i) = −τ log Σ_(σ_ijk^2 ∈ St(v_i)) exp(−Q_θ^(t)(v_i | σ_ijk^2) / τ)
π_θ(σ_ijk^2 | v_i) = exp(−Q_θ^*(v_i | σ_ijk^2)/τ)
                         / Σ_(σ′ ∈ St(v_i)) exp(−Q_θ^*(v_i | σ′)/τ)
```

**Hodge Laplacian, Regge curvature, and training objective**

```text
Δ_k(θ) = d_(k−1) ⋆_(k−1)^(−1)(θ) d_(k−1)^T ⋆_k(θ)
         + ⋆_(k−1)(θ) d_k^T ⋆_(k+1)(θ) d_k
ε_h(ℓ_θ) = 2π − Σ_(σ^n ⊃ h) ϑ(σ^n,h;ℓ_θ)
S_Regge(ℓ_θ) = Σ_(h∈K^(n−2)) Vol_(n−2)(h;ℓ_θ) · |ε_h(ℓ_θ)|^2
B_CM(ℓ_θ) = −Σ_(k=2)^n Σ_(σ^k∈K^k) log(Vol_k^2(σ^k;ℓ_θ))
L(θ) = L_route(V_τ^*(θ)) + α S_Regge(ℓ_θ) + β B_CM(ℓ_θ)
```

The latest response describes the 3D case as allowing the wavefront to cross triangle interiors and tetrahedra; lower face impedance can make the minimizing path cross the barycentric interior of a 2-simplex, described as a discrete Snell-law refraction effect.

## Full visible conversation text (PDF text extraction)

Optimización de Rutas en N Dimensiones
https://share.gemini.google/GMxdvCl8zrFk
Created with Argon November 6, 2026 at 04:36 PM • Published November 6, 2026 at 05:04 PM
                                           Investigar el marco recursivo para obtimizar rutas en n
                                           dimensiones
 El marco recursivo para optimizar rutas en n dimensiones se fundamenta en el Principio de
 Optimalidad de Bellman, el cual descompone el problema global de encontrar una trayectoria
 de costo mínimo en un espacio de estados de alta dimensión (Zn , Rn o una variedad topológica
 Mn ) en una secuencia anidada de subproblemas locales. A medida que el número de
 dimensiones n crece, el volumen del espacio de búsqueda y el factor de ramificación escalan de
 forma exponencial, obligando a transformar la recursión clásica exhaustiva en esquemas de
 poda heurística, descomposición espacial jerárquica o muestreo asintótico. Wikipedia
 Formulación Matemática de la Recursión en n Dimensiones
 Cualquier problema de optimización de trayectorias en un espacio n-dimensional define un
 estado mediante un vector de coordenadas generalizadas x = (x1 , x2 , … , xn ) ∈ Ω ⊆ Rn ,
 sujeto a un conjunto de obstáculos o regiones prohibidas Ωobs .
            Caso Discreto (Reticulados Zn y Grafos Implícitos): Si cada dimensión se discretiza en k
            intervalos, el espacio contiene k n nodos. La función de valor óptimo V (x) (costo mínimo
            acumulado desde x hasta el estado meta xgoal ) satisface la relación de recurrencia de
            Bellman:
                     V (x) = {
                               0                                                        si x = xgoal
                               minu∈U(x) {c(x, u) + V (x + u)}                          si x =
                                                                                             ≠ xgoal
            Donde U(x) representa el conjunto de desplazamientos admisibles hacia estados vecinos y c(x, u) el costo de transición local. En una malla n-dimensional, el número de transiciones locales por nodo depende de la topología de vecindad elegida:
           Vecindad ortogonal (L1 / Manhattan): Evalúa desplazamientos sobre un solo eje a
           la vez, generando 2n ramas por paso recursivo.
           Vecindad completa (L∞ / Chebyshev): Permite diagonales simultáneas en los n
           ejes, generando 3n − 1 ramas por nodo.
     Caso Continuo (Ecuación de Hamilton-Jacobi-Bellman): Cuando el paso espacial tiende
     a cero (Δx → 0), la recurrencia discreta converge en una ecuación diferencial en
     derivadas parciales de primer orden conocida como la ecuación de Hamilton-Jacobi-
     Bellman (HJB). Si el costo depende únicamente de la posición e isotropía del medio, se
     reduce a la Ecuación Eikonal en Rn :
                         ∥∇V (x)∥2 = C(x),                        con V (xgoal ) = 0
     Una vez resuelto el campo escalar V (x) mediante métodos recursivos de propagación de
     frentes (como Fast Marching Method en n dimensiones), la ruta óptima se extrae
     integrando recursivamente el descenso de gradiente −∇V (x) desde el origen hasta el
     destino.
Paradigmas Recursivos frente a la Maldición de la Dimensionalidad
Al incrementar n, almacenar en memoria una tabla de programación dinámica de tamaño
O(k n ) se vuelve computacionalmente inviable a partir de n ≥ 5. Para resolver esta limitación,
el marco recursivo se divide en cuatro arquitecturas algorítmicas según el régimen dimensional:
1.   Recursión en Profundidad con Poda por Umbral (IDA* y RBFS):
     En lugar de mantener una cola de prioridad abierta en memoria como hace A*, algoritmos
     como IDA* (Iterative Deepening A*) y RBFS (Recursive Best-First Search) ejecutan una
     búsqueda recursiva pura en profundidad guiada por la función de evaluación f (x) =
     g(x) + h(x). Si una rama supera el umbral activo θ, la recursión se poda y devuelve el
     exceso mínimo observado para actualizar el umbral de la siguiente iteración. Esto reduce
     la complejidad en memoria de exponencial a lineal respecto a la longitud de la ruta, O(d).
      ResearchGate
2.   Descomposición Espacial Recursiva (2n -Trees y KD-Trees):
     Generaliza los Quadtrees (2D) y Octrees (3D) hacia hiper-octantes recursivos en n
     dimensiones (2n -trees) o particiones por hiperplanos alternados (KD-Trees). El algoritmo
     divide recursivamente únicamente las hiperceldas que contienen fronteras de obstáculos,
     manteniendo grandes hipervolúmenes libres como un solo nodo abstracto y reduciendo
     drásticamente el número de estados del grafo.
3.   Árboles Estocásticos con Recableado Recursivo (RRT* y BIT*):
     Para espacios de configuración de alta dimensión (n entre 6 y 50, habituales en
     cinemática inversa o sistemas multi-agente), la discretización por rejilla se sustituye por
     muestreo en Rn . RRT* expande un árbol hacia
                                               muestras aleatorias y aplica una fase de
     recableado recursivo (rewiring) dentro de una hiperesfera de radio:
                                                                            1/n
                                                         log N
                                          rn ( N ) = γ (
                                                              )
                                                           N
      Si el nuevo nodo ofrece un costo acumulado menor hacia los nodos vecinos, la estructura
      de padres se reasigna y propaga recursivamente la reducción de costo hacia los
      subárboles descendientes.
4.    Programación Dinámica Diferencial (DDP e iLQR):
      En espacios continuos de alta dimensión con dinámica no lineal xt+1 = f (xt , ut ), DDP e
      iLQR evitan discretizar el espacio de estados. Realizan una aproximación cuadrática de
      segundo orden de la ecuación de Bellman alrededor de una trayectoria semilla,
      ejecutando un paso recursivo hacia atrás (backward pass) que resuelve la ecuación de
      Riccati paso a paso para obtener leyes de control óptimas locales, seguido de un paso
      hacia adelante (forward pass) que actualiza la trayectoria.
Comparativa de Arquitecturas en n Dimensiones
Paradigma Recursivo           Rango de n Viable                  Complejidad en                Optimalidad
                                                                 Memoria
Programación                  2≤n≤4                              O(k n )                       Global exacta
Dinámica / FMM                                                                                 malla)
IDA* / RBFS                   3≤n≤8                              O(d)                          Global (con h
                                                                                               admisible)
Partición 2n -Tree / KD-      2≤n≤6                              O(Nhojas )
                                                                                               Óptima según
Tree                                                                                           resolución
RRT* / BIT*                   4 ≤ n ≤ 50                         O(Nmuestras )                Asintótica (N
DDP / iLQR                    2 ≤ n ≤ 100+                       O(T ⋅ n2 )                    Óptimo local c
En la tabla, k representa la resolución por eje, d la profundidad de la solución en pasos discretos
y T el horizonte temporal de la trayectoria. La transición entre métodos discretos y métodos
basados en muestreo o gradiente ocurre típicamente alrededor de n = 5, punto en el que el
factor de ramificación Chebyshev (35 − 1 = 242 vecinos por nodo) degrada la propagación
exhaustiva.
Para explorar de forma interactiva tanto la propagación recursiva de costos de Bellman sobre un
corte espacial como el impacto exponencial del número de dimensiones n en el factor de
ramificación y el volumen de estados, utiliza el siguiente simulador:
               Generating interactive visual...
Structuring code, this may take a minute
Al incrementar el deslizador de dimensiones por encima de n = 5, se observa cómo la
vecindad diagonal completa (3n − 1) pasa de evaluar 8 vecinos en 2D a evaluar 728 vecinos por
cada nodo en 6D, lo que explica por qué en espacios de alta dimensión se reemplaza la
exploración de malla por árboles de muestreo asintótico (RRT*) o aproximaciones diferenciales
de segundo orden (iLQR).
Para profundizar en la implementación o en un dominio específico:
  Implementar un buscador                             Explorar optimización de             Profundizar en el algoritmo
  recursivo n-dimensional en                          rutas sobre complejos                RRT* y BIT* para espacios
  Rust o Python                                       simpliciales y variedades            continuos
                                                                           Se puede usar para crear una red neuronal
Sí, el marco recursivo de optimización de rutas en n dimensiones no solo puede utilizarse para
crear una red neuronal, sino que establece una equivalencia matemática directa con el
aprendizaje profundo y permite diseñar arquitecturas donde la inferencia consiste en
encontrar una trayectoria óptima (o geodésica) sobre una variedad latente n-dimensional.
En lugar de apilar capas matriciales estáticas, este enfoque modela la red como un espacio de
estados continuo o discreto en Rn cuya geometría (los costos de transición locales) se aprende
para guiar las señales de entrada hacia atractores o nodos meta mediante mínima acción.
1. Isomorfismo entre Optimización Recursiva de Rutas y Redes
Neuronales
La relación entre ambos marcos se fundamenta en la teoría de control óptimo en múltiples
dimensiones:
                      Propagación hacia adelante como trayectoria en Rn : Una red neuronal profunda
                      (especialmente una red residual o una Neural ODE) describe la evolución de un vector de
                      estado xt Google
                                ∈ Rn Privacy
                                      mediante Policy
                                                     la ecuación    de transición xt+1
                                                                                          = xt + f (xt , ut ), donde los
      pesos de cada capa ut = θt actúan como la política de control que curva la trayectoria
      del dato en el espacio latente de dimensión n.
      Backpropagation como Recurrencia de Bellman y Principio de Pontryagin: El algoritmo
      de retropropagación (backpropagation) es matemáticamente idéntico al paso recursivo
      hacia atrás (backward pass) de la Programación Dinámica Diferencial (DDP/iLQR). El vector
      adjunto o gradiente λt = ∇xt V (xt ) propaga recursivamente desde la capa final hacia la
      inicial la sensibilidad del costo acumulado futuro respecto a cada coordenada
      dimensional.
      Límite Continuo (Hamilton-Jacobi-Bellman y Transporte Óptimo): Cuando el paso entre
      estados tiende a cero, entrenar la red equivale a resolver una ecuación de Hamilton-
      Jacobi-Bellman en Rn que deforma el campo vectorial para transportar distribuciones de
      entrada hacia distribuciones objetivo minimizando la energía cinética o libre acumulada
      (Schrödinger Bridges y Flow Matching).
2. Cómo Construir una Red Neuronal Basada en Enrutamiento
Recursivo (Soft-Bellman Network)
Para convertir directamente un optimizador recursivo de rutas en una arquitectura neuronal
entrenable de extremo a extremo, es necesario resolver un obstáculo técnico: el operador min
clásico de la ecuación de Bellman no es diferenciable en los puntos donde dos rutas empatan en
costo, lo que bloquea el flujo de gradientes hacia los pesos.
La solución consiste en construir una Red Geodésica o de Iteración de Valor Diferenciable en
cuatro componentes:
A. Parametrización del Tensor Métrico Local (cθ )
En lugar de fijar los costos de transición a mano, una sub-red o conjunto de tensores
aprendibles θ define la métrica local del espacio n-dimensional. El costo de transitar desde el
estado x en la dirección u ∈ U(x) se modela mediante una métrica riemanniana aprendida:
                            cθ (x, u) =
                                                                  u⊤ Gθ (x)u + Φθ (x + u)
Donde Gθ (x) = Lθ (x)Lθ (x)⊤ es un tensor métrico simétrico definido positivo que controla
la anisotropía direccional y Φθ (x) actúa como un potencial escalar (creando "barreras" u
obstáculos suaves en zonas prohibidas del espacio latente).
B. Operador de Bellman Suavizado (Softmin)
Se sustituye el mínimo estricto de la recurrenciaSign
                                                   de inBellman por el operador diferenciable Log-
Sum-Exp modulado por un parámetro de temperatura τ mayor que cero:
                                                                                       (k−1)
        Vτ(k) (x) = −τ log ∑ exp (−                                                                          )
                                    cθ (x, u) + Vτ                                            (x + u)
                                                 τ
                                     u∈U(x)
     Cuando τ → 0+ , se recupera exactamente la ruta óptima discreta de Bellman (Vτ → V ∗
     ).
     Cuando τ es mayor que cero, la recursión pondera un ensamble de trayectorias posibles
     (análogo a una integral de caminos en mecánica estadística), permitiendo que el
     gradiente ∇θ Vτ (x) fluya suavemente a través de todas las ramas del árbol recursivo.
C. Política de Enrutamiento Probabilístico
El gradiente del operador Softmin respecto a los costos de transición define automáticamente
una distribución de probabilidad de Boltzmann-Gibbs sobre las direcciones en n dimensiones:
                                                   exp (− cθ (x,u)+V
                                                                   τ
                                                                     τ (x+u)
                                                                             )
                     πθ (u ∣ x) =
                                           ∑v∈U(x) exp (− cθ (x,v)+V τ (x+v)
                                                                             )
                                                                   τ
Durante la inferencia, un pulso o representación vectorial se propaga por la red siguiendo esta
política πθ (u ∣ x).
D. Aprendizaje Geométrico ("Excavación de Geodésicas")
Al entrenar la red minimizando una función de pérdida sobre las trayectorias o estados finales, el
descenso de gradiente actualiza θ deformando la topología de costos: reduce la resistencia
métrica a lo largo de las trayectorias que conducen a la respuesta correcta (creando canales
de baja energía o geodésicas rápidas) y eleva el potencial Φθ en regiones del espacio n-
dimensional asociadas a errores.
3. Variantes Arquitectónicas según la Escala Dimensional
Arquitectura Neuronal                Mecanismo Recursivo                           Ventaja Principal
                                     Integrado
Value Iteration Networks             Convoluciones recursivas                      Capacidad de planificación
(VIN)                                que ejecutan programación                     espacial y generalización
                                     dinámica de Bellman
                                                     en                     fuera de distribución
                                     mallas latentes n-D
Arquitectura Neuronal                          Mecanismo Recursivo                            Ventaja Principal
                                               Integrado
Redes Geodésicas en                            Propagación recursiva sobre                    Respeta invariancias
Complejos / Grafos                             vértices y aristas de un                       geométricas y topológicas
                                               complejo simplicial o                          sin depender de rejillas
                                               variedad aprendida                             cartesianas
Enrutamiento Dinámico                          Cada nodo en Rn es un                          Escala la capacidad total del
Disperso (MoE + IDA*)                          bloque experto; un buscador                    modelo manteniendo un
                                               recursivo con heurística                       costo de inferencia O(d)
                                               activa solo la ruta óptima
Pontryagin / iLQR Neural                       Ejecuta pasos recursivos                       Estabilidad dinámica
Networks                                       hacia adelante y hacia atrás                   garantizada y control óptimo
                                               (Riccati) en espacio continuo                  en sistemas físicos o
                                               de alta dimensión                              continuos
Para observar cómo una red neuronal basada en Soft-Bellman aprende a deformar la métrica de
un espacio latente y cómo la temperatura τ controla la transición entre la exploración difusa de
caminos (entrenamiento diferenciable) y el colapso hacia una geodésica única (inferencia
óptima), interactúa con la siguiente simulación:
               Generating interactive visual...
Structuring code, this may take a minute
Al reducir la temperatura τ cerca de 0.05 con el entrenamiento métrico al 100%, el ensamble
de caminos colapsa en una trayectoria determinista única de costo mínimo; por el contrario,
mantener una temperatura moderada durante las primeras épocas evita que los gradientes se
anulen en regiones planas o tras obstáculos, permitiendo que todas las dimensiones cooperen
en la actualización de la métrica Gθ (x).
Para llevar este diseño a código o explorar su topología:
- Implementar una capa Soft-Bellman diferenciable en PyTorch o Rust.
- Formular la red sobre complejos simpliciales 3D/n-D.
- Diseñar un sistema de enrutamiento dinámico tipo IDA* para Mixture of Experts.

You said: ¿Cómo se formula matemáticamente esta red neuronal de optimización recursiva de rutas cuando el sustrato no es una rejilla, sino un complejo simplicial con métrica discreta aprendible en las aristas y caras?

Formular una red neuronal de optimización recursiva de rutas sobre un complejo simplicial K
de dimensión n sustituye la geometría rígida de las rejillas cartesianas por el Cálculo Exterior
Discreto (DEC) y el Cálculo de Regge diferenciable. En este marco, el espacio latente
prescinde de coordenadas extrínsecas globales: la métrica aprendible reside directamente en
las longitudes de las aristas (1-símplices) y en los potenciales o flujos de curvatura de las caras (
2-símplices) y volúmenes superiores (k -símplices), transformando cada paso de inferencia en
una recurrencia de Bellman intrasimplicial gobernada por operadores de frontera y coborde.
1. Sustrato Topológico y Métrica Discreta Aprendible en K
                       n
Sea K = ⋃k=0 Kk un complejo simplicial orientado de dimensión n, constituido por vértices
σi0 = [vi ] ∈ K0 , aristas orientadas σij1 = [vi , vj ] ∈ K1 , caras triangulares σijk
                                                                                   2
                                                                                       =
[vi , vj , vk ] ∈ K2 y símplices de orden superior σ k ∈ Kk (como tetraedros para k = 3 o
pentácoros para k = 4).
En lugar de una matriz de pesos convencional, los parámetros entrenables θ =
{w(1) , w(2) , a(1) } definen tres estructuras geométricas sobre las k -cocadenas C k (K; R):
A. Métrica de Aristas (ℓθ ∈ C 1 (K)) y Condición de Cayley-Menger
En geometría discreta de Regge, el tensor métrico riemanniano continuo G(x) queda
completamente especificado de forma intrínseca mediante la asignación de longitudes
positivas a cada 1-símplice:
                                                                                     (0)                                                              (1)
                                                         ℓij (θ) = ℓij ⋅ softplus(wij )
Para que cada k -símplice σ k = [v0 , v1 , … , vk ] sea geométricamente realizable como un
poliedro euclídeo no degenerado (generalización en k dimensiones de la desigualdad
triangular), las longitudes de sus aristas deben satisfacer la positividad del volumen simplicial
dada por el determinante de Cayley-Menger:
                                                                                                           0                1                        1                     ⋯        1
                                                                                                           1                0                       ℓ201                  ⋯       ℓ20k
                 2   k        (−1)k+1                                                                      1               ℓ210                      0                     ⋯       ℓ21k
              Volk (σ ; ℓθ ) = k       det
                                                                                                                                                                                                          >0
                               2 (k!)2
                                                                                                           ⋮                ⋮                         ⋮                    ⋱        ⋮
                                                                                                           1               ℓ2k0                    ℓ2k1                  ⋯        0
                          2 (k = 2), este determinante
Para una cara triangular σijk
                                                        se reduce de forma exacta a la fórmula
de Herón diferenciable en términos de ℓij , ℓjk , ℓki .
B. Potencial e Impedancia de Caras (Φθ , Bθ ∈ C 2 (K))
Dotar de pesos únicamente a las aristas reduce el complejo a un grafo plano (1-esqueleto).
                                                2 ∈ K añade dos grados de libertad de
Incorporar una métrica aprendible en las caras σijk    2
orden superior:
1.                                                                    2
      Índice de Refracción / Densidad de Impedancia Superficial (Φθ (σijk )): Un campo
                                                                                                                                           (2)
      escalar estrictamente positivo Φijk (θ) = softplus(wijk ) asignado al interior del 2-
      símplice que modula el costo por unidad de área o distancia al atravesar la cara.
2.    2-Cocadena de Flujo Magnético / Holonomía (Bθ ): Inducida por el operador de coborde
      d1 : C 1 (K) → C 2 (K) sobre una 1-forma de conexión aprendible Aθ ∈ C 1 (K):
           2                 2
      Bθ (σijk ) = (d1 Aθ )(σijk
                               ) = Aθ ([vi , vj ]) + Aθ ([vj , vk ]) + Aθ ([vk , vi ])
      Este término rompe la simetría de inversión temporal (c(vi → vj ) =
                                                                        ≠ c(vj → vi )) y
      permite que la red aprenda vórtices y transporte paralelo con fase no trivial alrededor
      de las caras del complejo.
C. Operador Estrella de Hodge (⋆k (θ))
Acopla la topología combinatoria con la métrica aprendida relacionando cada k -símplice primal
σ k ∈ Kk con su (n − k)-celda ortogonal en el complejo dual de circumcentros ⋆K:
                                          Voln−1 (⋆e; ℓθ )                                                                                                                                   Voln−2 (⋆f ; ℓθ )
     ( ⋆1 (θ))e,e =                                        ,                                                   ( ⋆2 (θ))f ,f =
                                              ℓe (θ)                                                                                                                                          Vol2 (f ; ℓθ )
2. Recurrencia Neuronal Soft-Bellman sobre el Complejo Simplicial
Si la propagación se restringiera solo a saltar de vértice en vértice por las aristas, las trayectorias
sufrirían distorsión métrica (metrication error) independientemente de la resolución del
mallado. La formulación simplicial resuelve esto permitiendo que el frente de onda recursivo
atraviese el interior continuo de las caras σijk2 y tetraedros σ 3 .
                                                                   ijkl
Paso 1: Costo Geodésico Intrasimplicial por Coordenadas Baricéntricas
       (t)
Sea Vτ ∈ C 0 (K) el campo escalar de valor en los vértices en la iteración recursiva t. Para
                                                2
calcular la contribución de una cara incidente σijk = [vi , vj , vk ] al vértice vi , la trayectoria
óptima puede provenir de cualquier punto intermedio p(λ) = λvj + (1 − λ)vk sobre la
arista opuesta [vj , vk ], con λ ∈ [0, 1].
Aplicando la ley de cosenos en el triángulo definido por ℓij (θ), ℓik (θ), ℓjk (θ), la distancia
intrínseca exacta desde el vértice vi hasta el punto baricéntrico p(λ) atravesando la cara σijk          2
es libre de coordenadas extrínsecas:
                    dℓθ (λ) =
                                                       λ ℓ2ij (θ) + (1 − λ) ℓ2ik (θ) − λ(1 − λ) ℓ2jk (θ)
                                                      2 minimiza sobre λ ∈ [0, 1] la suma del
El funcional de acción local a través del 2-símplice σijk
costo métrico interior (ponderado por la impedancia de la cara Φijk ) y el valor interpolado en la
frontera opuesta:
              ) = min {Φijk (θ) dℓθ (λ) + λVτ(t) (vj ) + (1 − λ)Vτ(t) (vk ) +
  (t)     2
Qθ (vi ∣ σijk
                                            λ∈[0,1]
                                                                    3
            En un k -símplice general (por ejemplo, un tetraedro σijkl en 3D), λ se generaliza a un
            vector baricéntrico λ ∈ Δk−1 sobre la (k − 1)-cara opuesta, y la distancia cuadrática se
            evalúa mediante la forma bilineal inversa de la matriz de Gram simplicial:
                                                                                                                                     1
                                                    d2ℓθ (λ) = ∑ λa ℓ2ia −                                                             ∑ λa λb ℓ2ab
                                                                                                                                     2
                                                                                         a                                                a,b
            El minimizador analítico λ∗ (θ) se obtiene en forma cerrada resolviendo una ecuación
            cuadrática escalar en cada cara (o proyectando sobre los extremos λ∗ ∈ {0, 1} si la
                                                                                                                                                                                (t)
            característica cae fuera del cono simplicial), lo que hace que ∇θ Qθ sea exacto por el
            Teorema de la Envolvente.
Paso 2: Agregación Recursiva sobre la Estrella Simplicial St(vi )
Definido el costo óptimo de llegada desde cada cara incidente en la estrella del vértice
              2                  2
St(vi ) = {σijk
                ∈ K2 ∣ vi ∈ σijk
                                  }, el estado neuronal se actualiza mediante el operador Soft-
Bellman con temperatura τ > 0:
                                                                                                                                Q(t)      2
                                                                                                                         exp (−                 )
                                                                                                                                 θ (vi ∣ σijk )
                    Vτ(t+1) (vi ) = −τ log                                                ∑
                                                                                        2
                                                                                                                                     τ
                                                                                       σijk ∈St(vi )
3. Dinámica de Cocadenas y Laplaciano de Hodge-de Rham
Una vez alcanzado el punto fijo Vτ∗ (θ) tras T pasos recursivos (o mediante diferenciación
implícita en el equilibrio), la red induce un flujo de información sobre todo el complejo que se
articula con capas convolucionales simpliciales:
1.   Distribución de Enrutamiento por Caras (πθ ):
                                                                                         2
     La probabilidad de que el flujo geodésico en el vértice vi atraviese el 2-símplice σijk viene
     dada por la distribución de Boltzmann sobre su estrella:
                         2
                                                                                       exp ( − Q∗θ (vi ∣ σijk
                                                                                                          2
                                                                                                              )/τ )
                    πθ (σijk ∣ vi ) =
                                                                                  ∑σ′ ∈St(vi ) exp ( − Q∗θ (vi ∣ σ ′ )/τ )
2.   Acoplamiento con el Laplaciano de Hodge (Δk ):
     Si cada símplice de dimensión k porta además un vector de características ocultas
     h(k) ∈ C k (K; Rd ), su propagación guiada por la métrica aprendida sigue el operador de
     Hodge-Laplace:
            Δk (θ) = dk−1 ⋆−1
                                    ⊤             −1     ⊤
                           k−1 (θ) dk−1 ⋆k (θ) + ⋆k (θ) dk ⋆k+1 (θ) dk
                                          Δdown
                                           k
                                                (frontera inferior)
                                                                                                                                                Δup
                                                                                                                                                 k
                                                                                                                                                    (cofrontera superior)
     Para k = 1 (señales en aristas), Δdown1    propaga gradientes de potencial entre vértices,
                    up
     mientras que Δ1 acopla las aristas a través de la métrica de las caras ⋆2 (θ). Por la
     Descomposición de Hodge, la red separa de manera natural el flujo irrotacional
     (gradiente exacto d0 Vτ ), el flujo solenoidal (rotacional en caras δ2 Φ) y los modos
     armónicos topológicos (ker Δ1 ≅ H 1 (K)) que rodean agujeros o cavidades del
     espacio de estados.
4. Funcional de Entrenamiento: Energía Libre y Curvatura de
Regge
Para entrenar los pesos métricos θ mediante retropropagación sin que los símplices colapsen
en volúmenes nulos o degenerados, la función de pérdida combina el objetivo de enrutamiento
con dos regularizadores geométricos intrínsecos:
               L(θ) = Lruta (Vτ∗ (θ)) + α SRegge (ℓθ ) + β BCM (ℓθ )
     Acción de Curvatura Discreta de Regge (SRegge ): En un complejo de dimensión n, la
     curvatura intrínseca reside en los (n − 2)-símplices o bisagras h ∈ Kn−2 (vértices en
     variedades 2D, aristas en complejos 3D). El ángulo de déficit en la bisagra h es:
                                              ϵh (ℓθ ) = 2π − ∑ ϑ(σ n , h; ℓθ )
                                                                                               σ n ⊃h
                                                       del n-símplice σ n en la bisagra h,
     Donde ϑ(σ n , h; ℓθ ) es el ángulo diedro interior
     calculado analíticamente mediante el coseno inverso de los productos internos de las
     normales de sus (n − 1)-caras. El regularizador penaliza o estructura la curvatura
     concentrada:
                                SRegge (ℓθ ) = ∑ Voln−2 (h; ℓθ ) ⋅ ∣ϵh (ℓθ )∣2
                                                                                       h∈Kn−2
      Barrera Logarítmica de Cayley-Menger (BCM ): Impide que el descenso de gradiente
      viole la desigualdad métrica simplicial manteniendo los determinantes estrictamente
      dentro del cono métrico válido:
                                                                                         n
                                BCM (ℓθ ) = − ∑ ∑ log (Vol2k (σ k ; ℓθ ))
                                                                                        k=2 σ k ∈Kk
5. Síntesis de Operadores en la Red Neuronal Simplicial
Entidad Simplicial                  Variable Aprendible                                                               Operador Diferencial                                        Rol en la Optim
                                                                                                                                                                                  de Ruta
0-Símplice (σi0 )                  Potencial nodal Vτ (vi )                                                        Gradiente discreto d0
                                                                                                                                                                                  Almacena la e
                                                                                                                                                                                  libre acumulad
                                                                                                                                                                                  el atractor
             1                                                                                                        Estrella de Hodge                                           Define distanc
1-Símplice (σij )
                                   Longitud ℓij (θ) y fase
                                    Aij (θ)
                                                                                                                      ⋆1 (θ)
                                                                                                                                                                                  intrínsecas y a
                                                                                                                                                                                  direccional
             2                                                                                                        Minimización                                                Permite cruce
2-Símplice (σijk )                 Impedancia Φijk (θ) y
                                    flujo Bijk
                                                                                                                      baricéntrica Qθ
                                                                                                                                                                                  geodésico inte
                                                                                                                                                                                  refracción sup
Bisagra (h ∈ Kn−2 )
                                    Ángulo de déficit                                                                 Acción de Regge                                             Controla la cu
                                    ϵh (ℓθ )
                                                                                                                    SRegge
                                                                                                                                                                                  intrínseca y
                                                                                                                                                                                  focalización d
Complejo Global (K)                 Espectro armónico                                                                 Laplaciano de Hodge                                         Conserva inva
                                    ker(Δk )                                                                         Δk (θ)                                                     homológicos (
                                                                                                                                                                                  durante el fluj
En la tabla, los operadores d^k y ⋆_k(θ) garantizan que la retropropagación actualice simultáneamente las longitudes de las aristas ℓ_ij y las impedancias de las caras Φ_ijk mediante diferenciación automática sobre la fórmula baricéntrica Q_θ.
Para inspeccionar geométricamente cómo la modificación de las longitudes de las aristas
ℓij (θ), la impedancia de las caras Φijk (θ) y el déficit de curvatura de Regge ϵh deforman un
complejo simplicial tridimensional y alteran en tiempo real la geodésica intrasimplicial Soft-
Bellman, explora el siguiente visualizador interactivo 3D:
               Generating interactive visual...
Structuring code, this may take a minute
Al alternar entre una impedancia de caras alta y baja en el modelo 3D, se aprecia la diferencia
esencial frente a un grafo estándar: cuando el peso superficial Φijk disminuye en una región de
caras triangulares, la trayectoria óptima abandona el 1-esqueleto de las aristas y corta
directamente a través del interior baricéntrico (λ∗ ∈ (0, 1)) de los 2-símplices, comportándose
como un rayo óptico que se refracta según la ley de Snell discreta sobre la variedad simplicial.
Para avanzar hacia la implementación o extensiones teóricas de esta
arquitectura:
  Programar la capa simplicial                      Incorporar métricas                               Conectar la recurrencia con
  Soft-Bellman en PyTorch o                         lorentzianas y                                    Energía Libre Variacional
  Rust                                              Triangulaciones Dinámicas                         multi-agente
                                                    Causales
