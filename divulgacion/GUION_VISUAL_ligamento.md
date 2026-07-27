# GUION VISUAL — «LIGAMENTO»: cómo la información atraviesa cada red

**Para:** generador de gráficos/animación (ChatGPT u otro).
**Objetivo:** recrear visualmente, paso a paso, cómo fluye la información por las arquitecturas del
proyecto Ligamento — con foco en las dos reglas de memoria (softmax vs. delta) y en los cuatro
experimentos (E1–E4). Fidelidad técnica: todos los números son los reales del proyecto.

> **Cómo usar este guion:** cada ESCENA trae `OBJETIVO`, `EN PANTALLA` (qué dibujar), `ANIMACIÓN`
> (qué se mueve), `TEXTO` (rótulos/narración) y `NOTA` (aclaración técnica para no dibujar algo
> falso). Podés pedir cada escena como una imagen fija o como un plano animado de 4–8 s.

---

## 0 · ESTILO Y CONVENCIONES (leer antes de dibujar)

**REFERENCIA DE ESTILO (imagen provista):** el look base es el de una **visualización de "flujo de
atención"**: fondo azul profundo, y **haces/hilos luminosos curvos** de azul claro que conectan una
hilera de nodos, como fibras de luz. Esa estética es la firma **de la atención softmax** y se usa tal
cual para ella. **Importante:** NO todas las redes deben verse así — ese enjambre *es* softmax. Cada
regla/arquitectura necesita su **propia firma visual** que contraste, o el video diría una sola cosa.
Firmas por red (mantener distinguibles):
- **Softmax** → el **enjambre de haces** de la referencia (muchas conexiones, todos-con-todos).
- **Delta** → **anti-enjambre**: un **núcleo compacto** (grilla 16×16) que late y se corrige; casi sin
  haces. El contraste "tormenta de hilos" vs. "núcleo que late" es el eje visual del proyecto.
- **Mixto (E1)** → mitad enjambre + mitad núcleo, **entretejidos** (telar).
- **Contexto/compuertas (E2/E3)** → **válvulas** que se abren/cierran y **carriles** que se saltan.
- **Modalidades (E4)** → **tres carriles de color** que corren separados o se funden.

**Tono visual:** limpio, tipo "explicación de laboratorio moderno". Fondo azul muy oscuro `#0B1020`
(como la referencia). Formas simples, líneas finas, luz que fluye. Nada de ruido. Sans-serif. Estilo
coherente en TODAS las escenas. El **azul** domina el fondo y el flujo; los colores de concepto
(verde/magenta/ámbar de abajo) se usan como **acentos** para distinguir, sin romper la unidad azul.

**Paleta fija (respetar en todo el guion):**
- **Claves (keys)** → azul cian `#38BDF8`
- **Valores (values)** → ámbar `#F59E0B`
- **Tokens especiales** (BOS/SEP/PAD/contexto) → gris/violeta `#8B8FA3`
- **Regla SOFTMAX** → verde `#34D399`
- **Regla DELTA** → magenta/violeta `#C084FC`
- **Sorpresa / error de predicción** → rojo cálido `#FB7185` (destella)
- **Flujo de información** → partículas o trazos blancos `#FFFFFF` que viajan por los caminos
- **Texto** → blanco hueso `#E5E7EB`; acentos en el color del concepto que nombran

**Metáforas maestras (mantener consistentes):**
- Un **token** = una ficha hexagonal con su símbolo, coloreada según su tipo.
- Un **vector/embedding** = una columna de 64 celditas tipo mini-heatmap (barra vertical).
- La **memoria softmax** = una **estantería que crece** (guarda cada par visto).
- La **memoria delta** = un **cuaderno de tamaño fijo** (una grilla que se sobrescribe).
- El **flujo de datos** = luz blanca que viaja por los cables/caminos.

**Metáfora textil de fondo (marca del proyecto):** "ligamento" es el patrón de entrelazado entre
urdimbre y trama en un telar. Úsese sutil: hilos que se cruzan de fondo, no protagonista.

---

## ACTO 0 · EL MARCO — ¿especializar o compartir?

### ESCENA 0.1 — Título
- **OBJETIVO:** abrir con la pregunta madre.
- **EN PANTALLA:** título central **«LIGAMENTO»**. Debajo, subtítulo: *"¿Dónde conviene especializar
  y dónde conviene compartir, dentro de una misma red?"*. De fondo, hilos finos cruzándose (telar).
- **ANIMACIÓN:** los hilos se tejen y forman brevemente la silueta de una red neuronal.
- **TEXTO:** «LIGAMENTO» · *especialización ↔ compartición*.

### ESCENA 0.2 — Las dos neuronas
- **OBJETIVO:** encarnar la tensión central antes de tecnicismos.
- **EN PANTALLA:** dos neuronas estilizadas. La de la izquierda **ABSORBE** (un remolino que atrae luz
  y la guarda: "especializa"). La de la derecha **DEJA PASAR** (la luz la atraviesa: "comparte").
- **ANIMACIÓN:** un mismo paquete de luz llega a ambas; una lo retiene, la otra lo transmite.
- **TEXTO:** *"Algunas unidades absorben. Otras dejan pasar. ¿Cuál conviene, y cuándo?"*
- **NOTA:** esta dualidad (absorber/compartir) es el hilo conductor de los 4 experimentos.

---

## ACTO 1 · EL VOCABULARIO Y LA TAREA (así se ve el dato)

### ESCENA 1.1 — El vocabulario (197 símbolos)
- **OBJETIVO:** mostrar de qué está hecho el "lenguaje" de juguete.
- **EN PANTALLA:** tres grupos de fichas:
  - **128 fichas azules** = claves (`key`)
  - **64 fichas ámbar** = valores (`value`)
  - **5 fichas gris/violeta** = especiales: `BOS` (inicio), `SEP` (separador), `PAD` (relleno),
    `CTX_A` y `CTX_B` (señal de contexto)
- **TEXTO:** *"Vocabulario = 128 + 64 + 5 = 197 símbolos."*
- **NOTA:** son EXACTOS. 128 claves permiten pedir hasta 128 asociaciones distintas sin repetir.

### ESCENA 1.2 — La tarea MQAR (memoria asociativa)
- **OBJETIVO:** mostrar la tarea básica: guardar pares clave→valor y luego recuperarlos.
- **EN PANTALLA:** una cinta horizontal de fichas:
  `BOS · (k₁ v₁) · (k₂ v₂) · … · (k_L v_L) · SEP · q₁ · q₂ · …`
  Los pares clave-valor (azul+ámbar) entran primero; tras `SEP` llegan las **consultas** (claves azules
  sueltas, en otro orden).
- **ANIMACIÓN:** cada par `(kᵢ vᵢ)` que entra deja una "huella" en una memoria (a definir en Acto 3).
  Luego llega una consulta `qⱼ = kᵢ`: un pulso viaja de la consulta a la memoria y **trae de vuelta el
  valor `vᵢ`** correcto, que se ilumina en ámbar.
- **TEXTO:** *"Guardá L pares. Después, dada una clave, devolvé su valor."*
- **NOTA:** esto es T1. La métrica es acc@1 (¿el valor top-1 es el correcto?).

### ESCENA 1.3 — La carga sube: ¿entra todo?
- **OBJETIVO:** instalar la tensión de capacidad.
- **EN PANTALLA:** el número **L** sube: `8 → 16 → 32 → 64 → 96 → 128`. La memoria (un recipiente) se va
  llenando. A cargas altas empieza a "desbordar" o a confundir claves parecidas.
- **ANIMACIÓN:** contador de L subiendo; el recipiente se satura y algunas recuperaciones titilan/fallan.
- **TEXTO:** *"¿Cuántos pares entran antes de que la memoria empiece a fallar?"*
- **NOTA:** esta es LA pregunta de capacidad. La respuesta depende de la regla de memoria (Acto 3).

---

## ACTO 2 · LA ARQUITECTURA BASE (el camino de un token)

> Todas las variantes comparten este esqueleto. Números reales: `d_model = 64`, `H = 4 cabezas`,
> `d_head = 16`, `4 bloques`, `FFN oculta = 192`.

### ESCENA 2.1 — De símbolo a vector (embedding)
- **EN PANTALLA:** una ficha-token entra a una **tabla de embedding** (una grilla `197 × 64`). Sale una
  **columna de 64 celditas** de colores (el vector del token).
- **ANIMACIÓN:** la ficha "cae" en su fila de la tabla y emerge como barra de 64.
- **TEXTO:** *"Cada símbolo → un vector de 64 números."*

### ESCENA 2.2 — Un bloque por dentro (×4)
- **OBJETIVO:** mostrar el interior de UN bloque; luego indicar que se repite 4 veces.
- **EN PANTALLA (de izquierda a derecha, el vector atraviesa):**
  1. **Pre-norm (LayerNorm):** el vector se "estabiliza" (normaliza).
  2. **Conv causal depthwise (kernel 3):** cada posición mezcla un poquito a sus **2 vecinos previos**
     (una ventanita local hacia atrás).
  3. **MIXER (el corazón):** aquí ocurre la atención/memoria → *se detalla en el Acto 3*. Caja destacada.
  4. **Suma residual (+):** la salida del mixer se SUMA al vector original (atajo).
  5. **Pre-norm** de nuevo.
  6. **FFN:** el vector de 64 se **expande a 192**, pasa por una no linealidad (GELU) y **vuelve a 64**.
  7. **Suma residual (+).**
- **ANIMACIÓN:** el mismo hilo de luz recorre los 7 pasos; en las sumas residuales se ve el "atajo" que
  rodea el bloque y se reencuentra.
- **TEXTO:** *"Bloque = normalizar → mezcla local → MIXER → FFN, con atajos residuales."* Al final:
  *"× 4 bloques."*

### ESCENA 2.3 — Las 4 cabezas y la norma por cabeza (invariante)
- **OBJETIVO:** mostrar el multi-cabeza y el detalle que unifica todo el proyecto.
- **EN PANTALLA:** el vector de 64 se **parte en 4 trozos de 16** (las 4 cabezas). Cada cabeza procesa su
  trozo por separado. **Antes de volver a unirlos**, cada trozo pasa por una **RMSNorm por cabeza**
  (una "regla de calibración" idéntica para todas). Luego se **concatenan** y pasan por una proyección
  común `W_O`.
- **ANIMACIÓN:** el vector se abre en 4 carriles; cada carril se normaliza (breve destello uniforme);
  se vuelven a unir.
- **TEXTO:** *"4 cabezas de 16. Todas se calibran igual (RMSNorm por cabeza) — siempre, en toda variante."*
- **NOTA CRÍTICA:** este "misma calibración para todas las cabezas" es un **invariante** del diseño: sin
  él, las comparaciones del experimento E1 quedarían falseadas. Vale destacarlo visualmente (un candado
  o sello de "idéntico para todos").

### ESCENA 2.4 — De vector a predicción
- **EN PANTALLA:** tras los 4 bloques, el vector final pasa por una última norma y una proyección a
  **197 salidas** → una **barra de probabilidades** sobre el vocabulario → se elige la más alta (argmax).
- **ANIMACIÓN:** la barra de 197 se ilumina; el pico ámbar (un `value`) gana y se materializa como la
  ficha predicha.
- **TEXTO:** *"El vector final apuesta por un símbolo. ¿Acertó el valor correcto?"*

---

## ACTO 3 · LAS DOS REGLAS DE MEMORIA (el contraste central)

> Este acto es el más importante. Todo el proyecto gira sobre esta diferencia. Dedicarle el mayor
> cuidado visual. El "MIXER" del Acto 2.2 es una de estas dos cosas.

### ESCENA 3.1 — SOFTMAX: la estantería que guarda todo
- **OBJETIVO:** mostrar la atención plena como memoria explícita y sin límite de capacidad.
- **EN PANTALLA:** a medida que cada token entra, deja su par **(key, value)** en una **estantería que
  crece** hacia la derecha (una repisa de cajitas azul+ámbar, una por token visto).
- **ANIMACIÓN — cuando llega una consulta `q`:**
  1. `q` (azul) se compara con **TODAS** las keys guardadas: salen **haces de luz** desde `q` hacia cada
     key. El **grosor/brillo de cada haz = qué tan parecidos son** (producto punto `q·k`, escalado por
     `√16`).
  2. Esos parecidos pasan por **softmax** → se convierten en **pesos que suman 1** (los haces se
     re-normalizan; el más parecido brilla, los demás se atenúan).
  3. La respuesta es la **mezcla ponderada de los values**: el value de la key ganadora domina y se
     ilumina en ámbar.
- **TEXTO:** *"Softmax guarda cada par y, para responder, compara la consulta contra todos. No tiene
  cuello de botella de capacidad — pero el costo crece con el cuadrado del largo (n²)."*
- **NOTA:** clave para la narrativa: softmax **no satura** por capacidad; su límite es el cómputo, no la
  memoria. Por eso en los experimentos softmax se mantiene casi perfecto incluso con muchas claves.

### ESCENA 3.2 — DELTA: el cuaderno de tamaño fijo que se corrige
- **OBJETIVO:** mostrar la regla delta como un ESTADO comprimido que se escribe por "sorpresa".
- **EN PANTALLA:** una sola **grilla de 16×16** por cabeza = el **estado `S`** (el "cuaderno"). Tamaño
  FIJO, no crece nunca.
- **ANIMACIÓN — por cada token con su (key `k`, value `v`):**
  1. **LEER primero:** el estado predice `pred = S·k` (lo que el cuaderno *cree* que va con esa clave).
  2. **SORPRESA:** se calcula el error `err = v − pred` = *lo que el estado todavía no sabía*. Se
     representa con un **destello rojo**: **grande** si el cuaderno se equivocó mucho, **pequeño** si ya
     lo sabía.
  3. **ESCRIBIR:** se actualiza `S ← S + β · (err ⊗ k)` — un **estampado** (producto externo) del error
     sobre la clave, que ilumina una zona de la grilla. `β` es una **válvula aprendida** (0…1) que
     regula **cuánto** se escribe.
  4. **LEER la respuesta:** `y = S·q`.
- **TEXTO:** *"Delta guarda todo en UN estado de tamaño fijo. Escribe solo la sorpresa (lo que aún no
  sabía) y con una válvula β que decide cuánto. Barato (costo lineal) y corregible — pero el estado
  tiene capacidad limitada."*
- **NOTA CRÍTICA:** el cuello de botella es el **rango del estado** (~16). Cuando entran muchas más
  asociaciones que ese rango, el cuaderno se satura y empieza a confundir. **Aquí vive el "plateau"** de
  capacidad (~67% en cargas altas): es una propiedad geométrica del estado, no falta de entrenamiento.

### ESCENA 3.3 — Cara a cara
- **OBJETIVO:** fijar el contraste que motiva todo.
- **EN PANTALLA:** split-screen. Izquierda: **softmax = estantería infinita** (verde, crece, cara).
  Derecha: **delta = cuaderno fijo** (magenta, no crece, barato pero se llena).
- **TEXTO (dos columnas):**
  - Softmax → *capacidad casi ilimitada · costo n² · guarda explícito.*
  - Delta → *costo lineal · corregible (sobrescribe) · capacidad = tamaño del estado.*
- **NOTA:** esta tensión "capacidad vs. costo/corregibilidad" es la pregunta que E1 pone a prueba.

---

## ACTO 4 · E1 — CABEZAS MIXTAS (ADN heterogéneo en la misma capa)

### ESCENA 4.1 — Los dos baselines
- **EN PANTALLA:** dos modelos:
  - **C1:** 4 cabezas **softmax** (4 estanterías verdes).
  - **C2:** 4 cabezas **delta** (4 cuadernos magenta).
- **TEXTO:** *"C1 = pura capacidad. C2 = pura corregibilidad."*

### ESCENA 4.2 — La mezcla (C3)
- **OBJETIVO:** la idea central de E1 — reglas distintas conviviendo en la MISMA capa.
- **EN PANTALLA:** una capa con **4 cabezas: 2 verdes (softmax) + 2 magenta (delta)**, alimentadas por el
  **mismo stream residual**. Sus 4 salidas pasan por la **RMSNorm por cabeza** (idéntica, del Acto 2.3),
  se **concatenan** y salen por `W_O` común.
- **ANIMACIÓN:** el mismo vector entra a las 4 cabezas; 2 lo procesan como "estantería", 2 como
  "cuaderno"; las salidas se entretejen (guiño al telar) en un solo vector.
- **TEXTO:** *"C3: cabezas de ADN distinto, tejidas en la misma capa."*

### ESCENA 4.3 — Las dos hipótesis
- **OBJETIVO:** mostrar qué se está preguntando (sin spoilear resultado).
- **EN PANTALLA:** dos futuros posibles, lado a lado:
  - **Herencia (deseado):** C3 conserva la **capacidad** de las cabezas softmax **y** hereda la
    **corregibilidad** de las delta. (Dos luces que se suman.)
  - **Interferencia (riesgo):** las reglas se estorban al compartir el stream y C3 rinde **peor que
    ambas**. (Dos luces que se cancelan.)
- **TEXTO:** *"¿Lo mejor de ambos mundos, o se interfieren? Eso mide E1."*

---

## ACTO 5 · E2 — RUTAS DE CONTEXTO (¿cuándo hace falta el contexto, y dónde?)

### ESCENA 5.1 — El canal de contexto
- **EN PANTALLA:** aparte del flujo principal, un **canal de contexto**: un codificador chico que resume
  el contexto en **8 vectores clave/valor** consultables.
- **TEXTO:** *"Un canal lateral con el contexto, disponible bajo demanda."*

### ESCENA 5.2 — Cuatro maneras de usarlo (A/B/C/D)
- **OBJETIVO:** contrastar 4 estrategias de inyección.
- **EN PANTALLA:** el mismo token principal, cuatro variantes:
  - **A (Perceiver):** el tronco **ignora** el contexto; solo la decisión **final** lo consulta.
  - **B (siempre):** **cross-attention** al contexto en **los 4 bloques**, sin filtro.
  - **C (como entrada):** el contexto entra como **tokens más** al principio.
  - **D (con compuerta):** cross-attention por bloque con una **válvula aprendida `tanh(g)`** que empieza
    **cerrada** (g=0) y se abre **solo cuando hace falta**.
- **ANIMACIÓN:** en D, la válvula se **abre más** a medida que sube la **ambigüedad ρ**.
- **TEXTO:** *"¿Alcanza con mirar el contexto al final (A), o hay que inyectarlo capa por capa (B/D)?"*

### ESCENA 5.3 — Por qué importa: la tarea polisémica (T3)
- **EN PANTALLA:** una clave con **dos sentidos** (dos values posibles). Una señal de contexto
  (`CTX_A` o `CTX_B`) al inicio **decide cuál** de los dos es el correcto.
- **ANIMACIÓN:** misma clave, dos contextos → dos respuestas distintas. Sin el contexto, es un volado
  50/50.
- **TEXTO:** *"Con contexto, la clave es unívoca. Sin contexto, es una moneda al aire."*

---

## ACTO 6 · E3 — DOBLE COMPUERTA (¿proceso? × ¿consulto?)

### ESCENA 6.1 — Dos decisiones por token
- **OBJETIVO:** mostrar dos ejes independientes de "gasto".
- **EN PANTALLA:** cada token, al entrar a un bloque, enfrenta **dos semáforos**:
  - **Compuerta de cómputo** (¿proceso este token en este bloque, o lo dejo pasar de largo?). Router
    **causal por umbral** (estilo Mixture-of-Depths): si su "puntaje" supera 0.5, se procesa.
  - **Compuerta de contexto** (¿consulto el canal lateral, o no?).
- **ANIMACIÓN:** tokens fáciles **saltan** bloques (van por el atajo); tokens difíciles se **procesan a
  fondo**. Tokens ambiguos **encienden** la consulta de contexto; los claros no.
- **TEXTO:** *"¿Cuánto pienso este token? ¿Y necesito mirar el contexto? Dos preguntas distintas."*

### ESCENA 6.2 — La disociación (hipótesis central de E3)
- **EN PANTALLA:** una grilla 2×2 de casos (fácil/difícil × claro/ambiguo). Se resalta que
  **dificultad → más profundidad de cómputo**, y **ambigüedad → más consulta de contexto**, y que **son
  ejes separados** (no la misma cosa).
- **TEXTO:** *"Dificultad y ambigüedad piden recursos distintos. ¿Las dos compuertas se separan?"*

---

## ACTO 7 · E4 — PARTICIÓN MODAL (especializar por modalidad, o compartir)

### ESCENA 7.1 — Tres modalidades, un concepto
- **EN PANTALLA:** un mismo **concepto** latente se "renderiza" en **3 modalidades** distintas:
  **texto** (secuencia), **imagen** (grilla 8×8), **audio** (onda). Tres flujos de color distinto que
  comparten un significado común.
- **TEXTO:** *"Mismo concepto, tres formas. ¿Conviene procesarlas separadas o juntas?"*

### ESCENA 7.2 — Tres arquitecturas (F / S / M)
- **EN PANTALLA:** tres variantes de la capa FFN:
  - **F (fija):** **expertos separados por modalidad** en **todas** las capas (3 carriles que nunca se
    tocan). Especialización total.
  - **S (compartida):** **un FFN común** para todo (los 3 flujos se **funden** en un solo carril).
    Compartición total.
  - **M (mixta):** **separados abajo** (bloques 1–2) y **compartidos arriba** (bloques 3–4).
- **ANIMACIÓN:** los 3 flujos de color entran; en F corren en paralelo sin mezclarse; en S se funden ya
  en el primer bloque; en M corren separados y luego confluyen.
- **TEXTO:** *"¿Dónde poner la frontera entre especializar y compartir?"*

### ESCENA 7.3 — Unidades multimodales
- **OBJETIVO:** mostrar el fenómeno emergente que se mide.
- **EN PANTALLA:** dentro de la red, algunas neuronas se **iluminan ante las 3 modalidades a la vez**
  (responden al *concepto*, no a la forma). Se marcan como **"unidades multimodales"**.
- **ANIMACIÓN:** llegan texto, imagen y audio del mismo concepto; una neurona se enciende con los tres.
- **TEXTO:** *"¿Emergen neuronas que entienden el concepto sin importar la modalidad? ¿A qué profundidad?"*

---

## ACTO 8 (OPCIONAL) · EL MÉTODO — pre-registro y honestidad

> Incluir solo si se quiere cerrar con el "cómo se hace ciencia" del proyecto. Se puede omitir para una
> pieza puramente técnica.

### ESCENA 8.1 — Congelar antes de mirar
- **EN PANTALLA:** el documento del protocolo se **sella** (candado + huella digital/hash) y se **fecha
  públicamente** ANTES de correr un solo experimento.
- **TEXTO:** *"Las predicciones se congelan antes de ver los datos. Si una falla, se reporta igual."*
- **NOTA:** metáfora: un sobre lacrado con la fecha estampada desde afuera (nadie puede cambiarla).

### ESCENA 8.2 — El plateau como geometría
- **EN PANTALLA:** volver al "cuaderno" delta (Acto 3.2): aunque se entrene infinito, el estado de rango
  ~16 **no puede** guardar 64 asociaciones nítidas → el techo ~67% es **geometría**, no falta de esfuerzo.
- **TEXTO:** *"El techo de memoria no es un problema de entrenamiento: es el tamaño del estado."*

---

## RESUMEN DE CONTINUIDAD (para que ChatGPT mantenga coherencia)

| Elemento | Representación fija |
|---|---|
| Clave / Valor / Especial | azul cian / ámbar / gris-violeta |
| Softmax | verde · "estantería que crece" · haces a todas las keys |
| Delta | magenta · "cuaderno 16×16 fijo" · escribe la sorpresa (destello rojo) con válvula β |
| Vector | columna de 64 celditas (mini-heatmap) |
| Flujo | luz blanca viajando por los caminos |
| Residual | atajo que rodea el bloque y se reencuentra en un "+" |
| RMSNorm por cabeza | destello uniforme idéntico en las 4 cabezas (candado "igual para todas") |
| 4 bloques · d=64 · 4 cabezas · d_head=16 · FFN 192 | números reales, no cambiar |

**Orden narrativo sugerido para un video de ~3–4 min:** 0 → 1 → 2 → **3 (el corazón, dale aire)** →
4 → 5 → 6 → 7 → (8 opcional). Los actos 3 y 4 son los que más recompensan una buena animación.

*Fin del guion. Todos los detalles técnicos corresponden a la implementación real del proyecto
Ligamento (arquitectura §5, vocabulario 128/64/5, reglas softmax y delta).*
