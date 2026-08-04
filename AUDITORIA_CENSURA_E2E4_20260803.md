# Auditoría de censura de E2–E4 — Paso 0 (0 h de GPU)

**Fecha:** 2026-08-03. Ejecuta el Paso 0 prescrito por la capa de razonamiento el 2026-08-03 en
respuesta a la pregunta 3 de `CIERRE_E1_20260803.md`. Clasifica cada veredicto pre-registrado de
E2, E3 y E4 según si puede correrse en el régimen actual o debe esperar al régimen calibrado.

Principio que ordena todo, y que se declara antes de clasificar: **las reglas de veredicto no pueden
condicionarse en datos vistos; las decisiones de instrumento sí pueden y deben.** Elegir el régimen
de E2–E4 a la luz de la censura medida en E1 es diseño de instrumento, no elección de hipótesis.

---

## 1. Criterio — fijado ANTES de mirar la clasificación

> Un veredicto pre-registrado **corre en el régimen actual si y solo si ambos brazos del contraste
> tienen al menos una celda esperada fuera del techo**, con el corte en **≤ 0,98**.

Y una precisión que el corte solo no cubre, derivada de lo que E1 ya mostró: **una predicción de
equivalencia se confirma trivialmente cuando ambos brazos saturan.** Si A = B = 1,000, «empatan
dentro del margen» es verdadero por saturación y la predicción es infalsable en ese régimen — el
mismo defecto por el que P1.1 quedó «no evaluable por saturación» (D2). Por eso las equivalencias se
clasifican por un criterio más exigente: **ambos brazos deben tener rango dinámico, no basta con que
la diferencia sea medible.**

Categorías:

- **VERDE** — corre como está.
- **AMARILLO** — corre, pero produce una **cota inferior**, no una medición: un brazo con rango
  dinámico contra un brazo en el techo. Es exactamente el patrón de PS-1.
- **ROJO** — no corre en el régimen actual: ambos brazos esperados en el techo. Un instrumento
  saturado no produce nulos informativos, produce **vacíos**.

**Evidencia que alimenta el prior.** E1 midió la arquitectura base (§5) sobre tareas de recuperación:
`softmax` dio 0/144 celdas bajo 0,99 hasta L = 128, techo duro de la tarea por E-001. E2–E4 **no
varían la regla de escritura de las cabezas** — varían rutas de contexto, compuertas y partición del
FFN — de modo que corren sobre cabezas softmax, el brazo que en E1 nunca cayó del techo.

---

## 2. Clasificación

| # | predicción | métrica | brazos | veredicto |
|---|---|---|---|---|
| **P2.1** | gap polisémico B − A crece con ρ | exactitud | A restringido **por diseño** (el tronco no accede al contexto) vs. B con acceso pleno | 🟡 **AMARILLO** |
| **P2.2** | D empata con B (**equivalencia**) | exactitud | ambos con acceso pleno al contexto | 🔴 **ROJO** |
| **P2.3** | apertura de compuertas crece con ρ | **mecanismo** (\|tanh g_l\|) | — | 🟢 **VERDE** |
| **P3.1** | profundidad efectiva: difícil > fácil | **mecanismo** (bloques/token) | — | 🟢 **VERDE** |
| **P3.2** | tasa de consulta: ambiguo > claro | **mecanismo** (razón ≥ 1,5×) | — | 🟢 **VERDE** |
| **P3.3** | disociación profundidad × consulta | **mecanismo** (dos IC bootstrap) | — | 🟢 **VERDE** |
| **P4.1** | M ≥ F y M ≥ S en emparejamiento cruzado | exactitud | T5-b es binaria balanceada sobre 32 conceptos latentes en 3 proxies de modalidad | 🟢 **VERDE** (con reserva, §3) |
| **P4.2** | fracción de unidades multimodales crece con la profundidad | **mecanismo** (probe vs. nula permutada) | — | 🟢 **VERDE** |
| **P4.3** | F empata con M (**equivalencia**, falsación) | exactitud | los mismos brazos de P4.1 | 🟡 **AMARILLO** (§3) |

**Salida mecánica: 5 verdes, 2 amarillos, 1 rojo.** El programa no está en riesgo global.

## 3. Lectura

**Lo que salva a E3 entero y a la mitad de E2 y E4: sus predicciones centrales no son de exactitud.**
Profundidad efectiva, tasa de consulta, apertura de compuertas y fracción de unidades multimodales
son magnitudes de mecanismo. El techo de accuracy no las toca: un modelo puede acertar el 100 % y aun
así rutear distinto según dificultad. **E3 es íntegramente verde** — sus tres predicciones son
disociaciones de compuertas, y su falsación (correlación token a token > 0,9) tampoco depende de
accuracy.

**El rojo es P2.2 y es del tipo peor.** Es una **equivalencia** entre dos rutas que tienen las dos
acceso pleno al contexto, sobre una tarea de recuperación con cabezas softmax. Si ambas dan 1,000, D
«empata» con B dentro de cualquier margen y P2.2 se confirma **sin haber medido nada**. Peor que un
vacío: un vacío que se lee como confirmación. Correrla en el régimen actual es gastar 8 semillas para
producir un renglón que no puede fallar.

**Los amarillos son honestos pero conocidos.** P2.1 mide, otra vez, el brazo débil: A está limitado
**por diseño** (el tronco no accede al contexto), así que tiene rango dinámico garantizado y el
contraste vive. Pero si B satura, el gap es una cota inferior sobre lo que A pierde, no una medición
de lo que B gana — la estructura exacta de PS-1. Es publicable con la limitación declarada; lo que no
puede es venderse como medición del beneficio de inyectar contexto por capa.

**La reserva sobre P4.1/P4.3.** T5-b no es una tarea de recuperación: es emparejamiento cruzado
binario balanceado (chance 50 %) entre proxies de modalidad generados por mapas distintos. No hay
razón teórica para esperar techo, y por eso P4.1 va en verde. Pero **no está medido**, y P4.3 es una
equivalencia: si T5-b resultara trivial para las tres condiciones, P4.3 se confirma por saturación
igual que P2.2. Por eso P4.3 queda en amarillo y **no en verde**, pese a compartir brazos con P4.1.
Es la única casilla del triage que un dato barato podría mover: un range-finding de T5-b con 2
semillas resuelve las dos a la vez.

## 4. Consecuencia para la agenda

- **Corre ya, sin esperar calibración:** **E3 completo** (P3.1, P3.2, P3.3), P2.3, P4.2.
- **Corre con limitación declarada de cota inferior:** P2.1.
- **Espera al régimen calibrado:** **P2.2**.
- **Corre, con range-finding previo de T5-b (2 semillas) que decide si P4.3 es evaluable:** P4.1,
  P4.3.

Ningún veredicto se reformula, ningún umbral se toca. Lo único que cambia es **el orden y el
régimen** en que se corren — decisión de instrumento.

---

## 5. Paso 1 — regla de banda, para congelar antes de calibrar

Propuesta de redacción, a congelar con hash **antes** de la primera hora de calibración, de modo que
la elección de régimen sea mecánica dado el dato:

> **R-BANDA.** El régimen de E2–E4 será la **menor extensión** (vocabulario, grilla de cargas) tal
> que la condición más fuerte de E1 —`softmax`— caiga dentro de **[0,30 · 0,98]** en al menos **tres
> cargas de la franja superior** de la grilla extendida. Entre extensiones que cumplan la banda se
> elige la de menor vocabulario; si empatan, la de menor carga máxima. La calibración se corre
> **solo sobre condiciones cuyos veredictos confirmatorios ya están cerrados** (`delta`, `mix22`,
> `softmax`), con **2 semillas, N = 2500**, declarada como *range-finding sin veredictos*, y con
> **tope duro de 6 h de T4**; la búsqueda es por bisección adaptativa al cruce de 0,5 y se corta al
> agotar el tope, no al completar una grilla.

Cortafuegos que la sostienen: (i) el régimen se elige con **condiciones cerradas**, neutral respecto
de los contrastes nuevos de E2–E4; (ii) la banda se fija antes de ver dónde cae el borde; (iii) el
range-finding no emite veredictos.

**Rama pre-escrita para «frontera inalcanzable»:** si `mix22` sigue en 1,0000 al agotar el tope de
6 h, su frontera excede lo medible en T4 a este `d_model`. Consecuencia mecánica: E2–E4 se re-alcanzan
a contrastes con al menos un brazo fuera del techo, y cualquier claim de frontera pasa a una campaña
futura con `d` menor — **como rediseño documentado, no como parche**, porque achicar `d` rompe el
«misma arquitectura» que hoy ancla la comparabilidad entre campañas.

**Además, sin GPU:** registrar el **margen logit** como métrica secundaria pre-registrada en E2–E4.
En el techo la accuracy queda ciega, pero los márgenes siguen ordenando condiciones. Cuesta logging,
no cómputo. Las primarias siguen siendo las de accuracy: la comparabilidad manda.

**Y celdas ancla:** cada campaña del régimen nuevo incluye un par de cargas del régimen viejo
(vocab 128) para comparabilidad con E1.

## 6. Observación del ejecutor sobre el costo declarado

La calibración se justifica por dos vías a la vez y conviene decirlo explícito: **~6 h aseguran ~40**,
y el número que produce —dónde está el borde de `mix22`— **es la cifra que hoy falta en el paper de
E1**, cuya sección de limitaciones dice que la grilla nunca encontró el borde. No es overhead: es la
figura pendiente y el seguro, con el mismo gasto.

Con una salvedad que el ejecutor declara porque afecta el presupuesto: la calibración exige **vocab
extendido** (256 o 512), y con vocab 128 y L = 128 la celda tope **degenera** —todas las claves del
pool están en juego, sin distractores no usados—, que es una razón adicional, independiente de la
censura, por la que ese punto mide mal. Cambiar el vocab cambia el embedding: la calibración **no**
puede reusar los checkpoints de E1, entrena de cero. Las 6 h son entrenamiento nuevo, no evaluación.
