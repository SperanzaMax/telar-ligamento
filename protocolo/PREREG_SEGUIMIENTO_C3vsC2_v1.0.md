# Pre-registro de SEGUIMIENTO — E1 · lectura de capacidad C3 vs. C2 (v1.0, CONGELADO)

**Estatus:** **RATIFICADO por Maxi y CONGELADO 2026-07-22**, antes de correr una sola semilla de C2
(delta). Hash SHA-256 y metadatos en el companion `FREEZE_PREREG_SEGUIMIENTO_v1.0.md`. Las predicciones
PS-1/PS-2/PS-3 son **idénticas** a las del borrador v0.1 anclado con timestamp server-side de GitHub en el
commit `8a6e690` (push 2026-07-22T19:08:03Z) — el único cambio de v0.1→v1.0 es este encabezado de estatus
(verificable con `git diff`). Doble ancla: timestamp del borrador (antes de datos) + hash del congelado.

**NO modifica el protocolo congelado v1.0** (`PROTOCOLO_v1.0.md`, SHA-256 `2f8ebb82…`, intacto). Es un
protocolo **hermano de seguimiento**: promueve a predicción pre-registrada un análisis que en v1.0 vivía
como **exploratorio** (el apéndice de E1: «C3 vs C2»).

## Motivación (con el dato que lo dispara)

El S0.9 preview (2026-07-22, `resultados/fase0/s09_c1_preview.json`) midió **C1 (softmax) acc@1 = 1.000 en
todas las cargas {8,16,32,64,96,128}**. Por la regla **D2**, esto activa la salida honesta **«P1.1 no
evaluable por saturación del baseline»**: MQAR con claves exactas es un *lookup trivial* para la atención
plena (KV completo, sin cuello de botella de rango), así que comparar C3 ≈ C1 **no informa** — C1 está en
techo por construcción, no por mérito.

La pregunta de capacidad **sí informativa** que queda es: **¿las 2 cabezas softmax de C3 rescatan la
capacidad que el estado de rango finito de C2 (delta) desperdicia a cargas altas?** Eso es C3 vs. C2, y
para que tenga valor confirmatorio (no post-hoc) se pre-registra **acá, antes** de ver C2 a esas cargas.

## Método

Hereda R1–R11 del protocolo v1.0 (mismas condiciones, arquitectura §5, vocab E-001, 8 semillas para
equivalencias/rescate, márgenes R11 apareados por semilla, tres veredictos R3). Baseline de referencia de
esta familia: **C2 (delta)** en T1 (reemplaza a C1, que satura).

**Carga de evaluación de capacidad (análoga a D2, con delta como referencia).** Se corre T1 en
L ∈ {8,16,32,64,96,128}. La **carga de evaluación** = la **menor L donde C2 (delta, media de 8 semillas)
cae por debajo de 95% acc@1**. Se fija en S0.9 (definitivo), registrada en `margenes_instanciados.md`,
**antes** de mirar C3. Si C2 no cae en todo el rango (improbable, dado el plateau ~67% esperado), se usa
la carga máxima (128) y se documenta.

## Predicciones pre-registradas

- **PS-1 (rescate — la central).** En la carga de evaluación, **C3 > C2** en acc@1, con
  `C3 − C2 > margen efectivo` (R11) e IC bootstrap 95% de la diferencia **apareada por semilla** que no
  cruce cero por debajo. Piso del margen: 2 puntos.
  - **Confirma:** las cabezas softmax rescatan capacidad dentro de la capa mixta.
  - **Falsa:** `C3 ≤ C2` bajo R3 (las softmax no rescatan; o interferencia — se cruza con P1.3 de v1.0).
  - **No concluyente:** el IC apareado cruza el margen.
- **PS-2 (¿techo o intermedio?) — descriptiva, sin veredicto binario.** Se reporta la posición de C3
  entre C2 (piso) y C1 (techo=1.0) en la carga de evaluación: `f = (C3 − C2) / (C1 − C2)`. f≈1 → C3 alcanza
  el techo; f≈0 → no rescata; 0<f<1 → rescate parcial. Solo se reporta (informa el mecanismo), no confirma
  ni falsa.
- **PS-3 (monotonía del rescate).** El rescate `C3 − C2` **no decrece** al subir la carga en el tramo donde
  C2 < 95% (si las softmax rescatan, la ventaja debería crecer o sostenerse con la presión de capacidad).
  Criterio: mediana sobre semillas de Spearman(C3−C2, L) ≥ 0 en ese tramo. Secundaria.

## Congelamiento

Este borrador se congela (hash SHA-256 + registro en un companion) tras la ratificación de Maxi y **antes
de la primera corrida del S0.9 definitivo** que mide C2 a cargas nuevas. Hasta entonces, el ancla es el
timestamp del push de este archivo. Ninguna predicción se ajusta después de ver datos.

*Autoría del diseño: Maxi + ejecutor Opus 4.8, sobre el marco de Fable 5. Programa TELAR / Ligamento.*
