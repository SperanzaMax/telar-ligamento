# TELAR-EXP · «Ligamento» — Protocolo experimental v1.0

**Estado:** **CONGELADO** (2026-07-22). Hash SHA-256 y metadatos de depósito en el archivo companion `FREEZE_v1.0.md` (un archivo no puede contener su propio hash). Las predicciones P1.1…P4.3 quedan pre-registradas e inmutables; cualquier ajuste posterior sería un protocolo nuevo, no una edición de este.
**Cambios v0.3.1 → v1.0** (decisiones de Maxi, consenso de los tres, 2026-07-22): **D1 resuelta** — se quita el factor √2 de R11 (revierte el reescalado introducido como O2 en v0.3): margen efectivo = `máx(piso, 1.5 × SD entre semillas del baseline)`, apareo por semilla conservado, tres veredictos de R3 como salvaguarda direccional. **D2 resuelta** — entra tal cual estaba redactada (carga de evaluación de E1 fijada empíricamente por saturación del baseline). La sección «Decisiones pendientes» se conserva como «Decisiones resueltas» (rastro del pre-registro dentro del doc congelado). Reconciliación hermana en TELAR-03 completada (`telar03/docs/reconciliation_softmax_L64.md`): softmax nunca saturó a ~67% — el plateau era de las reglas de estado; el resumen simplificaba de más. Esto motivó y valida D2.
**Cambios v0.3 → v0.3.1** (verificación del ejecutor, 2026-07-22): (Ajuste 1, aplicado) la RMSNorm por cabeza pasa de rasgo de C3 a **invariante de la arquitectura base** (§5), de modo que las cabezas softmax de C3 sean arquitecturalmente idénticas a las de C1 — sin esto se confundían P1.1/P1.2; (nuevo) toda equivalencia se reporta a **tres veredictos** (empate / diferencia / no concluyente, R3); (nuevo) sección de decisiones con D1 (dirección del factor √2 en R11) y D2 (regla de carga por saturación del baseline en E1, motivada por los datos de TELAR-03 Fase 0: softmax dio 0.994 de acc@1 a L64 en el re-entreno).
**Cambios v0.2 → v0.3** (ronda de lectura del ejecutor, 2026-07-22): **Bloqueantes:** (A1) R2 pasa a régimen por experimento: en E1 se igualan solo parámetros y se declara el sesgo conservador (C3 corre con desventaja de FLOPs frente a C1); (A2) E4 suma el control S⅓ y una regla de desambiguación pre-registrada para P4.1 (allí el desbalance de FLOPs NO era conservador); (A3) P2.1 corrige su dominio: Spearman sobre ρ ∈ {0.25…1.0}, el chequeo de ρ=0 se muda a exactitud total. **Operacionalización:** (O1) umbral de unidad multimodal sube a percentil 95 del nulo por permutación; (O2) margen efectivo reescalado por √2 y tests de equivalencia con diferencias apareadas por semilla *(el reescalado por √2 fue revertido en v1.0 por D1; el apareo se conserva)*; (O3) P1.2 pasa de cociente a contraste lineal; (O4) igualación de largo de secuencia entre rutas en E2. **Técnicos:** (T1) S0.7 fija hilos de XLA y tolerancias numéricas; (T2) el top-k de MoD se reemplaza por router causal por umbral con pérdida de carga. **Menores:** especificación de la concatenación de cabezas heterogéneas en C3 (RMSNorm por cabeza), selección de LR por exactitud de validación, tests del generador T3.
**Cambios v0.1 → v0.2** (ronda de umbrales): R11 (márgenes anclados a ruido), P1.2 por fracción de brecha, 8 semillas en equivalencias, P2.1 por tendencia, P2.3 a 0.7.
**Programa:** Cortex-Nexus / TELAR · **Fecha de diseño:** 2026-07-21 · **Fecha de freeze:** 2026-07-22 · **Autor:** Maxi (SperanzaMax)
**Ejecutor previsto:** Claude Opus 4.8 vía Claude Code. **Diseño adversarial:** Fable 5 + Maxi + lectura de ejecutor (Opus 4.8).
**Numeración:** pendiente de asignación dentro del programa TELAR. Nombre de trabajo: **«Ligamento»** (provisional, ex-«Trama», cedido para no colisionar con TELAR-03-capacidad): en tejeduría, el ligamento es el patrón técnico de entrelazado entre urdimbre y trama — apropiado para un protocolo cuyo experimento central entreteje cabezas de ADN distinto en la misma capa.

---

## 0. Instrucciones para el modelo ejecutor

1. **Leé este documento completo antes de escribir una sola línea de código.**
2. Orden de ejecución estricto: **Fase 0 → E1 → E2 → E3 → E4.** No se inicia un experimento sin haber emitido el informe parcial del anterior.
3. Las predicciones (P1.1 … P4.3) están **pre-registradas**. No se modifican, no se reinterpretan después de ver datos. Si una predicción resulta mal formulada, se documenta el problema y se reporta como «no concluyente por defecto de formulación».
4. Toda desviación operativa (bug, cambio de hiperparámetro, reducción de pasos) se anota en `desviaciones.md` con fecha, motivo y alcance, **antes** de mirar los resultados afectados.
5. Si un chequeo de sanidad de la Fase 0 falla: **frenar y reportar.** No improvisar arreglos silenciosos.
6. Los resultados negativos son resultados. Se reporta todo lo corrido, con el mismo nivel de detalle que un resultado positivo.
7. Cada experimento produce tres artefactos obligatorios: `EX_resultados.json`, `EX_informe.md` y las figuras listadas en su sección.
8. Los datos sintéticos se generan con generadores con semilla propia y **tests unitarios** que verifiquen sus propiedades de diseño (ver §4).
9. Idioma de informes: castellano; términos técnicos en inglés donde sea el estándar del campo.
10. Si el cómputo disponible no alcanza para las cuatro líneas, la prioridad es **E1 > E2 > E3 > E4** (E4 es el más caro). Se documenta qué quedó sin correr.

---

## 1. Contexto y motivación

**Linaje.** TELAR-01 midió reglas de escritura de memoria (softmax, linear/Hebbiana, delta, Titans) sobre MQAR (~117k parámetros, JAX, CPU) y encontró que la escritura basada en gradiente compra **correctabilidad** (delta: 96,4% de exactitud de sobreescritura) pero **no capacidad** (las reglas de estado plafonean ~60–70% a carga 64; softmax, atención plena, no satura en ese régimen), con orden delta > titans robusto en 5 semillas.

**Origen de este protocolo.** Conversación de diseño del 2026-07-21 que exploró cuatro ideas arquitecturales: (a) representación universal con contexto reintegrado al final vs. inyectado por capa; (b) canal de contexto paralelo consultable bajo demanda mediante compuertas aprendidas; (c) cómputo condicional con bypass residual por token (estilo Mixture-of-Depths); (d) especialización por modalidad dentro de la capa (partición fija vs. emergente vs. mixta) y heterogeneidad arquitectural («ADN» distinto) entre cabezas de una misma capa.

**Pregunta paraguas.** ¿Dónde conviene **especializar** (ADN heterogéneo, compuertas selectivas, expertos por modalidad) y dónde conviene **compartir** (stream residual común, capas multimodales, cómputo denso)? ¿La frontera es medible en régimen de escala chica?

**Referencias conceptuales mínimas** (para el informe, no hace falta reproducirlas): Perceiver IO (decodificación por queries), Flamingo (gated cross-attention, init 0), Mixture-of-Depths (routing por capacidad con bypass residual), VLMo/BEiT-3 (expertos por modalidad), LIMoE (especialización emergente y acaparamiento), Hymba (cabezas heterogéneas en paralelo), MAD (perfilado de primitivas con tareas sintéticas chicas), Goh et al. 2021 (neuronas multimodales en CLIP).

---

## 2. Reglas metodológicas globales (obligatorias)

- **R1 — Pre-registro.** Este protocolo se congela (hash SHA-256 del archivo + depósito) antes de la primera corrida principal. Los pilotos previos sirven solo para verificar infraestructura y estimar tiempos; está prohibido elegir hiperparámetros mirando métricas de tarea en pilotos.
- **R2 — Presupuestos igualados, con régimen por experimento.** Regla general: toda comparación exige parámetros dentro de ±5% **y** FLOPs por token dentro de ±5%, ambos reportados en tabla. Excepciones pre-registradas donde la arquitectura hace imposible igualar ambos ejes a la vez: **E1** — softmax es O(n²) y delta es O(n) en secuencia, así que C1 > C3 > C2 en FLOPs de atención por construcción; se igualan parámetros, se reportan FLOPs por condición, y se declara la dirección del sesgo: C3 corre con **desventaja** de cómputo frente a C1, por lo que toda confirmación de P1.1/P1.2 es conservadora respecto de la hipótesis. **E3** — cómputo condicional: se igualan y reportan los FLOPs efectivos promedio realizados. **E4** — el ruteo por modalidad computa 1 de 3 expertos (mismos parámetros, ~⅓ de los FLOPs de FFN de S); acá el desbalance **no** es conservador (favorecería a M en P4.1) y se resuelve con el control S⅓ y la regla de desambiguación de §9, no con una relajación silenciosa.
- **R3 — Semillas y criterio de orden.** Mínimo 5 semillas por condición (0–4). Los experimentos que contienen predicciones de **equivalencia** (P1.1, P2.2, P4.3) corren **8 semillas (0–7)** en todas las condiciones involucradas: demostrar un empate dentro de un margen exige más potencia que demostrar una diferencia, no menos. Se reporta media ± desvío estándar. Una afirmación de orden «A > B» requiere: (i) signo consistente en ≥80% de las semillas (≥4/5 o ≥7/8 según corresponda), y (ii) intervalo de confianza bootstrap del 95% de la diferencia de medias (10 000 remuestreos sobre semillas) que no cruce cero. Una afirmación de equivalencia «A ≈ B» se reporta a **tres veredictos**: **empate** si el IC bootstrap del 95% de la diferencia apareada queda íntegramente dentro de ±margen efectivo (R11); **diferencia** si queda íntegramente fuera de un lado; **no concluyente** si cruza el margen. Nunca se declara empate ni diferencia por defecto.
- **R4 — Barrido de learning rate idéntico.** AdamW, warmup 5%, decaimiento coseno. Barrido {3e-4, 1e-3, 3e-3} aplicado por igual a todas las condiciones; se selecciona por **exactitud de validación de la métrica primaria de la tarea** (desempates por pérdida de validación) y se reporta el LR elegido por condición. Motivo: los veredictos se emiten sobre exactitud; seleccionar por pérdida abría una brecha entre criterio de selección y criterio de veredicto.
- **R5 — Entrenamiento acotado.** Tope duro de 20 000 pasos por corrida; early stopping por validación con paciencia de 2 000 pasos. Batch 64. Estos valores solo pueden cambiar por la regla de escala de §12, nunca por condición individual.
- **R6 — Sin métricas post-hoc.** Las métricas y análisis válidos son los pre-especificados en cada experimento. Cualquier análisis adicional va en un apéndice rotulado «exploratorio, no pre-registrado» y no participa de los veredictos.
- **R7 — Sanidad primero.** Ningún experimento principal corre antes de aprobar la Fase 0 completa (§10).
- **R8 — Registro total.** Cada corrida archiva: config completa (JSON), semilla, curvas de train/val, tiempo de pared, hardware. Nada se descarta.
- **R9 — Criterios de decisión por adelantado.** Cada experimento define qué resultado **confirma**, qué resultado **falsa** y qué queda **no concluyente**. El informe emite un veredicto explícito por predicción usando solo esos criterios.
- **R10 — Alcance declarado.** Escala chica, CPU, tareas sintéticas: ninguna conclusión se extrapola a gran escala sin señalarlo explícitamente como especulación.
- **R11 — Márgenes anclados al ruido, en la escala correcta.** Ningún margen numérico de §13 se usa tal cual: lo que se congela es la **regla**, no el número. El margen efectivo de cada predicción es `máx(piso de la tabla, 1.5 × desvío estándar entre semillas del baseline de referencia)`. Los tests de equivalencia usan **diferencias apareadas por semilla** (`d_s = A_s − B_s`, mismas semillas en ambas condiciones): el apareo elimina la varianza compartida por semilla — que es la fuente principal de variabilidad de la diferencia — y recupera la potencia, sin necesidad de inflar el margen. *(D1, 2026-07-22: el reescalado por √2 de la v0.3 se quitó; en equivalencias empujaba en la misma dirección que el apareo —empate más fácil— en vez de compensar. La salvaguarda direccional son los tres veredictos de R3, no el tamaño del margen.)* Los baselines de referencia se corren **primero** (sus corridas se reutilizan como corridas principales, no se repiten), y los márgenes efectivos se instancian y registran en `resultados/fase0/margenes_instanciados.md` antes de correr o mirar cualquier condición de comparación. Instanciados, no se tocan. Baselines de referencia por familia: P1.x → C1 en T1@(carga de evaluación, ver D2) y en T2; P2.x → ruta B en T3; P3.x → condición densa en T4; P4.x → S en T5-b. Si un margen efectivo resulta mayor que el efecto predicho por la propia predicción, la predicción se reporta como «no evaluable con esta potencia» — nunca se afloja el criterio a posteriori.

---

## 3. Entorno, stack y estructura de repo

- **Stack:** JAX puro + optax (continuidad con TELAR-01). float32. Determinismo por semilla verificado (dos corridas con la misma semilla deben dar curvas idénticas dentro de tolerancia numérica; se verifica en Fase 0).
- **Hardware:** CPU por defecto. Si hay GPU disponible puede usarse solo si el determinismo por semilla se mantiene; se documenta el hardware real en cada corrida.
- **Presupuesto por modelo:** 100k–160k parámetros salvo indicación contraria; tabla obligatoria por experimento.
- **Estructura de repo:**

```
telar-ligamento/
  protocolo/PROTOCOLO_v1.0.md        # este archivo, congelado
  protocolo/FREEZE_v1.0.md           # companion: hash SHA-256 + metadatos de depósito
  desviaciones.md
  src/
    datos.py          # generadores T1–T5 + tests
    modelos.py        # tronco base, cabezas por regla, canal de contexto, compuertas
    reglas_memoria.py # softmax, delta (port de TELAR-01); linear y titans disponibles
    entrenar.py       # loop, barrido LR, early stopping, logging
    probes.py         # probes lineales + controles permutados
    analisis.py       # bootstrap, consistencia de signo, figuras
  experimentos/
    E1/ E2/ E3/ E4/   # configs YAML + runner por experimento
  resultados/
    fase0/ E1/ E2/ E3/ E4/   # fase0 incluye margenes_instanciados.md (S0.9)
```

---

## 4. Suite de tareas sintéticas

Todas las tareas: vocabulario ≤ 96 tokens, generadores con semilla independiente, particiones train/val/test de 50 000 / 5 000 / 5 000 ejemplos, longitud de secuencia ≤ 128. Cada generador incluye tests unitarios de sus propiedades de diseño.

- **T1 — MQAR estándar** (idéntico en espíritu a TELAR-01). Pares clave→valor presentados en secuencia, luego consultas. Carga L ∈ {8, 16, 32, 64, 96, 128} (rango extendido por D2). Métrica: exactitud de recuperación.
- **T2 — MQAR-sobreescritura** (correctabilidad). Como T1, pero una fracción de claves se reasigna a un valor nuevo a mitad de secuencia. Métrica: exactitud sobre las claves reasignadas (responder el valor nuevo, no el viejo).
- **T3 — MQAR-polisémico** (ambigüedad graduada). Una señal de contexto (token especial ∈ {ctx_a, ctx_b} en posición 0) determina el «sentido» de las claves polisémicas: una fracción ρ ∈ {0, 0.25, 0.5, 0.75, 1.0} de las claves mapea a valores distintos según el contexto (2 sentidos por clave). **Propiedad de diseño verificable:** con el contexto enmascarado, la exactitud máxima alcanzable sobre claves polisémicas es 50%.
- **T4 — Dificultad × ambigüedad** (para E3). Diseño 2×2. Eje dificultad: composición de 1 salto (clave→valor) vs. 3 saltos (clave→clave→clave→valor). Eje ambigüedad: ρ = 0 vs. ρ = 0.5 sobre la clave inicial. Cuatro celdas: fácil-claro, fácil-ambiguo, difícil-claro, difícil-ambiguo, balanceadas dentro de cada batch.
- **T5 — Multimodal sintético** (para E4). K = 32 conceptos latentes; cada concepto c se «renderiza» en tres proxies de modalidad mediante mapas generativos fijos y distintos:
  - **M-texto:** secuencia con estadística zipfiana y n-gramas a distancia dependientes de c.
  - **M-imagen:** grilla 8×8 con estructura local 2D dependiente de c, aplanada en orden raster (64 tokens).
  - **M-audio:** señal 1D suave (suma de 3 senoidales con frecuencias/fases parametrizadas por c), cuantizada a tokens.
  Dos tareas: (a) clasificación del concepto (32 clases) desde una entrada de cualquier modalidad; (b) emparejamiento cruzado: dadas dos entradas de modalidades distintas, decidir si comparten concepto (binaria, balanceada). **Declaración de alcance:** son proxies con estadísticas distintas y semántica compartida; ninguna afirmación se extiende a imágenes o audio reales.

---

## 5. Arquitectura base común

- **Tronco:** transformer pre-norm de 4 bloques, d_model = 64, H = 4 cabezas (16 dim/cabeza), FFN de expansión 3 (dim oculta 192; divisible por 3 para E4), residual estándar. **Invariante (Ajuste 1):** todas las cabezas de atención del tronco, en todos los experimentos y todas las condiciones, aplican RMSNorm **por cabeza** antes de la concatenación y la proyección W_O — no es un rasgo de las condiciones mixtas: si solo C3 la tuviera, sus cabezas softmax dejarían de ser idénticas a las de C1 y el contraste de E1 quedaría confundido. Presupuesto resultante ≈ 100k–150k parámetros según variante; tabla obligatoria.
- **Reglas de escritura por cabeza** (port de la lib TELAR-01): `softmax` (atención estándar) y `delta` (regla delta / DeltaNet) son las reglas de las comparaciones principales; `linear` y `titans` quedan disponibles solo para el apéndice exploratorio de E1.
- **Canal de contexto** (E2, E3): codificador chico (2 capas, mismo d_model) que produce k = 8 vectores K/V a partir de los tokens de contexto. Acceso por cross-attention con compuerta escalar por bloque: `h ← h + tanh(g_l) · CA(h, C)`, con `g_l` inicializado en 0 (estilo Flamingo).
- **Compuerta de cómputo** (E3): router **causal por umbral**, que corrige la no-causalidad del top-k por lote del MoD original (donde puntajes de tokens futuros compiten en la selección del token presente): el token t se procesa si `sigmoid(score_t) > 0.5`, con una pérdida auxiliar de carga que empuja la tasa media de procesamiento hacia la capacidad objetivo c = 0.5. El puntaje suave multiplica la salida de la rama procesada para mantener diferenciabilidad. Se reporta la **capacidad efectiva realizada** por corrida (con umbral ya no es exactamente 0.5 por construcción), y esa capacidad realizada es la que entra en la igualación de FLOPs efectivos de R2/E3.
- **Probes** (E4): clasificadores lineales entrenados con el tronco congelado; control obligatorio con etiquetas permutadas.

---

## 6. E1 — Cabezas mixtas softmax + delta («ADN» heterogéneo en la misma capa) — PRIORIDAD 1

**Pregunta.** ¿Una capa con cabezas de reglas distintas hereda las virtudes de ambas (capacidad de softmax + correctabilidad de delta) o las reglas se interfieren al compartir el mismo stream residual?

**Condiciones** (parámetros igualados ±5%; FLOPs reportados por condición según R2/E1; misma cantidad total de cabezas):
- **C1:** 4 cabezas softmax (baseline de capacidad).
- **C2:** 4 cabezas delta (baseline de correctabilidad).
- **C3:** mixta 2 softmax + 2 delta, en los 4 bloques.
- **C4 (exploratoria, no participa de veredictos):** 3+1 y 1+3.
- **Especificación de C3 (mezcla de reglas):** ambas reglas emiten un vector por posición y por cabeza (softmax: atención causal estándar; delta: lectura del estado recurrente); la RMSNorm por cabeza es el invariante de §5 — presente por igual en C1, C2, C3 y C4 — de modo que las cabezas softmax de C3 son arquitecturalmente idénticas a las de C1 y las delta a las de C2; la concatenación pasa por la proyección W_O compartida estándar.
- **Sesgo declarado (R2/E1):** C3 corre con desventaja de FLOPs de atención frente a C1; toda confirmación de P1.1/P1.2 ocurre contra un baseline ventajado en cómputo y es, por lo tanto, conservadora.

**Carga de evaluación de capacidad (D2, 2026-07-22).** T1 se corre en L ∈ {8, 16, 32, 64, 96, 128}. La **carga de evaluación de capacidad** para P1.1 es la **menor L del rango donde C1 cae por debajo de 95% de exactitud** — no un L64 fijo. Motivo: en TELAR-01 y en TELAR-03 Fase 0, softmax rinde ~0.99 a L64 (`telar03/docs/reconciliation_softmax_L64.md`); a d_head = 16 el número exacto no es importable, así que se determina empíricamente. Si C1 no cae por debajo de 95% en todo el rango, **P1.1 se reporta como «no evaluable por saturación del baseline»** (salida honesta pre-registrada). Esta carga se fija en S0.9 junto con los márgenes, antes de mirar C2/C3.

**Tareas.** T1 (capacidad, todas las cargas) y T2 (correctabilidad).

**Métricas primarias.** Exactitud T1 por carga; exactitud T2.

**Semillas.** C1, C2 y C3 corren con 8 semillas (contienen la equivalencia P1.1); C4 exploratoria con 5.

**Predicciones pre-registradas.**
- **P1.1 (equivalencia):** C3 ≈ C1 en T1 a la carga de evaluación de capacidad (D2), con margen efectivo según R11 (piso: 2 puntos, elegido por debajo del efecto mínimo tratado como real en TELAR-01, ~3.5 pts): la capacidad la sostienen las cabezas softmax.
- **P1.2 (herencia como contraste lineal):** `L = C3 − 0.5·C1 − 0.5·C2 ≥ 0` en T2, con IC bootstrap del 95% de L (apareado por semilla) que no cruce cero por debajo. Equivale a «C3 recupera al menos la mitad de la brecha (C2 − C1)» cuando C2 > C1, pero sin división: el cociente de la v0.2 era numéricamente inestable cuando el denominador se acercaba al margen. La fracción de brecha se reporta igual, como descriptivo. Condición de aplicabilidad: C2 − C1 > margen efectivo; si no, no hay brecha que heredar y P1.2 se reporta como no evaluable.
- **P1.3 (falsación):** si C3 < min(C1, C2) en ambas tareas bajo el criterio R3, hay interferencia destructiva por stream compartido y la hipótesis de herencia queda falsada.

**Criterios de decisión.** Confirmación = P1.1 y P1.2 satisfechas bajo R3. Falsación = P1.3 bajo R3. Cualquier otro patrón = no concluyente (se reporta el patrón completo).

**Análisis secundario pre-registrado.** Atribución por cabeza: ablación en test (poner a cero la salida de cada cabeza, una por vez) para verificar que la correctabilidad de C3 se concentra en las cabezas delta y la capacidad en las softmax.

**Figuras.** (1) Exactitud vs. carga por condición; (2) barras de correctabilidad; (3) matriz de ablación por cabeza.

---

## 7. E2 — Estrategias de inyección de contexto (rutas A/B/C/D)

**Pregunta.** ¿Cuándo alcanza con que el contexto llegue al final (ruta A) y cuándo hace falta inyectarlo por capa (ruta B) o bajo demanda con compuertas (ruta D)?

**Condiciones** (el canal de contexto y su codificador existen en las cuatro para igualar parámetros; lo que cambia es el acceso):
- **A:** el tronco no accede al contexto; solo la decodificación final lo usa (queries sobre el estado final, estilo Perceiver IO).
- **B:** cross-attention al contexto en los 4 bloques, siempre activa (sin compuerta).
- **C (baseline):** contexto concatenado a la entrada como tokens; sin cross-attention.
- **D:** cross-attention por bloque con compuerta aprendida `tanh(g_l)`, init 0.
- **Igualación de largo (O4):** todas las rutas reciben secuencias del mismo largo total: las primeras k = 8 posiciones están reservadas — en C llevan los tokens de contexto, en A/B/D llevan un token PAD neutro excluido de la pérdida. Sin esto, C entrenaba con secuencias más largas que el resto y rompía R2 dentro del propio experimento.

**Tarea.** T3 con el barrido completo ρ ∈ {0, 0.25, 0.5, 0.75, 1.0}.

**Métricas.** Exactitud total; exactitud sobre claves polisémicas; perfil de compuertas |tanh(g_l)| por bloque a lo largo del entrenamiento (solo D); FLOPs efectivos de acceso a contexto.

**Semillas.** 8 por condición (el experimento contiene la equivalencia P2.2).

**Predicciones pre-registradas.**
- **P2.1 (tendencia, dominio corregido):** el gap por semilla `g_s(ρ) = exactitud_B − exactitud_A` sobre claves polisémicas se define solo donde ese conjunto es no vacío: **ρ ∈ {0.25, 0.5, 0.75, 1.0}**. Criterio: mediana sobre semillas de Spearman(g_s, ρ) > 0, con IC bootstrap sobre semillas que no cruce cero. El chequeo de ρ = 0 se muda a **exactitud total**: |B − A| dentro del margen efectivo (piso: 1 punto). (La v0.2 metía ρ = 0 en una métrica indefinida: sin claves polisémicas no hay gap polisémico que medir — bloqueante A3 del ejecutor.)
- **P2.2 (equivalencia):** D empata con B en exactitud (margen efectivo, piso: 1.5 puntos) usando menos contexto efectivo cuando ρ es bajo (compuertas ≈ 0 en ρ = 0).
- **P2.3:** en D, la apertura media de compuertas crece con ρ: mediana sobre semillas de Spearman(apertura, ρ) ≥ 0.7. Umbral reducido desde 0.8 a propósito: con 5 puntos de ρ el coeficiente está groseramente cuantizado y una sola inversión local lo baja a ~0.7; exigir 0.8 equivalía a exigir monotonía casi perfecta, más severo de lo que el número aparenta.
- **Falsación:** si A empata con B en ρ altos (diferencia dentro del margen efectivo en ρ ≥ 0.75 bajo R3), la hipótesis «el cómputo intermedio necesita contexto temprano» queda falsada en este régimen — resultado publicable pro-Perceiver.

**Sanidad específica.** Con contexto enmascarado en test, la exactitud sobre claves polisémicas debe caer a ~50% en todas las condiciones (verifica el diseño de T3 y la ausencia de fuga).

**Figuras.** (1) Exactitud polisémica vs. ρ, cuatro curvas; (2) mapa de calor de compuertas bloque × ρ (ruta D); (3) exactitud vs. FLOPs efectivos de contexto.

---

## 8. E3 — Doble compuerta: ¿proceso este token? × ¿consulto contexto?

**Pregunta.** ¿Las dos compuertas capturan ejes distintos (dificultad → cómputo; ambigüedad → contexto) o son redundantes?

**Arquitectura.** Bloques con compuerta de cómputo (router causal por umbral, §5, capacidad objetivo c = 0.5) y compuerta de contexto (como ruta D de E2). Cuatro estados posibles por token × bloque. La capacidad efectiva realizada se reporta por condición y entra en la igualación de FLOPs efectivos.

**Condiciones** (FLOPs efectivos promedio igualados y reportados):
- Doble compuerta · solo compuerta de cómputo · solo compuerta de contexto · sin compuertas (denso, ruta B).

**Tarea.** T4 (2×2 dificultad × ambigüedad).

**Métricas.** Exactitud por celda; **profundidad efectiva** media por token (nº de bloques que lo procesan); **tasa de consulta** de contexto media por token; relación entre ambas señales por celda.

**Predicciones pre-registradas.**
- **P3.1:** profundidad efectiva: difícil > fácil, con diferencia media ≥ 0.5 bloques.
- **P3.2:** tasa de consulta: ambiguo > claro, con razón ≥ 1.5×.
- **P3.3 (disociación, la predicción central):** en la celda **difícil-claro**, profundidad alta con consulta baja; en **fácil-ambiguo**, profundidad baja con consulta alta. Operacionalización pre-especificada: IC bootstrap 95% de [profundidad(difícil-claro) − profundidad(fácil-ambiguo)] > 0 **y** IC bootstrap 95% de [consulta(fácil-ambiguo) − consulta(difícil-claro)] > 0.
- **Falsación:** si la correlación token a token entre las dos señales de compuerta supera 0.9 en las cuatro celdas, las compuertas son redundantes y una sola basta — simplificación válida y reportable como hallazgo.

**Figuras.** (1) Exactitud por celda y condición; (2) dispersión profundidad × consulta coloreada por celda; (3) perfiles de compuerta por bloque.

---

## 9. E4 — Partición modal: fija vs. compartida vs. mixta

**Pregunta.** ¿Dónde conviene poner la frontera entre especialización por modalidad y compartición? ¿Emergen «unidades multimodales» y a qué profundidad?

**Condiciones** (parámetros igualados entre F, S y M; cuando hay partición, el FFN de 192 unidades se divide en 3 expertos de 64 ruteados por etiqueta de modalidad; FLOPs de FFN reportados por condición — ver R2/E4):
- **F:** partición fija por modalidad en los 4 bloques (la versión «5/5/5 en todas las capas»). Computa 1 de 3 expertos por entrada: ~⅓ de los FLOPs de FFN de S.
- **S:** FFN compartido denso en los 4 bloques.
- **M:** partición fija en bloques 1–2, FFN compartido en bloques 3–4.
- **S⅓ (control de cómputo):** FFN compartido denso con dimensión oculta 64 (⅓ de S): FLOPs de FFN ≈ F, parámetros ≈ ⅓. No compite en P4.1; existe únicamente para desambiguar cómputo de fusión.

**Regla de desambiguación pre-registrada para P4.1** (bloqueante A2: en este experimento el desbalance de FLOPs favorece a M, no a la hipótesis nula). La ventaja de M sobre F solo se interpreta como evidencia de **fusión** (y no de cómputo) si se cumple al menos una de: (i) `S − S⅓ ≤ margen efectivo` en T5-b — el cómputo extra de FFN, por sí solo, no compra emparejamiento cruzado en este régimen; o (ii) `(M − F) > (S − S⅓) + margen efectivo` — el efecto excede el dividendo de cómputo. Si no se cumple ninguna, P4.1 se reporta como **«confundida por cómputo»**, no como confirmada.

**Tareas.** T5-a (clasificación de concepto por modalidad) y T5-b (emparejamiento cruzado).

**Métricas.** Exactitud por modalidad (T5-a); exactitud de emparejamiento cruzado (T5-b); **fracción de unidades multimodales** por bloque: una unidad cuenta como multimodal si el probe lineal sobre su activación decodifica el concepto por encima del **percentil 95 de la distribución nula** (construida con 50 probes de etiquetas permutadas por bloque) **en las tres modalidades a la vez**. (El umbral «2× control» de la v0.2 ≈ 2× azar ≈ 6% era tan bajo que volvía P4.2 casi infalsable — defecto O1 del ejecutor.)

**Semillas.** 8 por condición (el experimento contiene la equivalencia P4.3). Si la regla de escala de §12 vuelve inviable el costo, se reduce a 5 documentándolo en `desviaciones.md` y P4.3 se reporta explícitamente como «de potencia reducida».

**Predicciones pre-registradas.**
- **P4.1:** M ≥ F y M ≥ S en emparejamiento cruzado, con ventaja mayor al margen efectivo (piso: 2 puntos) sobre la mejor de las otras dos.
- **P4.2:** la fracción de unidades multimodales crece con la profundidad en S y en M; en F es ≈ 0 en todos los bloques.
- **P4.3 (falsación, equivalencia):** si F empata con M en emparejamiento cruzado (dentro del margen efectivo, piso: 1 punto, bajo R3), la necesidad de compartición para la fusión queda cuestionada en este régimen — contra la lectura estándar de las neuronas multimodales, y por eso mismo interesante.

**Figuras.** (1) Emparejamiento cruzado por condición; (2) fracción de unidades multimodales por bloque × condición; (3) exactitud por modalidad.

---

## 10. Fase 0 — Infraestructura y chequeos de sanidad (previa a todo)

- **S0.1:** el baseline softmax reproduce el comportamiento de TELAR-01 en T1: > 95% a carga 8 y curva de degradación coherente con TELAR-01/TELAR-03 (softmax se mantiene alto a L64; ver S0.9/D2 para la carga de evaluación).
- **S0.2:** delta reproduce correctabilidad alta en T2 (> 90%).
- **S0.3 — Tests del generador T3:** (i) los dos sentidos de cada clave polisémica son equiprobables en el corpus generado (test 50/50 por clave); (ii) un clasificador bayesiano óptimo con contexto enmascarado rinde 50 ± 1% sobre polisémicas (verificación analítica o por conteo); (iii) un modelo entrenado con contexto visible, evaluado con contexto enmascarado, cae a ~50% en polisémicas.
- **S0.4:** compuertas vivas: tras 500 pasos de un piloto de D, algún |g_l| > 0.01 (no colapso en cero permanente). Verificación binaria; prohibido ajustar hiperparámetros mirando exactitud de tarea.
- **S0.5:** calibración de probes: la distribución nula de 50 permutaciones queda centrada en azar, y su percentil 95 se calcula y registra por bloque — es el umbral operativo de las unidades multimodales de E4.
- **S0.6:** tabla de parámetros y FLOPs por condición de cada experimento, todas dentro de ±5%.
- **S0.7 — Determinismo operacionalizado:** se fija el número de hilos de XLA en CPU y se documenta en la config (las reducciones multihilo de XLA no garantizan orden de suma determinista en float32). Criterio numérico: dos corridas con la misma semilla deben diferir en pérdida final < 1e-6 y en métricas de tarea exactamente 0. Si con hilos fijos no se alcanza, se reduce a un solo hilo y se documenta el costo en tiempo.
- **S0.8:** registro de tiempo de pared por corrida piloto, para la regla de escala de §12.
- **S0.9 — Medición de ruido, instanciación de márgenes (R11) y carga de evaluación de E1 (D2):** correr primero los baselines de referencia de cada familia con todas sus semillas; registrar el desvío estándar entre semillas de la métrica correspondiente; calcular los márgenes efectivos con la regla de R11 (`máx(piso, 1.5×SD)`, sin √2) y escribirlos en `resultados/fase0/margenes_instanciados.md` **antes** de correr o mirar cualquier condición de comparación. Estas corridas de baseline se reutilizan como corridas principales. **Además (D2):** de la curva de C1 (softmax) sobre T1 en L ∈ {8,16,32,64,96,128}, fijar y registrar en el mismo archivo la carga de evaluación de capacidad de P1.1 = menor L con C1 < 95%; si no existe, dejar registrado «P1.1 no evaluable por saturación del baseline». Chequeo asociado: si algún margen efectivo supera el efecto predicho por su predicción, se anota de inmediato que esa predicción queda «no evaluable con esta potencia» (no se ajustan pisos, no se agregan semillas por fuera de lo pre-registrado).

Si cualquier chequeo falla: frenar, reportar en `resultados/fase0/informe_fase0.md`, esperar decisión.

---

## 11. Formato de resultados e informes

**`EX_resultados.json`** (por experimento): lista de corridas con `{experimento, condicion, semilla, lr_elegido, params, flops_token, pasos_efectivos, metricas: {...}, tiempo_pared_s, hardware}`.

**`EX_informe.md`** (por experimento), estructura fija:
1. Tabla de condiciones y presupuestos (params, FLOPs, LR elegido).
2. Resultados: media ± desvío por condición y métrica, con las semillas listadas.
3. **Veredicto por predicción:** confirmada / falsada / no concluyente, citando el criterio pre-registrado aplicado.
4. Desviaciones que afectaron al experimento (o «ninguna»).
5. Figuras (las listadas en la sección del experimento).
6. Apéndice exploratorio (opcional, rotulado, sin peso en veredictos).

**Informe final (`sintesis.md`):** qué se aprendió sobre el eje especializar ↔ compartir; tabla resumen de los veredictos P1.1–P4.3; límites; propuestas concretas para la siguiente iteración (sin ejecutarlas).

---

## 12. Límites declarados y regla de escala

- Escala chica, CPU, tareas sintéticas: los hallazgos describen este régimen. Toda extrapolación se rotula como especulación.
- T5 usa proxies de modalidad, no modalidades reales.
- **Regla de escala:** si una corrida principal supera 4 horas de pared en el hardware disponible, se reduce el tope de pasos un 50% en **todas** las condiciones del experimento afectado (nunca en una sola) y se documenta en `desviaciones.md`.
- Prioridad ante cómputo insuficiente: E1 > E2 > E3 > E4.

---

## Decisiones resueltas (pre-registro — rastro dentro del doc congelado)

*Esta sección era «Decisiones pendientes antes del freeze» en v0.3.1. Se conserva convertida a «resueltas» para que el rastro de cómo se decidió cada criterio sobreviva dentro del documento congelado: el pre-registro fue un proceso, no un decorado.*

**D1 — Dirección del factor √2 en R11. RESUELTA 2026-07-22: se quita el √2.**
Consenso de los tres (Maxi + Fable5 + ejecutor Opus 4.8). El ×√2 (margen más grande) y el apareo por semilla (IC más angosto) empujaban la equivalencia en la **misma** dirección — más fácil de declarar —, no se compensaban; el rótulo «conservador» de v0.3 era correcto para superioridad e incorrecto para equivalencia. Además ninguna dirección es uniformemente prudente: margen estricto dificulta confirmar P1.1/P2.2 (empate confirmatorio) pero también dificulta disparar P4.3 (empate falsador). **Aplicado:** en R11, `1.5 × √2 × SD` → `1.5 × SD`; se eliminó la oración del factor √2; el apareo por semilla se conserva; la salvaguarda direccional son los tres veredictos de R3.

**D2 — Regla de carga por saturación del baseline en E1. RESUELTA 2026-07-22: entra tal cual.**
Motivada por TELAR-03 Fase 0 (softmax acc@1 = 0.994 @ L64) y confirmada por la reconciliación (`telar03/docs/reconciliation_softmax_L64.md`): softmax nunca saturó a ~67% — el plateau era de las reglas de estado; el resumen de TELAR-01 simplificaba de más. Sin esta regla, P1.1 evaluaría un empate contra un baseline en techo (trivial). **Aplicado:** (i) T1 se extiende a L ∈ {8,16,32,64,96,128}; (ii) la carga de evaluación de capacidad de P1.1 = menor L donde C1 < 95%; (iii) si C1 no cae en todo el rango, P1.1 = «no evaluable por saturación del baseline». Refuerzo del ejecutor: la arquitectura base corre a d_head = 16 (no 32), así que el número no es importable de TELAR-01/03 → la carga se fija empíricamente en S0.9.

**Pendiente hermano (fuera de este protocolo, en TELAR-03).** Reconciliar por qué linear cae de 0.898 (TELAR-01, 5 semillas) a 0.724 (TELAR-03 Fase 0) @ L64, mientras delta/titans se mantienen o suben — cambio de setup real (criterio de lectura, generador o presupuesto de pasos). No afecta a Ligamento; anotado en `telar03/docs/reconciliation_softmax_L64.md`.

---

## 13. Registro resumido de predicciones (para el congelamiento)

Los márgenes listados son **pisos**; el margen efectivo de cada predicción se instancia por R11 (máx entre piso y 1.5 × desvío entre semillas del baseline) en S0.9, antes de las corridas de comparación.

| ID | Tipo | Semillas | Enunciado corto | Confirma si | Falsa si |
|----|------|----------|----------------|-------------|----------|
| P1.1 | Equivalencia | 8 | Mixta conserva capacidad de softmax | IC de (C3−C1) en T1@(carga D2) dentro de ±margen ef. (piso 2 pts) | IC fuera del margen con C3 < C1 |
| P1.2 | Contraste lineal | 8 | Mixta hereda correctabilidad de delta | L = C3−0.5·C1−0.5·C2 ≥ 0 en T2, IC apareado no cruza 0 | L < 0 bajo R3; no evaluable si C2−C1 ≤ margen ef. |
| P1.3 | Falsación | 8 | Sin interferencia destructiva | — | C3 < min(C1,C2) en T1 y T2 bajo R3 |
| P2.1 | Tendencia | 8 | Brecha B−A crece con ambigüedad | Mediana Spearman(g_s,ρ) > 0 en ρ∈{0.25…1}; |B−A| total ≤ margen ef. en ρ=0 | Tendencia nula/negativa o brecha nula en ρ altos |
| P2.2 | Equivalencia | 8 | Compuertas empatan con inyección total | IC de (D−B) dentro de ±margen ef. (piso 1.5 pts) | D < B fuera del margen |
| P2.3 | Tendencia | 8 | Apertura de compuertas sigue a ρ | Mediana de Spearman ≥ 0.7 | Mediana < 0.7 |
| P3.1 | Superioridad | 5 | Difícil → más profundidad efectiva | Δ ≥ máx(0.5 bloques, margen ef.) bajo R3 | Δ menor o signo invertido |
| P3.2 | Superioridad | 5 | Ambiguo → más consulta de contexto | razón ≥ 1.5× bajo R3 | razón < 1.5× |
| P3.3 | Disociación | 5 | Las dos compuertas se disocian | Ambos IC bootstrap > 0 | Correlación > 0.9 en todas las celdas |
| P4.1 | Superioridad | 8 | Mixta gana en fusión cruzada | M ≥ F y M ≥ S por > margen ef. (piso 2 pts) **y** pasa la regla de desambiguación de §9 | M ≤ max(F,S), o veredicto «confundida por cómputo» |
| P4.2 | Tendencia | 8 | Unidades multimodales crecen con profundidad | Tendencia creciente en S y M con umbral p95 del nulo; ≈0 en F | Sin tendencia o presentes en F |
| P4.3 | Equivalencia | 8 | La partición total limita la fusión | F < M fuera del margen ef. (piso 1 pt) | IC de (M−F) dentro de ±margen ef. bajo R3 |

---

*Fin del protocolo v1.0 — CONGELADO 2026-07-22. D1 y D2 resueltas; hash SHA-256 y depósito en `FREEZE_v1.0.md`. Las predicciones son pre-registradas e inmutables: cualquier ajuste posterior es un protocolo nuevo. Próximo paso del ejecutor: Fase 0 → E1.*
