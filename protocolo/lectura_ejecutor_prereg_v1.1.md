# Lectura de ejecutor — prereg de seguimiento v1.1 (DRAFT) + runner de E1

**Autor:** Opus 4.8 (ejecutor). **Fecha:** 2026-07-23. **Destino:** pasada de contra-lectura de Fable5,
antes de congelar el v1.1 y lanzar E1.

Cuatro observaciones surgidas de implementar el runner (`experimentos/E1/e1_runner.py`) contra el texto del
draft. Ninguna toca el protocolo madre (`2f8ebb82…` intacto). Una es candidata a bloqueante.

---

## O1 — [CANDIDATA A BLOQUEANTE] Presupuesto de pasos asimétrico entre condiciones

**El problema.** El early stopping ratificado por Maxi es **colectivo por condición**: las 8 semillas de una
condición se entrenan en bloques de +2500 hasta que **todas** convergen, y ahí paran juntas (`all(conv_flags)`
en `run_condition`). Eso garantiza régimen único *dentro* de la condición — la condición (a) de la nota de
cierre — pero **no entre condiciones**: `N_final` puede ser 2500 para C1 y 10 000 para C2.

**Por qué importa.** PS-1 compara `acc@1(C3) − acc@1(C2)` en la carga de evaluación. Si C3 (mixta, con dos
cabezas softmax que convergen rápido) para en 2500 y C2 (delta pura) sigue hasta 10 000, el contraste mezcla
**arquitectura con presupuesto de cómputo**. R2 del protocolo madre exige paridad de recursos (params ±5% +
FLOPs reportados + cláusula de sesgo conservador); esta asimetría es del mismo género y no está cubierta,
porque R2 habla de recursos por paso, no de número de pasos.

**Dirección del sesgo.** Plausiblemente **conservadora para PS-1**: C3 mediría con menos entrenamiento que su
baseline, lo que reduce el rescate observado. Pero no está garantizado — si delta sobre-entrena y se degrada en
las cargas altas, el sesgo se invierte y PS-1 se confirmaría por una razón espuria.

**Propuestas (ordenadas por preferencia del ejecutor):**
1. **`N_final` común a todas las condiciones** = el máximo de los `N_final` individuales. Todas las condiciones
   se llevan al mismo presupuesto; se reporta además el paso de convergencia propio de cada una. Costo: hasta
   4× cómputo en las condiciones que convergen temprano (softmax converge a 2500 según S0.9).
2. **Doble reporte:** veredicto principal a `N_final` común (como 1) + tabla secundaria «cada condición en su
   propia convergencia». Cuesta lo mismo que 1, pero deja explícito si el veredicto depende del criterio.
3. **Dejarlo como está** y declarar la asimetría con su dirección de sesgo en el prereg, con la cláusula
   conservadora de R2. Barato, pero deja PS-1 abierto a la objeción obvia.

**Recomiendo la 2** (el cómputo extra recae sobre softmax, que es la condición barata y ya saturada, y cierra
la objeción de raíz). Necesita decisión antes del freeze porque cambia el criterio del veredicto.

---

## O2 — PS-4(i) es de bajo riesgo dado su origen, y su umbral es filoso donde no debería

`L₀` = menor carga con acc@1 < 0.99, y se predice `L₀ ≥ 64`.

- **Bajo riesgo:** la grilla es discreta {8,16,32,64,96,128} y el snapshot que *generó* la predicción ya ubica
  L₀ exactamente en 64. Con convergencia, L₀ solo puede quedarse en 64 o subir a 96 — ambos confirman. Para
  falsarla haría falta que L32 (hoy 0.9990) **baje** con más entrenamiento.
- **Filoso donde no importa:** justamente ese L32 = 0.9990 está a 0.001 del umbral. La predicción se juega su
  falsación en el ruido de una carga saturada, no en el fenómeno de interés (la caída).

**Sugerencia:** o bien usar el umbral **0.95**, que es el mismo criterio con el que ya se define la carga de
evaluación (coherencia interna, y L₀ pasa a ser 96 en el snapshot → predicción no trivial), o bien declarar
PS-4(i) explícitamente como **descriptiva** y dejar el contenido falsable en (ii) y (iii). Como está, aporta
poca información y puede caer por un artefacto.

---

## O3 — PS-5 puede confundirse con «cuánto se entrenó cada semilla»

Con el early stopping colectivo, **todas las semillas de C2 paran en el mismo paso**, que es el que necesita
la semilla más lenta. Las semillas rápidas quedan entonces *sobre*-entrenadas respecto de su propio criterio.

Si el sobre-entrenamiento afecta de manera distinta a capacidad y a correctabilidad — que es precisamente lo
que este proyecto sospecha —, entonces una anticorrelación entre acc@1 y T2 **puede emerger del punto de
parada** en vez de un trade-off intrínseco de rango. Es la hipótesis rival directa de PS-5.

**Mitigación (barata, no cambia el diseño):** registrar por semilla el paso en que **ella** alcanzó el criterio
(`val_hist` ya lo permite: es el primer múltiplo de 500 con mejora < 0.5 pts) y reportar la correlación
parcial controlando por ese paso, junto a la correlación cruda. Si ambas son negativas, PS-5 se sostiene; si
solo lo es la cruda, el trade-off era del régimen. Propongo agregarlo al texto de PS-5 como análisis obligatorio.

---

## O4 — PS-5 cruza dos cargas distintas sin declararlo

En el runner, la capacidad se mide en la **carga de evaluación** (L96 o L128) y la correctabilidad T2 en
**L=32 fijo** (`T2_LOAD = 32`). PS-5 correlaciona ambas. Es legítimo — son los dos ejes del protocolo — pero
el texto no dice que viven en cargas distintas, y eso admite la lectura de que un trade-off «a la misma carga»
fue medido, cuando no lo fue.

**Sugerencia:** una de dos, explícita en el texto: (a) declarar que la correlación es entre *capacidad a carga
alta* y *correctabilidad a carga media*, que es lo que se mide; o (b) medir T2 **también** a la carga de
evaluación y usar esa para PS-5 (es una llamada más a `eval_overwrite`, costo despreciable, y da la versión
«misma carga» del trade-off). (b) es más informativa y me parece barata; puede reportarse junto con la de L32.

---

## Resumen para la decisión

| # | Tipo | Necesita decisión antes del freeze |
|---|---|---|
| O1 | Criterio de veredicto (presupuesto de pasos) | **Sí** |
| O2 | Umbral de PS-4(i) | **Sí** (cambia el texto de la predicción) |
| O3 | Análisis obligatorio en PS-5 (correlación parcial) | Sí, es texto del prereg |
| O4 | Carga de medición de T2 en PS-5 | Sí, es texto del prereg |

O1 y O2 cambian qué contaría como confirmación; O3 y O4 solo agregan precisión a lo que ya se hará. Las cuatro
son integrables en una pasada. El runner ya está escrito y se adapta a cualquiera de las opciones en minutos.
