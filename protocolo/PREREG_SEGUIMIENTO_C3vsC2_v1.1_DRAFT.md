# Pre-registro de SEGUIMIENTO — E1 · C3 vs. C2 (v1.1, DRAFT para contra-lectura de Fable5)

**Estatus:** DRAFT. Amplía el v1.0 (congelado, SHA-256 `8b85aed7…`) con dos predicciones nuevas pedidas por
Maxi (inicio/pendiente y anticorrelación) + las tres condiciones de la nota de cierre de S0.9. **No se congela
hasta la pasada de Fable5.** Una vez integrado, se congela (hash + tag + push) **ANTES de lanzar E1** (la
convergencia de C2 en E1 es el primer dato que toca estas predicciones).

## Estatus epistémico de las predicciones nuevas (declaración honesta)
PS-1..PS-3 (v1.0) se anclaron **antes de ver delta**. **PS-4 y PS-5 se generaron VIENDO el snapshot @2500 de
S0.9** (no son pre-datos puros); se **confirman en datos independientes**: la **C2 convergida de E1** (8 semillas,
convergencia real, régimen único). Esto es hipótesis-generada-por-exploración confirmada en un conjunto nuevo —
válido y declarado como tal. El snapshot @2500 queda como su origen documentado, no como su test.

## Condiciones heredadas de la nota de cierre S0.9 (Maxi, 2026-07-22)
- **(a) Régimen único de las 8 delta de E1:** 4 extendidas desde checkpoint @2500 + 4 nuevas, **mismo criterio de
  convergencia** (acc@1 en {L96,L128} cada 500 pasos, mejora < 0.5 pts en ventana de 500, tope 10 000).
- **(b) Márgenes R11 del seguimiento se instancian desde la C2 CONVERGIDA de E1** (no del snapshot @2500).
- **(c) La carga de evaluación** = menor L con C2 convergido < 95% acc@1 (el L96 del snapshot es **preliminar**;
  con convergencia puede moverse a L128 si L96 cruza sobre 0.95).

## Predicciones nuevas

### PS-4 — Forma de la degradación (inicio y pendiente)
De la C2 convergida de E1 (media sobre 8 semillas), la curva acc@1 vs. carga cumple:
- **(i) Inicio de la caída interior:** existe una carga de inicio `L₀` = menor L con acc@1 < 0.99, y **`L₀ ≥ 64`**
  (delta se mantiene ~perfecto hasta cargas medias; el snapshot lo ubicaba en L64).
- **(ii) Monotonía:** acc@1 no crece al aumentar L en el tramo `L ≥ L₀` (Spearman(acc@1, L) < 0 con IC apareado
  que no cruza 0 en ese tramo).
- **(iii) Pendiente creciente (caída que acelera):** la caída entre cargas consecutivas es mayor en el tramo alto
  que en el medio: `[acc@1(L96) − acc@1(L128)] > [acc@1(L64) − acc@1(L96)]`, con IC apareado que no cruza 0.
  (En el snapshot: 0.114 vs 0.086 — la caída acelera hacia L128.)
- **Confirma:** la degradación por saturación del estado tiene forma característica (meseta → caída acelerada),
  no lineal ni abrupta. **Falsa:** si la caída es lineal (pendiente constante) o si L₀ < 64.

### PS-5 — Anticorrelación capacidad ↔ correctabilidad (entre semillas)
Entre las **8 semillas** de C2 convergida, la **capacidad** (acc@1 a la carga de evaluación) y la
**correctabilidad** (T2, acc de sobreescritura) están **anti-correlacionadas**: Pearson(acc@1_evalL, T2) **< 0**,
con IC bootstrap del 95% (sobre semillas) que **no cruza 0** por arriba.
- **Origen (snapshot @2500):** seed2/3 mayor capacidad (L128 ~0.79) / menor T2 (~0.85); seed0/1 al revés
  (0.755 / ~0.92). Correlación negativa aparente.
- **Confirma:** existe un **trade-off** intrínseco capacidad↔correctabilidad en delta — el estado de rango finito
  no puede maximizar ambas a la vez, y las semillas se distribuyen sobre esa frontera. **Falsa:** si la
  correlación es ≥ 0 (el aparente trade-off del snapshot era ruido no convergido).
- **Nota de potencia:** con 8 semillas el IC de una correlación es ancho; si cruza 0, se reporta «no concluyente»
  (no se fuerza veredicto). Es la predicción de mayor riesgo por potencia — declarado.

## Integración
En la v1.1 final: PS-4 y PS-5 se agregan a la §7 del v1.0; las condiciones (a)-(c) se anexan a la §Método; el
resto del v1.0 (PS-1..PS-3, gate, alcance) queda intacto y byte-idéntico. Criterios de decisión: Puente 2 (del
v1.0) sin cambios; PS-4 y PS-5 se reportan como **secundarias** (no condicionan la confirmación de Puente 2,
son caracterización del plateau).

*Draft de Opus para la pasada de Fable5. Tras integrar sus comentarios → v1.1 congelada (hash + tag + push)
ANTES de E1.*
