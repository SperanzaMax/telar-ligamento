# Pre-registro de SEGUIMIENTO — E1 · lectura de capacidad C3 vs. C2 (v1.1, CONGELADO)

**Estatus:** **CONGELADO 2026-07-23**, antes de correr una sola semilla de E1. Extiende el v1.0 (congelado
2026-07-22, SHA-256 `8b85aed730d82004bc2eba6836404b52ccbdd8150d082e73e64a9b00f6f8a20a`) con dos predicciones
nuevas (**PS-4**, **PS-5**) y con las cláusulas de método que surgieron de la lectura de ejecutor sobre el
runner (**O1–O4**, resueltas por Fable 5 el 2026-07-23; ver `lectura_ejecutor_prereg_v1.1.md`).

Incluye además una **precisión del ejecutor (O5)** sobre la regla de veredicto de PS-5, detectada al
implementarla y agregada **antes del freeze y antes de ver datos** (ver PS-5). Es una corrección de
**alcance** de la regla de Fable 5, no un cambio de dirección: sin ella, la regla habría reportado
«confirmada» en el escenario que fue diseñada para detectar.

**NO modifica el protocolo madre** (`PROTOCOLO_v1.0.md`, SHA-256 `2f8ebb82…`, intacto). **NO modifica el
contenido de PS-1/PS-2/PS-3**: el bloque que va desde «## Motivación» hasta el final de PS-3 es
**byte-idéntico** al del v1.0 congelado (verificable, ver companion). Lo que el v1.1 agrega sobre esas
predicciones es **el punto de medición** que el v1.0 no fijaba (presupuesto de pasos, Anexo B) y una regla de
discordancia que **solo puede restar poder confirmatorio, nunca agregarlo**: su único desenlace nuevo es «no
concluyente». No introduce grados de libertad para confirmar nada que el v1.0 no confirmaría.

## Estatus epistémico de las predicciones nuevas (declaración honesta)

PS-1..PS-3 (v1.0) se anclaron **antes de ver delta**. **PS-4 y PS-5 se generaron VIENDO el snapshot @2500 de
S0.9** (no son pre-datos puros); se **confirman en datos independientes**: la **C2 convergida de E1** (8
semillas, convergencia real, régimen único). Esto es hipótesis-generada-por-exploración confirmada en un
conjunto nuevo — válido y declarado como tal. El snapshot @2500 queda como su origen documentado, no como su
test.

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

---

# Anexos de método (v1.1)

## Anexo A — Condiciones de la nota de cierre S0.9 (Maxi, 2026-07-22)

- **(a) Régimen único de las 8 delta de E1:** mismo criterio de convergencia para las ocho (acc@1 en
  {L96,L128} cada 500 pasos, mejora < 0.5 pts en ventana de 500, tope duro 10 000). *Nota de
  implementación:* no existen checkpoints de pesos de S0.9, de modo que las «4 extendidas» se re-entrenan
  desde cero bajo el mismo régimen que las 4 nuevas — el resultado es **más** uniforme que lo pedido, no menos.
- **(b) Márgenes R11 del seguimiento** se instancian desde la **C2 convergida de E1**, no del snapshot @2500.
- **(c) La carga de evaluación** = menor L con C2 convergido < 95% acc@1 (el L96 del snapshot es
  **preliminar**; con convergencia puede moverse a L128 si L96 cruza sobre 0.95).

## Anexo B — Presupuesto de pasos y doble reporte (resuelve O1)

El early stopping es **colectivo por condición**: las 8 semillas de una condición paran juntas cuando todas
convergen. Eso da régimen único *dentro* de la condición, pero `N_final` puede diferir **entre** condiciones,
y entonces PS-1 (C3 vs C2) mezclaría arquitectura con presupuesto de cómputo. Se resuelve así:

- **(B1) `N_common`** = **máximo** de los `N_final` de convergencia colectiva entre condiciones. **Todas** las
  condiciones entrenan hasta `N_common`, incluidas las que ya convergieron (no se puede evaluar un checkpoint
  que no existe).
- **(B2) Checkpoint de convergencia propia.** Al cruzar su propia convergencia colectiva, cada condición
  registra sus métricas en ese punto. De ahí sale la **tabla secundaria**.
- **(B3) Regla de discordancia.** El **veredicto de PS-1 lo da la tabla primaria** (`N_common`). La secundaria
  (cada condición en su propia convergencia) es **chequeo de robustez**. **Si las dos tablas dan veredictos
  distintos, PS-1 se reporta «no concluyente por sensibilidad al presupuesto».** Nunca se elige la tabla que
  conviene.
- **(B4) Nota de asimetría residual.** `N_common` casi con certeza lo fija C2 (delta es la lenta), de modo que
  **C2 queda evaluada exactamente en su convergencia** y el sobre-entrenamiento recae sobre las condiciones
  baratas y ya saturadas (C1, y en parte C3). La asimetría residual apunta, por lo tanto, en la dirección
  inofensiva. Se reporta `N_final` por condición junto a `N_common`.

Estas cláusulas se aplican **igual** a PS-2, PS-3, PS-4 y PS-5 y a las predicciones P1.2/P1.3 del madre.

## Anexo C — Carga de medición de T2 en PS-5 (resuelve O4)

La correctabilidad se mide en **L=32** (como en S0.9) **y también en la carga de evaluación**. Para PS-5:

- **(C1) Primaria = versión misma-carga** (T2 medida en la carga de evaluación), **si** su distribución entre
  semillas **no es degenerada**: media entre semillas **> 0.20** (el azar es 1/64 = 0.0156; el umbral está ~13×
  por encima y deja recorrido para variar) **y** SD entre semillas **≥ 0.01** (un orden por debajo de la SD
  observada en S0.9, 0.0415).
- **(C2) Fallback pre-registrado:** si la versión misma-carga queda **pisada contra el suelo** (falla
  cualquiera de las dos condiciones de C1), se declara **«T2 no evaluable a la carga de evaluación»** y la
  versión **L32 pasa a primaria**, con su condición **cross-carga declarada** en el reporte (capacidad a carga
  alta vs. correctabilidad a carga media).
- **(C3)** La elección se decide **por esta regla, no mirando cuál da mejor correlación**. Ambas versiones se
  reportan siempre.

---

# Predicciones nuevas (v1.1)

## PS-4 — Forma de la degradación (inicio y pendiente)

De la C2 convergida de E1, la curva acc@1 vs. carga cumple:

- **(i) Inicio de la caída — predicción de punto.** Para cada semilla `s` se define
  `L₀(s)` = **menor L con acc@1 < 0.99** (umbral de «salir del techo»; **no** se usa 0.95, que ya es
  degradación avanzada y desconectaría L₀ de la escala teórica n*≈35). Se predice
  **`mediana(L₀) = 64` exactamente** sobre las 8 semillas.
  - **Por qué de punto y no `≥ 64`:** la mediana blinda contra el parpadeo de una semilla suelta en una carga
    saturada (L32 = 0.9990 en el snapshot, a 0.001 del umbral), y el «= 64» vuelve la predicción **falsable
    hacia arriba**, que es la dirección informativa: si la convergencia cura L64 (todas las semillas de vuelta
    sobre 0.99), `mediana(L₀)` salta a 96 y la hipótesis «el inicio es geometría del d² por cabeza» queda
    tocada — significaría que el inicio observado era artefacto de presupuesto, no umbral.
  - **Confirma:** `mediana(L₀) = 64`. **Falsa:** `mediana(L₀) ∈ {8,16,32}` (inicio más temprano) o `= 96`/`128`
    (el inicio era artefacto de presupuesto). **No concluyente:** la mediana cae **entre** dos valores de la
    grilla (p. ej. 80, empate 4–4 entre semillas) → se reporta «no concluyente por dispersión entre semillas»,
    con la distribución completa de `L₀(s)`.
- **(ii) Monotonía.** acc@1 no crece al aumentar L en el tramo `L ≥ mediana(L₀)`: Spearman(acc@1, L) < 0 con
  IC apareado por semilla que no cruce 0 en ese tramo.
- **(iii) Pendiente creciente (la caída acelera).** `[acc@1(L96) − acc@1(L128)] > [acc@1(L64) − acc@1(L96)]`,
  con IC bootstrap apareado por semilla que no cruce 0. (En el snapshot: 0.114 vs 0.086.)
- **Confirma (conjunto):** la degradación por saturación del estado tiene forma característica — meseta,
  luego caída acelerada. **Falsa:** caída lineal (pendiente constante) o inicio desplazado según (i).

## PS-5 — Anticorrelación capacidad ↔ correctabilidad (entre semillas)

Entre las **8 semillas** de C2 convergida, la **capacidad** (acc@1 en la carga de evaluación) y la
**correctabilidad** (T2, según el Anexo C) están **anti-correlacionadas**: Pearson **< 0**.

- **Origen (snapshot @2500):** seed2/3 mayor capacidad (L128 ~0.79) / menor T2 (~0.85); seed0/1 al revés
  (0.755 / ~0.92).
- **Hipótesis rival explícita (régimen de parada).** Con parada colectiva, las semillas rápidas quedan
  **sobre-entrenadas** respecto de su propio criterio. Si el sobre-entrenamiento afecta distinto a capacidad y
  a correctabilidad, la anticorrelación podría **emerger del punto de parada** y no de un trade-off intrínseco
  de rango. Para separarlas se registra, por semilla, `paso_conv(s)` = primer múltiplo de 500 en que **esa**
  semilla cumplió el criterio (recuperable de `val_hist`), y se reportan:
  1. **Pearson crudo** (capacidad, T2) con IC bootstrap 95% sobre semillas;
  2. **Pearson parcial** (capacidad, T2 | `paso_conv`);
  3. **diagnóstico:** Pearson(capacidad, `paso_conv`) y Pearson(T2, `paso_conv`) **por separado** — si ninguna
     de las dos correlaciona con el punto de parada, la hipótesis rival muere sola y la parcial es formalidad;
     si ambas correlacionan con signos opuestos, la rival queda identificada con nombre y apellido.
- **Regla de veredicto (explícita, porque con n=8 la parcial es frágil):**
  - **Confirmada:** cruda **negativa** con IC bootstrap que **excluya el cero** **y** parcial **del mismo
    signo** reteniendo **≥ 50 %** de la magnitud de la cruda (`|parcial| / |cruda| ≥ 0.5`).
  - **«Confundida por régimen de parada»:** la parcial **invierte** el signo respecto de la cruda
    (*inversión*) **o** se **atenúa hacia cero** conservando el signo (`|parcial| / |cruda| < 0.5`,
    *atenuación*).
  - **No concluyente:** el IC de la cruda cruza cero (potencia insuficiente) — no se fuerza veredicto.
- **Precisión O5 (ejecutor, 2026-07-23, antes del freeze — por qué existe la banda del 50 %).** La regla
  original decía «confundida si la parcial invierte el signo». Al implementarla se verificó que **el caso
  canónico del confound no invierte el signo: lo atenúa hacia cero**. Simulando la hipótesis rival exacta
  (capacidad y T2 gobernadas por `paso_conv` con signos opuestos, sin relación residual) se obtiene cruda
  = −1.00 y parcial ≈ 0; **con n = 8, el signo de una parcial ≈ 0 lo decide el ruido**, de modo que la regla
  literal habría reportado «confirmada» en el escenario que fue diseñada para detectar, según de qué lado del
  cero cayera el azar. La banda de retención cierra ese agujero por magnitud en vez de por signo. El umbral
  0.5 se fija **acá, antes de ver datos**. Verificado en `experimentos/E1/test_analisis_e1.py` (caso
  construido con residuos ortogonalizados: retención 0.30 → atenuación).
- **Las sub-etiquetas *inversión* / *atenuación* son descriptivas**, no cambian el veredicto: con n=8 la
  frontera entre ambas es ruidosa. Lo que tiene fuerza de veredicto es «confundida por régimen de parada».
- Se reporta además Spearman como robustez. PS-4 y PS-5 son **secundarias**: caracterizan el plateau, no
  condicionan la confirmación de PS-1 (Puente 2).

## Congelamiento

Este documento se congela (hash SHA-256 + companion `FREEZE_PREREG_SEGUIMIENTO_v1.1.md` + tag firmado +
push al repo público) **antes de la primera corrida de E1**. Ninguna predicción se ajusta después de ver
datos. El v1.0 permanece congelado e inalterado; el v1.1 lo extiende sin reescribirlo.

*Autoría del diseño: Maxi + Fable 5 (marco y veredictos O1–O4) + ejecutor Opus 4.8 (draft, lectura de
ejecutor e implementación). Programa TELAR / Ligamento.*
