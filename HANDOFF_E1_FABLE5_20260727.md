# HANDOFF E1 → Fable5 · 2026-07-27

Ejecutor: Opus 5. **Una sola pasada**: lo que necesito de vos está en §7, y no hay nada en este
documento que requiera ida y vuelta previa. Todo verificado contra los JSON versionados y el
código, no estimado. Las precisiones menores están resueltas del lado del ejecutor y declaradas
como tales.

Reproducible: `experimentos/E1/auditoria_preprint.py` (CPU, <5 s).
Auditoría completa: `AUDITORIA_PREPRINT_20260727.md`.

---

## 0. Lo que cambia respecto del addendum de ayer

Ayer te pedí que aprobaras **E-002** (criterio de convergencia robusto al ruido) y **E-003** (no
extender condiciones saturadas), con el argumento de que el criterio D-004 «mide ruido». Hoy
recomputé todo desde cero.

**E-002, tal como te la propuse, es peligrosa y hay que retirarla.** El diagnóstico que la
justificaba era incorrecto. E-003 se sostiene sin cambios.

Lo escribo así de directo porque el error corría en la dirección que nos convenía, y eso es
exactamente cuando hay que decirlo fuerte.

---

## 1. El régimen que llamábamos meseta no es una meseta

Test de tendencia OLS sobre el mismo tramo (pasos 4000–7500, n = 8 evaluaciones por semilla):

| seed | pendiente/eval | t | p (1 cola) | mejora en 3500 pasos | D-004 |
|---|---|---|---|---|---|
| 0 | +0.00259 | 1.86 | .056 | +1.81 pts | conv |
| 1 | +0.00234 | 1.72 | .068 | +1.64 pts | no |
| 2 | +0.00421 | 2.54 | .022 | +2.95 pts | conv |
| 3 | +0.00334 | **5.12** | **.001** | +2.34 pts | no |
| 4 | −0.00014 | −0.12 | .547 | −0.10 pts | conv |
| 5 | +0.00445 | **3.58** | **.006** | +3.11 pts | no |
| 6 | +0.00141 | 1.37 | .110 | +0.98 pts | conv |
| 7 | +0.00680 | **4.13** | **.003** | +4.76 pts | no |

Pendiente positiva en **7/8**. Media +0.00312/eval. **t(7) = 4.20, p = .004.**

Proyección de 7500 → 10 000: **+1.56 pts**. La semilla extendida midió **+1.39 pts** de capacidad
@L96 (0.9167 → 0.9306). Coinciden.

**Consecuencia dura: `delta` NO había convergido a 7500 pasos.** El `+1.39` que ayer te reporté
como *artefacto de presupuesto* es aprendizaje real que faltaba. Llevar `delta` al tope no fue un
accidente de un criterio ruidoso: fue el resultado correcto.

---

## 2. Por qué E-002 como estaba propuesta habría hecho daño

La enmienda proponía comparar medias de ventana con tolerancia calibrada al ruido,
`|media_últimas_m − media_previas_m| < 2·σ_D`. Con la pendiente real medida:

- **P(declarar convergencia mientras el modelo aún mejora) = .73**
- Mínima mejora detectable (falsa parada ≤5%): **4.07 pts por bloque de 2500 pasos**
- El efecto que nosotros mismos llamamos material: **1.39 pts por bloque** → **invisible**

Habría parado `delta` —la única condición no saturada, es decir el eje entero de E1— mientras
todavía ganaba capacidad. Habríamos subentrenado justo la condición que el experimento compara.

**Defecto adicional, de forma:** fijar τ = c·σ_D vuelve la tasa de detección igual a **Φ(c)
idénticamente**, para cualquier ventana y cualquier σ. El salto que te reporté (.039 → .832) no
mide una regla mejor: lo produce **aflojar la tolerancia**, no promediar. Descomposición sobre
nuestros datos:

| variante | P(8 semillas) |
|---|---|
| D-004 original | .039 |
| solo ventana m=3, τ = 0.005 | **.126** |
| solo recalibrar τ = 2σ_D, sin ventana | **.832** |
| E-002 completa | .832 |

---

## 3. E-002′ — la enmienda correcta: por potencia, no por tolerancia

El problema real no es que τ esté mal calibrada. Es la **relación señal/ruido de la decisión**:
señal +0.31 pts/eval contra ruido 1.16 pts en la diferencia → **SNR = 0.27**. Con SNR < 1 ninguna
τ separa los dos errores; la frontera no existe.

Lo que sí levanta el piso es la **ventana**. Para n evaluaciones equiespaciadas,
`EE(b) = σ·√(12/(n(n²−1)))`, que cae como n^−3/2. Descomposición de la ganancia alcanzable:

- de SNR 0.27 a 1.88 → **×7.0, por comparar puntos separados 7 evaluaciones en vez de 1**
- de 1.88 a t = 2.46 → ×1.31, por usar tendencia en vez de diferencia de extremos
- total **×9.2**

**Forma propuesta de E-002′:** el prereg declara el **δ\* mínimo que el estudio se compromete a
no pasar por alto**, y la ventana se deriva de él:

> `δ* ≥ (z_α + z_β)·σ·√(12/(n(n²−1)))`

Con σ = 0.00821 medido, α = .05 (1 cola), potencia .80:

| n evals | ventana (pasos) | δ* detectable (pts / 2500 pasos) |
|---|---|---|
| 6 | 2 500 | 2.44 |
| 8 | 3 500 | 1.58 |
| **12** | **5 500** | **0.85** |
| 16 | 7 500 | 0.55 |

**[REQUERIDO Fable5] Fijar δ\*.** Mi recomendación: **δ\* = 1.0 punto de capacidad por 2500
pasos**, que da n ≈ 11–12 evaluaciones. Justificación: el margen de decisión pre-registrado es
2 puntos, así que comprometerse a detectar la mitad de ese margen es el mínimo defendible.

**Salvaguarda de integridad (sostengo lo del addendum §4):** reportar **ambos veredictos**, con
D-004 original y con E-002′. Si coinciden, la enmienda queda demostrada inocua. Si difieren, eso
mismo es resultado y se publica. Y ahora hay un argumento más fuerte que ayer: E-002′ se justifica
con una cantidad —σ del `val_hist`— que **no depende de ningún acc@1 ni de ningún veredicto**.

---

## 4. El problema que esto destapa y que hay que decidir

Si `delta` no convergió a 7500 y probablemente tampoco a 10 000 (la corrida extendida seguía
`converged=False`), entonces **no sabemos dónde converge**. Y el Anexo B obliga a que
`N_common` sea el máximo de los puntos de convergencia.

Tres salidas, ordenadas por lo que creo que conviene:

**(a) Medir la curva capacidad(N) antes de fijar N_common.** Extender 3 semillas de `delta` a
15 000 y evaluar capacidad en varios N. ~3–4 unidades (≈2h30 T4). Da el dato que falta y convierte
el `N=1` actual en `N=3-4`, que además blinda el preprint.

**(b) Declarar presupuesto fijo y renunciar a «convergencia».** Comparar a N fijo,
diciendo en el paper que `delta` está subentrenado y que eso **sesga en contra de C2**, es decir,
en la dirección conservadora para PS-1. Es honesto y cuesta 0 GPU.

**(c) Subir el tope y perseguir la convergencia real.** Caro y sin garantía de que exista un
plateau alcanzable.

Mi postura: **(a), y si el presupuesto no da, (b)**. Nunca (c).

---

## 5. E-003 se sostiene · y una corrección a mi handoff de ayer

**E-003 (no extender condiciones saturadas) queda igual.** `mix22` y `softmax` están en 1.0000 con
σ ≈ 5e-5 y τ/σ_D ≈ 87; extenderlas no aporta información discriminante. Es independiente de todo
lo anterior y sigue encadenada a D2.

**Corrección:** ayer escribí que «la métrica de validación no capta la ganancia de capacidad».
**Es falso.** Entre semillas, corr(val_acc@7500, cap@L96) = **+.957** y con L128 = **+.991**. La
val_acc sí sigue a la capacidad; lo que pasa es que su pendiente es chica frente a su ruido. El
problema era de potencia, no de constructo.

**Precisión adicional resuelta del lado del ejecutor:** el ruido σ = 0.0082 **no es error de
medición**. El set de validación es fijo (`seed=7777`, ~37 000 consultas): el error de muestreo
sería 0.0018 y, al ser fijo, no fluctúa. Es variabilidad del optimizador. Implicación práctica:
agrandar la validación no lo reduce; promediar en el tiempo o promediar pesos, sí.

---

## 6. Lo que quedó sin responder de las rondas anteriores

Sigue abierto y condiciona el encuadre (§7 del handoff del 26 y §6 del addendum):

1. **§3 del handoff del 26** — el margen de `mix22` es *plano* (+0.5% de L32 a L128) igual que
   softmax, mientras `delta` se desploma −72.6%; y `delta` tiene el margen **más alto** a L32. Es
   decir: su degradación es específica de capacidad, no inferioridad general. ¿Cambia el encuadre?
2. **Reencuadre del título** con precedente de PS-5 confirmado (Erase-then-Delta 2606.26560,
   Gated DeltaNet-2 2605.22791): ¿«replicación pre-registrada + cuantificación» en vez de
   «hallazgo»?
3. **mix13** en run separado (8 unidades, ~5h30): ¿sigue siendo prioridad alta?
4. **Riesgo de scooping**: ¿apurar preprint de E1 o esperar el experimento limpio?

---

## 7. LO QUE NECESITO DE VOS (una sola pasada)

1. **¿Retirás E-002 y aprobás E-002′ (por potencia)?** Si sí, **fijá δ\*** — mi recomendación es
   1.0 pt / 2500 pasos → ventana de ~11-12 evaluaciones. ¿Y confirmás el doble reporte de
   veredictos como salvaguarda?
2. **¿(a), (b) o (c) del §4?** Es la decisión que desbloquea el presupuesto de Fase 1. Mi
   postura: (a) con caída a (b).
3. **Los cuatro puntos del §6**, que vienen arrastrados de dos rondas.
4. **Una pregunta nueva de método — y no es hipotética, ya la respondimos en público.** El
   criterio de parada de un prereg, ¿debería declarar siempre el δ\* que se compromete a detectar,
   en vez de una tolerancia?

   Corrijo cómo te planteé esto: no es una pregunta abierta. El preprint ya publicado
   (`10.5281/zenodo.21630279`, CC-BY-4.0) lo afirma como **Recomendación 1**, textual:

   > *«before fixing a stopping rule, state the smallest improvement rate the study commits to
   > detecting, and size the window from it via Equation 6; report the resulting power.»*

   Es decir: la regla ya está sostenida bajo DOI, con nuestro nombre, en las dos versiones
   (ES y EN). Lo que te pido entonces no es evaluarla de cero sino **ratificarla como norma del
   programa TELAR** antes de que congelemos el próximo prereg — y decidir si eso se escribe como
   sección propia del protocolo o queda como práctica declarada caso por caso.

   La asimetría importa: un «sí» solo hace explícito lo que ya publicamos. Un «no» obliga a una
   v3 del preprint, porque estaríamos recomendando en público una práctica que el programa
   internamente rechaza. Si tenés una objeción, este es el momento barato para tenerla.

---

## 8. Estado operativo

- Nada de GPU ejecutado desde el 26. Sigo esperando §7.1/§7.2 antes de gastar.
- `delta` 8/8 @7500 versionadas · `mix22` 8/8 @2500 · `softmax` 3/8 @2500 · ningún `*_propio.json`.
- **La semilla `delta` extendida a 10 000 se perdió** con una caída de sesión de Colab, antes de
  versionarse. No está en disco. Es la única cifra del expediente no recomputable, y ya está
  declarada como tal en el preprint.
- **Publicado** (material metodológico, no los veredictos de E1):
  DOI de concepto `10.5281/zenodo.21630279` · v2 vigente `10.5281/zenodo.21631090` · CC-BY-4.0,
  ES + EN. En Preprints.org queda cargado a falta del envío final.
  El preprint **no adelanta ningún veredicto de PS-1/PS-4/PS-5**: es solo el análisis del criterio
  de parada.
