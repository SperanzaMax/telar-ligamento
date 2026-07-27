# Auditoría del preprint «When the Stopping Criterion Measures Noise» + revisión de Gemini

**Fecha:** 2026-07-27 · **Auditor:** Opus 5 · **Método:** recómputo independiente sobre
`resultados/E1/*.json` y `src/entrenar.py`. No se reutilizó `analisis_criterio.py`.
Script reproducible: `experimentos/E1/auditoria_preprint.py` (CPU, <5 s).

---

## Veredicto en una línea

La tesis central **no se sostiene como está redactada**: el tramo que el paper trata como meseta
no es una meseta. Hay material para un paper más fuerte en los mismos datos, sin GPU adicional,
pero exige reescritura sustantiva del argumento (no del formato).

---

## 1. El defecto que hunde la tesis actual: la meseta no existe

El paper modela las observaciones de meseta como `acc(N) = μ + ε`, con μ constante. Test de
tendencia OLS sobre el mismo tramo que usa el paper (pasos 4000–7500, n = 8 evaluaciones):

| seed | pendiente/eval | SE | t | p (1 cola) | mejora en 3500 pasos | D-004 |
|---|---|---|---|---|---|---|
| 0 | +0.00259 | 0.00139 | 1.86 | 0.056 | +1.81 pts | True |
| 1 | +0.00234 | 0.00136 | 1.72 | 0.068 | +1.64 pts | False |
| 2 | +0.00421 | 0.00166 | 2.54 | 0.022 | +2.95 pts | True |
| 3 | +0.00334 | 0.00065 | 5.12 | 0.001 | +2.34 pts | False |
| 4 | −0.00014 | 0.00116 | −0.12 | 0.547 | −0.10 pts | True |
| 5 | +0.00445 | 0.00124 | 3.58 | 0.006 | +3.11 pts | False |
| 6 | +0.00141 | 0.00103 | 1.37 | 0.110 | +0.98 pts | True |
| 7 | +0.00680 | 0.00165 | 4.13 | 0.003 | +4.76 pts | False |

- Pendiente positiva en **7/8** semillas. Media +0.00312/eval. **t = 4.20, p = 0.004** contra H₀: pendiente = 0.
- Proyección de 7500 → 10000 pasos: **+1.56 pts de val_acc**.
- Observado en s0 extendida: **+1.45 pts de capacidad @L96**. Coinciden.

**Consecuencia.** El `+1.45` que el paper presenta como *artefacto de presupuesto* es
**aprendizaje real que todavía no había terminado**. En el tramo analizado el modelo mejora
≈ 0.31 pts cada 500 pasos. Llevar `delta` al tope de 10 000 no fue un accidente del criterio:
fue la decisión correcta. La afirmación «el criterio mide ruido, no convergencia» no está
demostrada — a 7500 pasos `delta` efectivamente **no había convergido**.

**Correlato secundario.** Comparando D-004 contra el test de tendencia como referencia, los dos
coinciden en 6 de 8 semillas (bajo azar puro, P(≥6/8) = 0.145). Con n = 8 no hay potencia para
afirmar que D-004 es azaroso *ni* que detecta. El claim honesto es: **el diseño no puede resolver
la señal**, no «el criterio mide ruido».

---

## 2. La corrección por autocorrelación es espuria (y Gemini la validó)

El paper (§Sensitivity) y Gemini afirman que ρ₁ ≈ −0.35 infla Var(D) y por lo tanto la Ec. 2
*sobreestima* P. El álgebra de Gemini —Var(D) = 2σ²(1−ρ), σ_D = 0.01379, P = 0.6415,
P⁸ = 0.028— es **aritméticamente correcta**. La cantidad de entrada **no existe**:

1. **Es el sesgo esperado del estimador.** Con n = 8 puntos y tendencia lineal removida, bajo
   ruido blanco **puro** (20 000 simulaciones): E[ρ̂₁] = **−0.287**, IC95 [−0.803, +0.310].
   El observado (mediana −0.315) cae en el **percentil 49** de esa nula. No hay señal.
2. **El propio paper lo refuta en la tabla de al lado.** Si ρ = −0.31 fuera real,
   σ_diff/σ_resid debería ser √(1−ρ) = **1.147**. El ratio observado es **1.0097**
   → ρ implicado = **−0.02**. Los dos estimadores «concuerdan estrechamente» exactamente
   porque ρ ≈ 0.

Hay que **eliminar** ese párrafo. Es un flanco gratuito: un revisor de estadística lo ve en un
minuto y contamina la credibilidad del resto.

---

## 3. La Ec. 3 tiene tasa de detección tautológica

Fijar τ = 2σ_D implica P(declarar convergencia en meseta) = Φ(2) = **0.977 siempre**, para
cualquier m y cualquier σ. El 0.832 = 0.977⁸ no mide el criterio de ventana: es identidad
algebraica de haber elegido τ como múltiplo de σ_D. Descomposición sobre los datos reales:

| variante | τ | σ_D | P (1 semilla) | P (8) |
|---|---|---|---|---|
| A · D-004 original | 0.005 | 0.01162 | 0.667 | 0.039 |
| B · sólo ventana m=3, τ = 0.005 | 0.005 | 0.00671 | 0.772 | **0.126** |
| C · sólo recalibrar τ = 2σ_D, sin ventana | 0.0232 | 0.01162 | 0.977 | **0.832** |
| D · Ec. 3 completa | 0.0134 | 0.00671 | 0.977 | 0.832 |

El salto 0.038 → 0.832 lo produce **enteramente la recalibración de τ**, no la ventana.

### Y la Ec. 3 es peor que D-004 en el eje que al paper le importa

Especificidad no reportada. Con la pendiente real medida (+0.0093 por ventana de 1500 pasos):

- **P(la Ec. 3 declare convergencia mientras el modelo aún mejora) = 0.73.**
- Mínima mejora detectable (falsa parada ≤ 5 %): **4.07 pts por bloque de 2500 pasos**.
- El efecto que el paper llama material: **+1.45 pts por 2500 pasos** → **invisible** para el
  criterio propuesto.

La solución propuesta habría subentrenado `delta` justo en el eje que el experimento compara.
Es una contradicción interna: el paper propone un remedio que agrava el daño que denuncia.

---

## 4. Estimación no paramétrica (reemplaza al modelo gaussiano)

Con eval cada 500 pasos y w = 500, el criterio D-004 **es exactamente la última diferencia
sucesiva**. No hace falta modelo: la fracción empírica de diferencias sucesivas < τ en el tramo
(N = 56) es

**P = 0.554, IC95 [0.423, 0.684]** → P(8 semillas) = **0.0088**

frente a 0.663 / 0.0375 del modelo gaussiano. Ajusta mejor a lo observado (4/8):
E[X] = 4.4 vs 5.3; P(X≤4) = 0.515 vs 0.259. Este estimador elimina de un golpe las objeciones
de gaussianidad, direccionalidad y autocorrelación.

---

## 5. El ruido no es de medición — es del optimizador

`_val_acc` (`src/entrenar.py:94`) usa `seed=7777` **fijo**: el conjunto de validación es idéntico
en cada evaluación. Con reps=2 × batch=64 × cargas {64,96,128} ≈ 36 864 queries, el error de
muestreo sería σ ≈ 0.0018 — y al ser el conjunto fijo, **ni siquiera fluctúa**. El σ = 0.0082
observado es **100 % variabilidad del proceso de optimización**.

Implicaciones: (a) agrandar el set de validación no reduce este ruido; (b) la calibración a
priori de τ por error de muestreo binomial **no aplica** acá; (c) el remedio correcto es
promediado temporal —de la métrica o de los pesos (EMA/SWA)— o un test de tendencia.

---

## 6. Errores factuales y de reproducibilidad

| ítem | paper | verificado |
|---|---|---|
| cargas de validación | «loads {96, 128}» | código: **(64, 96, 128)** |
| capacidad s0 @7500 | 0.9161 | JSON: **0.9167** |
| «4-of-9 split» | 9 runs | en el repo hay **8**; el 9.º (s0@10000) **no está versionado** |
| σ de saturadas (τ/σ = 122.8) | 4 cifras | estimado con **4 puntos** por run; el número exacto no es interpretable |
| Open Practices | «reproduce every number» | **falso hoy**: falta el run extendido |
| `referencias.bib` | — | 6 entradas con `VERIFICAR-AUTORES`; `sae2026reliable` con autoría **pendiente** |

---

## 7. Qué se sostiene intacto

1. **Heterocedasticidad dependiente de arquitectura.** σ varía ~3 órdenes de magnitud
   (0.0082 vs 0.00005). Un τ fijo compartido es estructuralmente incapaz de ser justo entre
   condiciones. Esto no depende de ningún supuesto de meseta.
2. **El confound de presupuesto es real** — con otro mecanismo: las saturadas cierran de
   inmediato mientras la no-saturada aún mejora, así que cualquier N_common o subentrena a una
   o sobreentrena a las otras, y **cuál de las dos ocurre lo decide la condición que satura**.
3. **La ventaja del pre-registro con DOI** para argumentar que el defecto no fue elegido post
   hoc. Sigue siendo el activo diferencial.

---

## 8. El paper que sí sale de estos datos

**Reencuadre:** de «la tolerancia es menor que el ruido» a **«la relación señal/ruido de la
decisión de parada»**.

- Señal real: +0.31 pts/eval. Ruido de la diferencia: 1.16 pts. **SNR = 0.27.**
- Con SNR < 1 **ningún τ funciona**: τ regula el trade-off entre parar tarde y parar temprano,
  pero no hay frontera — cualquier τ con detección alta en meseta también para durante mejora
  real (demostrado en §3).
- La salida no es calibrar τ: es **cambiar el estadístico**. Test de pendiente OLS sobre la
  misma ventana de 8 puntos: SE(b) = σ·√(12/(n(n²−1))) = 0.00127 → **t = 2.45** para la señal
  media. La misma ventana, los mismos datos: el test de dos puntos no ve nada (SNR 0.27), el
  test de tendencia detecta (t = 2.45). La potencia crece con n^{3/2} en vez de con √n.
- Recomendación accionable y pre-registrable: elegir w por **potencia** — dado el δ mínimo que
  te importa no perderte y el σ del proceso, w tal que δ·w/SE(b,w) ≥ z_α + z_β.

Con eso el paper: (a) no depende de un supuesto de meseta falso; (b) no necesita el argumento de
autocorrelación; (c) explica por qué D-004 falló *y* por qué la Ec. 3 fallaría peor; (d) da un
criterio derivado, no elegido; (e) mantiene intactos los dos resultados fuertes del §7.

**Título sugerido:** *Stopping Criteria Below the Signal-to-Noise Floor: Why Two-Point Tolerance
Rules Cannot Detect Convergence, and What Replaces Them.*

---

## 9. Consecuencia inmediata para Ligamento/E1

**E-002, tal como está propuesta en el addendum a Fable5, es peligrosa** y no debe aprobarse en
esa forma: habría declarado convergida a `delta` mientras aún mejoraba ~0.31 pts/eval
(P = 0.73), subentrenando la única condición no saturada — precisamente el eje que E1 compara.
La enmienda correcta es **por potencia**, no por tolerancia, y debe declarar explícitamente el
δ mínimo que se compromete a no perder.

**E-003 (no extender condiciones saturadas) se sostiene** y es independiente de todo lo anterior:
mix22 y softmax están en 1.0000 con σ = 5e-5; extenderlas no aporta información discriminante.

**Dato nuevo y no negociable:** `delta` **no había convergido a 7500 pasos**. Cualquier veredicto
de PS-1 que compare capacidades a ese presupuesto está midiendo una condición subentrenada.

**Corrección al handoff del 26-07:** la afirmación «la métrica de validación no capta la ganancia
de capacidad» es **falsa**. Entre semillas, corr(val_acc@7500, cap@L96) = **+0.957** y
corr(val_acc@7500, cap@L128) = **+0.991**. La val_acc sí sigue a la capacidad; lo que pasa es que
su pendiente es pequeña frente a su ruido.
