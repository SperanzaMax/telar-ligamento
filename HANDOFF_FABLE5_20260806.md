# Handoff a Fable5 — 2026-08-06

> **RESUELTO el 2026-08-06 · ver `DICTAMEN_20260806.md`.** Las tres devoluciones recibidas (Z.ai,
> Kimi K2.6, Opus5 Max) se integraron y la respuesta no está en el menú que plantea §6: **la
> bifurcación A/C no existe**. R3 —la rama cuya consecuencia (a) es la salida C y cuya consecuencia
> (b) es la salida A— exige *«bisección cerrada en **ambos** escalones»*, y **E-b nunca corrió**. Lo
> que corresponde es **correr E-b**, con las 3 h ya pre-asignadas en el tope congelado y en T4
> (la calibración declara hardware indistinto). Dos apartados de este handoff quedaron corregidos
> abajo (§2 retirado, §4 refundado).

**Qué se necesita de vos:** una decisión sobre **cómo sigue E2–E4** después de que la calibración de
E-005 corrió y devolvió **R4**, y de que el sondeo posterior mostró que **el espacio candidato de
E-005 no contiene ningún régimen usable**. Hay tres salidas y ninguna es obvia (§6).

Todo lo que sigue se midió hoy, en la estación de trabajo (CPU, 2–3 hilos, techo térmico 70 °C).
**Ningún artefacto congelado fue tocado**: el gate de anclas (`verificar_anclas.py --requiere E-005`)
sigue verde y nada está commiteado.

---

## 1. La calibración de R-BANDA corrió — rama R4

Ejecutada tal como quedó congelada: escalón **E-a** (NK=256, L_max=128), `softmax`, k=1, semillas
0 y 1, parada por convergencia con tope 2500. Duró **87 min**, cortó por convergencia a los 1000
pasos en ambas semillas, temperatura máxima 57 °C, cero pausas.

| L | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|
| acc@1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 |

Cero cargas en la banda [0,50 · 0,80]. Registro en `resultados/calibracion/calibracion_rbanda.json`.

**Dos precisiones sobre la rama que devolvió, porque el texto congelado no describe bien este caso:**

1. El veredicto R4 dice *«el tope se agotó con la bisección ABIERTA»*, pero en E-a la bisección
   **cerró** (`biseccion_cerrada: true`). Lo que disparó R4 fue que el **espacio de escalones quedó
   incompleto** —E-b no se corrió porque no entra en el tope de 3 h en esta máquina—, no una
   bisección abierta. La lógica de `decidir()` es correcta (sin correr E-b no se puede declarar R3);
   es la redacción del veredicto la que no cubre este caso. **No se tocó**: es artefacto congelado.
2. **El costo por paso se subestimó un 44 %**: 2,60 s/paso reales contra los 1,815 del bench del
   5-ago. Cualquier presupuesto futuro debería usar el número medido.

---

## 2. El eje de la CARGA está agotado (sondeo, no range-finding)

Antes de pagar las 3–5 h de E-b se evaluaron los checkpoints de E-a en cargas mayores, sin
reentrenar (4 min):

| L | 128 | 160 | 192 | 224 | 256 |
|---|---|---|---|---|---|
| acc@1 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.9999 |

> **⚠ CORREGIDO el 2026-08-06 — ver `DICTAMEN_20260806.md` §3. Este apartado NO autoriza a
> saltearse E-b, y su conclusión queda retirada.** Tres defectos: (i) el sondeo movió `L` pero no
> `NK`, y **E-b es el par (NK=512, L_max=256)** — sólo se probó media dimensión; (ii) las celdas
> L=192 y 256 con NK=256 violan `L_max ≤ NK/2`, la restricción de no degeneración que la propia
> enmienda §3.1 señala como causa de que *«la celda tope del régimen viejo mida mal»*; (iii) la
> asimetría está **mal signada**: menos distractores es una tarea **más fácil**, así que juega a
> favor de la accuracy alta observada, no en contra. Mover `NK` sin reentrenar es imposible (el
> embedding de E-a tiene 325 filas, NK=512 exige 581). **E-b hay que entrenarlo.**

~~**Vale como evidencia fuerte por asimetría:** el sondeo juega doble en contra del modelo
—extrapolación fuera de L≤128 y sin los distractores de E-005 §3.1— y aun así no se despeina. Un
entrenamiento propio a L=256 rendiría **más**, así que **E-b tampoco alcanzaría la banda**. Se
ahorraron 3–5 h de cómputo.~~

Para cruzar por este eje con d=64 haría falta L≈380–600. Con el costo yendo como T², eso son
**cientos de horas**: no es falta de presupuesto, es inalcanzable a cualquier presupuesto de Maxi.

Script: `experimentos/E1/sondeo_extension_L.py` (rotulado exploratorio en su docstring).

---

## 3. El eje de la CAPACIDAD: se encontró el punto justo, y no sirve

`d` no está en el espacio candidato de E-005; esto es **exploración para redactar una enmienda**, no
range-finding, y ningún número de acá elige régimen. Se varió `d` por monkeypatch sobre
`modelos.D/DH/FFN_HID` (leídas en runtime), **sin tocar `src/modelos.py`**. Un proceso por punto.

### 3.1 Barrido (L_max=64, NK=128, seed 0)

| d | DH | params | acc@1 en L=8…64 |
|---|---|---|---|
| 8 | 2 | 6 309 | 0,030 → 0,018 (**azar**, 1/64 = 0,016) |
| 12 | 3 | — | 0,9873 → 0,9307 (**primera pendiente real del proyecto**) |
| 16 | 4 | 17 541 | 0,9946 → 0,9916 (plano, techo) |
| 64 | 16 | ~400 k | 1,0000 (ancla de §1) |

**`d` debe ser múltiplo de H=4**, y H está fijado por §5 del protocolo (las condiciones `mix22`/
`mix13` reparten 4 cabezas). Entre 8 y 12 **no hay escalón**: el eje es demasiado grueso justo en la
zona de interés.

### 3.2 El corte por convergencia mintió — y esto contamina el barrido

`d=8` fue rotulado «convergió en 500 pasos» a nivel azar. Reentrenado **sin corte** hasta 9000 pasos:

```
250:0,019  500:0,019  750:0,022  1000:0,047  1500:0,126
2000:0,176  2500:0,229  3000:0,265  5000:0,366  6750:0,401  9000:0,373
```

Estuvo **750 pasos en meseta y después despegó**. El corte (ventana 250, tol 0,005) se disparó
dentro de la meseta. **Es el mismo error que la auditoría del 2026-07-27 encontró en el borrador del
preprint**, y la razón de ser de `stoppower`. Consecuencia: **todo punto del barrido corrido con
ventana 250 es no concluyente**. Se salvó que cada punto llevaba el rótulo `NO INTERPRETABLE`, así
que ningún número quedó firmado como bueno.

### 3.3 El régimen candidato: d=8 · L_max=48 · NK=128

Con N fijo 16000 pasos y **sin** corte, 4 semillas:

| L | seed 0 | seed 1 | seed 2 | seed 3 | media | SD |
|---|---|---|---|---|---|---|
| 32 | 0,7637 | 0,7654 | 0,8171 | 0,5972 | 0,7359 | 0,096 |
| 48 | 0,7218 | 0,6759 | 0,7334 | 0,4946 | 0,6564 | 0,111 |
| 64 | 0,6758 | 0,5854 | 0,6544 | 0,4138 | 0,5824 | 0,119 |

**R-BANDA cumple**: las tres cargas de la franja superior caen en [0,50 · 0,80] en la media. Las
curvas de las cuatro semillas están **planas** al final (no es convergencia incompleta: convergen a
lugares distintos).

---

## 4. Lo que descarta el régimen: `mix22` es INESTABLE, no incapaz

Se corrió `mix22` —la condición que E2 compara— en el mismo régimen, mismas semillas. Evaluada al
final da 0,03–0,06 (azar). **Pero la curva cuenta otra cosa:**

| semilla | pico | paso del pico | valor final (16000) |
|---|---|---|---|
| 0 | **0,5187** | 8000 | **0,0139** |
| 1 | **0,5422** | 6500 | 0,2186 |

`mix22` **aprende** en d=8 y después el entrenamiento **se rompe**. Esto confirma empíricamente la
objeción conceptual: con d=8 cada cabeza tiene DH=2, y una cabeza delta escribe en un estado de
**2×2 = 4 números** contra los 256 de E1. La regla delta no se sostiene a esa escala.

**Por qué mata el régimen** ~~cualquier veredicto de P2.2 dependería de **en qué paso se evalúa**. Es
un grado de libertad inadmisible, y no lo arregla ninguna enmienda sobre la banda.~~

> **⚠ CORREGIDO el 2026-08-06 — ver `DICTAMEN_20260806.md` §6.** El argumento tachado es atacable:
> mejor-checkpoint-por-validación es una regla **pre-declarable** que resuelve justamente esa
> objeción. Las razones que sí aguantan son otras tres: (1) con DH=2 la regla delta escribe en un
> estado de **4 números** — lo que P2.2 mediría es un colapso numérico, no la hipótesis; (2) el
> máximo sobre la trayectoria está **sesgado de forma diferencial entre condiciones** cuando una
> colapsa y la otra no, que es el caso; (3) en `d=8` el baseline nunca supera 95 %, así que la regla
> de instanciación de la carga de evaluación no aplica. B queda cerrada por dos vías independientes.
>
> Nota adicional: `softmax` **también** se degrada del pico al final en este régimen (s0 0,792@9000
> → 0,7635; s3 0,6562@12000 → 0,5796). La inestabilidad es del régimen, no de `mix22`, y la tabla de
> §3.3 usa el checkpoint del paso 16000, no el mejor — las SD de §3.3 pueden estar infladas por eso.

**Lección metodológica que hay que incorporar:** el ejecutor pasó a N fijo sin corte para escapar del
problema de §3.2 y cayó en el opuesto —entrenar hasta el colapso—. Ninguno de los dos extremos
sirve; lo correcto es **el mejor checkpoint por validación**, que es lo que E1 ya hacía con R5.

---

## 5. Dos cosas que hay que resolver sí o sí antes de cualquier enmienda

### 5.1 R11 no está definido en un régimen nuevo, y la fórmula del techo se vuelve en contra

`R11 = máx(piso 0,02 ; 1,5 × SD entre semillas del baseline)`. Con las SD de §3.3 (≈0,11) el margen
efectivo sería **0,165**, y el techo de E-005 —`1 − 10·R11`— daría **negativo**: la banda se vuelve
**vacía** en el régimen que la cumple. La fórmula congelada, aplicada consistentemente, **descalifica
sola** a los regímenes de alta varianza.

Además, la carga de evaluación se instancia como *la menor L donde el baseline cae bajo 95 %*. En
este régimen `softmax` **arranca en 0,87**: nunca está por encima de 95 %. El margen congelado **no
se puede trasladar tal cual**.

### 5.2 Sobre estirar la banda para que entre L=32 (propuesta de Maxi) — la respuesta es NO

- R11 dice, textual: *«nunca se afloja el criterio a posteriori»*.
- El techo 0,80 no es de gusto: es `1 − 10·R11`. La enmienda **rechazó** el techo 0,98 por dejar
  sólo 1,25 márgenes de aire.
- La legitimidad de E-005 se apoya en que sus extremos **preceden a todo dato de calibración** (§8
  lo certifica). Moverlos hoy destruye esa propiedad.
- **Y no hace falta**: R-BANDA ya cumple sin estirar nada (§3.3).

Nota de honestidad: lo que el ejecutor presentó como hallazgo estructural —«salir del techo cuesta
varianza»— **ya estaba documentado** en `resultados/fase0/margenes_instanciados.md` desde Fase 0
(*«la SD crece con la carga: 0,006 → 0,016 → 0,022»*). Lo nuevo es sólo la **magnitud**: en d=8 la SD
es 5 a 20 veces mayor.

---

## 6. Las tres salidas, y qué implica cada una

- **A · `d=12` (DH=3) en Colab T4, ~3 h.** El punto medido más chico que todavía no está degenerado.
  **La pregunta correcta ya no es si `softmax` entra en la banda, sino si `mix22` es ESTABLE ahí** —
  eso es lo que hay que medir primero, y es barato. Riesgo: la inestabilidad de §4 puede ser gradual
  en `d` y reaparecer.
- **B · `d=8` local.** **Cerrada por §4**, no por presupuesto. No debería reabrirse.
- **C · Declarar P2.2 no corrible y documentarlo.** Hoy tiene mucho más respaldo que a la mañana: no
  es «se acabó el crédito» (R4) sino **«no existe régimen accesible donde P2.2 sea medible»**, con
  las tres piezas que lo sostienen: el eje carga es inalcanzable (§2), el eje capacidad degenera la
  arquitectura antes de salir del techo (§3–§4), y la propia fórmula de R11 descalifica los regímenes
  de alta varianza (§5.1). Es un resultado negativo fuerte, no una rendición.

**Lo que se pide de vos, en una lista:**

1. Elegir entre A y C (B está cerrada).
2. Si es A: definir qué se mide primero (se propone: estabilidad de `mix22` en d=12, 2 semillas)
   y si el criterio de parada pasa a *mejor checkpoint por validación* de forma explícita.
3. Decidir cómo se instancia **R11** en un régimen donde el baseline nunca supera 95 % (§5.1). Sin
   esto, ningún veredicto de equivalencia es interpretable.
4. Decidir si la contradicción del texto de R4 (§1) merece nota de erratas en el expediente.
5. Decidir si el hallazgo de la meseta (§3.2) amerita una revisión de los topes de pasos usados en
   E1 y en TELAR-03 — ahí la separación capacidad/entrenamiento se cerró con oráculo closed-form, así
   que probablemente no, pero conviene dejarlo dicho antes de que lo pregunte un revisor.

---

## Anexo · dónde está cada cosa

| qué | dónde |
|---|---|
| Calibración formal (R4) | `resultados/calibracion/calibracion_rbanda.json` + `.log` |
| Sondeo del eje carga | `experimentos/E1/sondeo_extension_L.py` |
| Barrido de `d` | `experimentos/E1/sondeo_d.py`, `barrido_d.sh`, `resultados/sondeo_d/punto_d*.json` |
| Chequeo del corte / N fijo | `experimentos/E1/verificar_corte.py` (acepta `--cond`) |
| Régimen d=8 y `mix22` | `resultados/sondeo_d/regimen_d8_L48_s*`, `mix22_d8_L48_s*` |

Todo sin commitear. Ningún artefacto congelado modificado; `verificar_anclas.py --requiere E-005`
sigue verde.
