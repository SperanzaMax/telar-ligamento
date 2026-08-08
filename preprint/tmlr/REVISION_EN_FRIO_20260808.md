# Revisión en frío del manuscrito TMLR — 2026-08-08

Cada cifra del paper recalculada **desde los JSON crudos**, sin importar el código de análisis
del proyecto: sólo `json`, `glob`, `statistics` y `random` de la biblioteca estándar. El
criterio de la revisión no fue «¿está bien escrito?» sino el de TMLR: **¿algún claim dice más
de lo que los datos sostienen?**

## 1. Lo que reproduce exacto

| Claim del paper | Valor recalculado | Estado |
|---|---|---|
| delta cae a `.921` y `.820` en L96 y L128 | `0.9208` · `0.8205` | ✅ |
| PS-1 (mix22 − delta @L96) `+.0792` | `+0.0792` | ✅ |
| «positivo en 8 de 8 semillas» | 8/8 | ✅ |
| `4.0×` el margen pre-registrado de `.0200` | `3.96×` | ✅ |
| mix13 − delta @L96 `+.0792` | `+0.0792`, 8/8 | ✅ |
| peor celda de mix22 en todo el grid `.999512` | `0.999512` | ✅ exacto |
| déficit peor celda de mix13 `1.22×10⁻⁴` | `1.221×10⁻⁴` | ✅ |
| `f = 1.001` | `1.0012` | ✅ |
| inicio de la degradación en `L₀ = 64` | 64 en 8/8, mediana 64 | ✅ |
| monotonicidad estricta con la carga | decreciente en las 6 cargas | ✅ |
| decaimiento acelerado `+.0361` | `+0.0361` | ✅ |
| `193 493` parámetros | `193493` | ✅ |
| E-b costó «diecisiete minutos de una T4» | `1021.7 s` = 17.03 min, `Tesla T4` | ✅ |
| el procedimiento congelado emite su tercera rama | `decidir_escalera.py` → `rama=R3` | ✅ |

El IC del paper es `[+.0747, +.0838]`; mi bootstrap independiente (20 000 remuestreos, semilla
distinta) da `[+.0747, +.0836]`. La diferencia está en el ruido de Monte Carlo, que es lo que
el propio paper declara para su verificación independiente.

## 2. Lo que corregí

**Tabla 2, celda `L=256`.** Decía `.9999`; el valor real es `0.9999771`, que **redondea a
`1.0000`**. En la misma tabla, `L=96` (`0.9999797`, prácticamente idéntico) sí figuraba como
`1.000`. Eran dos criterios de redondeo distintos para dos celdas iguales — trunc en una,
redondeo en la otra. Uniformada la tabla a cuatro decimales con redondeo correcto, y el texto
ahora dice «worst cell `.9999` (**at L=32**)», que es donde efectivamente está el peor valor
(`0.999939`). El claim no cambia; deja de ser atacable.

## 3. Lo que documenté sin tocar (ver `desviaciones.md`, D-007)

`calibracion_rbanda_E-b.json` lleva escrito en su campo `veredicto` el texto de **R4**
(«DECISIÓN SUSPENDIDA … Leer R4 como R3 sería convertir una restricción de crédito en un
hallazgo»), que **contradice al propio archivo**: el mismo objeto registra
`biseccion_cerrada: true`. Es un remanente del estado previo a la corrida.

**El paper no está mal.** El veredicto de la escalera no vive en ese campo: lo emite
`decidir_escalera.py` con los dos registros, y hoy volvió a emitir `rama=R3`. Pero un revisor
que abra ese JSON antes que el informe va a leer un R4 y sospechar precisamente lo que ese
texto prohíbe. Queda registrado en `desviaciones.md` en vez de editar el artefacto.

## 4. Claims que miré con lupa y aguantan

- **«one softmax head in four restores the ceiling»** — mix13 está a `1.0000` de media en las
  seis cargas y su peor celda individual es `0.999878`. Sostenido.
- **«the effect is censored … `+.0792` is a lower bound»** — correcto, y es la formulación
  conservadora: mix22 nunca baja de `0.999512`, así que su capacidad no se midió.
- **«sufficiency, not necessity»** — se sigue de que mix13 iguale a mix22 (`+0.0792` ambos).
  El paper no afirma que una cabeza sea óptima, sólo que la mínima dosis probada alcanza.
- **«the delta rule falls below ceiling at loads where hybrids do not»** — es el enunciado
  correcto: lo medido es la caída de delta, no la magnitud de la ventaja híbrida.
- Un detalle que **juega a favor y el paper no explota**: la condición `softmax` tiene peor
  celda `0.999023`, *peor* que la de mix22 (`0.999512`). El híbrido no queda por debajo de la
  referencia en ningún punto del grid. Está implícito en `f = 1.001`; no hace falta agregarlo,
  pero conviene saberlo si un revisor pregunta si el híbrido paga algo.

## Veredicto

**El manuscrito puede enviarse.** No encontré ningún claim que exceda lo medido; el sesgo del
texto es sistemáticamente conservador. Las dos observaciones eran de presentación y de
trazabilidad del expediente, no de resultados, y quedaron resueltas.
