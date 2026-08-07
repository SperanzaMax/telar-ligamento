# Cierre de la calibración R-BANDA — 2026-08-06 · rama **R3**

La escalera de `ENMIENDA_E-005_RBANDA_20260805.md` §3.1 está **completa**. Con los dos escalones
corridos, `decidir()` emite **R3 — frontera inalcanzable**. No es una lectura: es la salida del
mismo código del runner, alimentado con los dos registros.

```
R3 — FRONTERA INALCANZABLE: ningún escalón cumple la banda y la bisección cerró en ambos.
E2-E4 se re-alcanzan a contrastes con al menos un brazo fuera del techo (los 5 verdes y P2.1
con cota inferior declarada) y P2.2 QUEDA SIN CORRER — se registra como NO CORRIDA, nunca
como confirmada. Cualquier claim de frontera pasa a una campaña futura con d menor, como
rediseño documentado.
```

Reproducible con:

```
python experimentos/E1/decidir_escalera.py \
    resultados/calibracion/calibracion_rbanda.json \
    resultados/calibracion/calibracion_rbanda_E-b.json
```

---

## 1. La escalera, medida

| escalón | NK | L_max | cargas medidas | peor celda | en banda | bisección | tiempo | hardware |
|---|---|---|---|---|---|---|---|---|
| **E-a** | 256 | 128 | 8…128 | 1,0000 | — | **cerrada** | 87,0 min | CPU |
| **E-b** | 512 | 256 | 8…256 | **0,9999** | — | **cerrada** | **17,0 min** | Tesla T4 |

`acc@1` de E-b, las ocho cargas:

| L | 8 | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|
| acc@1 | 1,0000 | 1,0000 | 0,9999 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,9999 |

Ambas semillas cortaron por convergencia a **1000 pasos** de un tope de 2500, con `val_acc = 1,0000`.

**Hardware heterogéneo entre escalones, y es legítimo:** §4 declara *«hardware indistinto»* para la
calibración —no exige T4 ni homogeneidad— porque **no emite veredictos**. La homogeneidad sigue
siendo obligatoria **dentro de** E2–E4, que es donde mezclarla metería el hardware adentro de la
comparación. Queda declarado en el registro de cada escalón.

## 2. Por qué esto no es la salida C del handoff, aunque termine en el mismo lugar

El handoff del 2026-08-06 ofrecía «declarar P2.2 no corrible» como una decisión a tomar, y dos de las
tres devoluciones la eligieron. `DICTAMEN_20260806.md` mostró que esa decisión no estaba disponible:
R3 exige *«bisección cerrada en **ambos** escalones»*, E-b nunca había corrido, y R4 —lo que el
runner devolvía— prohíbe explícitamente esa lectura (*«leer R4 como R3 sería convertir una
restricción de crédito en un hallazgo»*).

Correr E-b costó **17 minutos de T4 (~0,35 unidades de cómputo)** y convirtió un R4 ambiguo en un R3
medido con la escalera completa. La diferencia no es de forma: es la que separa un resultado negativo
fuerte de uno que un revisor abre preguntando «¿por qué no corrieron el segundo escalón?».

**El presupuesto congelado alcanzó de sobra.** §4 asignaba hasta 3 h a E-b; usó 17 min. No hizo falta
enmienda, ni presupuesto adicional, ni tocar una sola constante.

## 3. Qué se confirma y qué se corrige del expediente

- **§2 del handoff (retirado) tenía la conclusión correcta por el argumento equivocado.** Predecía
  que E-b no alcanzaría la banda, pero lo apoyaba en un sondeo que movía `L` sin mover `NK`, con
  celdas que violaban `L_max ≤ NK/2` y una asimetría mal signada. Ahora el resultado está **medido en
  el régimen real** (NK=512, VOCAB=581, entrenado de cero), no extrapolado. La corrección del
  dictamen se mantiene: el argumento no servía, aunque la conclusión coincidiera.
- **El diagnóstico del corte no disparó ninguna bandera.** Las dos semillas cortaron **en el techo**
  (1,0000), donde el patrón meseta-y-despegue de D-006(c) es imposible: no hay a dónde despegar. E-b
  queda en la misma situación que E-a y que `softmax` en E1 — inmune por saturación, verificado, no
  supuesto.
- **El hallazgo de fondo se refuerza.** El modelo satura con 512 claves y 256 pares en juego. La
  frontera de la hibridación no está donde el presupuesto pueda alcanzarla a este `d_model`: es la
  conclusión que R3 ya tenía escrita, ahora con los dos escalones que la sostienen.

## 4. Consecuencias mecánicas (las escribe R3, no el ejecutor)

1. **P2.2 se registra como NO CORRIDA.** Nunca como confirmada. La cláusula de §5 es explícita:
   correrla en un régimen saturado produce un renglón que no puede fallar.
2. **E2–E4 corren igual**, re-alcanzados a contrastes con al menos un brazo fuera del techo: los 5
   verdes y **P2.1 con cota inferior declarada**.
3. **Bajar `d` pasa a campaña futura**, como rediseño documentado y no como parche —achicar `d` rompe
   el «misma arquitectura» que hoy ancla la comparabilidad entre campañas—. Esto **cierra la salida A**
   del handoff por la vía del protocolo, no por el resultado de `d=8`.
4. Siguen vigentes los dos resguardos de §6, cualquiera sea la rama: **margen logit** como métrica
   secundaria y **celdas ancla** del régimen viejo (NK=128) en cada campaña.

## 5. Lo que queda abierto

- **R11 en un régimen nuevo** ya no bloquea nada: sin cambio de régimen, los márgenes de Fase 0 siguen
  valiendo. El gate de admisibilidad sobre la SD que propone `DICTAMEN_20260806.md` (punto 3) queda
  como pre-registro para cualquier campaña futura con `d` menor.
- **El texto del veredicto R4 sigue sin cubrir el caso de escalón faltante con bisección cerrada.**
  Registrado en `desviaciones.md` D-006(a); el artefacto congelado no se toca.
- **Decidir si E2 corre sobre `softmax` o `mix13`** — pendiente desde el cierre de E-004, no lo
  resuelve la calibración.

---

**Anclas verdes.** `verificar_anclas.py --requiere E-005` pasó antes de la corrida (en la VM, sobre
el repo recién clonado) y después. Ninguna constante congelada fue modificada en ningún momento.
