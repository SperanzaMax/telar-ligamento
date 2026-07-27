# ADDENDUM E1 → Fable5 · 2026-07-26 (posterior al plan de 3 fases)

Ejecutor: Opus 5. Respuesta al plan "Cierre y publicación de E1". **No se ejecutó nada de GPU todavía**:
antes de gastar hay dos hallazgos que cambian el plan. Todo verificado contra código y JSON reales.
Precisiones menores resueltas del lado del ejecutor y declaradas. Lo que necesito de vos está en §6.

---

## 1. El presupuesto de Fase 1 está subestimado ~8×

El plan dice "~5h GPU". Corrí `planificador.py` sobre el estado real:

| CONDS | unidades | cómputo |
|---|---|---|
| delta, softmax, mix22 | 13 | 7h55m |
| + mix13 | 21 | 14h01m |
| + mix13, mix31 | 29 | 20h07m |

**Y eso es solo hasta cerrar fase A.** El planificador corta ahí a propósito (`planificador.py:89`:
*"todavía en fase A: N_common no se puede fijar"*).

La fase B es la cara y no está en esa tabla. Si delta cierra por tope a 10 000 — va camino a eso —
entonces `N_common = 10 000` y el Anexo B obliga a extender **toda condición del run** hasta ahí.
mix22 converge en 1000 pasos pero tendría que entrenarse hasta 10 000: 3 bloques × 8 semillas =
**24 unidades ≈ 16 h**. Ídem softmax, mix13, mix31.

**Total realista: ~40 h de T4 con tres condiciones · ~80 h con las cinco.**

Costo medido (auto-calibrado del propio planificador): delta 0.8365 s/paso · softmax 0.82 · mix22 0.8913.
Una unidad = 2500 pasos ≈ 35-39 min en T4.

---

## 2. RAÍZ DEL PROBLEMA: el criterio de convergencia mide ruido, no convergencia

Esto es lo importante del addendum. **No es que el experimento sea caro: es un criterio defectuoso
multiplicado por la regla de extensión.**

`converged()` (`src/entrenar.py:152`, criterio D-004):

```python
def converged(val_hist, target_steps, window=500, tol=0.005):
    return (at[target_steps] - at[target_steps - window]) < tol
```

Tolerancia **0.5 puntos**, comparando **dos mediciones puntuales** separadas por 500 pasos.

Ruido real de `val_acc` en delta (últimas 4 evaluaciones de cada semilla, todas @7500):

| seed | últimas 4 | rango | converged |
|---|---|---|---|
| s0 | 0.8728 · 0.8669 · 0.8817 · 0.8643 | 0.0173 | **True** |
| s1 | 0.8701 · 0.8645 · 0.8696 · 0.8772 | 0.0127 | False |
| s2 | 0.8667 · 0.8521 · 0.8752 · 0.8785 | 0.0264 | **True** |
| s3 | 0.8483 · 0.8544 · 0.8558 · 0.8683 | 0.0201 | False |
| s4 | 0.8448 · 0.8513 · 0.8584 · 0.8561 | 0.0135 | **True** |
| s5 | 0.8446 · 0.8539 · 0.8562 · 0.8616 | 0.0170 | False |
| s6 | 0.8520 · 0.8594 · 0.8513 · 0.8448 | 0.0146 | **True** |
| s7 | 0.8450 · 0.8570 · 0.8618 · 0.8688 | 0.0238 | False |

**El ruido (1.3–2.6 pts) es 3–5× la tolerancia (0.5 pts).**

Evidencia de que es azar y no señal:
- **4/8 True y 4/8 False**, en semillas estadísticamente indistinguibles (todas planas en 0.84–0.88).
- s0 dio True porque su última medición **bajó**; s1 dio False porque la suya **subió**. Mismo régimen.
- `delta_seed0` extendido a **10 000**: sigue `converged=False`, con `val_hist` plano entre 0.85 y 0.88
  **desde el paso 3500**. Estuvo ~6500 pasos sin aprender nada medible y el criterio no lo detectó.

### La cadena de costo

criterio ruidoso → delta nunca logra 8/8 `True` simultáneo → va siempre al tope 10 000 →
`N_common = 10 000` → el doble reporte extiende **todas** las condiciones hasta ahí →
mix22 entrena 10× más de lo que necesita, sin aportar información.

**Ese es ~80% del presupuesto.**

---

## 3. Propuesta: dos enmiendas → el expediente entra en 3 sesiones

**E-002 — criterio de convergencia robusto al ruido.** Comparar **medias de ventanas** (p. ej. promedio
de las últimas 3 evaluaciones contra las 3 previas, o pendiente de regresión sobre las últimas k) en vez
de dos puntos sueltos. Con eso delta cierra donde ya está plano, ~3500–5000 pasos.

**E-003 — no extender condiciones saturadas.** Si una condición cierra fase A con acc@1 ≥ 0.999 en todas
las cargas y todas las semillas, extenderla a `N_common` no aporta nada discriminante.
**Precedente propio: D2 ya declaró softmax "no evaluable por saturación".** E-003 es su extensión natural.

Plan resultante:

| paso | unidades | tiempo |
|---|---|---|
| delta — ya en 7500, cierra fase A por E-002 | **0** | — |
| softmax — completar 5 semillas a 2500 | 5 | ~3h20 |
| mix22 — ya está; saturado, no se extiende (E-003) | **0** | — |
| **subtotal: expediente PS-1 / PS-4 / PS-5 / P1.x** | **5** | **~1 sesión** |
| mix13 en **run separado** (exploratoria, no comparte N_common) | 8 | ~5h30 |
| **TOTAL** | **13** | **~3 sesiones** |

mix31 queda para una cuarta sesión si el presupuesto lo permite (coincide con tu prioridad
"mix13 antes que mix31").

**Nota sobre runs separados:** mix13/mix31 no dan veredictos pre-registrados. Si van en el mismo run que
delta, quedan atadas a su `N_common` y se las obliga a entrenar saturadas. En run aparte convergen en
~1000–2500 pasos. Ahorro ≈ 30 h sin tocar ningún veredicto. Es el mismo razonamiento que vos ya aplicaste
en `PLAN_PS1.md` al sacar softmax del camino crítico.

---

## 4. ADVERTENCIA DE INTEGRIDAD (el ejecutor la marca explícitamente)

Enmendar un prereg congelado **después de ver los datos** es exactamente la maniobra que el
pre-registro existe para impedir. Que acá sea legítimo depende de dos condiciones que hay que
sostener **por escrito en el paper**, no solo en el repo:

1. **E-002 se justifica con evidencia instrumental independiente de los resultados.** Que el ruido
   supere la tolerancia se demuestra con el `val_hist` solo, sin mirar ninguna acc@1 ni ningún
   veredicto. No cambia qué se predice ni cómo se testea: cambia **cuándo se deja de gastar GPU**.
2. **Se reportan ambas versiones.** Veredicto con criterio original y con criterio corregido. Si
   coinciden, la enmienda queda demostrada como inocua. Si difieren, **eso mismo es un resultado**
   y hay que publicarlo como tal.

Sin esa doble presentación, un revisor va a leer "cambiaron el criterio de parada después de ver los
datos" y va a tener razón en desconfiar. El activo principal de Ligamento es la disciplina metodológica;
esta es justo la decisión donde se puede perder.

E-003 es menos delicada (no toca ningún test, solo evita cómputo sin información discriminante) pero
conviene documentarla igual, encadenada a D2.

---

## 5. Ítem 10 iniciado: aparecieron precedentes de LOS DOS claims propios

Búsqueda **no sistemática** (4 queries en total). Aun así, en la primera pasada ya salieron precedentes
cercanos de las dos cosas que ibas a proponer como propias. Eso es señal, no ruido.

### PS-5 (anticorrelación capacidad ↔ correctabilidad)

La tensión es la **motivación explícita** de trabajo reciente:

- **Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention** —
  arXiv **2606.26560** (jun 2026). Agrega un paso de borrado con dirección independiente antes de la
  escritura correctiva.
- **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention** — arXiv **2605.22791** (NVIDIA, may 2026).
- Textual de la literatura: *"un gate de escritura escalar debe transar entre direcciones: subirlo ayuda a
  sobrescribir memoria vieja, pero puede sobre-corregir otra dirección que ya está bien calibrada"*.
  Eso **es** PS-5, enunciado como problema conocido.

**Lo que NO vi reportado:** la **medición correlacional entre semillas** con control parcial
(Pearson crudo −0.678, IC95 [−0.933, −0.187]; parcial −0.645, retención 0.95). El fenómeno tiene
precedente; la cuantificación por semillas, no la encontré.

### Firma del margen

- **Sharp Capacity Thresholds in Linear Associative Memory: From Winner-Take-All to Listwise Retrieval** —
  arXiv **2605.05189** (may 2026).
- **Attractor Geometry of Transformer Memory** — arXiv **2605.05686**: margen geométrico vs saturación de logits.
- **Understanding Transformer from the Perspective of Associative Memory** — arXiv **2505.19488**: introduce
  *retrieval SNR* como medida.
- **Understanding Factual Recall in Transformers via Associative Memories** — arXiv **2412.06538**.

**Lectura:** el margen como sonda de capacidad remanente **no es nuevo**. Lo que podría serlo es su uso
para mostrar que una híbrida se comporta como softmax (margen plano) y no como intermedio.

### Consecuencia según tu propio criterio

Escribiste: *"Si aparece precedente de PS-5, se cita y se reencuadra como replicación; eso decide el
título."* → **Ese disparador ya se activó.** Y con 4 queries; la sistemática probablemente encuentre más.

---

## 6. LO QUE NECESITO DE VOS

1. **¿Aprobás E-002 y E-003?** Sin ellas, Fase 1 son ~40 h de T4 (o ~80 h con las cinco condiciones), no 5 h.
   Con ellas, 3 sesiones. Si las aprobás, ¿te parece bien el doble reporte de veredictos del §4.2 como
   salvaguarda?
2. **¿O preferís pagar las ~40 h y no tocar el prereg?** Es defendible: máxima pureza metodológica al
   costo de ~12 sesiones de Colab free, con caídas de sesión de por medio (hoy hubo 2, ~50 min perdidos).
3. **Reencuadre del título/abstract**: con precedente de PS-5 confirmado, ¿el paper pasa a ser
   "replicación pre-registrada + cuantificación" en vez de "hallazgo"? El ejecutor cree que sí y que
   conviene decirlo desde el título.
4. **¿Sigue en pie mix13 como prioridad alta** dado que ahora sabemos que cuesta 8 unidades en run
   separado (~5h30)? El ejecutor cree que sí: es barata así y es la única pregunta de suficiencia mínima.
5. **El §3 del handoff anterior** (margen plano de mix22/softmax vs desplome de delta −72.6%) sigue sin
   respuesta tuya y condiciona el encuadre.

---

## 7. Estado operativo (sin cambios desde el handoff)

- mix22 8/8 completo y **pusheado** (`5423270`).
- delta 8/8 @7500 (s0 tiene una corrida a 10 000 sin sincronizar al repo, a propósito: rompería la homogeneidad).
- softmax 3/8. `*_propio.json`: ninguno.
- Nada de Fase 1 ejecutado. **Esperando tu decisión sobre §6.1 / §6.2 antes de gastar GPU.**
