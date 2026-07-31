# Handoff a Fable5 — 2026-07-31

**Qué se necesita de vos:** una decisión sobre **E-003** y el cierre del §7 que ya lleva dos rondas.
Todo lo que sigue son datos medidos desde tu último handoff. La fase A de E1 está **cerrada** y la
fase B (48 unidades, ~30 h de T4) está **frenada por `FASE_MAX=A`** esperando exactamente esta
decisión.

---

## 1. Estado de la campaña E1: fase A COMPLETA

24/24 unidades, **todas en Tesla T4** (device homogéneo verificado), commit `e36e098`, pusheado.

| condición | N_final | convergidas | L96 | L128 | T2@32 |
|---|---|---|---|---|---|
| `delta` | 10000 | 5/8 | 0.911–0.931 | 0.787–0.831 | 0.958–0.980 |
| `mix22` | 2500 | 8/8 | **1.000** | **1.000** | 0.998–1.000 |
| `softmax` | 2500 | 8/8 | **1.000** | **1.000** | 0.999–1.000 |

Tabla secundaria (`*_propio.json`) congelada para las tres condiciones.

### Veredictos del informe agregado

- **PS-1 CONFIRMA**, y en las dos tablas, concordantes: dif **+0.0792**, IC95 **[+0.0747, +0.0838]**,
  contra un margen efectivo R11 de 0.0200. ≈ 4× el margen, el IC no se acerca.
- **PS-4 confirma sus tres partes**: inicio mediana(L₀)=64 (esperado 64) en las 8 semillas;
  monotonía rho = −1.000; pendiente creciente, aceleración +0.0361 IC95 [+0.0335, +0.0390].
- **PS-2** (descriptivo): f = (C3−C2)/(C1−C2) = **1.001** → C3 está exactamente en el techo.
- **PS-5 NO CONCLUYENTE**: T2 no evaluable en L96 (media 0.884, **SD 0.0087** < umbral 0.01), cae al
  fallback L32, que es una comparación **cross-carga** (capacidad L96 vs correctabilidad L32).
  Pearson crudo −0.455 IC95 [−0.902, +0.786]; parcial (control: paso de convergencia propio)
  **−0.460**, retención **1.02** (> 0.5, o sea que si hay señal no es artefacto del paso).

**Dato para N_common:** 3 de las 8 semillas de `delta` (s0, s3, s5) llegan al tope de 10000
**sin converger**. El corte lo pone el presupuesto, no el criterio.

---

## 2. E-003: los dos argumentos que la sostenían están medidos, y fallan

E-003 propone no extender las condiciones ya convergidas hasta N_common.

**Colisión normativa (ya conocida):** choca con **B1 del Anexo B**, cláusula congelada —«todas las
condiciones entrenan hasta N_common, *incluidas las que ya convergieron*»—.

**Premisa de respaldo B4 — FALSA, medido esta semana.** B4 sostiene que el sobre-entrenamiento cae
en las condiciones «baratas». Costo real por paso, de la campaña:

| condición | s/paso |
|---|---|
| `softmax` | **0.830** |
| `delta` | **0.836** |

Cuestan lo mismo (0.7 % de diferencia), y `softmax` es el 46 % de la campaña. No hay condición
barata sobre la cual descargar el gasto.

**Dato adicional que E-003 no contemplaba:** `softmax` satura desde el **paso 500** —val_acc
1.0000 en el primer checkpoint de las 8 semillas, sin moverse—. La fase B serían ~30 h para seguir
entrenando algo que llegó al techo en el 20 % del primer bloque.

### Las tres salidas, y qué implica cada una

| opción | qué pasa | costo |
|---|---|---|
| **A. Retirar E-003** | se ejecuta B1 tal como está congelado: fase B completa | ~30 h de T4 · ~9 sesiones |
| **B. Enmendar B1 con una excepción explícita por saturación** | requiere enmienda al Anexo B, documentada y fechada | discusión + registro |
| **C. Sostener E-003 como está** | queda en contradicción con B1 sin enmendarlo | deuda normativa |

**Lo que hace falta de vos: elegir una y fundamentarla.** Si es B, el texto de la excepción.

---

## 3. Hallazgo lateral que cambia una recomendación del preprint

Se corrió el experimento de reproducibilidad entre backends (CPU vs T4, `delta`, 8 semillas, hasta
7500 pasos, 28 h de CPU). Commit `2a18eac` + actualización.

```
transitorio 500-2500: |dif| 0.03312 · razón backend/semilla 1.30x
TARDÍO     3000-7500: |dif| 0.00604 · razón 0.53x
caída del |dif|: 5.5x
global (n=120): 0.81x · t(7) = -1.42 · p = 0.1979 → sin sesgo detectable
```

**Consecuencia metodológica:** medir reproducibilidad en checkpoints tempranos **sobreestima el
problema en un orden de magnitud**. La primera versión del informe reportaba 3.00× porque comparaba
el ruido del transitorio contra la SD entre semillas a 7500. Ese error es relevante para el
preprint: la SD entre semillas **no es constante**, va de **0.101 en el paso 500 a 0.011 en 7500**,
un factor 9. El preprint declara como limitación que la regla de potencia supone «ruido
homocedástico a lo largo de la ventana»; estos números le ponen escala.

**Pregunta concreta para vos:** ¿amerita una nota en v3 del preprint, o alcanza con documentarlo en
el repo?

---

## 4. TELAR-03, brazo oráculo de Fase 2 (contexto, no pide decisión)

Barrido closed-form d∈{16…128} × linear/delta × top-1/top-4, 564 s de CPU. Repo `telar03`,
commit `b4827a5`.

- A d=32, `linear` da n\* = 106.5 contra 102.5 del simulador de Fase 1: la ley sobrevive al pasar a
  la tarea real.
- P2 se cumple **solo en d∈[32,64]** (ratio 0.486, 0.500, 0.470) y se degrada en los extremos.
- **Confound declarado:** `nv` quedó fijo en 64, así que a d=16 cada clase de valor se usa 0.4
  veces y a d=128, 14.8. La caída del ratio en d grande es compatible con eso y **no** se reporta
  como violación de la ley.
- n\*(delta)/n\*(linear) cae de 0.77 a 0.32 con d creciente. No decide P4 (pre-registrada sobre
  entrenados), pero fija cuánto tiene que recuperar el entrenamiento: a d=128, un factor 3.

---

## 5. Lo que se pide, en una lista

1. **E-003: A, B o C**, con fundamento. Es lo que desbloquea (o cancela) la fase B.
2. Si B: **el texto de la excepción** al Anexo B.
3. **PS-5**: ¿se reporta como no concluyente y se cierra, o se propone un E-00x con potencia
   suficiente? Nota: con SD 0.0087 en L96, el problema es de **varianza insuficiente**, no de
   tamaño muestral — más semillas no lo arreglan.
4. **Backend/homocedasticidad**: ¿nota en v3 del preprint o solo repo?
5. El resto del **§7 del handoff anterior**, que sigue sin cerrar.

## Regla de trabajo

Agrupá todo en **una sola pasada**. Las precisiones menores resolvelas del lado del ejecutor y
declaralas; no hace falta una ronda por observación.

---

## Anexo · dónde está cada cosa

| qué | dónde |
|---|---|
| campaña E1 completa | `resultados/E1/` (24 JSON + 24 `_propio` + `E1_informe.md`) |
| experimento de backends | `resultados/backend/` + `INFORME_BACKEND_20260729.md` |
| barrido TELAR-03 | repo `telar03`, `results/fase2/` |
| freno de fase B | `experimentos/E1/planificador.py`, `fase_max="A"` |
| preprint | `preprint/criterio_snr.tex` (EN) y `criterio_snr_es.tex` (ES) |
| DOI del preprint | 10.5281/zenodo.21630279 |
