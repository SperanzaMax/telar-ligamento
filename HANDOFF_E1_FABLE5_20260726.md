# HANDOFF E1 → Fable5 · 2026-07-26

Ejecutor: Opus 5. Todo lo de abajo está **verificado contra los JSON reales**, no estimado.
Las precisiones menores ya están resueltas del lado del ejecutor y declaradas como tales.
Una sola pasada: lo que necesito de vos está en §7.

---

## 1. Estado de la campaña

| condición | semillas | pasos | estado |
|---|---|---|---|
| **mix22 (C3)** | **8/8** | 2500 | **COMPLETO**, las 8 `converged=True`, `paso_conv_propio=1000` |
| delta (C2) | 8/8 | 7500 (s0 en 10000) | heterogéneo; 4/8 `converged=True` |
| softmax (C1) | 3/8 | 2500 | 3/3 `converged=True` |

`*_propio.json`: **NINGUNO**. Fase A sin cerrar.

Commits pusheados a `github.com/SperanzaMax/telar-ligamento` (`22074c0 → 5423270`).

---

## 2. Resultado principal — mix22 8/8

acc@1 por semilla (redondeo a 4 decimales; los valores "1.0000" con asterisco son 0.9999x):

| seed | L32 | L64 | L96 | L128 | T2@32 |
|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9993 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9983 |
| 2 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 0.9993 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9993 |
| 4 | 1.0000 | 0.9999 | 1.0000* | 0.9999 | 0.9988 |
| 5 | 1.0000 | 0.9999 | 1.0000* | 0.9998 | 0.9978 |
| 6 | 1.0000 | 1.0000 | 0.99996 | 1.0000 | 0.9993 |
| 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9995 |

Contraste (medias):

| cond | L64 | L96 | L128 | T2@32 |
|---|---|---|---|---|
| mix22 (8) | 1.000 | 1.000 | 1.000 | 0.999 |
| softmax (3) | 1.000 | 1.000 | 1.000 | 1.000 |
| **delta (8) @7500** | **0.983** | **0.916** | **0.814** | **0.960** |

**Rescate C3−C2**: +0.084 (L96) · +0.186 (L128). Margen efectivo **R11 = 0.02**. SD entre semillas de mix22 ≈ 2e-5.

**Aviso**: delta todavía sube. `delta_seed0` extendido a 10000 pasos da L96 **0.9306** (era 0.9167 @7500) y L128 0.8297, con `converged=False`. Si las otras 7 acompañan, el rescate baja a **~+0.070** en L96. Sigue 3.5× sobre R11, pero el número final del informe preliminar (+0.084) **va a ser menor**.

---

## 3. NUEVO — sondeo de margen (exploratorio, no pre-registrado)

**Motivo**: mix22 y softmax saturan los dos en acc@1 = 1.000. Un techo saturado no discrimina: no dice dónde está el límite de C3. Métrica alternativa: `margen = logit(valor correcto) − max(logit de los otros valores)`, promediado sobre columnas de query.

Ejecutado sobre los checkpoints reales (CPU, JAX 0.10.2, batch 32, reps 4).

| cond | n | L32 | L64 | L96 | L128 | Δ L32→L128 |
|---|---|---|---|---|---|---|
| mix22 | 8 | 10.94 ± 0.59 | 10.92 ± 0.65 | 10.95 ± 0.66 | **11.00 ± 0.66** | **+0.5%** |
| softmax | 3 | 11.58 ± 0.57 | 11.61 ± 0.40 | 11.64 ± 0.31 | **11.66 ± 0.30** | **+0.7%** |
| delta | 8 | 12.50 ± 0.22 | 8.64 ± 0.17 | 5.53 ± 0.18 | **3.42 ± 0.16** | **−72.6%** |

Fracción de queries con margen ≤ 0 (errores): mix22 0.01% · softmax 0.00% · delta 0.02% → 1.61% → 8.37% → **18.75%**.

### Tres lecturas

1. **El techo de mix22 está lejos de L128, no apenas alcanzado.** El margen es *plano* (incluso sube levemente) en todo el rango. No hay ninguna señal de acercamiento al límite. Subir a L192 o L256 probablemente tampoco lo rompería.
2. **delta tiene el margen MÁS ALTO de las tres a L32** (12.50 vs 11.58 softmax vs 10.94 mix22). No es un modelo peor: es el mejor a carga baja y el peor a carga alta. Su degradación es *específica de capacidad*, no una deficiencia general. Esto es más fino que "delta pierde".
3. mix22 se comporta como softmax (plano), no como un intermedio entre softmax y delta. En esta métrica el rescate es **completo**, no parcial.

---

## 4. LÍMITE ESTRUCTURAL descubierto hoy

`gen_mqar` (src/datos.py:31) exige **L claves distintas por fila**, muestreadas de un pool `NK = 128`:

```python
assert L <= NK, f"L={L} excede el pool de claves NK={NK} (E-001)"
```

**L = 128 es el máximo estructural de la tarea**, no una elección de conveniencia. Está bajo la enmienda **E-001 (2026-07-22)**.

Consecuencia para el experimento "romper el techo": subir la carga exige ampliar `NK`, lo que corre `V0` (128), cambia `VOCAB` (197) y por lo tanto la matriz de embedding y la capa de salida → **los checkpoints actuales quedan incompatibles, hay que re-entrenar todo desde cero**, con secuencias 2-4× más largas y atención cuadrática. Estimación: decenas de horas de T4, no las ~4 h que el ejecutor había estimado antes de leer el generador (estimación corregida y declarada). Además requiere **enmienda nueva al prereg congelado**.

---

## 5. Veredictos al día

- **PS-1**: **NO COMPUTABLE**. Verificado 3 veces (con 3, 6 y 8 semillas de mix22). La causa es la **ausencia de `*_propio.json`** (tabla SECUNDARIA de la regla de discordancia, Anexo B3), **no** la cantidad de semillas. Lo destraba el cierre de fase A, nada más.
- **PS-4**: **CONFIRMA** robusto. Inicio L₀ = 64 (8/8), monotonía ρ = −1.000, aceleración +0.0349 (IC95 [+0.0318, +0.0379]).
- **PS-5**: **CONFIRMA**. Pearson crudo −0.678 (IC95 [−0.933, −0.187]), Spearman −0.762, parcial −0.645 con retención 0.95. T2 primaria = misma_carga (L96).
- **PS-2**: no evaluable por saturación de C1 (D2).
- **P1.2**: +0.0193 (≥0 ✓). **P1.3**: T2(C3) 0.999 vs min(C1,C2) 0.960 (sin interferencia ✓).
- Lateral no pre-registrado: **mix22 converge en ~1000 pasos; delta necesita 7500+ y varias no convergen ni ahí.** ~7× más rápido.
- Lateral no pre-registrado: el `val_hist` de delta s0 está **plano (0.85–0.88) del paso 3500 al 10000**, pero la capacidad medida a L96 igual subió (0.9167 → 0.9306). La métrica de validación no capta la ganancia de capacidad.

---

## 6. Contexto de literatura (búsqueda del ejecutor, 2 queries, no sistemática)

El resultado "las cabezas híbridas rescatan el recall que la atención lineal pura no alcanza" **ya está establecido**, y medido sobre MQAR:

- **Zoology: Measuring and Improving Recall in Efficient Language Models** — arXiv **2312.04927** (2023). Introduce MQAR, muestra que el cuello de botella es la dimensión del estado.
- **Simple linear attention LMs balance the recall-throughput tradeoff** (Based) — arXiv **2402.18668** (2024). Linear attention + sliding-window softmax para recuperar recall. Híbridos cierran ~97.4% de la brecha.
- **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention** — arXiv **2605.22791** (NVIDIA, mayo 2026). Ataca la misma tensión que nuestro eje capacidad↔correctabilidad.

Lectura del ejecutor: **PS-1 tal como está redactado es un redescubrimiento.** El diferencial real de Ligamento es (a) el prereg congelado con DOI — metodología por encima de la norma del campo; (b) el eje correctabilidad como dimensión separada; (c) posiblemente el §3 de este documento, que es lo único que no vi en la literatura consultada.

---

## 7. LO QUE NECESITO DE VOS (Fable5)

1. **¿El §3 (margen plano vs desplome) cambia el encuadre del paper?** El ejecutor cree que "el rescate es completo, no parcial, y el techo de C3 está lejos" es más fuerte y más publicable que "C3 > C2 en acc", que es lo ya sabido. ¿Lo ves así o es sobre-lectura de una métrica no pre-registrada?
2. **¿Vale las decenas de horas de T4 romper el techo** (ampliar NK, re-entrenar todo, enmendar el prereg)? ¿O el margen plano ya es evidencia suficiente de que el límite está lejos y el experimento caro es redundante?
3. **Orden de lo que queda**: delta s1–s7 → 10000 (~4h30) y cierre de fase A destraban PS-1 formal, que es el resultado **menos** novedoso. ¿Se cierra igual por integridad del prereg (postura del ejecutor: sí) o se reordena?
4. **El punto 2 del §3** (delta con el margen más alto a L32) sugiere reformular la degradación como *específica de capacidad* y no como inferioridad general. ¿Merece predicción propia en E2, o alcanza como observación?
5. **Riesgo de scooping**: el campo está muy activo (§6). ¿Corresponde apurar un preprint con lo que hay, o esperar al experimento limpio?

---

## 8. Pendientes operativos

- delta s1–s7 → 10000 (7 unidades ≈ 4h30 T4). Se perdió el parcial de s1 (step 8500) por caída de sesión.
- Cierre de fase A → `*_propio.json`. **Es lo único que destraba PS-1.**
- Sincronizar delta al repo cuando quede homogéneo (hoy se omitió a propósito: s0@10000 vs resto@7500 rompería la homogeneidad del informe).
- 2 caídas de sesión de Colab hoy (~50 min de GPU perdidos). Al cambiar de cuenta: montar la carpeta canónica por **acceso directo**, nunca por copia (hoy generó una bifurcación que costó ~1 h de diagnóstico).

**Reproducir el §3**: `sondeo_margen.py` en el scratchpad de la sesión; corre en CPU, ~2 min las 19 semillas-condición.
