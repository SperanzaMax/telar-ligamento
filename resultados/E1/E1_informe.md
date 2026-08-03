# E1 — informe (prereg de seguimiento v1.1) — REGENERADO LOCAL (parcial)

> Regenerado sin JAX desde `resultados/E1/*.json`. Semillas presentes por condición: {'delta': 8, 'softmax': 8, 'mix22': 8}. Las condiciones con <8 semillas están INCOMPLETAS.

> ⚠️ **N HETEROGÉNEO dentro de una condición** (viola N_common; PS-1/PS-5 NO válidos hasta nivelar): {'mix22': [2500, 7500, 10000]}

**N_common = 10000** · **carga de evaluación (desde C2): L96** · **margen efectivo R11 = 0.0200**

N_final por condición (convergencia colectiva propia): delta=10000, softmax=2500, mix22=2500

### Tabla PRIMARIA — N_common = 10000 (da el veredicto) · † NO extendidas, a su N real: softmax, mix22

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|
| delta | 1.000 | 1.000 | 1.000 | 0.985 | 0.921 | 0.820 | 0.972 | 10000 |
| softmax | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2500 |
| mix22 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2500–10000 ⚠ |

### Tabla SECUNDARIA — cada condición en su propia convergencia (robustez)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|
| delta | 1.000 | 1.000 | 1.000 | 0.985 | 0.921 | 0.820 | 0.972 | 10000 |
| softmax | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2500 |
| mix22 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 2500 |

## PS-1 — rescate de capacidad (C3 vs C2)

- **VEREDICTO: CONFIRMA**
- primaria (N_common): confirma · dif = +0.0792 · IC95 [+0.0747, +0.0838]
- secundaria (convergencia propia): confirma · dif = +0.0792 · IC95 [+0.0747, +0.0838]
- tablas **comparables sólo en parte**: 2/8 semillas de C3 siguen con el mismo checkpoint en las dos tablas (extensión en curso) → **B3 todavía no es un chequeo de robustez completo**

## B1-ter — degradación de C3 al extender (enmienda E-003′, congelada antes de correr)

- **NO APLICA**: extensión EN CURSO (5/8 semillas a N_common=10000; N mínimo = 2500); el criterio se declaró sobre la extensión completa.

## PS-2 — posición de C3 entre piso y techo (descriptiva)

- f = (C3−C2)/(C1−C2) = **1.001** (C1=1.000, C2=0.921, C3=1.000)

## PS-4 — forma de la degradación

- **(i) inicio:** confirma · mediana(L₀) = 64 (esperado 64, umbral 0.99) · L₀ por semilla = [64, 64, 64, 64, 64, 64, 64, 64]
- **(ii) monotonía:** confirma · rho medio = -1.000
- **(iii) pendiente creciente:** confirma · aceleración = +0.0361 · IC95 [+0.0335, +0.0390]

## PS-5 — anticorrelación capacidad ↔ correctabilidad

- **VEREDICTO: NO CONCLUYENTE**
- T2 primaria: **L32** (L32). T2 NO evaluable a L96 (media=0.864, SD=0.0087; requiere media>0.2 y SD≥0.01). Fallback a L32 — condición CROSS-CARGA: capacidad a L96 vs. correctabilidad a L32.
- Pearson crudo = -0.453 · IC95 [-0.902, +0.786] · Spearman = -0.299
- Pearson parcial (control: paso de convergencia propio) = -0.460 · retención = 1.02 (umbral 0.5)
- diagnóstico: corr(capacidad, paso) = +0.111 · corr(T2, paso) = -0.490

## Protocolo madre

- **P1.1** (C3≈C1 capacidad): softmax en techo → «no evaluable por saturación» (D2).
- **P1.2** (herencia de correctabilidad): T2(C3) − ½T2(C1) − ½T2(C2) = +0.0139 (≥0 ✓).
- **P1.3** (no interferencia): T2(C3) = 1.000 vs min(C1,C2) = 0.972 (sin interferencia ✓).

---
*Veredictos automáticos según el prereg de seguimiento v1.1. Informe PARCIAL regenerado localmente sin JAX; el definitivo lo emite la celda 9 del notebook con la campaña completa.*