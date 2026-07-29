# E1 — informe (prereg de seguimiento v1.1)

**N_common = 10000** · **carga de evaluación (desde C2 convergida): L96** · **margen efectivo R11 = 0.0200**

N_final por condición (convergencia colectiva propia): delta=10000, softmax=—, mix22=2500

### Tabla PRIMARIA — todas las condiciones a N_common = 10000 (da el veredicto)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|
| delta | 1.000 | 1.000 | 1.000 | 0.985 | 0.921 | 0.820 | 0.972 | 10000 |
| softmax | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2500 |
| mix22 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 2500 |

### Tabla SECUNDARIA — cada condición en su propia convergencia (robustez)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|
| delta | 1.000 | 1.000 | 1.000 | 0.985 | 0.921 | 0.820 | 0.972 | 10000 |
| mix22 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 2500 |

## PS-1 — rescate de capacidad (C3 vs C2)

- **VEREDICTO: CONFIRMA**
- primaria (N_common): confirma · dif = +0.0792 · IC95 [+0.0747, +0.0838]
- secundaria (convergencia propia): confirma · dif = +0.0792 · IC95 [+0.0747, +0.0838]
- tablas CONCORDANTES

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
- **P1.2** (herencia de correctabilidad): T2(C3) − ½T2(C1) − ½T2(C2) = +0.0132 (≥0 ✓).
- **P1.3** (no interferencia): T2(C3) = 0.999 vs min(C1,C2) = 0.972 (sin interferencia ✓).

---
*Veredictos automáticos según el prereg de seguimiento v1.1 (SHA en `FREEZE_PREREG_SEGUIMIENTO_v1.1.md`). El informe final los revisa a mano.*