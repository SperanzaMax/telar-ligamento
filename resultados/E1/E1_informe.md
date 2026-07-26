# E1 — informe (prereg de seguimiento v1.1) — REGENERADO LOCAL (parcial)

> Regenerado sin JAX desde `resultados/E1/*.json`. Semillas presentes por condición: {'delta': 8, 'softmax': 3, 'mix22': 0}. Las condiciones con <8 semillas están INCOMPLETAS.

> N homogéneo dentro de cada condición.

**N_common = 7500** · **carga de evaluación (desde C2): L96** · **margen efectivo R11 = 0.0200**

N_final por condición (convergencia colectiva propia): delta=—, softmax=—, mix22=—

### Tabla PRIMARIA — todas las condiciones a N_common = 7500 (da el veredicto)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|
| delta | 1.000 | 1.000 | 1.000 | 0.983 | 0.916 | 0.814 | 0.960 | 7500 |
| softmax | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2500 |

### Tabla SECUNDARIA — cada condición en su propia convergencia (robustez)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|

## PS-1 — rescate de capacidad (C3 vs C2)

- **NO COMPUTABLE**: falta la condición mix22 (C3) y/o las tablas de convergencia propia (`*_propio.json`). Es la predicción estrella del prereg; sin mix22 no hay veredicto.

## PS-4 — forma de la degradación

- **(i) inicio:** confirma · mediana(L₀) = 64 (esperado 64, umbral 0.99) · L₀ por semilla = [64, 64, 64, 64, 64, 64, 64, 64]
- **(ii) monotonía:** confirma · rho medio = -1.000
- **(iii) pendiente creciente:** confirma · aceleración = +0.0349 · IC95 [+0.0318, +0.0379]

## PS-5 — anticorrelación capacidad ↔ correctabilidad

- **VEREDICTO: CONFIRMA**
- T2 primaria: **misma_carga** (L96). T2 medida en la carga de evaluación (L96).
- Pearson crudo = -0.678 · IC95 [-0.933, -0.187] · Spearman = -0.762
- Pearson parcial (control: paso de convergencia propio) = -0.645 · retención = 0.95 (umbral 0.5)
- diagnóstico: corr(capacidad, paso) = +0.297 · corr(T2, paso) = -0.323

---
*Veredictos automáticos según el prereg de seguimiento v1.1. Informe PARCIAL regenerado localmente sin JAX; el definitivo lo emite la celda 9 del notebook con la campaña completa.*