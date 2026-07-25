# E1 — informe (prereg de seguimiento v1.1) — REGENERADO LOCAL (parcial)

> Regenerado sin JAX desde `resultados/E1/*.json`. Semillas presentes por condición: {'delta': 8, 'softmax': 1, 'mix22': 0}. Las condiciones con <8 semillas están INCOMPLETAS.

> ⚠️ N HETEROGÉNEO dentro de una condición (viola N_common; PS-1/PS-5 NO válidos hasta nivelar): {'delta': [2500, 5000]}

**N_common = 5000** · **carga de evaluación (desde C2): L96** · **margen efectivo R11 = 0.0200**

N_final por condición (convergencia colectiva propia): delta=—, softmax=—, mix22=—

### Tabla PRIMARIA — todas las condiciones a N_common = 5000 (da el veredicto)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|
| delta | 1.000 | 1.000 | 0.999 | 0.976 | 0.896 | 0.783 | 0.925 | 5000 |
| softmax | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2500 |

### Tabla SECUNDARIA — cada condición en su propia convergencia (robustez)

| cond | L8 | L16 | L32 | L64 | L96 | L128 | T2@32 | N |
|---|---|---|---|---|---|---|---|---|

## PS-1 — rescate de capacidad (C3 vs C2)

- **NO COMPUTABLE**: falta la condición mix22 (C3) y/o las tablas de convergencia propia (`*_propio.json`). Es la predicción estrella del prereg; sin mix22 no hay veredicto.

## PS-4 — forma de la degradación

- **(i) inicio:** confirma · mediana(L₀) = 64 (esperado 64, umbral 0.99) · L₀ por semilla = [64, 64, 64, 64, 64, 64, 64, 64]
- **(ii) monotonía:** confirma · rho medio = -1.000
- **(iii) pendiente creciente:** confirma · aceleración = +0.0327 · IC95 [+0.0297, +0.0358]

## PS-5 — anticorrelación capacidad ↔ correctabilidad

- **VEREDICTO: NO CONCLUYENTE**
- T2 primaria: **misma_carga** (L96). T2 medida en la carga de evaluación (L96).
- Pearson crudo = +0.370 · IC95 [-0.123, +0.967] · Spearman = +0.524
- Pearson parcial (control: paso de convergencia propio) = +0.395 · retención = 1.07 (umbral 0.5)
- diagnóstico: corr(capacidad, paso) = -0.078 · corr(T2, paso) = +0.199

---
*Veredictos automáticos según el prereg de seguimiento v1.1. Informe PARCIAL regenerado localmente sin JAX; el definitivo lo emite la celda 9 del notebook con la campaña completa.*