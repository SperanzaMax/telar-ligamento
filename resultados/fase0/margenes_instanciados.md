# S0.9 — márgenes instanciados y cargas de evaluación (R11) · SNAPSHOT @ 2500

> **ASTERISCO (D-003/D-004):** delta NO convergió a 2500 pasos (train aún subía). Estos números son un
> **snapshot documentado**, no el resultado principal. La **convergencia y los márgenes definitivos se
> instancian en E1** (que entrena C2/delta con early stopping, R5). softmax sí converge a 2500.

**Hardware:** Tesla T4 (GPU) · **Semillas:** softmax = 4, delta = 4 · S0.7 determinismo = 0.00 · matmul highest.

## softmax (C1) — acc@1 por carga (media ± SD, 4 semillas)
| L | media | SD | margen efectivo |
|---|---|---|---|
| 8 | 1.0000 | 0.0000 | 0.0200 |
| 16 | 1.0000 | 0.0000 | 0.0200 |
| 32 | 1.0000 | 0.0000 | 0.0200 |
| 64 | 1.0000 | 0.0000 | 0.0200 |
| 96 | 1.0000 | 0.0000 | 0.0200 |
| 128 | 1.0000 | 0.0000 | 0.0200 |

**Carga de evaluación (softmax):** NINGUNA — no cae bajo 95% en [8..128] → **P1.1 «no evaluable por
saturación del baseline» (D2)**. Correctabilidad T2 = 1.000. Confirma la reconciliación: softmax no satura.

## delta (C2) — acc@1 por carga (media ± SD, 4 semillas @ 2500)
| L | media | SD | margen efectivo | acc@4 | acc@16 |
|---|---|---|---|---|---|
| 8 | 1.0000 | 0.0000 | 0.0200 | 1.0000 | 1.0000 |
| 16 | 1.0000 | 0.0000 | 0.0200 | 1.0000 | 1.0000 |
| 32 | 0.9990 | 0.0004 | 0.0200 | 1.0000 | 1.0000 |
| 64 | 0.9737 | 0.0055 | 0.0200 | 0.9972 | 0.9998 |
| 96 | 0.8874 | 0.0159 | 0.0238 | 0.9753 | 0.9978 |
| 128 | 0.7731 | 0.0219 | 0.0328 | 0.9265 | 0.9889 |

**Correctabilidad T2 (L32):** media = 0.8901, SD = 0.0415.
**Carga de evaluación (delta, PRELIMINAR):** L=96 (menor L con media acc@1 < 0.95; L64 aún pasa con 0.974).
Se re-instancia en E1 con delta convergido — podría moverse a L128 si la convergencia sube L96 sobre 0.95.

## Hallazgos consolidados (4 semillas)
- **Plateau real con barras de error.** delta cae 0.974 → 0.887 → 0.773 (L64→96→128); la SD crece con la
  carga (0.006 → 0.016 → 0.022): más presión de capacidad, más dispersión. Márgenes R11 despegan del piso
  en L96/L128.
- **Top-1 vs listwise robusto.** A L128: acc@1 = 0.773 pero acc@16 = 0.989 (gap 0.216). La información está
  en el estado; satura la lectura top-1. Replica P1 de TELAR-03 y la distinción de los papers.
- **Capacidad ↔ correctabilidad (matiz de 4 semillas).** seed2/3 mejor capacidad (L128 ~0.79) / peor T2
  (~0.85); seed0/1 al revés (0.755 / ~0.92). Posible trade-off o ruido del currículum — a confirmar con
  convergencia y más semillas en E1.
- **Contraste central:** softmax = techo perfecto en ambos ejes; delta = capacidad que cae + correctabilidad
  preservada. «El gradiente compra correctabilidad, no capacidad», con datos.

*Snapshot generado 2026-07-22 (4+4 semillas @ 2500, Tesla T4). Resultado principal (convergido) → E1.*
