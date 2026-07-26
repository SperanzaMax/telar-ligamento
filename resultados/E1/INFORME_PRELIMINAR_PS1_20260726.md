# E1 — Informe preliminar PS-1 (2026-07-26)

> **Estado:** campaña en curso en Colab (Tesla T4). Corte al 2026-07-26 ~07:33 UTC.
> Este informe da la **señal direccional temprana de PS-1**; el veredicto formal se emite con las
> 8 semillas de mix22 + N_common + tablas secundarias (prereg de seguimiento v1.1).

## Muestra disponible

| condición | semillas | pasos |
|---|---|---|
| delta (C2) | 8/8 | 7500 (aún sub-entrenada, 4/8 sin converger — capacidad sigue subiendo) |
| mix22 (C3) | **3/8** | 2500 (convergió = satura) |
| softmax (C1) | 3/8 | 2500 (satura) |

## Capacidad (acc@1) — el resultado

| carga | delta C2 (n=8) | **mix22 C3 (n=3)** | softmax C1 | dif C3−C2 | margen R11 |
|---|---|---|---|---|---|
| L64 | 0.983 | **1.000** | 1.000 | +0.017 | 0.02 |
| **L96** (evaluación) | 0.916 | **1.000** | 1.000 | **+0.084** | 0.02 |
| L128 | 0.814 | **1.000** | 1.000 | **+0.186** | 0.02 |

**Correctabilidad (T2@L32):** delta 0.960 · **mix22 0.999** · softmax 1.000

## Lectura de PS-1 (la predicción estrella)

**PS-1 = ¿las cabezas mixtas (C3) rescatan la capacidad frente a delta puro (C2) en la carga donde
delta cae bajo 95%?** → En L96 y L128, delta cae; **mix22 se va al techo (1.000)**.

- **Dirección: CONFIRMA, y fuerte.** dif C3−C2 = **+0.084 en L96** y **+0.186 en L128**, muy por
  encima del margen efectivo R11 = 0.02.
- El rescate **crece con la carga** (L64 +0.017 → L96 +0.084 → L128 +0.186): justo donde delta más
  sufre, las cabezas softmax más lo salvan. Rescate monótono (apunta también a PS-3).
- **Sin trade-off:** mix22 no paga el rescate con correctabilidad — T2@32 = 0.999 (mejor que delta
  0.960). C3 se lleva capacidad **y** correctabilidad al techo. (Relevante para P1.2/P1.3 «no
  interferencia».)

**Interpretación mecanicista:** mix22 = 2 cabezas softmax + 2 delta. Basta con inyectar cabezas
softmax para que la memoria asociativa recupere la capacidad plena que delta puro no alcanza →
el cuello de botella de capacidad de linear-attention es **rescatable por composición**, no
intrínseco de la mezcla.

## Caveats (para no cantarlo como final)

1. **Solo 3/8 semillas** de mix22. Faltan 5 (aunque las 3 dan 1.000 exacto, SD=0 → muy estable).
2. mix22 satura a su convergencia (2500). El PS-1 **formal** usa N_common (= convergencia de delta,
   ~7500-10000); mix22 **se mantiene en 1.000** ahí (saturado) → no cambia la conclusión.
3. delta aún sube (a convergencia L96 podría trepar algo, ~0.92-0.94), pero **no llega a 1.000**
   (plateau) → el margen +0.06/+0.08 se sostiene con holgura sobre R11=0.02.
4. Falta la **regla de discordancia** (tabla primaria vs secundaria) para el sello formal.

## Otros veredictos (al día, delta 8/8 @7500)

- **PS-4 (forma de la degradación): CONFIRMA robusto** — inicio L₀=64 (8/8), monotonía ρ=−1.000,
  aceleración +0.035 (IC no cruza 0).
- **PS-5 (anticorrelación capacidad↔correctabilidad en delta): CONFIRMA preliminar** — crudo −0.678,
  IC95 [−0.933, −0.187] excluye cero; parcial −0.645, retención 0.95. (Emergió al subir el N; no
  final hasta convergencia.)

## Próximo paso
Completar mix22 s3-s7 (faltan 5 semillas ≈ 1-2 sesiones), dejar converger delta, fijar N_common,
extender ambos y cerrar fase A → **PS-1 formal**. Restante ≈ 11 h ≈ 4 sesiones.
