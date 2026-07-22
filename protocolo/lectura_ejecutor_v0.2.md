# Lectura de ejecutor — TELAR-EXP «Trama»/«Ligamento» v0.2 → insumos para v0.3

Opus 4.8, 2026-07-22. Ordenado por severidad. Objetivo: que v0.3 se congele una vez y limpio.

## A. Bloqueantes de congelamiento (obligan desviación si se congela así)

**A1 · R2/E1 — FLOPs ±5% infactible en E1.** [YA ACEPTADO] Resolución acordada: en E1,
params ±5% obligatorio, FLOPs reportados por condición, cláusula de sesgo conservador (C3
corre con desventaja de cómputo vs C1 → toda confirmación de P1.1/P1.2 es conservadora).

**A2 · R2 TAMBIÉN rompe en E4 — y ahí NO es conservador.** El mismo problema que E1, sin
carve-out (E3 sí lo tiene). En E4 los expertos ruteados por modalidad computan **1 de 3**
(mismos params: 3×FFN-64 = 24576 = 192-denso; pero 1/3 de FLOPs). Entonces F (todo ruteado)
≈ ⅓ de los FLOPs de FFN de S (denso), y M en el medio. **Peor que E1:** P4.1 predice M ≥ F,S;
si M gana, corre con MÁS FLOPs que F → la ventaja de M puede ser cómputo, no fusión. El sesgo
NO favorece automáticamente la hipótesis. Fix v0.3: reportar FLOPs efectivos por condición en
E4 **y** agregar una variante FLOP-matched de F (o M) como control, o declarar el confound de
cómputo como límite pre-registrado de P4.1. P4.3 (equivalencia F≈M) sí queda conservadora (F
más barato empatando a M refuerza "la partición NO limita la fusión").

**A3 · P2.1 indefinida en ρ=0.** El gap `g_s(ρ)=acc_B−acc_A` se define "sobre claves
polisémicas". En ρ=0 **no hay claves polisémicas** → métrica vacía. Pero P2.1 mete ρ=0 en el
enunciado ("|B−A| dentro del margen en ρ=0") y el Spearman(g_s,ρ) barre todos los ρ. Fix:
computar el Spearman sobre ρ∈{0.25,0.5,0.75,1.0} (excluir ρ=0), y mover el chequeo de ρ=0 a
**exactitud total** (no polisémica). Si no, el análisis se rompe al correr.

## B. Defectos de operacionalización (no bloquean, pero sesgan veredictos)

**B1 · Umbral de "unidad multimodal" = 2×control-permutado ≈ 2×chance ≈ 6%.** El control
permutado rinde ~chance = 1/32 ≈ 3%; "2×" = 6% de accuracy de probe cuenta como multimodal a
casi cualquier unidad débilmente informativa. Bar demasiado bajo → P4.2 casi no-falsable. Fix:
definir el umbral como "accuracy del probe supera el **percentil 95 de la distribución nula
permutada**" (umbral de significancia real), en las tres modalidades.

**B2 · Margen de equivalencia mal escalado (P1.1, P2.2, P4.3 — las de 8 semillas).** R11 fija
el margen efectivo con `1.5×SD_entre-semillas(baseline)` = ruido de **una** condición. Pero
R3 testea equivalencia con el **CI bootstrap de la diferencia** (C3−C1), cuya varianza es la
de **dos** condiciones (~SD_C3²+SD_C1² si no pareás por semilla). Margen de 1 condición vs
estadístico de 2 → el test de equivalencia puede quedar arbitrariamente duro o blando, y las 8
semillas no lo arreglan si el margen está mal escalado. Fix: derivar el margen del **SD de la
diferencia pareada por semilla** sobre la familia baseline, o declarar explícitamente que el
piso es un SESOI (smallest effect size of interest) y el 1.5×SD es solo el override por ruido.

**B3 · P1.2 ratio inestable.** `(C3−C1)/(C2−C1)` es cociente de dos cantidades ruidosas → CI
de cola pesada aun sobre el guard de aplicabilidad. El guard `(C2−C1)<margen → no evaluable`
cubre el denominador chico, pero cerca del borde el CI del ratio explota. Fix: bootstrapear el
ratio con CI percentil robusto (ya implícito) y reportar además la diferencia cruda (C3−C1) en
puntos, para que el veredicto no dependa solo del cociente.

**B4 · Ruta C de E2 cambia el largo de secuencia.** C concatena el contexto como tokens → T
mayor que A/B/D (que usan cross-attention a un banco separado). FLOPs de softmax y footprint
posicional distintos → rompe R2 dentro de E2. Fix: padear A/B/D al mismo T, o reportar FLOPs y
declarar que C no es FLOP-comparable.

## C. Flags técnicos (ya anunciados)

**C1 · S0.7 determinismo en CPU.** XLA multihilo puede variar reducciones float32 run-to-run.
Fix: `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` + `OMP_NUM_THREADS=1`, **o** redefinir S0.7
con tolerancia numérica explícita (p.ej. |Δloss|<1e-5) en vez de "curvas idénticas".

**C2 · E3 top-k de MoD y causalidad.** La decisión de procesar el token t compite contra
puntajes de tokens futuros (top-k sobre la secuencia) → no-causalidad suave / fuga. En T4
(clasificación, no LM autorregresivo) el impacto es menor, pero hay que **declararlo** o usar
router causal / capacidad por bloque. Decidir cuál en v0.3.

## D. Menores (spec de implementación, para el src, no para el freeze)

- **E1/C3:** especificar que la O-proyección ve un **concat de cabezas heterogéneas** (2
  softmax O(n²) + 2 delta recurrentes); el `mixer()` de TELAR-01 asume una sola regla → hay que
  splitear por tipo dentro de la capa. La ablación por cabeza (análisis secundario) queda bien
  definida sobre ese concat.
- **R4/R1:** en MQAR la val-loss ≈ task-acc; "elegir LR por val-loss" es casi "por métrica de
  tarea". Es defendible (pre-registrado, simétrico entre condiciones), pero conviene declarar el
  split de validación y que la selección es idéntica para todas las condiciones.
- **T3 (S0.3):** el test unitario del generador debe garantizar que, marginalizando sobre
  contextos, cada sentido de una clave polisémica es 50/50 — si no, el techo enmascarado no es
  exactamente 50% y S0.3 falla por diseño, no por bug del modelo.

## Resumen para v0.3
3 bloqueantes (A1 aceptado, A2 nuevo en E4, A3 ρ=0), 4 defectos de operacionalización (B1–B4),
2 técnicos (C1–C2), 3 menores (D). Todo en una pasada → v0.3 → freeze.
