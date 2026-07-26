# Estrategia para llegar a PS-1 (decidida 2026-07-26)

**Objetivo:** obtener el veredicto de **PS-1** (¿C3=mix22 rescata capacidad frente a C2=delta
en la carga de evaluación L96?) lo antes posible **y válido**, sin gastar cómputo de Colab free
en lo que no aporta al veredicto.

## Decisión
Cambiar la celda 8 del notebook a:
```
%env CONDS=mix22,delta
```
(se quitan `softmax`, `mix31`, `mix13` del camino crítico).

## Fundamento
- **PS-1 = C3 vs C2. No usa softmax.** softmax (C1) solo entra en PS-2 (calculable con C1=1.0 por
  saturación/D2) y en P1.2/P1.3 (madre). Como el doble-reporte **extiende TODA condición del run
  hasta el N_common**, tener softmax en el run lo obliga a entrenar saturado (~1.000) hasta N_common
  → cómputo tirado. Sacarlo **no cambia ningún número de PS-1**: N_common = máx de la convergencia de
  las condiciones del run, y softmax converge a 2500 → nunca es el máximo.
- **mix31/mix13** son exploratorias: no dan ningún veredicto pre-registrado. Segunda tanda.
- **mix22 primero en la lista** = se agenda antes → señal direccional temprana.

## Hito de señal temprana
Tras el **primer bloque de mix22 (8 semillas @2500)**, comparar informalmente:
- delta (C2) @2500, L96 = **0.892** (referencia S0.9).
- mix22 (C3) @2500, L96 = ← observado.
- mix22 claramente > 0.892 → PS-1 tiende a **confirma**; ≈ o < → tiende a **falsa/no rescate**.
Es informal (N bajo, no convergido) pero da la dirección de la estrella.

## Secuencia completa
1. **Sesiones con `CONDS=mix22,delta`:** el planner construye mix22 (fase A, desde 0) y termina la
   convergencia de delta (reanuda desde 7500). Fija N_common = máx(delta, mix22), extiende ambos a
   N_common (fase B), cierra fase A (tablas secundarias `*_propio.json`).
2. **Regenerar informe** (celda 9 en Colab, o `regenerar_informe_local.py` en la PC) → sale **PS-1**
   con la regla de discordancia + PS-2 + PS-4 + PS-5.
3. **Al final, para el madre:** `CONDS=delta,softmax,mix22` para P1.2/P1.3 (softmax se extiende a
   N_common, caro pero no bloquea PS-1 → va último).
4. **mix31/mix13:** segunda tanda exploratoria.

## Notas
- Cambiar CONDS **no pierde progreso**: es reanudable, lee los checkpoints de Drive; solo cambia qué
  se agenda. delta @7500 intacto.
- mix22 arranca con loss ~5.5 y val_acc bajo = correcto (no tiene checkpoint), NO es reinicio erróneo.
- **Costo inevitable:** mix22 desde 0 hasta N_common (8 semillas × ~7500-10000 pasos) es el grueso;
  un PS-1 válido lo exige. La estrategia adelanta la *dirección*, no elimina ese costo.
- Estado al 2026-07-26: delta 8/8 @7500 (4/8 conv), softmax 3/8, mix22 0/8. PS-4 confirmado robusto;
  PS-5 confirma preliminar @7500 (no final hasta convergencia); PS-1 pendiente de mix22.
