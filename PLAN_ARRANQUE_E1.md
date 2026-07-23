# PLAN DE ARRANQUE — E1 (decisiones de Maxi 2026-07-22, para ejecutar 2026-07-23)

Este documento fija el arranque automático de la próxima sesión. El ejecutor (Opus) lo lee y ejecuta la
secuencia sin re-preguntar. Las decisiones ya están tomadas.

## Decisiones de Maxi (cerradas)

1. **E1 — arrancamos ya.** Hardware: **Colab Pro** (lo activa Maxi). **Semillas: 8.** **Early stopping
   ratificado**, con criterio sobre **eval por carga congelada** (acc@1 en **L96 y L128** evaluada cada
   **500 pasos**), **tope duro 10 000**. → el criterio de convergencia se mide sobre las cargas del veredicto,
   no sobre una val genérica.
2. **Nota de cierre S0.9 — aprobada con TRES condiciones:**
   - (a) las **8 delta de E1 bajo un solo régimen**: 4 extendidas desde checkpoint + 4 nuevas, **mismo criterio**
     de convergencia para las 8.
   - (b) los **márgenes del prereg de seguimiento se instancian desde la C2 (delta) CONVERGIDA de E1**, no del
     snapshot 2500.
   - (c) la **tabla S0.9 (2500) queda rotulada snapshot / cota inferior** (ya hecho en `margenes_instanciados.md`).
3. **Secuencia OBLIGATORIA antes de lanzar E1:**
   `nota de cierre` → **prereg de seguimiento C3-vs-C2 actualizado, congelado y pusheado** (incorpora
   **inicio/pendiente** de la caída y **anticorrelación** capacidad↔correctabilidad como predicciones formales;
   drafteado por Fable5 + Opus en una pasada rápida) → **recién ahí se lanza E1.**
4. **Puente:** los 3 slots los pasa Maxi cuando tenga el corpus a mano; **gestiona ya el token HF de Gemma**.
   Nombre definitivo: **«Puente».**
5. **TELAR-03 Fase 2:** **Fable5 la diseña en paralelo** mientras E1 corre; ejecución después.
6. **Flujo:** cruzado (Fable5 diseña / Opus ejecuta y critica), como hasta ahora.

## Las DOS únicas tareas de Maxi
- **Activar Colab Pro.**
- **Pasar los 3 slots de Puente** (cuando tenga el corpus V6 a mano) + el token HF de Gemma.

## Secuencia del ejecutor (orden estricto)

**Preparado HOY (2026-07-22), listo para mañana:**
- [x] `margenes_instanciados.md` (S0.9 snapshot 4+4, rotulado cota inferior).
- [x] Draft del prereg de seguimiento **v1.1** (`protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.1_DRAFT.md`) con las
      predicciones nuevas (**PS-4** inicio/pendiente, **PS-5** anticorrelación) — mi pasada, para que Fable5 la critique.
- [x] **C3 implementado** (cabezas mixtas softmax+delta en `modelos.py`) + test de sanidad.
- [x] **Runner de E1** (`experimentos/E1/e1_runner.py`): C1/C2/C3/C4, 8 semillas, early stopping con criterio
      L96/L128 cada 500, tope 10k, régimen único para las 8 delta (extiende 4 desde checkpoint + 4 nuevas).

**Mañana, al arrancar (bloqueado hasta la pasada de Fable5 + Colab Pro):**
1. Fable5 hace su pasada sobre el prereg v1.1 → incorporo → **congelo v1.1 (hash + tag + push)**. Esto
   DESBLOQUEA E1 (condición 3).
2. Maxi confirma Colab Pro activo.
3. Adapto el notebook Colab a E1 (celda maestra reanudable, ya con GPU estable de Pro).
4. **Lanzo E1** (8 semillas, 4 condiciones). Delta se entrena con el régimen único + early stopping.
5. Monitoreo por Telegram (mecanismo ya listo, token por env).

## Detalles técnicos de E1 (fijados)
- **Condiciones:** C1 (4 softmax), C2 (4 delta), C3 (2 softmax + 2 delta, mixta), C4 (3+1, 1+3 exploratoria).
- **Tareas:** T1 (capacidad, 6 cargas) + T2 (correctabilidad).
- **Convergencia (todas las condiciones, uniforme):** acc@1 en {L96, L128} evaluada cada 500 pasos; converge
  cuando la mejora en la ventana de 500 < 0.5 pts; tope 10 000. Las 8 semillas de delta bajo el mismo régimen.
- **Márgenes R11:** se instancian desde la C2 convergida (condición 2b). Carga de evaluación del prereg =
  menor L con delta convergido < 95% (se re-mide; el L96 del snapshot es preliminar).
- **Predicciones:** P1.1 (no evaluable, D2), P1.2 (herencia correctabilidad), P1.3 (no interferencia) del
  protocolo madre; PS-1..PS-7 del prereg de seguimiento v1.1.

*Estado al cierre 2026-07-22: todo lo delegable, hecho. Cuello de botella = Colab Pro + slots (Maxi) y la
pasada de Fable5 sobre el prereg v1.1.*
