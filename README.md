# «Ligamento» (nombre de trabajo PROVISIONAL) — especialización ↔ compartición

**Proyecto hermano de TELAR-03 «Trama» (umbrales de capacidad).** Colisión de nombre
resuelta: TELAR-03-capacidad llegó primero y conserva «Trama»; este proyecto se rebautiza.
Nombre provisional **«Ligamento»** (patrón textil de entrelazado urdimbre/trama). Pendiente
de confirmación de Maxi — alternativa ofrecida: «Lanzadera». Renombre = `mv` del directorio.

**Ejecutor:** Claude Opus 4.8. **Diseño:** Fable 5 + Maxi.

## Estado
- **CONGELADO 2026-07-22 → `protocolo/PROTOCOLO_v1.0.md`.**
  SHA-256 `2f8ebb829ddcaf9de0b4409b44567e84eff52bd5fa62bf1233836b166712f7f1`
  (companion `protocolo/FREEZE_v1.0.md`). Recorrido: v0.2 → v0.3 (lectura de ejecutor) →
  v0.3.1 (verificación) → v1.0 (D1: sin √2 en R11; D2: carga de E1 por saturación del baseline).
- Depósito externo (Zenodo/OSF/git tag) pendiente de Maxi — registrar DOI/tag en el companion.
- **Próximo paso del ejecutor: Fase 0 → E1.** Ya no se discute diseño; se corre. Toda desviación
  operativa va a `desviaciones.md` con fecha y motivo antes de mirar los resultados afectados.

## Decisiones registradas para v0.3 (aceptadas por Maxi 2026-07-22)

### R2/E1 — bloqueante aceptado
`R2` (FLOPs/token ±5%) es **infactible en E1** por construcción: softmax es O(n²), delta es
O(n); a n≤128, d_head=16, C1 (4×softmax) > C3 (2+2) > C2 (4×delta) en FLOPs de atención.
**Resolución v0.3:** en E1, **params ±5% obligatorio; FLOPs reportados por condición en
tabla** (no igualados). Más **cláusula de sesgo conservador pre-registrada**: toda
confirmación de P1.1/P1.2 ocurre con C3 corriendo en **desventaja** de cómputo frente a C1 →
el desbalance es conservador respecto de la hipótesis (si la mixta empata capacidad y hereda
correctabilidad con menos FLOPs, el resultado es más fuerte, no confundido). La violación se
convierte en sesgo documentado a favor.

### Otros flags de ejecutor (para incorporar en v0.3)
- **S0.7 determinismo en CPU:** XLA con múltiples hilos puede variar reducciones float32. Fijar
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` + `OMP_NUM_THREADS=1`, o definir S0.7 con
  tolerancia numérica explícita (p.ej. |Δloss| < 1e-5) en vez de "idénticas".
- **E3 top-k de MoD y causalidad:** la decisión de procesar el token t compite contra puntajes
  de tokens futuros (top-k sobre la secuencia) → no-causalidad suave. Mitigar (router causal /
  predictor auxiliar estilo MoD) o declararlo como límite pre-registrado. En T4 (clasificación,
  no LM autorregresivo) el impacto es menor, pero hay que declararlo.

## Estructura (§3 del protocolo, adaptada al nombre)
```
telar-ligamento/
  protocolo/          # PROTOCOLO_v1.0.md (CONGELADO) + FREEZE_v1.0.md (hash/ancla) + lecturas
  src/                # datos.py, modelos.py, reglas_memoria.py, entrenar.py, probes.py, analisis.py
  experimentos/E1..E4/
  resultados/fase0/ E1..E4/
```

## Licencia y cita
- **Código** (`src/`, `experimentos/`, scripts): **MIT** — ver [`LICENSE`](LICENSE).
- **Protocolo y documentación** (`protocolo/`, este README, docs): **CC-BY-4.0** — ver
  [`LICENSE-PROTOCOL`](LICENSE-PROTOCOL).
- **Cita sugerida:** Speranza, M. (2026). *«Ligamento» — Experimental protocol for
  specialization vs. sharing in small-scale transformers* (v1.0, pre-registered and frozen).
  Programa TELAR. Ancla de integridad: SHA-256 en `protocolo/FREEZE_v1.0.md`.
- **Pre-registro:** el diseño se congeló antes de cualquier corrida principal. El tag firmado
  `ligamento-v1.0-freeze` y el commit del freeze son el ancla pública de fecha; el hash SHA-256
  del protocolo permite verificar que no se tocó (`sha256sum -c` contra el companion).
