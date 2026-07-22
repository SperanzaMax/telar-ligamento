# Resultados — Fase 0

## Estado de los artefactos (rótulos)

| Archivo | Rótulo | Qué es | Hardware |
|---|---|---|---|
| `s09_c1_preview.json` / `.log` | **PREVIEW** | C1 (softmax), **1 semilla**, orientativo. acc@1=1.000 en todas las cargas → veredicto D2: «P1.1 no evaluable por saturación». **No es baseline de campaña.** | CPU local (4 núcleos) |
| `s0_leakage.json` | control de sanidad | Test de fuga sobre el 1.000 de C1 (clave nunca presentada → azar) | CPU local |
| `s09/` + `margenes_instanciados.md` + `s09/s09_summary.json` | **DEFINITIVO** | S0.9 de campaña: **C1+C2, 8 semillas**, márgenes R11, cargas de evaluación. Corre con `fase0_s09.py` (Colab/GPU, notebook en `notebooks/`). | GPU (mismo hardware para C1 y C2) |
| `s07_check.json` | S0.7 | Determinismo re-verificado en el hardware real de la campaña | (el de la corrida) |

**Regla:** la tabla de C1 del preview es orientativa y está rotulada como tal. Los veredictos y márgenes
de Fase 0 salen **solo** del S0.9 definitivo (8 semillas, mismo hardware para ambas condiciones), con
`jax_default_matmul_precision='highest'`, XLA determinista y S0.7 re-corrido en esa GPU.
