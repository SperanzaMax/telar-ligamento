"""Test de humo del agregador de E1 (`aggregate`) con datos sintéticos, sin JAX ni GPU.

Inyecta stubs de los módulos que solo hacen falta para ENTRENAR (jax, entrenar, modelos, fase0_s09),
escribe JSONs de semilla como los que produce la campaña, y verifica que el informe sale con los
veredictos correctos — incluida la regla de discordancia del Anexo B3.

Uso: python3 experimentos/E1/test_agregador_e1.py
"""
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
LOADS = [8, 16, 32, 64, 96, 128]
fallos = []


def check(nombre, cond, extra=""):
    print(f"  {'✓' if cond else '✗'} {nombre}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        fallos.append(nombre)


def _stubs():
    """Módulos falsos: el agregador no los usa, pero e1_runner los importa al cargarse."""
    jax = types.ModuleType("jax")
    jax.config = types.SimpleNamespace(update=lambda *a, **k: None)
    jax.devices = lambda: [types.SimpleNamespace(platform="cpu", device_kind="stub")]
    sys.modules["jax"] = jax
    entrenar = types.ModuleType("entrenar")
    for n in ("train_resumable", "converged", "eval_capacity", "eval_overwrite"):
        setattr(entrenar, n, lambda *a, **k: None)
    sys.modules["entrenar"] = entrenar
    modelos = types.ModuleType("modelos")
    modelos.count_params = lambda p: 0
    sys.modules["modelos"] = modelos
    s09 = types.ModuleType("fase0_s09")
    s09.device_info = lambda: {"device_kind": "stub"}
    s09.notify = lambda *a, **k: None
    sys.modules["fase0_s09"] = s09


def escribir_semilla(d, cond, seed, curva, t2, steps, paso_conv, sufijo=""):
    out = {"exp": "E1", "cond": cond, "seed": seed, "steps": steps, "converged": True,
           "paso_conv_propio": paso_conv, "params": 192453, "device": {"device_kind": "stub"},
           "capacity": {str(L): {"1": float(a), "4": float(min(1.0, a + .1)),
                                 "16": float(min(1.0, a + .2))} for L, a in zip(LOADS, curva)},
           "T2": {str(L): float(v) for L, v in t2.items()},
           "val_hist": [{"step": 500, "val_acc": .5}], "wall_s": 1.0}
    json.dump(out, open(os.path.join(d, f"e1_{cond}_seed{seed}{sufijo}.json"), "w"), indent=1)


def escenario(d, rescate_primaria, rescate_secundaria, n_seeds=8, semilla=0):
    """Escribe C1/C2/C3 con el rescate pedido en cada tabla."""
    rng = np.random.default_rng(semilla)
    base = np.array([1.0, 1.0, .999, .974, .887, .773])
    for s in range(n_seeds):
        ruido = rng.normal(0, .004, 6)
        c2 = np.clip(base + ruido, 0, 1)
        escribir_semilla(d, "softmax", s, np.ones(6), {32: 1.0, 96: 1.0, 128: 1.0}, 5000, 2000)
        escribir_semilla(d, "softmax", s, np.ones(6), {32: 1.0, 96: 1.0, 128: 1.0}, 2500, 2000, "_propio")
        escribir_semilla(d, "delta", s, c2, {32: .89 - .5 * (c2[5] - .773), 96: .60, 128: .45},
                         5000, 2500 + 500 * s)
        escribir_semilla(d, "delta", s, c2, {32: .89 - .5 * (c2[5] - .773), 96: .60, 128: .45},
                         5000, 2500 + 500 * s, "_propio")
        for suf, resc in (("", rescate_primaria), ("_propio", rescate_secundaria)):
            c3 = np.clip(c2 + np.where(np.array(LOADS) >= 96, resc, 0), 0, 1)
            escribir_semilla(d, "mix22", s, c3, {32: .93, 96: .70, 128: .55},
                             5000 if not suf else 2500, 2000, suf)


def correr_aggregate(d):
    os.environ["RESULTS_DIR"] = d
    os.environ["CONDS"] = "delta,softmax,mix22"
    os.environ["N_SEEDS"] = "8"
    for m in list(sys.modules):
        if m in ("e1_runner", "analisis_e1"):
            del sys.modules[m]
    sys.path.insert(0, AQUI)
    import e1_runner
    e1_runner.aggregate()
    return open(os.path.join(d, "E1_informe.md"), encoding="utf-8").read()


_stubs()
print("=== Agregador de E1 · escenario CONCORDANTE (rescate real en ambas tablas) ===")
d = tempfile.mkdtemp()
escenario(d, rescate_primaria=0.10, rescate_secundaria=0.10)
inf = correr_aggregate(d)
check("carga de evaluación instanciada en L96 desde C2", "carga de evaluación (desde C2 convergida): L96" in inf)
check("declara N_common", "N_common = 5000" in inf, [l for l in inf.split("\n") if "N_common" in l][:1])
check("emite las dos tablas", "Tabla PRIMARIA" in inf and "Tabla SECUNDARIA" in inf)
check("PS-1 confirma con tablas concordantes", "**VEREDICTO: CONFIRMA**" in inf and "CONCORDANTES" in inf)
check("PS-2 reporta la posición f", "f = (C3−C2)/(C1−C2)" in inf)
check("PS-4(i) confirma con la curva del snapshot", "mediana(L₀) = 64" in inf)
check("PS-5 aplica el fallback a L32 (T2@96 = 0.60 tiene SD nula)", "T2 primaria: **L32**" in inf)
check("el fallback declara la condición cross-carga", "CROSS-CARGA" in inf)
check("P1.2 y P1.3 del madre presentes", "**P1.2**" in inf and "**P1.3**" in inf)
shutil.rmtree(d)

print("\n=== escenario DISCORDANTE (rescate solo en la tabla primaria) ===")
d = tempfile.mkdtemp()
escenario(d, rescate_primaria=0.10, rescate_secundaria=-0.05)
inf = correr_aggregate(d)
check("PS-1 → «no concluyente por sensibilidad al presupuesto»",
      "NO CONCLUYENTE POR SENSIBILIDAD AL PRESUPUESTO" in inf)
check("y lo marca como DISCORDANTES", "DISCORDANTES" in inf)
check("no oculta que la primaria confirmaba", "primaria (N_common): confirma" in inf)
shutil.rmtree(d)

print("\n=== escenario SIN rescate (C3 = C2) ===")
d = tempfile.mkdtemp()
escenario(d, rescate_primaria=0.0, rescate_secundaria=0.0)
inf = correr_aggregate(d)
check("PS-1 falsa cuando no hay rescate", "**VEREDICTO: FALSA**" in inf)
shutil.rmtree(d)

print("\n" + "=" * 60)
if fallos:
    print(f"✗ {len(fallos)} FALLO(S): " + "; ".join(fallos))
    sys.exit(1)
print("✓ AGREGADOR VERIFICADO — doble tabla, regla de discordancia y fallback de T2 funcionan.")
