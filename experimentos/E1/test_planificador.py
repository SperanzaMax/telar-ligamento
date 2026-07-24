"""Tests del planificador de sesiones de E1. Sin JAX ni GPU.

Verifica que la campaña se fracciona en unidades atómicas correctas, que las fases A y B se
encadenan bien y que el recorte por presupuesto no parte una unidad al medio.

Uso: python3 experimentos/E1/test_planificador.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planificador as pl  # noqa: E402

CONDS = ["delta", "softmax", "mix22"]
N = 8
fallos = []


def check(nombre, cond, extra=""):
    print(f"  {'✓' if cond else '✗'} {nombre}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        fallos.append(nombre)


def escribir(d, cond, seed, steps, converged, propio=False, wall=2400.0, pasos=2500):
    for suf in ([""] + (["_propio"] if propio else [])):
        json.dump({"cond": cond, "seed": seed, "steps": steps, "converged": converged,
                   "wall_s": wall, "pasos_en_corrida": pasos},
                  open(os.path.join(d, f"e1_{cond}_seed{seed}{suf}.json"), "w"))


def poblar(d, cond, steps, converged, propio=False):
    for s in range(N):
        escribir(d, cond, s, steps, converged, propio)


print("=== Campaña vacía ===")
d = tempfile.mkdtemp()
est = pl.leer_estado(d, CONDS, N)
acc = pl.plan(est, CONDS, N)
train = [a for a in acc if a[0] == "entrenar"]
check("arranca entrenando la primera condición al bloque 2500",
      train[0] == ("entrenar", "delta", 0, 2500), train[0])
check("prioriza delta (fija N_common y la carga de evaluación)",
      [a[1] for a in train[:8]] == ["delta"] * 8)
check("encola el primer bloque de las tres condiciones (24 unidades, todo trabajo seguro)",
      len(train) == 24 and all(a[3] == 2500 for a in train), len(train))
check("NO especula con bloques futuros de una condición sin saber si convergió",
      max(a[3] for a in train) == 2500)
check("no propone informe todavía", not any(a[0] == "informe" for a in acc))
shutil.rmtree(d)

print("\n=== Bloque terminado SIN converger → siguiente bloque ===")
d = tempfile.mkdtemp()
poblar(d, "delta", 2500, converged=False)
acc = pl.plan(pl.leer_estado(d, CONDS, N), CONDS, N)
train = [a for a in acc if a[0] == "entrenar" and a[1] == "delta"]
check("propone llevar delta a 5000", all(a[3] == 5000 for a in train), train[0])
check("son 8 unidades de delta (una por semilla)", len(train) == 8)
shutil.rmtree(d)

print("\n=== Bloque terminado CON convergencia → cierra fase A y sigue con la próxima condición ===")
d = tempfile.mkdtemp()
poblar(d, "delta", 5000, converged=True)
acc = pl.plan(pl.leer_estado(d, CONDS, N), CONDS, N)
check("primera acción: cerrar fase A de delta con N_final=5000",
      acc[0] == ("cerrar_faseA", "delta", None, 5000), acc[0])
check("después arranca softmax en 2500",
      acc[1] == ("entrenar", "softmax", 0, 2500), acc[1])
shutil.rmtree(d)

print("\n=== Semilla rezagada (sesión cortada a mitad de bloque) ===")
d = tempfile.mkdtemp()
poblar(d, "delta", 2500, converged=False)
for s in (3, 4, 5, 6, 7):
    escribir(d, "delta", s, 5000, False)          # 3 semillas quedaron atrás
acc = pl.plan(pl.leer_estado(d, CONDS, N), CONDS, N)
train = [a for a in acc if a[0] == "entrenar" and a[1] == "delta"]
check("solo re-encola las semillas rezagadas", len(train) == 3 and {a[2] for a in train} == {0, 1, 2},
      [(a[2], a[3]) for a in train])
check("y las lleva al bloque en que están las demás", all(a[3] == 5000 for a in train))
shutil.rmtree(d)

print("\n=== Fase B: N_common y fraccionamiento en unidades de 2500 ===")
d = tempfile.mkdtemp()
poblar(d, "delta", 10000, converged=True, propio=True)     # la lenta fija N_common
poblar(d, "softmax", 2500, converged=True, propio=True)
poblar(d, "mix22", 5000, converged=True, propio=True)
est = pl.leer_estado(d, CONDS, N)
check("detecta fase A cerrada en las tres", all(est[c]["faseA_cerrada"] for c in CONDS))
acc = pl.plan(est, CONDS, N)
train = [a for a in acc if a[0] == "entrenar"]
check("delta no se re-entrena (ya está en N_common)", not any(a[1] == "delta" for a in train))
sm = [a for a in train if a[1] == "softmax" and a[2] == 0]
check("softmax s0 se fracciona 2500→5000→7500→10000",
      [a[3] for a in sm] == [5000, 7500, 10000], [a[3] for a in sm])
check("ninguna unidad salta más de 2500 pasos de una",
      all(a[3] - b[3] == 2500 for a, b in zip(sm[1:], sm[:-1])))
mx = [a for a in train if a[1] == "mix22" and a[2] == 0]
check("mix22 s0 se fracciona 5000→7500→10000", [a[3] for a in mx] == [7500, 10000], [a[3] for a in mx])
check("cierra con la acción de informe", acc[-1][0] == "informe")
total = 8 * 3 + 8 * 2
check(f"total de unidades de fase B = {total}", len(train) == total, len(train))
shutil.rmtree(d)

print("\n=== Campaña completa → solo informe ===")
d = tempfile.mkdtemp()
for c in CONDS:
    poblar(d, c, 7500, converged=True, propio=True)
acc = pl.plan(pl.leer_estado(d, CONDS, N), CONDS, N)
check("no queda nada que entrenar", not any(a[0] == "entrenar" for a in acc))
check("la única acción es el informe", acc == [("informe", None, None, None)], acc)
shutil.rmtree(d)

print("\n=== Recorte por presupuesto ===")
d = tempfile.mkdtemp()
est = pl.leer_estado(d, CONDS, N)
acc = pl.plan(est, CONDS, N)
costos = {"delta": 0.9, "softmax": 0.9, "mix22": 1.02}
hacer, restan, seg = pl.sesion(acc, 210 * 60, costos)     # 3.5 h
check("entran 5 unidades en 3.5 h a 0.9 s/paso", len(hacer) == 5, f"{len(hacer)} unidades, {pl.fmt(seg)}")
check("no se pasa del presupuesto", seg <= 210 * 60, pl.fmt(seg))
check("el resto queda para las próximas sesiones", len(restan) == len(acc) - len(hacer))
check("nunca parte una unidad al medio", all(a[0] in ("entrenar", "cerrar_faseA", "informe") for a in hacer))
hacer2, _, _ = pl.sesion(acc, 60, costos)                  # presupuesto ridículo
check("con presupuesto insuficiente hace al menos una unidad (no se cuelga)", len(hacer2) == 1)
shutil.rmtree(d)

print("\n=== Auto-calibración de costos desde corridas reales ===")
d = tempfile.mkdtemp()
escribir(d, "delta", 0, 2500, False, wall=2258.0 + pl.SEG_EVAL, pasos=2500)
c = pl.costos_medidos(d, CONDS)
check("mide 0.903 s/paso de la corrida real de delta", abs(c["delta"] - 0.9032) < 1e-3, f"{c['delta']:.4f}")
check("no inventa costos para condiciones sin datos", "mix22" not in c)
shutil.rmtree(d)

print("\n" + "=" * 60)
if fallos:
    print(f"✗ {len(fallos)} FALLO(S): " + "; ".join(fallos))
    sys.exit(1)
print("✓ PLANIFICADOR VERIFICADO — fases A/B, fraccionamiento y presupuesto funcionan.")
