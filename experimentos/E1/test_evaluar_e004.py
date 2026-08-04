"""Tests de la tabla de interpretación de E-004: las tres filas y sus bordes.

Cada test construye el caso que DEBE disparar cada fila, incluidos los bordes del umbral.
Uso: python3 experimentos/E1/test_evaluar_e004.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_e004 import evaluar, LOADS, KS, R11, UMBRAL_TECHO  # noqa: E402

fallos = []


def run(seed, cond, valor_grilla, valor_L96=None):
    """Un run sintético: toda la grilla en `valor_grilla`, con L96/k=1 opcionalmente distinto."""
    cap = {str(L): {k: valor_grilla for k in KS} for L in LOADS}
    if valor_L96 is not None:
        cap["96"]["1"] = valor_L96
    return {"seed": seed, "cond": cond, "steps": 2500, "capacity": cap,
            "T2": {"32": 1.0, "96": 1.0, "128": 1.0}}


def mix13(grilla, L96=None):
    return [run(s, "mix13", grilla, L96) for s in range(8)]


def delta(L96):
    return [run(s, "delta", 1.0, L96) for s in range(8)]


def chequear(nombre, res, fila_esperada):
    ok = res["fila"] == fila_esperada
    print(f"  {'✓' if ok else '✗'} {nombre} → fila {res['fila']} "
          f"(esperada {fila_esperada}) · peor déficit {res['peor_deficit']:.5f} · "
          f"dif {res['dif']:+.4f}")
    if not ok:
        fallos.append(nombre)


print("\n=== fila T · techo en toda la grilla ===")
# mix13 en 1,0 exacto contra delta en 0,92: la dif es enorme, pero T manda porque va primero.
chequear("grilla en 1,0 exacto (dif grande, T tiene prioridad)",
         evaluar(mix13(1.0), delta(0.92)), "T")
chequear("déficit JUSTO en el umbral (0,0020) → todavía T",
         evaluar(mix13(1.0 - UMBRAL_TECHO), delta(0.92)), "T")

print("\n=== fila B · cae de la grilla y supera a delta por más de R11 ===")
# Un pelo por encima del umbral de techo: ya no es T; la dif sigue siendo grande → B.
chequear("déficit apenas sobre el umbral, dif >> R11",
         evaluar(mix13(1.0 - UMBRAL_TECHO - 1e-6), delta(0.92)), "B")
chequear("mix13 intermedio (0,96) contra delta 0,92 → dif 0,04 > R11",
         evaluar(mix13(0.96), delta(0.92)), "B")

print("\n=== fila D · empata con delta ===")
chequear("mix13 0,93 contra delta 0,92 → dif 0,01 ≤ R11",
         evaluar(mix13(0.93), delta(0.92)), "D")
chequear("mix13 por DEBAJO de delta → dif negativa, sigue siendo D",
         evaluar(mix13(0.90), delta(0.92)), "D")

print("\n=== borde exacto de R11 ===")
# dif exactamente R11 NO supera R11 (la tabla dice «supera»), así que cae en D.
r = evaluar(mix13(0.92 + R11), delta(0.92))
chequear(f"dif exactamente R11 ({R11}) → D, porque la tabla exige SUPERAR", r, "D")
r = evaluar(mix13(0.92 + R11 + 1e-6), delta(0.92))
chequear("dif un pelo por encima de R11 → B", r, "B")

print("\n=== la grilla completa se inspecciona, no solo L96 ===")
# Techo en todo salvo UNA celda fuera de L96: debe salir de T.
runs = mix13(1.0)
runs[3]["capacity"]["16"]["4"] = 0.99          # déficit 0,01 > 0,0020, en L16/k=4
r = evaluar(runs, delta(0.92))
chequear("una sola celda caída en L16/k=4 saca de la fila T", r, "B")
ok = r["peor_celda"] == (3, 16, "4")
print(f"  {'✓' if ok else '✗'} y la reporta: {r['peor_celda']}")
if not ok:
    fallos.append("localización de la peor celda")

print("\n=== exhaustividad: toda entrada cae en exactamente una fila ===")
casos = [(g, d) for g in (1.0, 0.999, 0.96, 0.93, 0.5) for d in (0.92, 0.80)]
filas = {evaluar(mix13(g), delta(d))["fila"] for g, d in casos}
ok = filas <= {"T", "B", "D"} and len(filas) == 3
print(f"  {'✓' if ok else '✗'} {len(casos)} combinaciones → filas observadas {sorted(filas)}")
if not ok:
    fallos.append("exhaustividad")

print("\n" + "=" * 60)
if fallos:
    print("✗ FALLOS:", fallos)
    sys.exit(1)
print("✓ TODOS LOS TESTS PASAN — la tabla congelada de E-004 se ejecuta como está escrita.")
