#!/usr/bin/env python3
"""Evalúa la TABLA DE INTERPRETACIÓN de `mix13` congelada por la enmienda E-004.

La tabla se congeló ANTES de que existiera un solo dato de `mix13`
(`ENMIENDA_E-004_mix13_20260803.md`, SHA-256 `6662724c…`, tag `ligamento-enmienda-e004`,
release 2026-08-04T01:44:44Z). Este script la ejecuta: no la interpreta, no la discute.

Las tres filas se evalúan EN ORDEN; la primera que se cumple decide. Son exhaustivas y
mutuamente excluyentes por construcción:

  T · `máx d(celda) ≤ 0,0020` sobre las 144 celdas  → techo en toda la grilla
  B · existe celda con `d > 0,0020` Y `dif(mix13 − delta)@L96 > R11`  → bracket dosis-respuesta
  D · `dif(mix13 − delta)@L96 ≤ R11`  → empata con delta

`d(celda) = 1 − capacity[L][k]`. La diferencia es APAREADA POR SEMILLA contra `delta` a su
`N_final = 10000`, con el mismo bootstrap percentil y la misma semilla del pipeline. `mix13` corre a
2500 y `delta` a 10000: asimetría conservadora contra `mix13`, declarada en la enmienda.

Uso:
    python3 evaluar_e004.py [dir_mix13] [dir_e1]
    (por defecto: resultados/E1_mix13 y resultados/E1)
"""
import json
import os
import random
import statistics
import sys

LOADS = [8, 16, 32, 64, 96, 128]
KS = ["1", "4", "16"]
CARGA_EVAL = 96
R11 = 0.0200            # margen efectivo instanciado en E1 (piso; D1 quitó el √2)
UMBRAL_TECHO = R11 / 10  # 0,0020 — el mismo de B1-bis
N_SEEDS = 8
N_BOOT = 10000
SEED_BOOT = 20260723
ALPHA = 0.05

# Tolerancia de punto flotante en las comparaciones de umbral. NO cambia ningún criterio de la
# tabla congelada: evita que el ruido de representación binaria decida una fila. Sin ella,
# `1.0 - 0.0020` da un déficit de 0,0020000000000000018 > 0,0020 y un caso que la tabla manda a T
# cae en B. Es el mismo defecto que la auditoría del 1-ago encontró en B1-bis, que exigía
# `capacity = 1,0000` EXACTO y fallaba en 12 de 48 celdas de `softmax`. Declarada como precisión de
# implementación del ejecutor (2026-08-03), posterior al congelamiento del .md y sin tocarlo.
TOL = 1e-9


def cargar(directorio, cond, n=N_SEEDS):
    runs = []
    for s in range(n):
        ruta = os.path.join(directorio, f"e1_{cond}_seed{s}.json")
        if not os.path.exists(ruta):
            raise SystemExit(f"[E-004] falta {ruta} — la tabla exige las {n} semillas")
        with open(ruta, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def deficit_max(runs):
    """Peor celda de toda la grilla: máx sobre semillas × cargas × k de (1 − capacity)."""
    peor, donde = 0.0, None
    for r in runs:
        for L in LOADS:
            for k in KS:
                d = 1.0 - r["capacity"][str(L)][k]
                if d > peor:
                    peor, donde = d, (r["seed"], L, k)
    return peor, donde


def acc1(runs, L):
    return [r["capacity"][str(L)]["1"] for r in runs]


def percentil(datos, p):
    xs = sorted(datos)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def bootstrap_ic(difs):
    rng = random.Random(SEED_BOOT)
    medias = []
    for _ in range(N_BOOT):
        muestra = [difs[rng.randrange(len(difs))] for _ in range(len(difs))]
        medias.append(statistics.fmean(muestra))
    return percentil(medias, ALPHA / 2), percentil(medias, 1 - ALPHA / 2)


AFIRMACIONES = {
    "T": ("La restitución del techo se logra YA CON LA DOSIS MÍNIMA PROBADA (1 de 4 cabezas). "
          "El claim sobre mix22 pasa a SUFICIENCIA, NO NECESIDAD, y el costo de la solución queda "
          "acotado por arriba en ¼ de las cabezas. PS-3 no obtiene dosis-respuesta: los tres puntos "
          "hibridados saturan. La frontera sigue FUERA de la grilla y la limitación de censura se "
          "mantiene, ahora sobre las tres condiciones."),
    "B": ("Queda el BRACKET DE DOSIS-RESPUESTA: delta < mix13 < techo. El +0,0792 de PS-1 CONSERVA "
          "su lectura de COTA INFERIOR y por primera vez el rescate tiene una medición intermedia "
          "no censurada. PS-3 se evalúa sobre puntos con rango dinámico."),
    "D": ("El efecto EXIGE ≥ 2 CABEZAS softmax y el claim se restringe a eso: la hibridación no es "
          "gradual en este régimen, hay un umbral entre ¼ y ½. mix22 conserva su veredicto; lo que "
          "cambia es el alcance de la afirmación, que deja de sugerir que «alguna» hibridación "
          "alcanza."),
}


def evaluar(runs_mix13, runs_delta):
    peor, donde = deficit_max(runs_mix13)
    m13, dlt = acc1(runs_mix13, CARGA_EVAL), acc1(runs_delta, CARGA_EVAL)
    difs = [a - b for a, b in zip(m13, dlt)]
    dif = statistics.fmean(difs)
    ic = bootstrap_ic(difs)

    if peor <= UMBRAL_TECHO + TOL:
        fila = "T"
    elif dif > R11 + TOL:
        fila = "B"
    else:
        fila = "D"

    return {"fila": fila, "peor_deficit": peor, "peor_celda": donde, "dif": dif, "ic": ic,
            "difs": difs, "n_por_encima_r11": sum(1 for d in difs if d > R11)}


def informe(res):
    L = ["=" * 78,
         "E-004 · TABLA DE INTERPRETACIÓN DE mix13 (congelada 2026-08-03, SHA 6662724c…)",
         "=" * 78, "",
         f"  peor celda de la grilla : déficit {res['peor_deficit']:.6f} "
         f"(seed {res['peor_celda'][0]}, L{res['peor_celda'][1]}, k={res['peor_celda'][2]})"
         if res["peor_celda"] else "  peor celda: sin déficit (grilla en 1,0 exacto)",
         f"  umbral de techo         : {UMBRAL_TECHO:.4f}  (= R11/10, el de B1-bis)",
         f"  dif apareada @L{CARGA_EVAL}      : {res['dif']:+.4f} · IC95 "
         f"[{res['ic'][0]:+.4f}, {res['ic'][1]:+.4f}] · margen R11 = {R11:.4f}",
         f"  semillas con dif > R11  : {res['n_por_encima_r11']}/{len(res['difs'])}", "",
         f"  ►► FILA {res['fila']}", ""]
    for linea in AFIRMACIONES[res["fila"]].split(". "):
        if linea.strip():
            L.append("     " + linea.strip().rstrip(".") + ".")
    L += ["", "La fila la decide la tabla congelada, no el ejecutor. No se relee ni se discute.",
          "=" * 78]
    return "\n".join(L)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    d13 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "resultados", "E1_mix13")
    de1 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "resultados", "E1")
    print(informe(evaluar(cargar(d13, "mix13"), cargar(de1, "delta"))))
