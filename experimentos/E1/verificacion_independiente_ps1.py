#!/usr/bin/env python3
"""Recomputación INDEPENDIENTE de PS-1, sin tocar una línea del pipeline.

Responde a la observación (d2) de la auditoría de Fable5 del 2026-08-01: la búsqueda de errores
fue asimétrica —el único bug encontrado (B3 vacía) era uno cuya reparación no desconfirma nada—,
así que los componentes que PRODUCEN el número favorable necesitan un chequeo con la misma energía.

Este script no importa `analisis_e1`, ni `e1_runner`, ni numpy: sólo la stdlib. Reimplementa desde
cero la lectura de los JSON, el apareo por semilla, el margen efectivo R11 y el bootstrap percentil.
Si el resultado difiere del informe oficial, el pipeline tiene un defecto; si coincide dígito a
dígito, el número no depende del código que lo produjo.

Uso:  python3 verificacion_independiente_ps1.py [resultados/E1]
"""

import json
import os
import random
import statistics
import sys

CARGA_EVAL = 96          # instanciada por Anexo A(c): menor L con C2 convergida < 0.95
PISO_MARGEN = 0.02       # R11 = máx(piso, 1.5·SD), con D1 (sin √2)
N_BOOT = 10000
SEED_BOOT = 20260723     # mismo método y semilla que el pipeline, para que sea comparable
ALPHA = 0.05


def acc1(directorio, cond, carga):
    """acc@1 de cada semilla, leído crudo del JSON. Sin caché, sin helpers del pipeline."""
    out = []
    for s in range(8):
        ruta = os.path.join(directorio, f"e1_{cond}_seed{s}.json")
        with open(ruta, encoding="utf-8") as fh:
            out.append(json.load(fh)["capacity"][str(carga)]["1"])
    return out


def percentil(datos, p):
    """Percentil por interpolación lineal, escrito a mano."""
    xs = sorted(datos)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_ic(muestra):
    """IC percentil de la media, remuestreando las semillas con reemplazo."""
    rng = random.Random(SEED_BOOT)
    n = len(muestra)
    medias = [statistics.fmean(rng.choices(muestra, k=n)) for _ in range(N_BOOT)]
    return percentil(medias, ALPHA / 2), percentil(medias, 1 - ALPHA / 2)


def buscar_ic(linea, cual):
    """Extrae el extremo `cual` (0 = inferior, 1 = superior) del IC impreso en el informe."""
    crudo = linea.split("IC95")[1].strip().strip("[]").split(",")[cual]
    return float(crudo.strip().strip("[]"))


def veredicto(dif, margen):
    lo, hi = bootstrap_ic(dif)
    media = statistics.fmean(dif)
    if media > margen and lo > 0:
        v = "CONFIRMA"
    elif hi < margen:
        v = "FALSA"
    else:
        v = "NO CONCLUYENTE"
    return v, media, lo, hi


def por_defecto():
    """`resultados/E1` del repo, no del directorio desde el que se invoque el script."""
    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(raiz, "resultados", "E1")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else por_defecto()

    c3 = acc1(d, "mix22", CARGA_EVAL)
    c2 = acc1(d, "delta", CARGA_EVAL)
    dif = [a - b for a, b in zip(c3, c2)]

    # R11 se instancia desde la C2 convergida (Anexo A/b), no desde el snapshot
    margen = max(PISO_MARGEN, 1.5 * statistics.stdev(c2))
    v, media, lo, hi = veredicto(dif, margen)

    print("=== PS-1 recomputada sin el pipeline ===")
    print(f"  carga de evaluación L{CARGA_EVAL} · margen efectivo R11 = {margen:.4f}")
    print(f"  C3 (mix22) por semilla: {[f'{x:.4f}' for x in c3]}")
    print(f"  C2 (delta) por semilla: {[f'{x:.4f}' for x in c2]}")
    print(f"  dif apareada          : {[f'{x:+.4f}' for x in dif]}")
    print(f"  dif media = {media:+.4f} · IC95 [{lo:+.4f}, {hi:+.4f}] · peor semilla {min(dif):+.4f}")
    print(f"  VEREDICTO: {v}  ({media / margen:.1f}× el margen)")

    # ---- contraste con el informe oficial
    #
    # La MEDIA no depende del generador aleatorio: tiene que coincidir exacto. El IC sí depende
    # —dos bootstraps con RNG distintos remuestrean distinto—, así que exigirle coincidencia
    # dígito a dígito sería un test mal especificado. Se calibra el ruido de Monte Carlo corriendo
    # el bootstrap con varias semillas y se compara contra esa dispersión.
    ruta_inf = os.path.join(d, "E1_informe.md")
    if os.path.exists(ruta_inf):
        with open(ruta_inf, encoding="utf-8") as fh:
            linea = next((l for l in fh if "primaria (N_common)" in l), "")
        print("\n=== contraste con el informe del pipeline ===")
        print(f"  informe : {linea.strip()}")

        global SEED_BOOT
        original, los, his = SEED_BOOT, [], []
        for k in range(12):
            SEED_BOOT = original + 1000 * (k + 1)
            a, b = bootstrap_ic(dif)
            los.append(a)
            his.append(b)
        SEED_BOOT = original
        mc = max(statistics.stdev(los), statistics.stdev(his))

        media_ok = f"{media:+.4f}" in linea
        tol = 3 * mc
        ic_ok = all(abs(x - y) <= tol
                    for x, y in ((lo, buscar_ic(linea, 0)), (hi, buscar_ic(linea, 1))))
        print(f"  ruido de Monte Carlo del IC (SD sobre 12 semillas de bootstrap) = {mc:.2e}")
        print(f"  media coincide exacto : {'✓' if media_ok else '✗ DISCREPA — revisar el pipeline'}")
        print(f"  IC dentro de 3·MC ({tol:.1e}) : "
              f"{'✓' if ic_ok else '✗ DISCREPA — no es ruido de bootstrap'}")

    # ---- (b) flanco del sustraendo: ¿el efecto depende de las semillas sin converger?
    conv = []
    for s in range(8):
        with open(os.path.join(d, f"e1_delta_seed{s}.json"), encoding="utf-8") as fh:
            conv.append(json.load(fh)["converged"])
    idx = [i for i in range(8) if conv[i]]
    print("\n=== robustez al sustraendo (delta sin converger en el tope) ===")
    print(f"  convergidas: {idx} · sin converger: {[i for i in range(8) if not conv[i]]}")
    dif_c = [dif[i] for i in idx]
    vc, mc, loc, hic = veredicto(dif_c, margen)
    print(f"  solo convergentes (n={len(idx)}): dif = {mc:+.4f} · IC95 [{loc:+.4f}, {hic:+.4f}] "
          f"· peor {min(dif_c):+.4f} → {vc}")
    print(f"  {'✓ el veredicto NO depende de las no convergidas' if vc == v else '✗ el veredicto CAMBIA'}")

    # ---- (b) defensa por consecuencia de B1-ter: ¿cuánto tiene que caer C3 para tumbar PS-1?
    print("\n=== sensibilidad: caída de mix22 necesaria para cambiar el veredicto ===")
    caida = 0.0
    while caida < 0.15:
        vv, _, _, _ = veredicto([x - caida for x in dif], margen)
        if vv != "CONFIRMA":
            print(f"  el veredicto deja de CONFIRMAR con una caída de {caida:.4f}")
            print(f"  B1-ter dispara con {PISO_MARGEN:.4f} → el fusible es {caida / PISO_MARGEN:.2f}× "
                  f"más sensible que lo que protege")
            break
        caida += 0.0001

    # ---- (d1) censura por techo: ¿la grilla encontró el borde de mix22?
    print("\n=== censura por techo (d1) ===")
    for cond in ("mix22", "softmax"):
        peor = min(min(acc1(d, cond, L)) for L in (8, 16, 32, 64, 96, 128))
        print(f"  {cond:8s} peor celda de toda la grilla = {peor:.6f} (déficit {1 - peor:.2e})")
    print("  → si ninguna celda cae del techo, la capacidad de C3 está CENSURADA, no medida:")
    print("    PS-1 acota por abajo el rescate; no mide dónde está el límite de C3.")


if __name__ == "__main__":
    main()
