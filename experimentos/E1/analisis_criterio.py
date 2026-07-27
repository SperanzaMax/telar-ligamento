#!/usr/bin/env python3
"""Análisis del criterio de convergencia D-004 (sin GPU, sobre los val_hist ya guardados).

Claim a sostener: el criterio `converged()` compara DOS mediciones puntuales con una
tolerancia (0.5 pts) menor que el ruido de la propia métrica, por lo que en régimen de
plateau la declaración de convergencia es esencialmente azarosa; y como ese criterio fija
N_common, el ruido se propaga al PRESUPUESTO DE ENTRENAMIENTO comparado entre arquitecturas.

Todo lo de acá sale de datos ya en disco. No entrena nada.
"""
import json, glob, os, sys
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "..", "resultados", "E1")
EXTRA = os.environ.get("EXTRA_JSON", "")     # p.ej. delta_seed0 @10000, que no está en el repo
TOL, WINDOW = 0.005, 500                     # parámetros de D-004


def cargar():
    series = []
    for f in sorted(glob.glob(os.path.join(RES, "e1_*_seed*.json"))):
        if f.endswith("_propio.json"):
            continue
        d = json.load(open(f))
        series.append(d)
    if EXTRA and os.path.exists(EXTRA):
        series.append(json.load(open(EXTRA)))
    return series


def plateau(vh, cond):
    """Puntos en régimen de plateau. delta: desde 4000 (conservador; paso_conv_propio≈2500-3500).
    Saturadas (mix22/softmax): desde 1000 (convergen a 1000)."""
    ini = 4000 if cond == "delta" else 1000
    return np.array([h["val_acc"] for h in vh if h["step"] >= ini])


def sigma_dos_metodos(y):
    """(1) SD de residuos de regresión lineal; (2) SD de diferencias consecutivas /sqrt(2)."""
    if len(y) < 4:
        return None, None, None
    x = np.arange(len(y))
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    s_reg = resid.std(ddof=2)
    s_dif = np.diff(y).std(ddof=1) / np.sqrt(2)
    # autocorrelación lag-1 de los residuos: valida (o no) el supuesto iid
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1] if len(resid) > 3 else np.nan
    return s_reg, s_dif, r1


def phi(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


# ---------------------------------------------------------------- A. ruido
series = cargar()
print("=" * 100)
print("A. RUIDO DE val_acc EN PLATEAU  (σ_reg = SD de residuos; σ_dif = SD de Δ consecutivos /√2)")
print("=" * 100)
print(f"{'cond':>8} {'seed':>4} {'N':>6} {'n_pts':>5} {'media':>7} {'σ_reg':>8} {'σ_dif':>8} {'ac_lag1':>8} {'conv':>6}")
por_cond = {}
for d in series:
    y = plateau(d["val_hist"], d["cond"])
    s_reg, s_dif, r1 = sigma_dos_metodos(y)
    if s_reg is None:
        print(f"{d['cond']:>8} {d['seed']:>4} {d['steps']:>6} {len(y):>5}   (serie corta, se omite)")
        continue
    por_cond.setdefault(d["cond"], []).append((s_reg, s_dif, d["converged"]))
    print(f"{d['cond']:>8} {d['seed']:>4} {d['steps']:>6} {len(y):>5} {y.mean():>7.4f} "
          f"{s_reg:>8.5f} {s_dif:>8.5f} {r1:>8.3f} {str(d['converged']):>6}")

print(f"\n{'cond':>8} {'σ medio (reg)':>14} {'σ medio (dif)':>14} {'tol/σ':>8}  interpretación")
sig_cond = {}
for c, v in por_cond.items():
    sr = float(np.mean([x[0] for x in v])); sd = float(np.mean([x[1] for x in v]))
    sig_cond[c] = sr
    ratio = TOL / sr if sr > 0 else float("inf")
    interp = "tol >> ruido: criterio informativo" if ratio > 3 else \
             "tol ~ ruido: criterio DEGRADADO" if ratio > 1 else "tol < ruido: criterio AZAROSO"
    print(f"{c:>8} {sr:>14.5f} {sd:>14.5f} {ratio:>8.2f}  {interp}")

# ---------------------------------------------------------------- B. probabilidad
print("\n" + "=" * 100)
print("B. P(declarar convergencia) BAJO PLATEAU REAL")
print("   D = val[N] - val[N-500] ~ N(0, 2σ²)  →  P(D < tol) = Φ(tol / (σ√2))")
print("=" * 100)
for c, s in sig_cond.items():
    if s == 0:
        print(f"{c:>8}: σ≈0 (saturado) → P≈1.00 · 8/8 simultáneo P≈1.00")
        continue
    p = phi(TOL / (s * np.sqrt(2)))
    n = len(por_cond[c])
    obs = sum(1 for x in por_cond[c] if x[2])
    print(f"{c:>8}: σ={s:.5f} → P(1 semilla)={p:.3f} · esperadas de {n}: {p*n:.1f} · "
          f"OBSERVADAS: {obs} · P(8/8 simultáneo)={p**8:.4f}")

# ---------------------------------------------------------------- C. criterio corregido
print("\n" + "=" * 100)
print("C. CRITERIO DE VENTANA:  mean(últimas m) - mean(m previas) < tol")
print("   σ_D baja de σ√2 a σ√(2/m). Con tol calibrada al ruido medido en vez de fija.")
print("=" * 100)
print(f"{'cond':>8} {'m':>3} {'σ_D':>9} {'P(1 semilla)':>13} {'P(8/8)':>9}   con tol fija 0.005")
for c, s in sig_cond.items():
    if s == 0:
        continue
    for m in (1, 3, 5):
        sD = s * np.sqrt(2 / m)
        p = phi(TOL / sD)
        print(f"{c:>8} {m:>3} {sD:>9.5f} {p:>13.3f} {p**8:>9.4f}")
    print(f"{'':>8}    → con tol CALIBRADA = 2·σ_D (m=3): P(1)={phi(2.0):.3f} · P(8/8)={phi(2.0)**8:.3f}")

# aplicación real del criterio de ventana a las series de delta
print("\n   Aplicado a las series reales de delta (m=3, tol calibrada = 2σ_D):")
for d in series:
    if d["cond"] != "delta":
        continue
    y = plateau(d["val_hist"], "delta")
    if len(y) < 6:
        continue
    s = sig_cond["delta"]; sD = s * np.sqrt(2 / 3); tol_cal = 2 * sD
    dif = y[-3:].mean() - y[-6:-3].mean()
    print(f"      s{d['seed']} @{d['steps']}: Δmedias={dif:+.5f} vs tol_cal={tol_cal:.5f} "
          f"→ {'CONVERGE' if dif < tol_cal else 'no converge'}   (D-004 decía: {d['converged']})")

# ---------------------------------------------------------------- D/E. costo y confound
print("\n" + "=" * 100)
print("D/E. PROPAGACIÓN AL PRESUPUESTO")
print("=" * 100)
d0 = [d for d in series if d["cond"] == "delta" and d["steps"] == 10000]
if d0:
    c = d0[0]["capacity"]
    print(f"   delta s0 @10000: L96={c['96']['1']:.4f} L128={c['128']['1']:.4f} converged={d0[0]['converged']}")
d75 = [d for d in series if d["cond"] == "delta" and d["steps"] == 7500]
if d75:
    m96 = np.mean([d["capacity"]["96"]["1"] for d in d75])
    print(f"   delta 8 semillas @7500: L96 medio={m96:.4f}")
    if d0:
        print(f"   → el MISMO modelo cambia {m96:.4f} → {d0[0]['capacity']['96']['1']:.4f} "
              f"({(d0[0]['capacity']['96']['1']-m96)*100:+.2f} pts) solo por presupuesto")
print("\n   Sesgo estructural: σ(saturadas)≈0 → cierran enseguida; σ(delta) alto → va al tope.")
print("   El criterio NO es simétrico: castiga con más pasos justamente a la arquitectura")
print("   que peor resuelve la tarea, que es la variable bajo estudio.")
