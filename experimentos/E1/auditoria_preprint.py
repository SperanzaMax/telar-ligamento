#!/usr/bin/env python3
"""Auditoría independiente de los números del preprint criterio_convergencia.tex
y de la revisión de Gemini. No reutiliza analisis_criterio.py a propósito."""
import json, glob, os
import numpy as np
from math import erf, sqrt, log, exp

RES = "/home/maxi/Documentos/Nuevo Transformer/telar-ligamento/resultados/E1"
TOL, W = 0.005, 500
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))

runs = []
for f in sorted(glob.glob(os.path.join(RES, "e1_*_seed*.json"))):
    runs.append(json.load(open(f)))
delta = [d for d in runs if d["cond"] == "delta"]

print("=" * 78)
print("1. RUIDO DE MESETA — delta, meseta desde 4000 (como el paper)")
print("=" * 78)
sr_all, sd_all, r1_all, resid_all, diffs_all = [], [], [], [], []
for d in delta:
    y = np.array([h["val_acc"] for h in d["val_hist"] if h["step"] >= 4000])
    x = np.arange(len(y))
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    s_reg = resid.std(ddof=2)
    dif = np.diff(y)
    s_dif = dif.std(ddof=1) / sqrt(2)
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    sr_all.append(s_reg); sd_all.append(s_dif); r1_all.append(r1)
    resid_all.append(resid); diffs_all.extend(dif.tolist())
    print(f"  s{d['seed']}  n={len(y)}  s_resid={s_reg:.5f}  s_diff={s_dif:.5f}  "
          f"rho1={r1:+.3f}  pend={b:+.5f}/eval  conv={d['converged']}")

sr = float(np.mean(sr_all)); sd = float(np.mean(sd_all)); r1m = float(np.median(r1_all))
print(f"\n  media s_resid = {sr:.5f}   media s_diff = {sd:.5f}   mediana rho1 = {r1m:+.3f}")
print(f"  ratio s_diff/s_resid observado = {sd/sr:.4f}")
print(f"  ratio PREDICHO si rho1={r1m:+.2f} fuera real: sqrt(1-rho) = {sqrt(1-r1m):.4f}")
print(f"  -> rho1 implicado por la concordancia de estimadores: {1-(sd/sr)**2:+.3f}")

n_pl = len(resid_all[0])
print(f"\n  SESGO DE MUESTRA PEQUEÑA (n={n_pl} puntos de meseta, tras quitar tendencia lineal):")
rng = np.random.default_rng(0)
sim = []
for _ in range(20000):
    z = rng.normal(size=n_pl)              # ruido blanco PURO
    xx = np.arange(n_pl); bb, aa = np.polyfit(xx, z, 1)
    rr = z - (aa + bb * xx)
    sim.append(np.corrcoef(rr[:-1], rr[1:])[0, 1])
sim = np.array(sim)
print(f"    E[rho1_hat] bajo ruido blanco = {sim.mean():+.3f}   IC95 empírico "
      f"[{np.percentile(sim,2.5):+.3f}, {np.percentile(sim,97.5):+.3f}]")
print(f"    observado (mediana) = {r1m:+.3f}  -> percentil dentro de la nula: "
      f"{100*(sim < r1m).mean():.1f}%")

print("\n" + "=" * 78)
print("2. PROBABILIDAD DE DECLARAR CONVERGENCIA")
print("=" * 78)
p_gauss = Phi(TOL / (sr * sqrt(2)))
print(f"  (a) Gaussiana del paper, sigma=s_resid={sr:.5f}: P = Phi({TOL/(sr*sqrt(2)):.4f}) = {p_gauss:.4f}")
print(f"      P(8 semillas)      = {p_gauss**8:.4f}")
p_rho = Phi(TOL / (sr * sqrt(2 * (1 - r1m))))
print(f"  (b) Corrección de Gemini con rho={r1m:+.2f}: P = {p_rho:.4f}   P(8) = {p_rho**8:.4f}")
dif_arr = np.array(diffs_all)
p_emp = float((dif_arr < TOL).mean())
lo = p_emp - 1.96 * sqrt(p_emp * (1 - p_emp) / len(dif_arr))
hi = p_emp + 1.96 * sqrt(p_emp * (1 - p_emp) / len(dif_arr))
print(f"  (c) NO PARAMÉTRICA (fracción de diferencias sucesivas reales < tau),"
      f" N={len(dif_arr)}: P = {p_emp:.4f}  IC95 [{lo:.3f},{hi:.3f}]")
print(f"      P(8 semillas)      = {p_emp**8:.4f}")
print(f"      [el criterio D-004 con w=500 y eval cada 500 ES exactamente una diferencia sucesiva]")
obs = sum(d["converged"] for d in delta)
print(f"\n  observado: {obs}/8 convergidas")
from math import comb
pv = sum(comb(8, k) * p_gauss**k * (1-p_gauss)**(8-k) for k in range(0, obs+1))
print(f"  P(X<={obs}) bajo Binom(8, {p_gauss:.3f}) = {pv:.3f}  (predicho E[X]={8*p_gauss:.1f})")
pv_e = sum(comb(8, k) * p_emp**k * (1-p_emp)**(8-k) for k in range(0, obs+1))
print(f"  P(X<={obs}) bajo Binom(8, {p_emp:.3f}) = {pv_e:.3f}  (predicho E[X]={8*p_emp:.1f})")

print("\n" + "=" * 78)
print("3. CRITERIO PROPUESTO (Ec. 3): descomposición del efecto")
print("=" * 78)
sD_pt = sr * sqrt(2)
sD_win = sr * sqrt(2 / 3)
print(f"  sigma_D punto a punto = {sD_pt:.5f} ;  con ventana m=3 = {sD_win:.5f}")
print(f"  A) D-004 original                          tau=0.00500 / sD={sD_pt:.5f} -> P={Phi(TOL/sD_pt):.3f}  P8={Phi(TOL/sD_pt)**8:.3f}")
print(f"  B) sólo ventana m=3, MISMA tau=0.005       tau=0.00500 / sD={sD_win:.5f} -> P={Phi(TOL/sD_win):.3f}  P8={Phi(TOL/sD_win)**8:.3f}")
print(f"  C) sólo recalibrar tau=2*sD punto a punto  tau={2*sD_pt:.5f} -> P={Phi(2):.3f}  P8={Phi(2)**8:.3f}")
print(f"  D) Ec.3 completa (ventana + tau=2*sD_win)  tau={2*sD_win:.5f} -> P={Phi(2):.3f}  P8={Phi(2)**8:.3f}")
print("  -> el salto 0.038 -> 0.832 lo produce la RECALIBRACIÓN de tau, no la ventana:")
print(f"     ventana sola: {Phi(TOL/sD_win)**8:.3f} ; tau sola: {Phi(2)**8:.3f}")

print("\n" + "=" * 78)
print("4. ESPECIFICIDAD DEL CRITERIO PROPUESTO (lo que el paper no reporta)")
print("=" * 78)
print("  Bajo mejora real de delta por evaluación (pendiente d), P(parar igual) = Phi(2 - d*k/sD_win)")
print("  con k = separación entre centros de ventana = 3 evaluaciones = 1500 pasos.")
for d_pts in [0.001, 0.002, 0.004, 0.006, 0.010]:
    dd = d_pts * 3
    print(f"    mejora real {d_pts*100:.1f} pts/eval ({dd*100:.1f} pts por ventana): "
          f"P(declarar convergido) = {Phi(2 - dd/sD_win):.3f}")
mde = 2 * sD_win + 1.645 * sD_win
print(f"  Mínima mejora detectable (P(falsa parada)<=0.05): {mde*100:.2f} pts por ventana "
      f"= {mde/3*100:.2f} pts/eval = {mde/3/500*2500*100:.2f} pts por bloque de 2500 pasos")
# NOTA: 1.45 = 0.9306 - 0.9161, tomando como base la cifra REDONDEADA que cita el paper.
# La cifra publicada es 1.39 = 0.9306 - 0.9167, con la base del JSON (e1_delta_seed0.json,
# capacity.96.1 = 0.9167). La diferencia de 0.06 pts es el redondeo de la base, no una
# discrepancia de medición: ambas quedan muy por debajo del piso de 4.07 pts, así que el
# argumento del §4 no depende de cuál se use. Citar 1.39 en cualquier texto público.
print(f"  *** El efecto de presupuesto que el paper llama material es +1.45 pts en 2500 pasos")
print(f"      (1.39 con la base del JSON; ver nota en el código). ***")

print("\n" + "=" * 78)
print("5. ¿LA VAL_ACC SIGUE A LA CAPACIDAD? (delta, 8 semillas)")
print("=" * 78)
va_fin = np.array([[h["val_acc"] for h in d["val_hist"] if h["step"] == 7500][0] for d in delta])
cap96 = np.array([d["capacity"]["96"]["1"] for d in delta])
cap128 = np.array([d["capacity"]["128"]["1"] for d in delta])
pend = np.array([np.polyfit(np.arange(len([h for h in d['val_hist'] if h['step']>=4000])),
                            [h["val_acc"] for h in d["val_hist"] if h["step"]>=4000], 1)[0] for d in delta])
print(f"  val_acc@7500 : {np.round(va_fin,4)}")
print(f"  cap L96      : {np.round(cap96,4)}")
print(f"  corr(val_acc@7500, cap96)  = {np.corrcoef(va_fin, cap96)[0,1]:+.3f}")
print(f"  corr(val_acc@7500, cap128) = {np.corrcoef(va_fin, cap128)[0,1]:+.3f}")
print(f"  corr(pendiente meseta, cap96) = {np.corrcoef(pend, cap96)[0,1]:+.3f}")
print(f"  corr(converged, cap96) = {np.corrcoef([float(d['converged']) for d in delta], cap96)[0,1]:+.3f}")
print(f"  rango val_acc@7500 = {va_fin.max()-va_fin.min():.4f} ; rango cap96 = {cap96.max()-cap96.min():.4f}")

print("\n" + "=" * 78)
print("6. VERIFICACIÓN DE CIFRAS DEL MANUSCRITO")
print("=" * 78)
print(f"  paper: 'improved from 0.9161' | JSON delta s0 L96 = {delta[0]['capacity']['96']['1']:.4f}")
print(f"  paper: '4-of-9'               | en el repo hay 8 runs delta, {obs} convergidas. El 9º (s0@10000) NO está en el repo.")
sat = [d for d in runs if d["cond"] != "delta"]
ys = [np.array([h["val_acc"] for h in d["val_hist"] if h["step"] >= 1000]) for d in sat]
print(f"  saturadas: n de puntos de meseta por run = {[len(y) for y in ys][:5]} ... "
      f"(el paper estima sigma con {len(ys[0])} puntos)")
allv = np.concatenate(ys)
print(f"  val_acc únicos en meseta saturada: {np.unique(allv)[:6]} ... -> sigma "
      f"= {allv.std(ddof=1):.6f}")
