"""Tests de la lógica de veredictos de E1 (prereg de seguimiento v1.1). Sin JAX ni GPU.

Cada test construye el caso que DEBE disparar cada veredicto, incluidos los terceros estados.
Uso: python3 experimentos/E1/test_analisis_e1.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analisis_e1 import (  # noqa: E402
    margen_efectivo, paso_convergencia_propio, veredicto_ps1, veredicto_ps1_tabla,
    l0_por_semilla, veredicto_ps4_inicio, veredicto_ps4_pendiente, veredicto_ps4_monotonia,
    elegir_carga_t2, veredicto_ps5, pearson_parcial,
)

LOADS = [8, 16, 32, 64, 96, 128]
fallos = []


def check(nombre, cond, extra=""):
    print(f"  {'✓' if cond else '✗'} {nombre}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        fallos.append(nombre)


print("=== R11 / margen efectivo (D1: sin √2) ===")
check("margen = piso cuando SD es chica", abs(margen_efectivo(0.006) - 0.02) < 1e-12)
check("margen = 1.5·SD reproduce S0.9 L96 (SD .0159 → .0238)",
      abs(margen_efectivo(0.0159) - 0.02385) < 1e-9, f"{margen_efectivo(0.0159):.5f}")
check("margen = 1.5·SD reproduce S0.9 L128 (SD .0219 → .0328)",
      abs(margen_efectivo(0.0219) - 0.03285) < 1e-9, f"{margen_efectivo(0.0219):.5f}")

print("\n=== paso de convergencia propio (O3) ===")
vh = [{"step": 500, "val_acc": .50}, {"step": 1000, "val_acc": .70}, {"step": 1500, "val_acc": .80},
      {"step": 2000, "val_acc": .803}, {"step": 2500, "val_acc": .81}]
check("detecta el primer paso con mejora < 0.5 pts", paso_convergencia_propio(vh) == 2000,
      paso_convergencia_propio(vh))
vh_no = [{"step": 500, "val_acc": .30}, {"step": 1000, "val_acc": .50}, {"step": 1500, "val_acc": .70}]
check("None si nunca converge", paso_convergencia_propio(vh_no) is None)

print("\n=== PS-1 · veredictos por tabla ===")
rng = np.random.default_rng(0)
c2 = 0.77 + rng.normal(0, .02, 8)
v, d, ic = veredicto_ps1_tabla(c2 + 0.10, c2, margen=0.0328)     # rescate grande y consistente
check("confirma con rescate de 10 pts", v == "confirma", f"dif={d:+.3f} IC={ic[0]:+.3f},{ic[1]:+.3f}")
v, d, ic = veredicto_ps1_tabla(c2 - 0.05, c2, margen=0.0328)     # C3 peor que C2
check("falsa con C3 < C2", v == "falsa", f"dif={d:+.3f}")
# rescate centrado JUSTO en el margen y con dispersión grande: el IC tiene que cruzarlo
disperso = np.array([.10, -.04, .09, -.03, .08, -.02, .07, -.02])   # media ≈ .0288, SD alta
v, d, ic = veredicto_ps1_tabla(c2 + disperso, c2, margen=0.0328)
check("no concluyente cuando el IC cruza el margen", v == "no concluyente",
      f"dif={d:+.3f} IC={ic[0]:+.3f},{ic[1]:+.3f}")

print("\n=== PS-1 · regla de discordancia (Anexo B3) ===")
r = veredicto_ps1({"c3": c2 + 0.10, "c2": c2}, {"c3": c2 + 0.10, "c2": c2}, margen=0.0328)
check("concordantes → se propaga el veredicto", r["veredicto"] == "confirma" and r["concordantes"])
r = veredicto_ps1({"c3": c2 + 0.10, "c2": c2}, {"c3": c2 - 0.05, "c2": c2}, margen=0.0328)
check("discordantes → «no concluyente por sensibilidad al presupuesto»",
      r["veredicto"] == "no concluyente por sensibilidad al presupuesto", r["veredicto"])
check("nunca elige la tabla que conviene (primaria confirmaba)", r["primaria"]["veredicto"] == "confirma")

print("\n=== PS-4(i) · mediana de L₀ con umbral 0.99 ===")
# curva del snapshot: cae bajo 0.99 recién en L64 → L₀ = 64 en todas
snap = np.array([[1.0, 1.0, .999, .974, .887, .773]] * 8)
r = veredicto_ps4_inicio(snap, LOADS)
check("confirma con la curva del snapshot (mediana=64)", r["veredicto"] == "confirma" and r["mediana_L0"] == 64,
      r["mediana_L0"])
# una semilla parpadea en L32 (0.989): la mediana NO se mueve — es lo que O2 quería blindar
parpadeo = snap.copy(); parpadeo[0, 2] = .989
r = veredicto_ps4_inicio(parpadeo, LOADS)
check("el parpadeo de UNA semilla en L32 no mueve la mediana", r["veredicto"] == "confirma",
      f"L0={r['L0_por_semilla']}")
# la convergencia cura L64 en todas → L₀ salta a 96: falsable hacia arriba
curada = snap.copy(); curada[:, 3] = .995
r = veredicto_ps4_inicio(curada, LOADS)
check("falsa hacia arriba si la convergencia cura L64 (mediana=96)",
      r["veredicto"] == "falsa" and r["mediana_L0"] == 96, r["mediana_L0"])
# empate 4-4 → tercer estado
empate = snap.copy(); empate[:4, 3] = .995
r = veredicto_ps4_inicio(empate, LOADS)
check("empate 4–4 → no concluyente por dispersión",
      r["veredicto"].startswith("no concluyente"), f"mediana={r['mediana_L0']}")
# semilla que nunca baja del umbral → centinela conservador (empuja a falsar)
nunca = np.ones((8, 6))
r = veredicto_ps4_inicio(nunca, LOADS)
check("semillas que nunca bajan del umbral → falsa (lectura conservadora)", r["veredicto"] == "falsa")
check("l0_por_semilla marca None cuando no hay cruce", l0_por_semilla(nunca, LOADS)[0] is None)

print("\n=== PS-4(ii)/(iii) ===")
acel = np.array([[1.0, 1.0, .999, .974, .887, .773]] * 8) + rng.normal(0, .005, (8, 6))
r = veredicto_ps4_pendiente(acel, LOADS)
check("confirma la caída acelerada del snapshot (.114 > .086)", r["veredicto"] == "confirma",
      f"aceleración={r['aceleracion']:+.3f}")
lineal = np.array([[1.0, 1.0, .95, .90, .85, .80]] * 8) + rng.normal(0, .003, (8, 6))
r = veredicto_ps4_pendiente(lineal, LOADS)
check("no confirma con caída lineal", r["veredicto"] != "confirma", r["veredicto"])
r = veredicto_ps4_monotonia(acel, LOADS, desde=64)
check("monotonía decreciente en el tramo alto", r["veredicto"] == "confirma", f"rho={r['rho_medio']:.2f}")

print("\n=== Anexo C · selección de carga de T2 (O4) ===")
r = elegir_carga_t2(t2_evalL=[.85, .88, .90, .87, .92, .86, .89, .91], t2_l32=[.89] * 8, carga_eval=96)
check("misma-carga es primaria si no es degenerada", r["primaria"] == "misma_carga" and not r["degenerada"])
r = elegir_carga_t2(t2_evalL=[.02, .03, .02, .03, .02, .03, .02, .03], t2_l32=[.89] * 8, carga_eval=128)
check("fallback a L32 si T2 quedó pisada contra el suelo", r["primaria"] == "L32" and r["degenerada"],
      f"media={r['media']:.3f}")
check("el fallback declara la condición cross-carga", "CROSS-CARGA" in r["nota"])
r = elegir_carga_t2(t2_evalL=[.85] * 8, t2_l32=[.89] * 8, carga_eval=96)
check("fallback también si la SD es degenerada (sin varianza no hay correlación)", r["primaria"] == "L32")

print("\n=== PS-5 · trade-off vs. hipótesis rival (O3) ===")
cap = np.array([.755, .757, .790, .788, .770, .765, .785, .760])
t2_anti = 1.6 - cap + np.array([.001, -.001, .002, -.002, .001, -.001, .002, -.002])  # anticorrelación fuerte
paso_plano = [3000] * 8
r = veredicto_ps5(cap, t2_anti, paso_plano)
check("confirma con anticorrelación real y paso de parada constante", r["veredicto"] == "confirma",
      f"crudo={r['pearson_crudo']:+.2f} parcial={r['pearson_parcial']:+.2f}")
t2_pos = cap + np.array([.001, -.001, .002, -.002, .001, -.001, .002, -.002])
r = veredicto_ps5(cap, t2_pos, paso_plano)
check("falsa con correlación positiva fuerte", r["veredicto"] == "falsa", f"crudo={r['pearson_crudo']:+.2f}")
r = veredicto_ps5(cap, rng.normal(.89, .04, 8), paso_plano)
check("no concluyente con ruido (IC cruza cero)", r["veredicto"] == "no concluyente",
      f"crudo={r['pearson_crudo']:+.2f}")
# hipótesis rival CANÓNICA: capacidad y T2 dependen del paso de parada con signos opuestos.
# La parcial NO invierte el signo: se ATENÚA hacia cero. Sin la banda O5 esto se leería «confirma».
# Construcción determinista: residuos ORTOGONALIZADOS en la muestra, para fijar la correlación
# residual en −0.3 exacto (con n=8 el ruido muestral basta para darla vuelta si se deja al azar).
def _ortogonalizar(v, *contra):
    v = v - v.mean()
    for c in contra:
        c = c - c.mean()
        v = v - (v @ c) / (c @ c) * c
    return v / np.linalg.norm(v)

rng_o5 = np.random.default_rng(12345)
paso = np.array([2500., 3000, 3500, 4000, 4500, 5000, 5500, 6000])
u = _ortogonalizar(rng_o5.normal(0, 1, 8), paso)
w = _ortogonalizar(rng_o5.normal(0, 1, 8), paso, u)
cap_r = 0.70 + 0.00002 * paso + .004 * u
t2_r = 1.00 - 0.00002 * paso - .004 * (0.3 * u + 0.95 * w)   # corr residual = −0.30
r = veredicto_ps5(cap_r, t2_r, paso)
check("detecta el confound canónico por ATENUACIÓN (O5)",
      r["veredicto"].startswith("confundida por régimen de parada"),
      f"crudo={r['pearson_crudo']:+.2f} parcial={r['pearson_parcial']:+.2f} "
      f"retención={r['retencion_parcial']:.2f}")
check("y lo etiqueta como atenuación, no como inversión", "atenuación" in r["veredicto"], r["veredicto"])
# variante con INVERSIÓN: el residuo (una vez quitado el paso) correlaciona POSITIVO
e = rng.normal(0, .004, 8)
cap_i = 0.70 + 0.00002 * paso + e
t2_i = 1.00 - 0.00002 * paso + e          # mismo residuo → parcial positiva, cruda negativa
r = veredicto_ps5(cap_i, t2_i, paso)
check("detecta el confound por INVERSIÓN de signo",
      r["veredicto"] == "confundida por régimen de parada (inversión)",
      f"crudo={r['pearson_crudo']:+.2f} parcial={r['pearson_parcial']:+.2f}")
check("el diagnóstico expone ambas correlaciones con el paso",
      r["diagnostico_paso_parada"]["cap_vs_paso"] > 0 > r["diagnostico_paso_parada"]["t2_vs_paso"],
      r["diagnostico_paso_parada"])
r = veredicto_ps5(cap, t2_anti, [None] * 8)
check("tolera semillas que nunca convergieron", np.isnan(r["pearson_parcial"]))

print("\n=== correlación parcial ===")
z = np.arange(8.)
x = z + rng.normal(0, .01, 8)
y = -z + rng.normal(0, .01, 8)
check("la parcial mata una correlación enteramente explicada por el control",
      abs(pearson_parcial(x, y, z)) < 0.9, f"{pearson_parcial(x, y, z):+.2f}")

print("\n" + ("=" * 60))
if fallos:
    print(f"✗ {len(fallos)} FALLO(S): " + "; ".join(fallos))
    sys.exit(1)
print("✓ TODOS LOS TESTS PASAN — la lógica de veredictos del prereg v1.1 está verificada.")
