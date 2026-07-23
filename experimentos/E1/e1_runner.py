"""E1 — cabezas mixtas softmax+delta (§6 del protocolo v1.0). Runner de campaña.

Condiciones: C1=softmax, C2=delta, C3=mix22 (2softmax+2delta), C4=mix31/mix13 (exploratorias).
Tareas: T1 (capacidad, 6 cargas) + T2 (correctabilidad). 8 semillas.

Convergencia (decisión de Maxi 2026-07-22, misma para todas las condiciones):
  acc@1 en {L96, L128} evaluada cada 500 pasos; converge cuando la mejora en la ventana de 500 < 0.5 pts;
  bloques uniformes de +2500, tope 10 000. Las 8 semillas de cada condición bajo el mismo régimen.

Prereg de seguimiento v1.1 (congelado 2026-07-23) — Anexos B y C:
  B) DOBLE REPORTE. Pasada 1: cada condición hasta su convergencia colectiva (tabla SECUNDARIA).
     N_common = máximo de esos N_final. Pasada 2: todas las condiciones se extienden hasta N_common
     (tabla PRIMARIA, la que da el veredicto). Si las dos tablas discrepan → PS-1 «no concluyente por
     sensibilidad al presupuesto».
  C) T2 se mide en L32 Y en las cargas altas, para poder usar la versión misma-carga en PS-5 (con el
     fallback pre-registrado si queda pisada contra el suelo).

Márgenes R11 y carga de evaluación se instancian desde la **C2 convergida** (no del snapshot S0.9 @2500).
Cierres de rigor heredados (matmul highest, XLA determinista, checkpoints, sync).

Uso: RESULTS_DIR=/content/drive/MyDrive/ligamento_e1 N_SEEDS=8 python e1_runner.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
import sys, json, time, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
from entrenar import train_resumable, converged, eval_capacity, eval_overwrite
from modelos import count_params
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fase0"))
from fase0_s09 import device_info, notify
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analisis_e1 as an

CONDS   = os.environ.get("CONDS", "delta,softmax,mix22,mix31,mix13").split(",")
LOADS   = [8, 16, 32, 64, 96, 128]
VAL_LOADS = (96, 128)                 # criterio de convergencia sobre las cargas del veredicto (Maxi)
N_SEEDS = int(os.environ.get("N_SEEDS", 8))
BLOCKS  = [2500, 5000, 7500, 10000]   # convergencia uniforme, tope duro 10k
MAXLOAD = 128
T2_LOADS = [32, 96, 128]              # Anexo C: T2 en L32 y en las cargas candidatas a evaluación
LR      = 3e-3
BASE    = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(BASE, "resultados", "E1"))
os.makedirs(RESULTS, exist_ok=True)

ruta = lambda n: os.path.join(RESULTS, n)


def evaluar_y_guardar(params, cond, seed, N, vh, t0, sufijo=""):
    """Evalúa capacidad (T1) + correctabilidad (T2, varias cargas) y persiste el JSON de la semilla."""
    cap = eval_capacity(params, cond, loads=LOADS, seed=1000 + seed, reps=4)
    t2 = {str(L): eval_overwrite(params, cond, L=L, seed=2000 + seed) for L in T2_LOADS}
    cv = converged(vh, N)
    out = {"exp": "E1", "cond": cond, "seed": seed, "steps": N, "converged": bool(cv),
           "paso_conv_propio": an.paso_convergencia_propio(vh),
           "params": count_params(params), "device": device_info(),
           "capacity": {str(L): cap[L] for L in LOADS}, "T2": t2,
           "val_hist": vh, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(ruta(f"e1_{cond}_seed{seed}{sufijo}.json"), "w"), indent=1)
    print(f"[E1] {cond} s{seed} @{N}{sufijo} conv={cv} L96={cap[96][1]:.3f} L128={cap[128][1]:.3f}", flush=True)
    return bool(cv)


def entrenar_hasta(cond, N, sufijo=""):
    """Entrena las N_SEEDS semillas de `cond` hasta N pasos (reanudable). Devuelve flags de convergencia."""
    flags = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        params, vh = train_resumable(cond, seed, N, ruta(f"e1_{cond}_seed{seed}.ckpt"),
                                     max_load=MAXLOAD, lr=LR, val_loads=VAL_LOADS)
        flags.append(evaluar_y_guardar(params, cond, seed, N, vh, t0, sufijo))
    return flags


def run_condition(cond):
    """Pasada 1: bloques de +2500 hasta que las 8 semillas convergen. Devuelve N_final.

    Al converger, congela las métricas de ese punto como tabla SECUNDARIA (`_propio.json`).
    """
    N_final = BLOCKS[-1]
    for N in BLOCKS:
        flags = entrenar_hasta(cond, N)
        notify(f"E1 {cond}: bloque {N} · {sum(flags)}/{N_SEEDS} convergidas")
        if all(flags):
            N_final = N
            break
    for seed in range(N_SEEDS):                       # B2: checkpoint de convergencia propia
        shutil.copyfile(ruta(f"e1_{cond}_seed{seed}.json"), ruta(f"e1_{cond}_seed{seed}_propio.json"))
    print(f"[E1] {cond}: convergencia colectiva a N={N_final}", flush=True)
    return N_final


def cargar(cond, sufijo=""):
    out = []
    for s in range(N_SEEDS):
        p = ruta(f"e1_{cond}_seed{s}{sufijo}.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def _acc1(runs, L):
    return np.array([r["capacity"][str(L)]["1"] for r in runs])


def _t2(runs, L):
    return np.array([r["T2"][str(L)] for r in runs])


def tabla_md(data, titulo):
    L = [f"### {titulo}", "", "| cond | " + " | ".join(f"L{Lc}" for Lc in LOADS) + " | T2@32 | N |",
         "|" + "---|" * (len(LOADS) + 3)]
    for c in CONDS:
        r = data.get(c) or []
        if not r:
            continue
        fila = " | ".join(f"{_acc1(r, Lc).mean():.3f}" for Lc in LOADS)
        L.append(f"| {c} | {fila} | {_t2(r, 32).mean():.3f} | {r[0]['steps']} |")
    return L + [""]


def aggregate():
    """Informe con doble tabla, veredictos y las reglas de los Anexos B y C."""
    prim = {c: cargar(c) for c in CONDS}
    sec = {c: cargar(c, "_propio") for c in CONDS}
    c2p, c2s = prim.get("delta") or [], sec.get("delta") or []
    if not c2p:
        print("[E1] sin datos de C2 (delta): no se puede instanciar la carga de evaluación", flush=True)
        return

    # (Anexo A/c) carga de evaluación desde la C2 CONVERGIDA — tabla primaria
    evalL = next((L for L in LOADS if _acc1(c2p, L).mean() < 0.95), 128)
    margen = an.margen_efectivo(_acc1(c2p, evalL).std(ddof=1))
    N_common = max(r[0]["steps"] for r in prim.values() if r)

    L = ["# E1 — informe (prereg de seguimiento v1.1)", "",
         f"**N_common = {N_common}** · **carga de evaluación (desde C2 convergida): L{evalL}** · "
         f"**margen efectivo R11 = {margen:.4f}**", "",
         "N_final por condición (convergencia colectiva propia): " +
         ", ".join(f"{c}={ (sec[c][0]['steps'] if sec.get(c) else '—') }" for c in CONDS), ""]
    L += tabla_md(prim, f"Tabla PRIMARIA — todas las condiciones a N_common = {N_common} (da el veredicto)")
    L += tabla_md(sec, "Tabla SECUNDARIA — cada condición en su propia convergencia (robustez)")

    # ---- PS-1 con la regla de discordancia (Anexo B3)
    if prim.get("mix22") and sec.get("mix22") and c2s:
        ps1 = an.veredicto_ps1({"c3": _acc1(prim["mix22"], evalL), "c2": _acc1(c2p, evalL)},
                               {"c3": _acc1(sec["mix22"], evalL), "c2": _acc1(c2s, evalL)}, margen)
        L += ["## PS-1 — rescate de capacidad (C3 vs C2)", "",
              f"- **VEREDICTO: {ps1['veredicto'].upper()}**",
              f"- primaria (N_common): {ps1['primaria']['veredicto']} · dif = {ps1['primaria']['dif']:+.4f} "
              f"· IC95 [{ps1['primaria']['ic'][0]:+.4f}, {ps1['primaria']['ic'][1]:+.4f}]",
              f"- secundaria (convergencia propia): {ps1['secundaria']['veredicto']} · "
              f"dif = {ps1['secundaria']['dif']:+.4f} "
              f"· IC95 [{ps1['secundaria']['ic'][0]:+.4f}, {ps1['secundaria']['ic'][1]:+.4f}]",
              f"- tablas {'CONCORDANTES' if ps1['concordantes'] else 'DISCORDANTES'}", ""]
        # PS-2 (descriptiva)
        c1m = _acc1(prim["softmax"], evalL).mean() if prim.get("softmax") else 1.0
        c2m, c3m = _acc1(c2p, evalL).mean(), _acc1(prim["mix22"], evalL).mean()
        f = (c3m - c2m) / (c1m - c2m) if c1m > c2m else float("nan")
        L += [f"## PS-2 — posición de C3 entre piso y techo (descriptiva)", "",
              f"- f = (C3−C2)/(C1−C2) = **{f:.3f}** (C1={c1m:.3f}, C2={c2m:.3f}, C3={c3m:.3f})", ""]

    # ---- PS-4 sobre C2 convergida (tabla primaria)
    A = np.column_stack([_acc1(c2p, Lc) for Lc in LOADS])
    p4i = an.veredicto_ps4_inicio(A, LOADS)
    p4ii = an.veredicto_ps4_monotonia(A, LOADS, desde=p4i["mediana_L0"] if p4i["mediana_L0"] in LOADS else 64)
    p4iii = an.veredicto_ps4_pendiente(A, LOADS)
    L += ["## PS-4 — forma de la degradación", "",
          f"- **(i) inicio:** {p4i['veredicto']} · mediana(L₀) = {p4i['mediana_L0']:.0f} "
          f"(esperado 64, umbral {p4i['umbral']}) · L₀ por semilla = {p4i['L0_por_semilla']}",
          f"- **(ii) monotonía:** {p4ii['veredicto']} · rho medio = {p4ii['rho_medio']:+.3f}",
          f"- **(iii) pendiente creciente:** {p4iii['veredicto']} · aceleración = "
          f"{p4iii['aceleracion']:+.4f} · IC95 [{p4iii['ic'][0]:+.4f}, {p4iii['ic'][1]:+.4f}]", ""]

    # ---- PS-5 con selección de carga (Anexo C) y control por paso de parada (O3/O5)
    sel = an.elegir_carga_t2(_t2(c2p, evalL), _t2(c2p, 32), evalL)
    p5 = an.veredicto_ps5(_acc1(c2p, evalL), np.array(sel["valores"]),
                          [r["paso_conv_propio"] for r in c2p])
    d = p5["diagnostico_paso_parada"]
    L += ["## PS-5 — anticorrelación capacidad ↔ correctabilidad", "",
          f"- **VEREDICTO: {p5['veredicto'].upper()}**",
          f"- T2 primaria: **{sel['primaria']}** (L{sel['carga']}). {sel['nota']}",
          f"- Pearson crudo = {p5['pearson_crudo']:+.3f} · IC95 "
          f"[{p5['ic_crudo'][0]:+.3f}, {p5['ic_crudo'][1]:+.3f}] · Spearman = {p5['spearman_crudo']:+.3f}",
          f"- Pearson parcial (control: paso de convergencia propio) = {p5['pearson_parcial']:+.3f} "
          f"· retención = {p5['retencion_parcial']:.2f} (umbral {an.RETENCION_MIN})",
          f"- diagnóstico: corr(capacidad, paso) = {d['cap_vs_paso']:+.3f} · "
          f"corr(T2, paso) = {d['t2_vs_paso']:+.3f}", ""]

    # ---- P1.2 / P1.3 del protocolo madre (T2 en L32, como en el madre)
    if all(prim.get(k) for k in ("softmax", "delta", "mix22")):
        t1, t2_, t3 = (_t2(prim[k], 32).mean() for k in ("softmax", "delta", "mix22"))
        lin = t3 - 0.5 * t1 - 0.5 * t2_
        L += ["## Protocolo madre", "",
              f"- **P1.1** (C3≈C1 capacidad): softmax en techo → «no evaluable por saturación» (D2).",
              f"- **P1.2** (herencia de correctabilidad): T2(C3) − ½T2(C1) − ½T2(C2) = {lin:+.4f} "
              f"({'≥0 ✓' if lin >= 0 else '<0'}).",
              f"- **P1.3** (no interferencia): T2(C3) = {t3:.3f} vs min(C1,C2) = {min(t1, t2_):.3f} "
              f"({'sin interferencia ✓' if t3 >= min(t1, t2_) else 'INTERFERENCIA'}).", ""]

    L += ["---", "*Veredictos automáticos según el prereg de seguimiento v1.1 (SHA en "
          "`FREEZE_PREREG_SEGUIMIENTO_v1.1.md`). El informe final los revisa a mano.*"]
    open(ruta("E1_informe.md"), "w").write("\n".join(L))
    print("\n".join(L), flush=True)


def main():
    print(f"=== E1 · {CONDS} · {N_SEEDS} semillas · {device_info()} ===", flush=True)
    notify(f"▶️ E1 iniciado · {CONDS} · {N_SEEDS} semillas · {device_info()['device_kind']}")

    N_finales = {}
    for cond in CONDS:                                  # pasada 1: convergencia propia
        N_finales[cond] = run_condition(cond)
    N_common = max(N_finales.values())
    print(f"[E1] N_common = {N_common} (fijado por "
          f"{[c for c, n in N_finales.items() if n == N_common]})", flush=True)
    notify(f"E1: pasada 1 lista. N_final por condición = {N_finales} → N_common = {N_common}")

    for cond, N in N_finales.items():                   # pasada 2: todas hasta N_common (Anexo B1)
        if N < N_common:
            print(f"[E1] extendiendo {cond}: {N} → {N_common}", flush=True)
            entrenar_hasta(cond, N_common)
    aggregate()
    notify("🏁 E1 COMPLETO. E1_informe.md listo (PS-1 con doble tabla, PS-2, PS-4, PS-5, P1.2/P1.3).")


if __name__ == "__main__":
    main()
