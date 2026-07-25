"""Regenera E1_informe.md desde los JSON de resultados, SIN JAX (solo numpy/scipy).

Replica bit a bit la lógica JAX-free de `e1_runner.aggregate()` (tabla_md + veredictos del
prereg de seguimiento v1.1), para poder correr el informe en la PC de Maxi sobre los JSON
descargados de Drive, sin tener que abrir Colab. Los veredictos salen de `analisis_e1.py`
(el mismo módulo congelado que usa el runner).

Uso: RESULTS_DIR=resultados/E1 CONDS=delta,softmax,mix22 python regenerar_informe_local.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analisis_e1 as an

LOADS = [8, 16, 32, 64, 96, 128]
N_SEEDS = int(os.environ.get("N_SEEDS", 8))
CONDS = os.environ.get("CONDS", "delta,softmax,mix22").split(",")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "resultados", "E1"))
ruta = lambda n: os.path.join(RESULTS, n)


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
    prim = {c: cargar(c) for c in CONDS}
    sec = {c: cargar(c, "_propio") for c in CONDS}
    c2p, c2s = prim.get("delta") or [], sec.get("delta") or []
    if not c2p:
        print("[E1] sin datos de C2 (delta): no se puede instanciar la carga de evaluación", flush=True)
        return

    evalL = next((L for L in LOADS if _acc1(c2p, L).mean() < 0.95), 128)
    margen = an.margen_efectivo(_acc1(c2p, evalL).std(ddof=1))
    N_common = max(r[0]["steps"] for r in prim.values() if r)

    n_por_cond = {c: len(prim.get(c) or []) for c in CONDS}
    mixtos = {c: sorted({r["steps"] for r in (prim.get(c) or [])})
              for c in CONDS if len({r["steps"] for r in (prim.get(c) or [])}) > 1}
    L = ["# E1 — informe (prereg de seguimiento v1.1) — REGENERADO LOCAL (parcial)", "",
         f"> Regenerado sin JAX desde `resultados/E1/*.json`. Semillas presentes por condición: "
         f"{n_por_cond}. Las condiciones con <{N_SEEDS} semillas están INCOMPLETAS.", "",
         (f"> ⚠️ N HETEROGÉNEO dentro de una condición (viola N_common; PS-1/PS-5 NO válidos hasta "
          f"nivelar): {mixtos}" if mixtos else "> N homogéneo dentro de cada condición."), "",
         f"**N_common = {N_common}** · **carga de evaluación (desde C2): L{evalL}** · "
         f"**margen efectivo R11 = {margen:.4f}**", "",
         "N_final por condición (convergencia colectiva propia): " +
         ", ".join(f"{c}={ (sec[c][0]['steps'] if sec.get(c) else '—') }" for c in CONDS), ""]
    L += tabla_md(prim, f"Tabla PRIMARIA — todas las condiciones a N_common = {N_common} (da el veredicto)")
    L += tabla_md(sec, "Tabla SECUNDARIA — cada condición en su propia convergencia (robustez)")

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
        c1m = _acc1(prim["softmax"], evalL).mean() if prim.get("softmax") else 1.0
        c2m, c3m = _acc1(c2p, evalL).mean(), _acc1(prim["mix22"], evalL).mean()
        f = (c3m - c2m) / (c1m - c2m) if c1m > c2m else float("nan")
        L += [f"## PS-2 — posición de C3 entre piso y techo (descriptiva)", "",
              f"- f = (C3−C2)/(C1−C2) = **{f:.3f}** (C1={c1m:.3f}, C2={c2m:.3f}, C3={c3m:.3f})", ""]
    else:
        L += ["## PS-1 — rescate de capacidad (C3 vs C2)", "",
              "- **NO COMPUTABLE**: falta la condición mix22 (C3) y/o las tablas de convergencia propia "
              "(`*_propio.json`). Es la predicción estrella del prereg; sin mix22 no hay veredicto.", ""]

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

    if all(prim.get(k) for k in ("softmax", "delta", "mix22")):
        t1, t2_, t3 = (_t2(prim[k], 32).mean() for k in ("softmax", "delta", "mix22"))
        lin = t3 - 0.5 * t1 - 0.5 * t2_
        L += ["## Protocolo madre", "",
              f"- **P1.1** (C3≈C1 capacidad): softmax en techo → «no evaluable por saturación» (D2).",
              f"- **P1.2** (herencia de correctabilidad): T2(C3) − ½T2(C1) − ½T2(C2) = {lin:+.4f} "
              f"({'≥0 ✓' if lin >= 0 else '<0'}).",
              f"- **P1.3** (no interferencia): T2(C3) = {t3:.3f} vs min(C1,C2) = {min(t1, t2_):.3f} "
              f"({'sin interferencia ✓' if t3 >= min(t1, t2_) else 'INTERFERENCIA'}).", ""]

    L += ["---", "*Veredictos automáticos según el prereg de seguimiento v1.1. Informe PARCIAL "
          "regenerado localmente sin JAX; el definitivo lo emite la celda 9 del notebook con la "
          "campaña completa.*"]
    open(ruta("E1_informe.md"), "w").write("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    aggregate()
