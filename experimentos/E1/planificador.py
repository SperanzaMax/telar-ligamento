"""Planificador de campaña por sesiones (Colab free corta a las ~4 h).

Divide E1 en UNIDADES atómicas de trabajo — (condición, semilla, bloque de 2500 pasos) — que caben
holgadamente en una sesión y terminan siempre escribiendo checkpoint en Drive. Cada sesión ejecuta
tantas unidades como entren en su presupuesto de tiempo y corta limpio; la siguiente retoma leyendo
el estado desde Drive. El orden de ejecución no afecta los resultados (cada semilla es determinista
e independiente), así que fraccionar la campaña no toca nada del pre-registro.

Fases (Anexo B del prereg v1.1):
  A) cada condición avanza por bloques de +2500 hasta que sus 8 semillas convergen (o tope 10 000).
     Al cerrar, se congelan sus métricas como tabla SECUNDARIA (`_propio.json`).
  B) con todas las condiciones cerradas: N_common = máx(N_final) y todas se extienden hasta ahí
     (tabla PRIMARIA, la del veredicto).

Sin dependencia de JAX: el runner lo importa para decidir, y este módulo solo lee JSONs y calcula.
"""
import json
import os

BLOCKS = [2500, 5000, 7500, 10000]
SEG_EVAL = 120.0          # costo estimado de la evaluación (capacidad 6 cargas + T2 3 cargas), conservador
SPASO_FALLBACK = 1.05     # s/paso si todavía no hay medición propia (el peor del benchmark en T4)


# ------------------------------------------------------------------ estado

def leer_estado(results_dir, conds, n_seeds):
    """Estado de la campaña desde los JSON de Drive.

    Devuelve {cond: {"semillas": {seed: {"steps", "converged"}}, "faseA_cerrada": bool, "N_final": int|None}}
    """
    est = {}
    for c in conds:
        semillas, propios = {}, 0
        for s in range(n_seeds):
            p = os.path.join(results_dir, f"e1_{c}_seed{s}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                semillas[s] = {"steps": d["steps"], "converged": bool(d.get("converged"))}
            if os.path.exists(os.path.join(results_dir, f"e1_{c}_seed{s}_propio.json")):
                propios += 1
        cerrada = propios == n_seeds and n_seeds > 0
        n_final = None
        if cerrada:
            p0 = os.path.join(results_dir, f"e1_{c}_seed0_propio.json")
            n_final = json.load(open(p0))["steps"] if os.path.exists(p0) else None
        est[c] = {"semillas": semillas, "faseA_cerrada": cerrada, "N_final": n_final}
    return est


def _pasos(est_cond, seed):
    return est_cond["semillas"].get(seed, {}).get("steps", 0)


def _convergio(est_cond, seed):
    return est_cond["semillas"].get(seed, {}).get("converged", False)


# ------------------------------------------------------------------ plan

def plan(estado, conds, n_seeds):
    """Lista ORDENADA de acciones pendientes hasta terminar la campaña.

    Acciones: ("entrenar", cond, seed, target_steps) | ("cerrar_faseA", cond, N_final) |
              ("informe", None, None, None)
    """
    acciones = []

    # ---- Fase A: condición por condición, bloque por bloque
    for c in conds:
        ec = estado[c]
        if ec["faseA_cerrada"]:
            continue
        for B in BLOCKS:
            faltan = [s for s in range(n_seeds) if _pasos(ec, s) < B]
            if faltan:
                # bloque en curso: completar las semillas rezagadas (p. ej. sesión cortada a mitad)
                acciones += [("entrenar", c, s, B) for s in faltan]
                break                      # el resto del plan de esta cond depende de estos resultados
            if any(_pasos(ec, s) > B for s in range(n_seeds)):
                continue                   # este bloque ya quedó atrás: mirar el siguiente
            if all(_convergio(ec, s) for s in range(n_seeds)):
                acciones.append(("cerrar_faseA", c, None, B))
                break
        else:
            acciones.append(("cerrar_faseA", c, None, BLOCKS[-1]))   # tope duro alcanzado

    if any(a[0] == "entrenar" for a in acciones):
        return acciones                    # todavía en fase A: N_common no se puede fijar

    # ---- Fase B: todas las condiciones cerradas → extender hasta N_common
    # OJO: las condiciones que cierran su fase A en ESTA misma pasada todavía no tienen N_final en
    # disco; su valor sale de la acción `cerrar_faseA` recién encolada. Sin esto, un fallback al tope
    # inflaba N_common y mandaba a TODAS las condiciones a 10 000 sin necesidad.
    n_por_cond = {c: estado[c]["N_final"] for c in conds if estado[c]["N_final"] is not None}
    for tipo, c, _, N in acciones:
        if tipo == "cerrar_faseA":
            n_por_cond[c] = N
    faltantes = [c for c in conds if c not in n_por_cond]
    if faltantes:                      # estado inconsistente: conservador, no se asume convergencia
        n_por_cond.update({c: BLOCKS[-1] for c in faltantes})
    n_common = max(n_por_cond.values())
    for c in conds:
        for s in range(n_seeds):
            hechos = _pasos(estado[c], s)
            while hechos < n_common:                 # fraccionado en bloques de 2500: unidades atómicas
                hechos = min(hechos + 2500, n_common)
                acciones.append(("entrenar", c, s, hechos))
    acciones.append(("informe", None, None, None))
    return acciones


# ------------------------------------------------------------------ costos

def costos_medidos(results_dir, conds):
    """s/paso observado por condición, a partir de las corridas ya hechas (se auto-calibra)."""
    import glob
    out = {}
    for c in conds:
        muestras = []
        for f in glob.glob(os.path.join(results_dir, f"e1_{c}_seed*.json")):
            if f.endswith("_propio.json"):
                continue
            d = json.load(open(f))
            pasos = d.get("pasos_en_corrida")          # pasos realmente entrenados en esa corrida
            if pasos and d.get("wall_s", 0) > SEG_EVAL:
                muestras.append((d["wall_s"] - SEG_EVAL) / pasos)
        if muestras:
            out[c] = sum(muestras) / len(muestras)
    return out


def estimar_seg(accion, costos):
    """Segundos estimados de una acción de entrenamiento (bloque de hasta 2500 pasos + evaluación)."""
    if accion[0] != "entrenar":
        return 5.0
    _, cond, _, target = accion
    return 2500 * costos.get(cond, SPASO_FALLBACK) + SEG_EVAL


def sesion(acciones, presupuesto_seg, costos):
    """Recorta el plan a lo que entra en el presupuesto. Devuelve (a_ejecutar, restantes, seg_estimados)."""
    usado, corte = 0.0, 0
    for i, a in enumerate(acciones):
        if a[0] == "informe":
            corte = i + 1
            break
        seg = estimar_seg(a, costos)
        if usado + seg > presupuesto_seg and corte > 0:
            break
        usado += seg
        corte = i + 1
    return acciones[:corte], acciones[corte:], usado


# ------------------------------------------------------------------ reporte de avance

def fmt(seg):
    h, m = divmod(int(seg + 0.5), 3600)
    return f"{h}h{m // 60:02d}m" if h else f"{m // 60}m{m % 60:02d}s"


def resumen(estado, conds, n_seeds, costos, presupuesto_seg):
    """Texto de avance: qué está hecho, qué falta y cuántas sesiones más."""
    acciones = plan(estado, conds, n_seeds)
    pend = [a for a in acciones if a[0] == "entrenar"]
    total_seg = sum(estimar_seg(a, costos) for a in pend)
    L = ["=== Avance de la campaña E1 ==="]
    for c in conds:
        ec = estado[c]
        pasos = [_pasos(ec, s) for s in range(n_seeds)]
        conv = sum(_convergio(ec, s) for s in range(n_seeds))
        etiqueta = (f"fase A CERRADA (N_final={ec['N_final']})" if ec["faseA_cerrada"]
                    else f"en curso · {conv}/{n_seeds} convergidas")
        L.append(f"  {c:<8} pasos por semilla: {pasos} · {etiqueta}")
    fase = "B (extendiendo a N_common)" if all(estado[c]["faseA_cerrada"] for c in conds) else "A"
    L += [f"  fase actual: {fase}",
          f"  unidades pendientes: {len(pend)} · cómputo restante ≈ {fmt(total_seg)}",
          f"  sesiones de {fmt(presupuesto_seg)} restantes ≈ "
          f"{int(total_seg / presupuesto_seg) + (1 if total_seg % presupuesto_seg else 0)}"]
    if not pend and acciones and acciones[-1][0] == "informe":
        L.append("  ✓ CAMPAÑA COMPLETA — falta solo generar el informe.")
    return "\n".join(L)
