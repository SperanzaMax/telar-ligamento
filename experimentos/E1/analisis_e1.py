"""Análisis y veredictos de E1 según el prereg de seguimiento v1.1 (congelado 2026-07-23).

Lógica estadística pura (solo numpy): sin JAX, para poder testearse sin GPU. El runner la importa.

Implementa:
  - margen efectivo R11 (D1: máx(piso, 1.5·SD), sin el √2)
  - PS-1 con la regla de discordancia primaria/secundaria del Anexo B (O1)
  - PS-4 (mediana de L₀ con umbral 0.99, tres veredictos) del prereg v1.1 (O2)
  - PS-5 con Pearson crudo + parcial controlando por el paso de convergencia propio (O3)
    y la selección de carga de T2 del Anexo C (O4)
"""
import numpy as np

PISO_MARGEN = 0.02
UMBRAL_L0 = 0.99          # «salir del techo» (PS-4i); NO 0.95
C1_MEDIA_MIN = 0.20       # Anexo C1: T2 no degenerada — media entre semillas
C1_SD_MIN = 0.01          # Anexo C1: T2 no degenerada — SD entre semillas
RETENCION_MIN = 0.5       # PS-5 (O5): la parcial debe retener ≥50% de la magnitud de la cruda


# ---------------------------------------------------------------- utilidades

def margen_efectivo(sd, piso=PISO_MARGEN):
    """R11 con D1 (sin √2): máx(piso, 1.5·SD)."""
    return max(piso, 1.5 * float(sd))


def paso_convergencia_propio(val_hist, window=500, tol=0.005):
    """Primer paso en que ESA semilla cumplió el criterio (mejora < tol en la ventana).

    Devuelve None si nunca lo cumplió dentro del historial (llegó al tope sin converger).
    """
    at = {h["step"]: h["val_acc"] for h in val_hist}
    for step in sorted(at):
        previo = step - window
        if previo in at and (at[step] - at[previo]) < tol:
            return step
    return None


def bootstrap_ic(muestra, n_boot=10000, seed=20260723, alpha=0.05):
    """IC percentil de la media, remuestreando semillas (apareado: la muestra ya son diferencias)."""
    x = np.asarray(muestra, float)
    rng = np.random.default_rng(seed)
    medias = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(np.percentile(medias, 100 * alpha / 2)), float(np.percentile(medias, 100 * (1 - alpha / 2)))


def bootstrap_ic_estadistico(func, *arrays, n_boot=10000, seed=20260723, alpha=0.05):
    """IC percentil de un estadístico arbitrario, remuestreando SEMILLAS (índices comunes)."""
    arrays = [np.asarray(a, float) for a in arrays]
    n = arrays[0].size
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = func(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    from scipy.stats import rankdata
    return pearson(rankdata(x), rankdata(y))


def pearson_parcial(x, y, z):
    """Correlación parcial de x,y controlando por z (residuos de las regresiones lineales sobre z)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if z.std() == 0:
        return pearson(x, y)          # sin variación en el control, la parcial es la cruda
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return pearson(rx, ry)


# ---------------------------------------------------------------- PS-1 (O1)

def veredicto_ps1_tabla(c3_acc, c2_acc, margen):
    """PS-1 en UNA tabla. Devuelve (veredicto, dif_media, (ic_low, ic_high)).

    Diferencia apareada por semilla. Tres veredictos R3:
      confirma       — dif > margen y el IC no cruza cero por debajo
      falsa          — el IC entero queda por debajo del margen (incluye C3 ≤ C2:
                       rescate ausente, o presente pero prácticamente equivalente)
      no concluyente — el IC cruza el margen
    """
    dif = np.asarray(c3_acc, float) - np.asarray(c2_acc, float)
    lo, hi = bootstrap_ic(dif)
    media = float(dif.mean())
    if media > margen and lo > 0:
        v = "confirma"
    elif hi < margen:
        v = "falsa"
    else:
        v = "no concluyente"
    return v, media, (lo, hi)


def veredicto_ps1(primaria, secundaria, margen):
    """Regla de discordancia del Anexo B3.

    `primaria` y `secundaria` son dicts {'c3': [...], 'c2': [...]} (N_common / convergencia propia).
    El veredicto lo da la PRIMARIA; si la secundaria discrepa → «no concluyente por sensibilidad al
    presupuesto». Nunca se elige la tabla que conviene.
    """
    vp, dp, icp = veredicto_ps1_tabla(primaria["c3"], primaria["c2"], margen)
    vs, ds, ics = veredicto_ps1_tabla(secundaria["c3"], secundaria["c2"], margen)
    final = vp if vp == vs else "no concluyente por sensibilidad al presupuesto"
    return {"veredicto": final, "primaria": {"veredicto": vp, "dif": dp, "ic": icp},
            "secundaria": {"veredicto": vs, "dif": ds, "ic": ics}, "concordantes": vp == vs,
            "margen": margen}


# ---------------------------------------------------------------- PS-4 (O2)

def l0_por_semilla(acc_por_carga, loads, umbral=UMBRAL_L0):
    """L₀(s) = menor L con acc@1 < umbral, por semilla. None si la semilla nunca baja del umbral.

    `acc_por_carga`: matriz (n_semillas, n_loads).
    """
    A = np.asarray(acc_por_carga, float)
    out = []
    for fila in A:
        idx = np.nonzero(fila < umbral)[0]
        out.append(loads[idx[0]] if idx.size else None)
    return out


def veredicto_ps4_inicio(acc_por_carga, loads, esperado=64, umbral=UMBRAL_L0):
    """PS-4(i): predicción de punto sobre la mediana de L₀ (tres veredictos).

    Las semillas sin L₀ (nunca bajan del umbral) se tratan como L₀ > max(loads): se les asigna un valor
    centinela por encima de la grilla, que es la lectura conservadora (empujan la mediana hacia arriba,
    la dirección que FALSA la predicción).
    """
    l0 = l0_por_semilla(acc_por_carga, loads, umbral)
    centinela = loads[-1] * 2
    vals = np.array([centinela if v is None else v for v in l0], float)
    med = float(np.median(vals))
    if med == esperado:
        v = "confirma"
    elif med in set(float(x) for x in loads) or med >= centinela:
        v = "falsa"
    else:
        v = "no concluyente por dispersión entre semillas"
    return {"veredicto": v, "mediana_L0": med, "L0_por_semilla": l0, "umbral": umbral,
            "esperado": esperado}


def veredicto_ps4_pendiente(acc_por_carga, loads):
    """PS-4(iii): [acc(96)−acc(128)] > [acc(64)−acc(96)], IC bootstrap apareado que no cruce 0."""
    A = np.asarray(acc_por_carga, float)
    i64, i96, i128 = loads.index(64), loads.index(96), loads.index(128)
    dif = (A[:, i96] - A[:, i128]) - (A[:, i64] - A[:, i96])
    lo, hi = bootstrap_ic(dif)
    v = "confirma" if lo > 0 else ("falsa" if hi < 0 else "no concluyente")
    return {"veredicto": v, "aceleracion": float(dif.mean()), "ic": (lo, hi)}


def veredicto_ps4_monotonia(acc_por_carga, loads, desde):
    """PS-4(ii): Spearman(acc@1, L) < 0 en el tramo L ≥ `desde`, IC apareado por semilla."""
    A = np.asarray(acc_por_carga, float)
    cols = [i for i, L in enumerate(loads) if L >= desde]
    if len(cols) < 3:
        return {"veredicto": "no evaluable (tramo con menos de 3 cargas)", "rho_medio": float("nan")}
    Ls = np.array([loads[i] for i in cols], float)
    rhos = np.array([spearman(Ls, fila[cols]) for fila in A])
    lo, hi = bootstrap_ic(rhos)
    v = "confirma" if hi < 0 else ("falsa" if lo > 0 else "no concluyente")
    return {"veredicto": v, "rho_medio": float(np.nanmean(rhos)), "ic": (lo, hi), "desde_L": desde}


# ---------------------------------------------------------------- PS-5 (O3+O4)

def elegir_carga_t2(t2_evalL, t2_l32, carga_eval):
    """Anexo C: la versión misma-carga es primaria si su distribución no es degenerada.

    Decidido por la regla, NO mirando cuál correlaciona mejor.
    """
    x = np.asarray(t2_evalL, float)
    media, sd = float(x.mean()), float(x.std(ddof=1))
    ok = (media > C1_MEDIA_MIN) and (sd >= C1_SD_MIN)
    if ok:
        return {"primaria": "misma_carga", "carga": carga_eval, "valores": list(x),
                "media": media, "sd": sd, "degenerada": False,
                "nota": f"T2 medida en la carga de evaluación (L{carga_eval})."}
    return {"primaria": "L32", "carga": 32, "valores": list(np.asarray(t2_l32, float)),
            "media": media, "sd": sd, "degenerada": True,
            "nota": (f"T2 NO evaluable a L{carga_eval} (media={media:.3f}, SD={sd:.4f}; requiere "
                     f"media>{C1_MEDIA_MIN} y SD≥{C1_SD_MIN}). Fallback a L32 — condición CROSS-CARGA: "
                     f"capacidad a L{carga_eval} vs. correctabilidad a L32.")}


def veredicto_ps5(capacidad, t2, paso_conv):
    """PS-5 con la regla de veredicto del prereg v1.1 (incluye la precisión O5).

    confirma                          — cruda < 0, IC excluye el cero, y parcial del mismo signo
                                        RETENIENDO ≥ RETENCION_MIN de su magnitud
    confundida por régimen de parada  — la parcial invierte el signo (inversión) o se atenúa hacia
                                        cero (atenuación). O5: el caso canónico del confound produce
                                        atenuación, no inversión — sin esta banda, una parcial ≈ 0
                                        se leería como confirmación según el signo del ruido.
    no concluyente                    — el IC de la cruda cruza cero (potencia)
    """
    cap = np.asarray(capacidad, float)
    t2 = np.asarray(t2, float)
    crudo = pearson(cap, t2)
    lo, hi = bootstrap_ic_estadistico(pearson, cap, t2)

    # las semillas que nunca convergieron (None) se controlan con el tope alcanzado
    pc = np.array([np.nan if p is None else p for p in paso_conv], float)
    if np.isnan(pc).all():
        parcial = float("nan")
        diag = {"cap_vs_paso": float("nan"), "t2_vs_paso": float("nan")}
    else:
        pc = np.where(np.isnan(pc), np.nanmax(pc), pc)
        parcial = pearson_parcial(cap, t2, pc)
        diag = {"cap_vs_paso": pearson(cap, pc), "t2_vs_paso": pearson(t2, pc)}

    retencion = abs(parcial) / abs(crudo) if np.isfinite(parcial) and crudo != 0 else float("nan")
    if not np.isfinite(crudo):
        v = "no evaluable (varianza nula en alguna métrica)"
    elif lo <= 0 <= hi:
        v = "no concluyente"
    elif np.isfinite(parcial) and np.sign(parcial) != np.sign(crudo):
        v = "confundida por régimen de parada (inversión)"
    elif np.isfinite(retencion) and retencion < RETENCION_MIN:
        v = "confundida por régimen de parada (atenuación)"
    elif crudo < 0:
        v = "confirma"
    else:
        v = "falsa"                      # correlación positiva significativa: lo contrario del trade-off
    return {"veredicto": v, "pearson_crudo": crudo, "ic_crudo": (lo, hi), "pearson_parcial": parcial,
            "retencion_parcial": retencion, "spearman_crudo": spearman(cap, t2),
            "diagnostico_paso_parada": diag}


# ------------------------------------------------- rótulos honestos del informe (E-003′/B1-bis)

def titulo_tabla_primaria(prim, conds, n_common):
    """Encabezado de la tabla primaria, sin afirmar que todas las condiciones están a N_common.

    Con la excepción por saturación (B1-bis) una condición puede quedar legítimamente a su N real;
    el encabezado tiene que decirlo, porque la columna `N` de la tabla ya lo dice.
    """
    sin_extender = [c for c in conds if prim.get(c) and prim[c][0]["steps"] < n_common]
    t = f"Tabla PRIMARIA — N_common = {n_common} (da el veredicto)"
    return t + (f" · † NO extendidas, a su N real: {', '.join(sin_extender)}" if sin_extender
                else " · todas las condiciones a N_common")


def linea_concordancia_ps1(prim, sec, c2p, c2s, ps1):
    """Línea de concordancia de PS-1, sin afirmar que B3 corrió cuando no corrió.

    `cerrar_faseA` congela la tabla secundaria copiando la primaria. Hasta que la fase B extienda
    las condiciones, las dos tablas contienen el MISMO checkpoint y su coincidencia es tautológica:
    reportarla como «concordantes» la haría pasar por un chequeo de robustez que no se hizo.
    """
    if (prim["mix22"][0]["steps"] == sec["mix22"][0]["steps"]
            and c2p[0]["steps"] == c2s[0]["steps"]):
        return ("- tablas **no comparables**: primaria y secundaria contienen el mismo checkpoint "
                "en C3 y C2 (la fase B no corrió) → **B3 no se ejecutó**, la coincidencia no es "
                "chequeo de robustez")
    return f"- tablas {'CONCORDANTES' if ps1['concordantes'] else 'DISCORDANTES'}"


# ------------------------------------------------- B1-ter (enmienda E-003′, 2026-08-01)

UMBRAL_B1TER = 0.0200        # = piso de R11; caída media que cuenta como degradación material
CARGAS_B1TER = (96, 128)


def veredicto_b1ter(c3_prim, c3_sec, umbral=UMBRAL_B1TER, cargas=CARGAS_B1TER):
    """B1-ter: ¿`mix22` se degradó al extenderla de su fase A hasta `N_common`?

    Se declaró ANTES de correr la extensión (enmienda congelada con hash): una caída media
    apareada ≥ `umbral` en capacity@1 de cualquiera de `cargas`, entre el checkpoint de cierre de
    fase A (tabla secundaria) y el de `N_common` (tabla primaria), obliga a reportar PS-1 como
    «no concluyente por sensibilidad al presupuesto». No se relee ni se discute: se ejecuta.

    `c3_prim` / `c3_sec` son listas de runs (dicts de JSON) de la MISMA condición.
    Devuelve dict con `veredicto` ∈ {no aplica, sin degradación material, DEGRADACIÓN MATERIAL}.
    """
    if not c3_prim or not c3_sec:
        return {"veredicto": "no aplica", "motivo": "faltan datos de una de las dos tablas",
                "caidas": {}, "umbral": umbral}
    if c3_prim[0]["steps"] == c3_sec[0]["steps"]:
        return {"veredicto": "no aplica",
                "motivo": f"la extensión no corrió (ambas tablas a N={c3_prim[0]['steps']})",
                "caidas": {}, "umbral": umbral}

    caidas = {}
    for L in cargas:
        antes = np.array([r["capacity"][str(L)]["1"] for r in c3_sec], float)
        despues = np.array([r["capacity"][str(L)]["1"] for r in c3_prim], float)
        caidas[L] = float((antes - despues).mean())      # positivo = se degradó al extender
    peor = max(caidas.values())
    v = "DEGRADACIÓN MATERIAL" if peor >= umbral else "sin degradación material"
    return {"veredicto": v, "caidas": caidas, "peor": peor, "umbral": umbral,
            "N_antes": c3_sec[0]["steps"], "N_despues": c3_prim[0]["steps"]}


def lineas_b1ter(res):
    """Bloque de informe para B1-ter. El veredicto se imprime siempre, aunque no aplique."""
    L = ["## B1-ter — degradación de C3 al extender (enmienda E-003′, congelada antes de correr)", ""]
    if res["veredicto"] == "no aplica":
        return L + [f"- **NO APLICA**: {res['motivo']}.", ""]
    det = " · ".join(f"L{k}: {v:+.4f}" for k, v in sorted(res["caidas"].items()))
    L += [f"- **{res['veredicto']}** (umbral {res['umbral']:.4f}, "
          f"N {res['N_antes']} → {res['N_despues']})",
          f"- caída media apareada por carga: {det} · peor {res['peor']:+.4f}"]
    if res["veredicto"] == "DEGRADACIÓN MATERIAL":
        L += ["- ⚠️ **PS-1 pasa a «no concluyente por sensibilidad al presupuesto»** por B1-ter."]
    return L + [""]
