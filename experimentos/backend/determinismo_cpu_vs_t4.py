"""¿Cuánto ruido mete cambiar de dispositivo, comparado con cambiar la semilla?

Todas las corridas de E1 se hicieron en Tesla T4. Una prueba con seed0 (2026-07-28) mostró que
reentrenar la MISMA semilla, con los MISMOS datos y el MISMO código en CPU mueve la val_acc hasta
1.95e-2 — más que la SD entre las 8 semillas (1.105e-2). Si eso se sostiene con n=8, quiere decir
que el «media ± SD sobre N semillas» con que el campo reporta incertidumbre omite un término del
mismo orden o mayor, y que dos laboratorios con distinta GPU no pueden replicarse entre sí dentro
del margen con que se deciden las comparaciones de arquitectura.

Este script reentrena en CPU las mismas semillas que ya existen en T4 y compara trayectorias. La
referencia sale de los `val_hist` ya versionados en resultados/E1/ — no hay que volver a correr
nada en GPU.

Reanudable: entrena en bloques de 500 pasos dejando checkpoint, y saltea las semillas ya
terminadas. Si se corta la corrida se pierden como mucho ~15 min.

Uso:
    python experimentos/backend/determinismo_cpu_vs_t4.py            # las 8 semillas (~10 h en i3)
    SEEDS=0,1 python experimentos/backend/determinismo_cpu_vs_t4.py  # un subconjunto
    MODO=informe python experimentos/backend/determinismo_cpu_vs_t4.py   # solo agrega, sin entrenar
"""
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(BASE, "src"))

import numpy as np  # noqa: E402

COND = os.environ.get("COND", "delta")
TARGET = int(os.environ.get("TARGET", 2500))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7").split(",")]
MODO = os.environ.get("MODO", "correr")
BLOQUE = 500                      # checkpoint cada 500 pasos: reanudable sin perder más que eso

REF_DIR = os.path.join(BASE, "resultados", "E1")
OUT_DIR = os.path.join(BASE, "resultados", "backend")
CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

# hiperparámetros EXACTOS de e1_runner.py — cualquier diferencia acá invalida la comparación
MAXLOAD, LR, VAL_LOADS = 128, 3e-3, (96, 128)

# OJO (corregido 2026-07-29): antes acá había SIGMA_SEMILLA = 0.01105 fijo, que es la SD entre
# semillas a 7500 pasos. Pero las diferencias de backend se miden en 500-2500, y la SD entre
# semillas NO es constante: vale 0.101 en el paso 500 y recién baja a ~0.011 en 7500. Comparar
# contra la SD tardía inflaba la razón backend/semilla de 1.30x a 3.00x. Ahora se calcula la SD
# paso por paso desde las corridas de T4 y se compara cada paso contra la SD de ESE paso.
SIGMA_TARDIA = 0.01105            # SD@7500, se conserva solo como referencia declarada


def ref_hist(seed):
    """val_hist de la corrida original en T4, recortado al tramo que vamos a comparar."""
    p = os.path.join(REF_DIR, f"e1_{COND}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if d["device"]["platform"] != "gpu":
        raise SystemExit(f"{p}: la referencia no es de GPU ({d['device']}) — comparación sin sentido")
    return {h["step"]: h["val_acc"] for h in d["val_hist"] if h["step"] <= TARGET}


def salida(seed):
    return os.path.join(OUT_DIR, f"backend_{COND}_seed{seed}.json")


def correr(seed):
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    from entrenar import train_resumable

    ref = ref_hist(seed)
    if ref is None:
        print(f"  seed{seed}: SIN referencia en T4, se saltea", flush=True)
        return None

    ckpt = os.path.join(CKPT_DIR, f"backend_{COND}_seed{seed}.ckpt")
    t0, vh = time.time(), []
    for hasta in range(BLOQUE, TARGET + 1, BLOQUE):     # bloques -> checkpoint frecuente
        params, vh = train_resumable(COND, seed, hasta, ckpt, max_load=MAXLOAD, lr=LR,
                                     val_loads=VAL_LOADS)
    wall = time.time() - t0

    cpu = {h["step"]: h["val_acc"] for h in vh}
    pasos = sorted(set(ref) & set(cpu))
    difs = [abs(cpu[s] - ref[s]) for s in pasos]
    firmados = [cpu[s] - ref[s] for s in pasos]

    out = {"cond": COND, "seed": seed, "target": TARGET, "wall_s": round(wall, 1),
           "s_paso": round(wall / TARGET, 4) if wall else None,
           "device_cpu": str(jax.devices()), "pasos": pasos,
           "val_t4": [ref[s] for s in pasos], "val_cpu": [cpu[s] for s in pasos],
           "difs": difs, "difs_firmadas": firmados,
           "dif_max": max(difs), "dif_media": float(np.mean(difs))}
    json.dump(out, open(salida(seed), "w"), indent=1)
    print(f"  seed{seed}: dif_max {max(difs):.3e}  media {np.mean(difs):.3e}  "
          f"({wall/60:.1f} min)", flush=True)
    return out


def sd_por_paso():
    """SD entre las 8 semillas en T4, calculada PASO POR PASO (no una SD global)."""
    import glob
    H = {}
    for p in sorted(glob.glob(os.path.join(REF_DIR, f"e1_{COND}_seed*.json"))):
        for h in json.load(open(p))["val_hist"]:
            H.setdefault(h["step"], []).append(h["val_acc"])
    return {s: float(np.std(v, ddof=1)) for s, v in H.items() if len(v) >= 3}


def informe():
    datos = [json.load(open(salida(s))) for s in SEEDS if os.path.exists(salida(s))]
    if not datos:
        print("\nNo hay resultados todavía.")
        return
    todas = np.concatenate([d["difs"] for d in datos])
    firmadas = np.concatenate([d["difs_firmadas"] for d in datos])
    SD = sd_por_paso()
    print("\n" + "=" * 66)
    print(f"=== BACKEND vs SEMILLA · {COND} · {len(datos)}/{len(SEEDS)} semillas ===")
    print("=" * 66)
    print("  cambiar el DISPOSITIVO (T4 -> CPU, misma semilla):")
    print(f"     |dif| media = {todas.mean():.5f}   máx = {todas.max():.5f}   n = {len(todas)}")

    # comparación PAREADA: cada paso contra la SD entre semillas de ESE paso
    print("\n  pareado por paso (la SD entre semillas no es constante):")
    print(f"     {'paso':>6} {'|dif| backend':>14} {'SD semillas':>12} {'razón':>7}")
    razones = []
    for s in sorted(SD):
        difs = [abs(d["val_cpu"][d["pasos"].index(s)] - d["val_t4"][d["pasos"].index(s)])
                for d in datos if s in d["pasos"]]
        if not difs:
            continue
        m, sd = float(np.mean(difs)), SD[s]
        razones.append(m / sd)
        print(f"     {s:>6} {m:>14.5f} {sd:>12.5f} {m/sd:>6.2f}x")
    if razones:
        print(f"\n  razón backend/semilla = {np.mean(razones):.2f}x  (media de las razones por paso)")
        print(f"  [referencia: contra la SD@7500 fija ({SIGMA_TARDIA:.5f}) daría "
              f"{todas.mean()/SIGMA_TARDIA:.2f}x, pero eso compara ruido temprano con SD tardía]")
    print(f"\n  |dif| media como fracción del margen R11 (0.0200): {100*todas.mean()/0.02:.0f}%")
    print("  (ojo: R11 se aplica a comparaciones en N_common, no en el transitorio 500-2500)")

    # ¿hay sesgo sistemático del backend, o solo dispersión?
    n_neg = int((firmadas < 0).sum())
    print(f"\n  signo de (CPU - T4): {n_neg} negativos de {len(firmadas)}")
    if len(datos) >= 3:
        from scipy import stats
        por_semilla = [np.mean(d["difs_firmadas"]) for d in datos]
        t, p = stats.ttest_1samp(por_semilla, 0.0)
        print(f"  media por semilla: {np.mean(por_semilla):+.5f}  "
              f"t({len(por_semilla)-1}) = {t:.2f}  p = {p:.4f}")
        print(f"  -> {'SESGO sistemático del backend' if p < 0.05 else 'sin sesgo detectable: es dispersión'}")
    else:
        print("  (hacen falta >=3 semillas para separar sesgo de dispersión)")
    print("=" * 66)


if __name__ == "__main__":
    print(f"=== Reproducibilidad entre backends · {COND} · semillas {SEEDS} · hasta {TARGET} ===")
    if MODO != "informe":
        def hecha(s):
            """Ya está sólo si el resultado guardado llega al menos hasta TARGET."""
            p = salida(s)
            if not os.path.exists(p):
                return False
            return json.load(open(p)).get("target", 0) >= TARGET

        pend = [s for s in SEEDS if not hecha(s)]
        # el checkpoint retiene lo ya entrenado: solo se pagan los pasos que faltan
        falta = sum(TARGET - (json.load(open(salida(s))).get("target", 0)
                              if os.path.exists(salida(s)) else 0) for s in pend)
        print("referencia: corridas en Tesla T4 ya versionadas en resultados/E1/")
        print(f"pendientes: {pend or 'ninguna'}  ({falta} pasos ~{falta*2.28/3600:.1f} h en esta CPU)\n")
        for s in SEEDS:
            if hecha(s):
                print(f"  seed{s}: ya está hasta {TARGET}, se saltea", flush=True)
                continue
            correr(s)
    informe()
