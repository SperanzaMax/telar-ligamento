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
SIGMA_SEMILLA = 0.01105           # SD de val_acc@7500 entre las 8 semillas de delta en T4


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


def informe():
    datos = [json.load(open(salida(s))) for s in SEEDS if os.path.exists(salida(s))]
    if not datos:
        print("\nNo hay resultados todavía.")
        return
    todas = np.concatenate([d["difs"] for d in datos])
    firmadas = np.concatenate([d["difs_firmadas"] for d in datos])
    print("\n" + "=" * 66)
    print(f"=== BACKEND vs SEMILLA · {COND} · {len(datos)}/{len(SEEDS)} semillas ===")
    print("=" * 66)
    print("  cambiar el DISPOSITIVO (T4 -> CPU, misma semilla):")
    print(f"     |dif| media = {todas.mean():.5f}   máx = {todas.max():.5f}   n = {len(todas)}")
    print("  cambiar la SEMILLA (8 semillas en T4):")
    print(f"     SD          = {SIGMA_SEMILLA:.5f}")
    print(f"\n  razón backend/semilla = {todas.mean()/SIGMA_SEMILLA:.2f}x (media) · "
          f"{todas.max()/SIGMA_SEMILLA:.2f}x (peor)")
    print(f"  como fracción del margen R11 (0.0200): {100*todas.mean()/0.02:.0f}%")

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
        pend = [s for s in SEEDS if not os.path.exists(salida(s))]
        print("referencia: corridas en Tesla T4 ya versionadas en resultados/E1/")
        print(f"pendientes: {pend or 'ninguna'}  (~{len(pend)*TARGET*1.805/3600:.1f} h en esta CPU)\n")
        for s in SEEDS:
            if os.path.exists(salida(s)):
                print(f"  seed{s}: ya está, se saltea", flush=True)
                continue
            correr(s)
    informe()
