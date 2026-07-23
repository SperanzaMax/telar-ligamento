"""E1 — cabezas mixtas softmax+delta (§6 del protocolo v1.0). Runner de campaña.

Condiciones: C1=softmax, C2=delta, C3=mix22 (2softmax+2delta), C4=mix31/mix13 (exploratorias).
Tareas: T1 (capacidad, 6 cargas) + T2 (correctabilidad). 8 semillas.

Convergencia (decisión de Maxi 2026-07-22, misma para todas las condiciones):
  acc@1 en {L96, L128} evaluada cada 500 pasos; converge cuando la mejora en la ventana de 500 < 0.5 pts;
  bloques uniformes de +2500, tope 10 000. Las 8 semillas de cada condición bajo el mismo régimen.

Márgenes R11 y carga de evaluación del prereg C3-vs-C2 se instancian desde la **C2 convergida** (no del
snapshot S0.9 @2500). Cierres de rigor heredados (matmul highest, XLA determinista, checkpoints, sync).

Uso: RESULTS_DIR=/content/drive/MyDrive/ligamento_e1 N_SEEDS=8 python e1_runner.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
import sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
from entrenar import train_resumable, converged, eval_capacity, eval_overwrite
from modelos import count_params
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fase0"))
from fase0_s09 import device_info, notify

CONDS   = os.environ.get("CONDS", "delta,softmax,mix22,mix31,mix13").split(",")
LOADS   = [8, 16, 32, 64, 96, 128]
VAL_LOADS = (96, 128)                 # criterio de convergencia sobre las cargas del veredicto (Maxi)
N_SEEDS = int(os.environ.get("N_SEEDS", 8))
BLOCKS  = [2500, 5000, 7500, 10000]   # convergencia uniforme, tope duro 10k
MAXLOAD = 128
T2_LOAD = 32
LR      = 3e-3
PISO    = 0.02
BASE    = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(BASE, "resultados", "E1"))
os.makedirs(RESULTS, exist_ok=True)


def run_condition(cond):
    """Entrena las 8 semillas con convergencia uniforme; devuelve N_final y la lista de resultados."""
    N_final = BLOCKS[-1]
    for N in BLOCKS:
        conv_flags = []
        for seed in range(N_SEEDS):
            ckpt = os.path.join(RESULTS, f"e1_{cond}_seed{seed}.ckpt")
            t0 = time.time()
            params, vh = train_resumable(cond, seed, N, ckpt, max_load=MAXLOAD, lr=LR, val_loads=VAL_LOADS)
            cap = eval_capacity(params, cond, loads=LOADS, seed=1000 + seed, reps=4)
            t2 = eval_overwrite(params, cond, L=T2_LOAD, seed=2000 + seed)
            cv = converged(vh, N)
            conv_flags.append(bool(cv))
            out = {"exp": "E1", "cond": cond, "seed": seed, "steps": N, "converged": bool(cv),
                   "params": count_params(params), "device": device_info(),
                   "capacity": {str(L): cap[L] for L in LOADS}, "T2_L%d" % T2_LOAD: t2,
                   "val_hist": vh, "wall_s": round(time.time() - t0, 1)}
            json.dump(out, open(os.path.join(RESULTS, f"e1_{cond}_seed{seed}.json"), "w"), indent=1)
            print(f"[E1] {cond} s{seed} @{N} conv={cv} L96={cap[96][1]:.3f} L128={cap[128][1]:.3f}", flush=True)
        notify(f"E1 {cond}: bloque {N} · {sum(conv_flags)}/{N_SEEDS} convergidas")
        if all(conv_flags):
            N_final = N; break
    return N_final


def _acc1(runs, L):
    return np.array([r["capacity"][str(L)]["1"] for r in runs])


def aggregate():
    data = {c: [json.load(open(os.path.join(RESULTS, f"e1_{c}_seed{s}.json")))
                for s in range(N_SEEDS) if os.path.exists(os.path.join(RESULTS, f"e1_{c}_seed{s}.json"))]
            for c in CONDS}
    L = ["# E1 — informe (convergido)", ""]
    # carga de evaluación desde C2 (delta) convergida
    d = data.get("delta", [])
    evalL = next((Lc for Lc in LOADS if _acc1(d, Lc).mean() < 0.95), 128) if d else None
    L.append(f"**Carga de evaluación (desde C2 convergida): L={evalL}**")
    L.append("")
    L.append("| cond | " + " | ".join(f"L{Lc}" for Lc in LOADS) + " | T2 |")
    L.append("|" + "---|" * (len(LOADS) + 2))
    for c in CONDS:
        r = data[c]
        if not r: continue
        row = " | ".join(f"{_acc1(r, Lc).mean():.3f}" for Lc in LOADS)
        t2 = np.mean([x["T2_L%d" % T2_LOAD] for x in r])
        L.append(f"| {c} | {row} | {t2:.3f} |")
    # veredictos (si están C1,C2,C3)
    if all(k in data and data[k] for k in ("softmax", "delta", "mix22")):
        c1, c2, c3 = data["softmax"], data["delta"], data["mix22"]
        t2_1 = np.mean([x["T2_L%d" % T2_LOAD] for x in c1]); t2_2 = np.mean([x["T2_L%d" % T2_LOAD] for x in c2])
        t2_3 = np.mean([x["T2_L%d" % T2_LOAD] for x in c3])
        lin = t2_3 - 0.5 * t2_1 - 0.5 * t2_2                         # P1.2: herencia de correctabilidad
        # PS-1: C3 vs C2 en la carga de evaluación (rescate de capacidad)
        rescate = _acc1(c3, evalL).mean() - _acc1(c2, evalL).mean()
        L += ["", "## Veredictos (preliminar automático — el informe final los emite con IC bootstrap)",
              f"- **P1.1** (C3≈C1 capacidad): softmax en techo → «no evaluable por saturación» (D2).",
              f"- **P1.2** (herencia correctabilidad): L = T2(C3) − ½T2(C1) − ½T2(C2) = {lin:+.3f} "
              f"({'≥0 ✓' if lin >= 0 else '<0'}).",
              f"- **P1.3** (no interferencia): C3<min(C1,C2)? T2 C3={t2_3:.3f} vs min={min(t2_1,t2_2):.3f}.",
              f"- **PS-1** (rescate C3>C2 @L{evalL}): C3−C2 = {rescate:+.3f} "
              f"({'las softmax rescatan ✓' if rescate > PISO else 'sin rescate claro'})."]
    open(os.path.join(RESULTS, "E1_informe.md"), "w").write("\n".join(L))
    print("\n".join(L), flush=True)


def main():
    print(f"=== E1 · {CONDS} · {N_SEEDS} semillas · {device_info()} ===", flush=True)
    notify(f"▶️ E1 iniciado · {CONDS} · {N_SEEDS} semillas · {device_info()['device_kind']}")
    for cond in CONDS:
        run_condition(cond)
    aggregate()
    notify("🏁 E1 COMPLETO. E1_informe.md listo (veredictos P1.1/P1.2/P1.3 + PS-1).")


if __name__ == "__main__":
    main()
