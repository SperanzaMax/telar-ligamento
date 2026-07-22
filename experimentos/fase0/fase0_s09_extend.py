"""S0.9 · EXTENSIÓN UNIFORME de delta hasta convergencia (D-004).

La foto a 2500 (fase0_s09.py) no convergió para delta. Esta fase extiende delta en bloques de +2500 pasos
aplicados a TODAS las semillas por igual (reanudando desde checkpoints de pesos), hasta que TODAS cumplan el
criterio de convergencia (mejora de val-acc < 0.5 pts en los últimos 500 pasos) o se alcance el tope 10 000.
Resultado principal = tabla convergida (delta_conv_seed*.json); el snapshot 2500 (delta_seed*.json) se preserva.
Softmax queda a 2500 (ya cumple el criterio).

Cierres heredados de fase0_s09.py: matmul 'highest', XLA determinista, GPU en config, sync a RESULTS_DIR.
Uso: RESULTS_DIR=/content/drive/MyDrive/ligamento_s09 N_SEEDS=8 python fase0_s09_extend.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
import sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
from entrenar import train_resumable, converged, eval_capacity, eval_overwrite, _val_acc  # noqa
from modelos import count_params, forward
from fase0_s09 import device_info, notify, LOADS, N_SEEDS, MAXLOAD, T2_LOAD, LR, RESULTS, PISO_CAP

BLOCKS = [5000, 7500, 10000]          # +2500 uniforme; tope duro 10 000 (D-004)


def main():
    print(f"=== S0.9 EXTENSIÓN delta · {N_SEEDS} semillas · bloques {BLOCKS} · {device_info()} ===", flush=True)
    notify(f"▶️ S0.9 extensión de delta (bloques {BLOCKS}, criterio val-acc<0.5pt/500) · {device_info()['device_kind']}")
    N_final = None
    for N in BLOCKS:
        conv_flags = []
        for seed in range(N_SEEDS):
            ckpt = os.path.join(RESULTS, f"delta_seed{seed}.ckpt")
            t0 = time.time()
            params, val_hist = train_resumable("delta", seed, N, ckpt, max_load=MAXLOAD, lr=LR)
            cap = eval_capacity(params, "delta", loads=LOADS, seed=1000 + seed, reps=4)
            t2 = eval_overwrite(params, "delta", L=T2_LOAD, seed=2000 + seed)
            conv = converged(val_hist, N)
            conv_flags.append(bool(conv))
            out = {"experimento": "S0.9-extend", "condicion": "delta", "seed": seed, "steps": N,
                   "converged": bool(conv), "params": count_params(params), "device": device_info(),
                   "capacity": {str(L): cap[L] for L in LOADS}, "overwrite_acc_L%d" % T2_LOAD: t2,
                   "val_hist": val_hist, "wall_s": round(time.time() - t0, 1)}
            with open(os.path.join(RESULTS, f"delta_conv_seed{seed}.json"), "w") as f:
                json.dump(out, f, indent=1)
            print(f"[extend] delta s{seed} @ {N} · conv={conv} · L128 acc@1={cap[128][1]:.3f}", flush=True)
        n_ok = sum(conv_flags)
        notify(f"S0.9 extensión: bloque {N} · {n_ok}/{N_SEEDS} semillas convergidas")
        if all(conv_flags):
            N_final = N; break
    N_final = N_final or BLOCKS[-1]
    aggregate(N_final)
    notify(f"🏁 S0.9 extensión COMPLETA · delta convergido a N={N_final}. margenes_instanciados.md listo.")


def aggregate(N_final):
    """Márgenes R11 y carga de evaluación sobre delta CONVERGIDO + softmax (base, 2500)."""
    def load(prefix, tag):
        return [json.load(open(os.path.join(RESULTS, f"{prefix}_seed{s}.json")))
                for s in range(N_SEEDS) if os.path.exists(os.path.join(RESULTS, f"{prefix}_seed{s}.json"))]
    data = {"softmax": load("softmax", "base"), "delta": load("delta_conv", "conv")}
    lines = ["# S0.9 — márgenes instanciados y cargas de evaluación (R11) · delta CONVERGIDO", "",
             f"Hardware: {data['delta'][0]['device'] if data['delta'] else 'n/d'} · "
             f"softmax@2500 (converge), delta@{N_final} (convergido, D-004) · "
             + ", ".join(f"{c}={len(data[c])} semillas" for c in data), ""]
    summary = {"N_final_delta": N_final}
    for c in ("softmax", "delta"):
        runs = data[c]
        if not runs:
            continue
        lines += [f"## {c} (acc@1 por carga, media ± SD)", "| L | media | SD | margen efectivo máx(piso,1.5·SD) |", "|---|---|---|---|"]
        per_L = {}
        for L in LOADS:
            v = np.array([r["capacity"][str(L)]["1"] for r in runs])
            mean, sd = float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            per_L[L] = {"mean": mean, "sd": sd, "margin": max(PISO_CAP, 1.5 * sd)}
            lines.append(f"| {L} | {mean:.4f} | {sd:.4f} | {per_L[L]['margin']:.4f} |")
        evalL = next((L for L in LOADS if per_L[L]["mean"] < 0.95), None)
        lines.append("")
        lines.append(f"**Carga de evaluación ({c})**: " + (f"L={evalL} (menor L con acc@1<95%)" if evalL is not None
                     else ("NINGUNA → P1.1 «no evaluable por saturación» (D2)" if c == "softmax" else "L=128")))
        lines.append("")
        summary[c] = {"per_L": per_L, "eval_load": evalL,
                      "overwrite": float(np.mean([r["overwrite_acc_L%d" % T2_LOAD] for r in runs]))}
    with open(os.path.join(RESULTS, "..", "margenes_instanciados.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(RESULTS, "s09_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
