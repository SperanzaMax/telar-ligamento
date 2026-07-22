"""S0.9 DEFINITIVO — baselines de referencia C1 (softmax) y C2 (delta), 8 semillas.

Produce, para P1.x (v1.0) y para el prereg de seguimiento C3-vs-C2:
  - curva de capacidad acc@1/@4/@16 por carga (T1) y correctabilidad (T2), media ± SD sobre semillas;
  - márgenes efectivos R11 = máx(piso, 1.5×SD del baseline), apareados por semilla;
  - carga de evaluación D2 (softmax) y su análoga con delta (menor L con acc@1<95%).

Cierres de rigor (pedidos antes de campaña):
  * jax_default_matmul_precision = 'highest'
  * XLA flags deterministas (GPU) — seteadas ANTES de importar jax
  * S0.7 re-corrido en el hardware real (2× misma semilla → diferencia < tol)
  * checkpoint/reanudación por (condición, semilla): si su JSON existe, se salta
  * sync fuera de sesión: RESULTS_DIR configurable (apuntar a Drive en Colab)
  * modelo de GPU/dispositivo registrado en la config de CADA corrida

Uso:
  RESULTS_DIR=/content/drive/MyDrive/ligamento_s09  N_SEEDS=8  STEPS=2500 \
    python fase0_s09.py
"""
import os
# --- flags deterministas ANTES de importar jax (GPU); en CPU no molesta ---
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")   # precisión numérica reproducible

from entrenar import train, eval_capacity, eval_overwrite
from modelos import count_params, init_params, forward

# --- configuración (env override) ---
CONDS   = ["softmax", "delta"]                    # C1, C2
LOADS   = [8, 16, 32, 64, 96, 128]
N_SEEDS = int(os.environ.get("N_SEEDS", 8))
STEPS   = int(os.environ.get("STEPS", 2500))
MAXLOAD = int(os.environ.get("MAX_LOAD", 128))
T2_LOAD = int(os.environ.get("T2_LOAD", 32))      # correctabilidad se mide a L=32 (como TELAR-01)
LR      = float(os.environ.get("LR", 3e-3))
BASE    = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(BASE, "resultados", "fase0", "s09"))
os.makedirs(RESULTS, exist_ok=True)

PISO_CAP = 2.0 / 100     # piso de margen para capacidad (P1.1), en fracción
PISO_T2  = 2.0 / 100


def notify(text):
    """Aviso opcional por Telegram. Token y chat SOLO por env (nunca en el repo público).
    En Colab: %env TELEGRAM_TOKEN=...  %env TELEGRAM_CHAT=..."""
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    if not (tok and chat):
        return
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=15)
    except Exception as e:
        print("[notify] fallo Telegram:", e, flush=True)


def device_info():
    d = jax.devices()[0]
    return {"platform": d.platform, "device_kind": getattr(d, "device_kind", str(d))}


def s07_determinism_check():
    """S0.7 en el hardware real: 2× el mismo forward determinista → diferencia < tol."""
    p = init_params(0, "softmax")
    x = jnp.array(np.random.default_rng(0).integers(0, 197, size=(8, 98)))
    f = jax.jit(lambda pp, xx: forward(pp, xx, "softmax"))
    a = np.array(f(p, x)); b = np.array(f(p, x))
    diff = float(np.abs(a - b).max())
    ok = diff < 1e-6
    print(f"[S0.7] determinismo en {device_info()}: max|Δ| entre 2 corridas = {diff:.2e} -> "
          f"{'OK' if ok else 'REVISAR'}", flush=True)
    return {"max_abs_diff": diff, "ok": bool(ok)}


def run_one(cond, seed):
    ckpt = os.path.join(RESULTS, f"{cond}_seed{seed}.json")
    if os.path.exists(ckpt):
        print(f"[skip] {cond} seed{seed} ya existe (reanudación)", flush=True)
        return json.load(open(ckpt))
    t0 = time.time()
    params, hist = train(cond, steps=STEPS, seed=seed, max_load=MAXLOAD, lr=LR, log_every=500)
    cap = eval_capacity(params, cond, loads=LOADS, seed=1000 + seed, reps=4)
    t2 = eval_overwrite(params, cond, L=T2_LOAD, seed=2000 + seed)
    out = {"experimento": "S0.9", "condicion": cond, "seed": seed, "params": count_params(params),
           "device": device_info(), "steps": STEPS, "max_load": MAXLOAD,
           "capacity": {str(L): cap[L] for L in LOADS}, "overwrite_acc_L%d" % T2_LOAD: t2,
           "wall_s": round(time.time() - t0, 1)}
    with open(ckpt, "w") as f:      # sync inmediato por semilla (sobrevive corte de sesión)
        json.dump(out, f, indent=1)
    print(f"[done] {cond} seed{seed} · {out['wall_s']:.0f}s · L128 acc@1="
          f"{cap[128][1]:.3f} · T2={t2:.3f}", flush=True)
    notify(f"✅ S0.9 · {cond} seed{seed} listo ({out['wall_s']:.0f}s)\n"
           f"acc@1: L64={cap[64][1]:.3f} L96={cap[96][1]:.3f} L128={cap[128][1]:.3f} · T2={t2:.3f}")
    return out


def aggregate():
    """Media±SD por (cond,L), márgenes R11 y cargas de evaluación. Escribe margenes_instanciados.md."""
    data = {c: [json.load(open(os.path.join(RESULTS, f"{c}_seed{s}.json")))
                for s in range(N_SEEDS) if os.path.exists(os.path.join(RESULTS, f"{c}_seed{s}.json"))]
            for c in CONDS}
    lines = ["# S0.9 — márgenes instanciados y cargas de evaluación (R11)", ""]
    lines.append(f"Hardware: {data['softmax'][0]['device'] if data['softmax'] else 'n/d'} · "
                 f"semillas por condición: " + ", ".join(f"{c}={len(data[c])}" for c in CONDS))
    lines.append("")
    summary = {}
    for c in CONDS:
        runs = data[c]
        if not runs:
            continue
        lines.append(f"## {c} (acc@1 por carga, media ± SD sobre {len(runs)} semillas)")
        lines.append("| L | media | SD | margen efectivo (máx(piso,1.5·SD)) |")
        lines.append("|---|---|---|---|")
        per_L = {}
        for L in LOADS:
            vals = np.array([r["capacity"][str(L)]["1"] for r in runs])
            mean, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            marg = max(PISO_CAP, 1.5 * sd)
            per_L[L] = {"mean": mean, "sd": sd, "margin": marg}
            lines.append(f"| {L} | {mean:.4f} | {sd:.4f} | {marg:.4f} |")
        # carga de evaluación: menor L con media < 0.95
        evalL = next((L for L in LOADS if per_L[L]["mean"] < 0.95), None)
        lines.append("")
        if evalL is None:
            lines.append(f"**Carga de evaluación ({c})**: NINGUNA — no cae bajo 95% en [8..128] → "
                         f"{'P1.1 «no evaluable por saturación» (D2)' if c=='softmax' else 'usar L=128 (prereg seguimiento)'}.")
        else:
            lines.append(f"**Carga de evaluación ({c})**: L={evalL} (menor L con acc@1<95%).")
        lines.append("")
        summary[c] = {"per_L": per_L, "eval_load": evalL,
                      "overwrite": float(np.mean([r["overwrite_acc_L%d" % T2_LOAD] for r in runs]))}
    with open(os.path.join(RESULTS, "..", "margenes_instanciados.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(RESULTS, "s09_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines), flush=True)
    print("\n=> escrito resultados/fase0/margenes_instanciados.md + s09/s09_summary.json", flush=True)


def main():
    print(f"=== S0.9 DEFINITIVO · {CONDS} · {N_SEEDS} semillas · {STEPS} pasos · {device_info()} ===",
          flush=True)
    s07 = s07_determinism_check()
    with open(os.path.join(RESULTS, "s07_check.json"), "w") as f:
        json.dump({**s07, "device": device_info()}, f, indent=1)
    notify(f"▶️ S0.9 campaña iniciada · {device_info()['device_kind']} · "
           f"S0.7 determinismo max|Δ|={s07['max_abs_diff']:.1e} ({'OK' if s07['ok'] else 'REVISAR'})")
    for cond in CONDS:
        for seed in range(N_SEEDS):
            run_one(cond, seed)
    aggregate()
    notify("🏁 S0.9 COMPLETA (C1+C2, todas las semillas). Márgenes y cargas de evaluación listos "
           "en margenes_instanciados.md + s09_summary.json.")


if __name__ == "__main__":
    main()
