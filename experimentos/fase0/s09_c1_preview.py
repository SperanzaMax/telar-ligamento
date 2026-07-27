"""S0.9 (preview) — baseline C1 (softmax) para P1.x: curva de capacidad y carga de evaluación (D2).

Entrena C1 (softmax puro, arquitectura §5) con currículum hasta L=128 y evalúa acc@1/@4/@16 en
L ∈ {8,16,32,64,96,128}. Objetivo: determinar la carga de evaluación de E1 (D2) = menor L donde
C1 < 95% acc@1; si C1 no cae en todo el rango → "P1.1 no evaluable por saturación del baseline".

Preview de 1 semilla para (a) responder la pregunta de D2 con datos reales y (b) estimar tiempos
antes de lanzar las 8 semillas del S0.9 definitivo. NO instancia márgenes todavía (eso requiere las
8 semillas, R11/R3). Resultado en resultados/fase0/s09_c1_preview.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from entrenar import train, eval_capacity
from modelos import count_params

LOADS = [8, 16, 32, 64, 96, 128]
SEED = 0
STEPS = 2500
MAX_LOAD = 128

def main():
    t0 = time.time()
    print(f"=== S0.9 preview · C1 softmax · seed={SEED} · steps={STEPS} · max_load={MAX_LOAD} ===", flush=True)
    params, hist = train('softmax', steps=STEPS, seed=SEED, max_load=MAX_LOAD, lr=3e-3, log_every=250)
    np_ = count_params(params)
    print(f"params={np_:,} · entrenamiento {time.time()-t0:.0f}s", flush=True)
    res = eval_capacity(params, 'softmax', loads=LOADS, seed=1234, reps=4)
    print("\ncarga  acc@1  acc@4  acc@16", flush=True)
    eval_load = None
    for L in LOADS:
        d = res[L]
        print(f"L={L:4d}  {d[1]:.3f}  {d[4]:.3f}  {d[16]:.3f}", flush=True)
        if eval_load is None and d[1] < 0.95:
            eval_load = L
    verdict = (f"carga de evaluación de E1 (D2) = {eval_load} (menor L con C1<95%)"
               if eval_load is not None
               else "C1 NO cae bajo 95% en [8..128] → P1.1 «no evaluable por saturación del baseline» (D2)")
    print("\n=>", verdict, flush=True)
    out = {"experimento": "S0.9-preview", "condicion": "C1-softmax", "seed": SEED, "steps": STEPS,
           "max_load": MAX_LOAD, "params": np_, "capacity": {str(L): res[L] for L in LOADS},
           "eval_load_D2": eval_load, "verdict": verdict, "wall_s": round(time.time()-t0, 1),
           "train_hist": hist}
    outdir = os.path.join(os.path.dirname(__file__), "..", "..", "resultados", "fase0")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "s09_c1_preview.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nguardado en resultados/fase0/s09_c1_preview.json · total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
