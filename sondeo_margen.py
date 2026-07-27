#!/usr/bin/env python3
"""SONDEO DE MARGEN (exploratorio, NO pre-registrado).

Motivo: la carga NO se puede subir por encima de L=128 (gen_mqar exige L claves distintas
de un pool NK=128, enmienda E-001). Como mix22 y softmax saturan los dos en acc@1=1.000,
la accuracy no discrimina en el extremo superior.

Alternativa: medir el MARGEN = logit(valor correcto) - max(logit de los otros valores),
promediado sobre las columnas de query. Con acc saturada, el margen sigue midiendo
"cuánto le sobra" al modelo. Margen chico = cerca del límite.

Se reporta tambien la fraccion de queries con margen <= 0 (los errores) para cruzar con acc@1.
"""
import os, sys, json, time, pickle, argparse
BASE = "/home/maxi/Documentos/Nuevo Transformer/telar-ligamento"
sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np, jax, jax.numpy as jnp
from functools import partial
jax.config.update("jax_default_matmul_precision", "highest")
from datos import gen_mqar, V0, NV
from modelos import forward
from entrenar import _pad_to

CKPT_DIR = "/home/maxi/Documentos/Nuevo Transformer/TRANFERENCIA/ligamento_e1-20260726T172138Z-1-001/ligamento_e1"

ap = argparse.ArgumentParser()
ap.add_argument("--loads", default="64,96,128")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--conds", default="mix22,delta,softmax")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out", default="/tmp/claude-1000/-home-maxi/701ac6a3-97c8-4a03-bd64-8a8bdfb38fbd/scratchpad/sondeo_margen.json")
a = ap.parse_args()
LOADS = [int(x) for x in a.loads.split(",")]
SEEDS = [int(x) for x in a.seeds.split(",")]
CONDS = a.conds.split(",")


def margen(params, cond, L, seed, batch, reps):
    rng = np.random.default_rng(seed)
    fwd = jax.jit(partial(forward, kind=cond))
    ms, errs = [], []
    for _ in range(reps):
        x, y = gen_mqar(rng, batch, L)
        xp, yp = _pad_to(x, y, 3 * L + 2)
        logits = np.array(fwd(params, jnp.array(xp)))
        m = yp >= 0
        vl = logits[..., V0:V0 + NV][m]                 # (Nq, NV)
        true = (yp - V0)[m]                             # (Nq,)
        correcto = vl[np.arange(len(true)), true]
        otros = vl.copy()
        otros[np.arange(len(true)), true] = -np.inf
        mg = correcto - otros.max(axis=1)
        ms.append(mg.mean()); errs.append(float((mg <= 0).mean()))
    return float(np.mean(ms)), float(np.mean(errs))


res = {}
for cond in CONDS:
    res[cond] = {}
    for seed in SEEDS:
        p = os.path.join(CKPT_DIR, f"e1_{cond}_seed{seed}.ckpt")
        if not os.path.exists(p):
            print(f"[{cond} s{seed}] sin checkpoint, salteo", flush=True); continue
        with open(p, "rb") as f:
            ck = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        t0 = time.time(); fila = {}
        for L in LOADS:
            mg, er = margen(params, cond, L, 3000 + seed, a.batch, a.reps)
            fila[str(L)] = {"margen": mg, "frac_err": er}
        res[cond][seed] = {"step": ck["step"], **fila}
        txt = "  ".join(f"L{L}: mg={fila[str(L)]['margen']:+.2f} err={fila[str(L)]['frac_err']*100:.2f}%"
                        for L in LOADS)
        print(f"[{cond} s{seed} @{ck['step']}] {txt}   ({time.time()-t0:.0f}s)", flush=True)

json.dump({"loads": LOADS, "reps": a.reps, "batch": a.batch, "res": res}, open(a.out, "w"), indent=1)
print("\nGuardado en", a.out)
