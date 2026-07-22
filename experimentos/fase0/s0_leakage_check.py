"""Control de fuga (sanidad S0, pedido antes de la campaña): validar que el acc@1=1.000 de C1 (softmax)
es LOOKUP real y no un atajo espurio.

Test: se almacenan (L-h) pares y se consultan las L claves; para las h claves 'holdout' el par nunca se
presentó. Si el modelo acierta esas por encima del azar (1/64≈0.0156), hay fuga. Esperado:
  - claves PRESENTES → acc ≈ 1.0
  - claves AUSENTES  → acc ≈ 1/NV (azar) y confianza ≈ plana

Rápido: entrena un softmax chico (max_load=32) y evalúa. Resultado a resultados/fase0/s0_leakage.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from entrenar import train, _pad_to
from datos import gen_mqar_leakage, V0, NV
import modelos
from modelos import forward


def eval_leakage(params, kind, L=32, h=8, reps=6, batch=64, seed=999):
    rng = np.random.default_rng(seed)
    fwd = jax.jit(partial(forward, kind=kind))
    acc_pres, acc_abs, conf_abs = [], [], []
    for _ in range(reps):
        x, y, absent = gen_mqar_leakage(rng, batch, L, h)
        t_max = 2 * (L - h) + 2 + L
        xp, yp = _pad_to(x, y, t_max)
        # alinear máscara absent con las columnas de query
        am = np.zeros_like(yp, dtype=bool); am[:, -L:] = absent
        logits = np.array(fwd(params, jnp.array(xp)))
        val_logits = logits[..., V0:V0 + NV]
        pred = val_logits.argmax(-1)
        true = np.where(yp >= 0, yp - V0, -1)
        m = yp >= 0
        pres = m & ~am
        absn = m & am
        acc_pres.append((( (pred == true) & pres).sum()) / max(pres.sum(), 1))
        acc_abs.append((( (pred == true) & absn).sum()) / max(absn.sum(), 1))
        # confianza (prob softmax del top-1) sobre ausentes
        probs = np.exp(val_logits - val_logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        conf = probs.max(-1)
        conf_abs.append(conf[absn].mean() if absn.sum() else np.nan)
    return (float(np.mean(acc_pres)), float(np.mean(acc_abs)), float(np.nanmean(conf_abs)))


def main():
    t0 = time.time()
    print("=== Control de fuga · softmax (max_load=32) ===", flush=True)
    params, _ = train('softmax', steps=1200, seed=0, max_load=32, lr=3e-3, log_every=400)
    ap, aa, ca = eval_leakage(params, 'softmax', L=32, h=8, reps=6)
    chance = 1.0 / NV
    print(f"\nclaves PRESENTES  acc@1 = {ap:.3f}   (esperado ~1.0)")
    print(f"claves AUSENTES   acc@1 = {aa:.3f}   (azar = {chance:.4f})")
    print(f"confianza media en AUSENTES = {ca:.3f}   (1/64={chance:.4f} = plana)")
    verdict = ("SIN FUGA ✓ (ausentes ≈ azar)" if aa < 3 * chance else
               f"POSIBLE FUGA ✗ (ausentes {aa:.3f} >> azar {chance:.4f})")
    print("=>", verdict, flush=True)
    out = {"check": "leakage", "kind": "softmax", "L": 32, "h": 8,
           "acc_present": ap, "acc_absent": aa, "chance": chance,
           "conf_absent": ca, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    outdir = os.path.join(os.path.dirname(__file__), "..", "..", "resultados", "fase0")
    with open(os.path.join(outdir, "s0_leakage.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"guardado en resultados/fase0/s0_leakage.json · {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
