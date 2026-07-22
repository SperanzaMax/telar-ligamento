"""Entrenamiento y evaluación para «Ligamento» (Fase 0+).

Loop AdamW con warmup+clip (R4/R5). Evaluación de capacidad por carga con acc@1 y acc@k
(la distinción top-1 vs listwise de P1). Incluye un smoke de S0.1 (softmax reproduce MQAR).

Uso: python entrenar.py smoke    # chequeo S0.1 rápido
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
import optax

from datos import gen_mqar, gen_overwrite, PAD, IGNORE, V0, NV
import modelos
from modelos import forward, init_params, count_params


def _pad_to(x, y, T):
    B = x.shape[0]
    xp = np.full((B, T), PAD, np.int32); yp = np.full((B, T), IGNORE, np.int32)
    xp[:, :x.shape[1]] = x; yp[:, :y.shape[1]] = y
    return xp, yp


def loss_fn(params, x, y, kind):
    logits = forward(params, x, kind)
    mask = (y >= 0)
    yl = jnp.where(mask, y, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    loss = (ce * mask).sum() / mask.sum()
    acc = ((logits.argmax(-1) == yl) * mask).sum() / mask.sum()
    return loss, acc


def train(kind, steps, seed=0, batch=64, lr=3e-3, max_load=16, t_max=None, log_every=100):
    """Currículum: pasos impares T1 (carga uniforme en [2,max_load]); pares T2 (correctabilidad)."""
    rng = np.random.default_rng(seed)
    params = init_params(seed, kind)
    if t_max is None:
        t_max = 4 * max_load + 2      # cubre T2 (overwrite): 3L+2r+2 ≤ 4L+2 con r≤L/2·… (r=L//2)
    sched = optax.warmup_constant_schedule(0.0, lr, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames='kind')
    def train_step(params, state, x, y, kind):
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, kind)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, a

    t0, hist = time.time(), []
    for s in range(1, steps + 1):
        if s % 2 == 0:
            L = int(rng.integers(4, max_load + 1))
            x, y, _ = gen_overwrite(rng, batch, L, r=max(1, L // 2))
        else:
            L = int(rng.integers(2, max_load + 1))
            x, y = gen_mqar(rng, batch, L)
        xp, yp = _pad_to(x, y, t_max)
        params, state, l, a = train_step(params, state, jnp.array(xp), jnp.array(yp), kind)
        if s % log_every == 0 or s == 1:
            l, a = float(l), float(a)
            hist.append({'step': s, 'loss': round(l, 4), 'acc': round(a, 4)})
            print(f'[{kind}] step {s:5d} loss {l:.4f} acc {a:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return params, hist


def eval_capacity(params, kind, loads, seed=1234, batch=64, reps=4, topk=(1, 4, 16)):
    """acc@k por carga sobre T1. Devuelve {L: {k: acc}}."""
    rng = np.random.default_rng(seed)
    fwd = jax.jit(partial(forward, kind=kind))
    out = {}
    for L in loads:
        accs = {k: [] for k in topk}
        for _ in range(reps):
            x, y = gen_mqar(rng, batch, L)
            t_max = 3 * L + 2
            xp, yp = _pad_to(x, y, t_max)
            logits = np.array(fwd(params, jnp.array(xp)))            # (B,T,VOCAB)
            m = yp >= 0
            # restringir a columnas de valor del vocab para la lectura (V0..V0+NV)
            val_logits = logits[..., V0:V0 + NV]
            order = np.argsort(-val_logits, axis=-1)                  # top índices (0..NV-1)
            true = np.where(m, yp - V0, -1)
            for k in topk:
                topk_idx = order[..., :k]
                hit = (topk_idx == true[..., None]).any(-1)
                accs[k].append((hit * m).sum() / m.sum())
        out[L] = {k: float(np.mean(v)) for k, v in accs.items()}
    return out


def eval_overwrite(params, kind, L=32, seed=4321, batch=64, reps=4):
    """Correctabilidad (T2): acc@1 sobre las claves REASIGNADAS (responder el valor nuevo)."""
    from datos import gen_overwrite
    rng = np.random.default_rng(seed)
    fwd = jax.jit(partial(forward, kind=kind))
    accs = []
    for _ in range(reps):
        x, y, upd = gen_overwrite(rng, batch, L, r=L // 2)
        t_max = 4 * L + 2
        xp, yp = _pad_to(x, y, t_max)
        um = np.zeros_like(yp, dtype=bool); um[:, -L:] = upd
        logits = np.array(fwd(params, jnp.array(xp)))
        pred = logits[..., V0:V0 + NV].argmax(-1)
        true = np.where(yp >= 0, yp - V0, -1)
        m = (yp >= 0) & um
        accs.append(((pred == true) & m).sum() / max(m.sum(), 1))
    return float(np.mean(accs))


def smoke():
    """S0.1: softmax debe reproducir MQAR (>95% a L=8) y degradarse suavemente."""
    print("=== SMOKE S0.1 · softmax en T1 (arquitectura §5, vocab E-001) ===")
    params, _ = train('softmax', steps=800, seed=0, max_load=16, lr=3e-3, log_every=200)
    print(f"params del modelo softmax: {count_params(params):,}")
    res = eval_capacity(params, 'softmax', loads=[8, 16, 32], reps=3)
    for L, d in res.items():
        print(f"  L={L:3d}  acc@1={d[1]:.3f}  acc@4={d[4]:.3f}  acc@16={d[16]:.3f}")
    ok = res[8][1] > 0.95
    print("S0.1:", "PASA ✓" if ok else "NO PASA ✗", f"(softmax acc@1 L=8 = {res[8][1]:.3f}, umbral 0.95)")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        print("uso: python entrenar.py smoke")
