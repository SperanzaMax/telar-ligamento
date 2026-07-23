"""Arquitectura base común de «Ligamento» (§5 del protocolo v1.0), port adaptado de TELAR-01.

Diferencias con TELAR-01 fijadas por §5:
  - d_model=64, H=4 cabezas, d_head=16 (TELAR-01 era H=2, DH=32).
  - 4 bloques pre-norm (TELAR-01: 2).
  - FFN de expansión 3 → hidden 192 (TELAR-01: 4x=256).
  - INVARIANTE (Ajuste 1, D1-freeze): RMSNorm POR CABEZA antes de concatenar y W_O, aplicada
    por igual a cabezas softmax y delta y en todas las condiciones. Sin esto, las cabezas softmax
    de C3 no serían idénticas a C1 y el contraste de E1 quedaría confundido.

Este módulo cubre reglas UNIFORMES por capa: 'softmax' (C1) y 'delta' (C2) — los baselines de
referencia de S0.9. Las cabezas mixtas de C3 (E1) extienden `mixer` con asignación por cabeza;
se añadirá al ejecutar E1 (prioridad E1>E2>E3>E4).

Vocab: importado de datos (E-001, VOCAB=197).
"""
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from datos import VOCAB

# --- dimensiones (§5) ---
D, H, DH = 64, 4, 16
NB = 4                      # número de bloques
FFN_HID = 192              # expansión 3

# Regla por cabeza (E1). 's'=softmax, 'd'=delta. Uniformes = C1/C2; mixtas = C3/C4.
KIND_RULES = {
    "softmax": ("s", "s", "s", "s"),   # C1
    "delta":   ("d", "d", "d", "d"),   # C2
    "mix22":   ("s", "s", "d", "d"),   # C3 (2 softmax + 2 delta)
    "mix31":   ("s", "s", "s", "d"),   # C4 exploratoria (3+1)
    "mix13":   ("s", "d", "d", "d"),   # C4 exploratoria (1+3)
}


def glorot(key, shape):
    lim = np.sqrt(6 / (shape[-2] + shape[-1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_params(seed, kind):
    ks = jax.random.split(jax.random.PRNGKey(seed), 128); i = iter(range(128))
    p = {'emb': jax.random.normal(ks[next(i)], (VOCAB, D)) * 0.02,
         'ln_f': {'g': jnp.ones(D), 'b': jnp.zeros(D)},
         'head': {'w': glorot(ks[next(i)], (D, VOCAB)), 'b': jnp.zeros(VOCAB)},
         'blocks': []}
    for _ in range(NB):
        blk = {'ln1': {'g': jnp.ones(D), 'b': jnp.zeros(D)},
               'ln2': {'g': jnp.ones(D), 'b': jnp.zeros(D)},
               'conv': {'w': jax.random.normal(ks[next(i)], (D, 3)) * 0.3, 'b': jnp.zeros(D)},
               'q': glorot(ks[next(i)], (D, D)), 'k': glorot(ks[next(i)], (D, D)),
               'v': glorot(ks[next(i)], (D, D)), 'o': glorot(ks[next(i)], (D, D)),
               'm1': {'w': glorot(ks[next(i)], (D, FFN_HID)), 'b': jnp.zeros(FFN_HID)},
               'm2': {'w': glorot(ks[next(i)], (FFN_HID, D)), 'b': jnp.zeros(D)}}
        if 'd' in KIND_RULES[kind]:                        # g_beta (zeros, no consume RNG) si hay cabezas delta
            blk['g_beta'] = {'w': jnp.zeros((D, H)), 'b': jnp.zeros(H)}
        p['blocks'].append(blk)
    return p


def ln(p, x):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * p['g'] + p['b']


def conv3(p, x):                       # depthwise causal, kernel 3
    xp = jnp.pad(x, ((0, 0), (2, 0), (0, 0)))
    return xp[:, :-2] * p['w'][:, 0] + xp[:, 1:-1] * p['w'][:, 1] + xp[:, 2:] * p['w'][:, 2] + p['b']


def split_heads(x):                    # (B,T,D) -> (B,H,T,DH)
    B, T, _ = x.shape
    return x.reshape(B, T, H, DH).transpose(0, 2, 1, 3)


def l2n(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def rmsn(x):                           # RMSNorm sobre la última dim (por cabeza)
    return x / jnp.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)


def _softmax_heads(q, k, v, T):
    att = jnp.einsum('bhtd,bhsd->bhts', q, k) / np.sqrt(DH)
    mask = jnp.tril(jnp.ones((T, T), bool))
    att = jnp.where(mask, att, -1e9)
    return jnp.einsum('bhts,bhsd->bhtd', jax.nn.softmax(att, -1), v)       # (B,H,T,DH)


def _delta_heads(blk, x, q, k, v, B):
    kn = l2n(jax.nn.silu(k)); qn = l2n(jax.nn.silu(q))
    beta = jax.nn.sigmoid(x @ blk['g_beta']['w'] + blk['g_beta']['b'])     # (B,T,H)
    tm = lambda a: a.transpose(2, 0, 1, 3) if a.ndim == 4 else a.transpose(1, 0, 2)
    S0 = jnp.zeros((B, H, DH, DH))

    def step(S, inp):
        qt, kt, vt, bt = inp                                  # (B,H,DH) y (B,H)
        yt = jnp.einsum('bhij,bhj->bhi', S, qt)               # leer antes de escribir
        pred = jnp.einsum('bhij,bhj->bhi', S, kt)
        err = vt - pred
        S2 = S + bt[..., None, None] * jnp.einsum('bhi,bhj->bhij', err, kt)
        return S2, yt

    _, ys = jax.lax.scan(step, S0, (tm(qn), tm(kn), tm(v), tm(beta)))
    return ys.transpose(1, 2, 0, 3)                           # (B,H,T,DH)


def mixer(blk, x, kind):
    """Salida de mezcla (B,T,D). Cada cabeza usa su regla (KIND_RULES); RMSNorm por cabeza SIEMPRE (invariante §5).
    Uniformes ('softmax'/'delta') reproducen exactamente C1/C2; mixtas ('mix22'…) combinan por cabeza (C3/C4)."""
    B, T, _ = x.shape
    rules = KIND_RULES[kind]
    has_s = 's' in rules; has_d = 'd' in rules
    q = split_heads(x @ blk['q']); k = split_heads(x @ blk['k']); v = split_heads(x @ blk['v'])
    if has_s and has_d:                                       # C3/C4: ambas ramas, selección por cabeza
        y_s = _softmax_heads(q, k, v, T)
        y_d = _delta_heads(blk, x, q, k, v, B)
        is_s = jnp.array([r == 's' for r in rules])           # (H,)
        y = jnp.where(is_s[None, :, None, None], y_s, y_d)
    elif has_s:                                               # C1
        y = _softmax_heads(q, k, v, T)
    else:                                                     # C2
        y = _delta_heads(blk, x, q, k, v, B)
    y = rmsn(y)                                               # INVARIANTE: RMSNorm por cabeza
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ blk['o']


def forward(params, x, kind):
    hx = params['emb'][x]
    for blk in params['blocks']:
        hx = hx + mixer(blk, conv3(blk['conv'], ln(blk['ln1'], hx)), kind)
        h2 = ln(blk['ln2'], hx)
        hx = hx + jax.nn.gelu(h2 @ blk['m1']['w'] + blk['m1']['b']) @ blk['m2']['w'] + blk['m2']['b']
    return ln(params['ln_f'], hx) @ params['head']['w'] + params['head']['b']


def count_params(params):
    return int(sum(np.prod(np.array(a).shape) for a in jax.tree_util.tree_leaves(params)))
