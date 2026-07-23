"""Micro-benchmark de costo de E1: mide s/paso reales y proyecta la campaña completa.

No produce ninguna métrica de tarea (no toca predicciones): solo cronometra el paso de
entrenamiento con la configuración exacta de E1 (max_load=128, batch=64) y el costo de una
evaluación de convergencia, y proyecta el peor caso (todas las condiciones al tope de 10k).

Uso: python3 experimentos/E1/bench_costo.py [pasos_de_medicion]
"""
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial

jax.config.update("jax_default_matmul_precision", "highest")

from datos import gen_mqar, gen_overwrite  # noqa: E402
from entrenar import loss_fn, _pad_to, _val_acc  # noqa: E402
from modelos import init_params  # noqa: E402

MAXLOAD, BATCH, LR = 128, 64, 3e-3
CONDS = ["delta", "softmax", "mix22", "mix31", "mix13"]
N_SEEDS = int(os.environ.get("N_SEEDS", 8))
BLOCKS = [2500, 5000, 7500, 10000]
VAL_EVERY = 500
VAL_LOADS = (96, 128)


def medir(cond, n_pasos):
    """Devuelve (s_por_paso, s_por_eval) para una condición, ya descontado el tiempo de compilación."""
    t_max = 4 * MAXLOAD + 2
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    params = init_params(0, cond)
    state = opt.init(params)
    rng = np.random.default_rng(0)

    @partial(jax.jit, static_argnames="kind")
    def train_step(params, state, x, y, kind):
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, kind)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, a

    def un_paso(s, params, state):
        if s % 2 == 0:
            L = int(rng.integers(4, MAXLOAD + 1))
            x, y, _ = gen_overwrite(rng, BATCH, L, r=max(1, L // 2))
        else:
            L = int(rng.integers(2, MAXLOAD + 1))
            x, y = gen_mqar(rng, BATCH, L)
        xp, yp = _pad_to(x, y, t_max)
        return train_step(params, state, jnp.array(xp), jnp.array(yp), cond)

    params, state, l, _ = un_paso(1, params, state)   # compilación (fuera del cronómetro)
    l.block_until_ready()
    params, state, l, _ = un_paso(2, params, state)   # el otro branch del currículum
    l.block_until_ready()

    t0 = time.time()
    for s in range(3, n_pasos + 3):
        params, state, l, _ = un_paso(s, params, state)
    l.block_until_ready()
    s_paso = (time.time() - t0) / n_pasos

    t0 = time.time()
    _val_acc(params, cond, val_loads=VAL_LOADS)
    s_eval = time.time() - t0
    return s_paso, s_eval


def fmt(seg):
    h, m = divmod(int(seg + 0.5), 3600)
    return f"{h}h{m // 60:02d}m" if h else f"{m // 60}m{m % 60:02d}s"


def main():
    n_pasos = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    dev = jax.devices()[0]
    print(f"=== Benchmark de costo E1 · {dev.platform.upper()} {getattr(dev, 'device_kind', '')} "
          f"· {n_pasos} pasos por condición ===\n")
    tot_min, tot_max = 0.0, 0.0
    print(f"| {'cond':<8} | {'s/paso':>7} | {'s/eval':>7} | {'1 semilla @2500':>16} | {'1 semilla @10k':>15} |")
    print("|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 18 + "|" + "-" * 17 + "|")
    for cond in CONDS:
        sp, se = medir(cond, n_pasos)
        c2500 = 2500 * sp + (2500 / VAL_EVERY) * se
        c10k = 10000 * sp + (10000 / VAL_EVERY) * se
        tot_min += N_SEEDS * c2500
        tot_max += N_SEEDS * c10k
        print(f"| {cond:<8} | {sp:7.3f} | {se:7.2f} | {fmt(c2500):>16} | {fmt(c10k):>15} |")
    print(f"\nCampaña completa ({len(CONDS)} condiciones × {N_SEEDS} semillas):")
    print(f"  mejor caso (todas convergen a 2500): {fmt(tot_min)}")
    print(f"  peor caso  (todas al tope de 10000): {fmt(tot_max)}")
    print("\nNota: el runner re-evalúa capacidad+T2 tras cada bloque (costo menor, no incluido).")
    print("Colab Pro corta sesiones largas: la celda maestra reanuda desde los checkpoints en Drive.")


if __name__ == "__main__":
    main()
