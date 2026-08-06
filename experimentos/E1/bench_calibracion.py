"""Micro-benchmark del costo por paso en el régimen de un escalón de R-BANDA (E-005 §3.1).

Existe porque el bench del 2026-08-05 **subestimó el costo un 44 %** (1,815 s/paso proyectados
contra 2,60 medidos en la corrida real de E-a), y ésa fue la causa raíz de que E-b no entrara en el
tope de 3 h y de que la calibración volviera R4 (desviaciones.md D-006(b)). Antes de gastar el
presupuesto de E-b hay que medir en el hardware donde va a correr, no proyectar.

Separado de `bench_costo.py` a propósito: aquél mide el régimen de E1 (max_load=128, VOCAB=197, las
5 condiciones) y está atado a las corridas versionadas. Éste mide **un escalón de la calibración**:
NK y L_max variables, sólo `softmax`, que es la única condición que la calibración corre.

NO produce ninguna métrica de tarea: sólo cronometra. No emite veredictos.

Uso:
    python experimentos/E1/bench_calibracion.py                 # E-b (el que falta)
    python experimentos/E1/bench_calibracion.py --escalon E-a   # control contra los 2,60 medidos
    python experimentos/E1/bench_calibracion.py --pasos 30
"""
import argparse
import os
import sys
import time
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

from datos import gen_mqar, gen_overwrite, hacer_vocab  # noqa: E402
from entrenar import loss_fn, _pad_to  # noqa: E402
from modelos import init_params  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar_rbanda import ESCALONES, COND, TOPE_PASOS, SEMILLAS, TOPE_SEG_POR_ESCALON  # noqa: E402

BATCH, LR = 64, 3e-3

# Referencia real, no proyectada: E-a corrió el 2026-08-06 en la estación de trabajo (3 núcleos)
# y tardó 43,4 y 43,3 min para 1000 pasos por semilla.
S_PASO_REAL_CPU = {"E-a": 2.60}


def medir(escalon, n_pasos):
    """s/paso en el régimen del escalón, descontada la compilación. Sólo cronometra."""
    voc = hacer_vocab(escalon["NK"], 64)
    max_load = escalon["L_max"]
    t_max = 4 * max_load + 2
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    params = init_params(0, COND, vocab=voc.VOCAB)
    state = opt.init(params)
    rng = np.random.default_rng(0)

    @partial(jax.jit, static_argnames="kind")
    def train_step(params, state, x, y, kind):
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, kind)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, a

    def un_paso(s, params, state):
        if s % 2 == 0:
            L = int(rng.integers(4, max_load + 1))
            x, y, _ = gen_overwrite(rng, BATCH, L, r=max(1, L // 2), voc=voc)
        else:
            L = int(rng.integers(2, max_load + 1))
            x, y = gen_mqar(rng, BATCH, L, voc=voc)
        xp, yp = _pad_to(x, y, t_max)
        return train_step(params, state, jnp.array(xp), jnp.array(yp), COND)

    for s in (1, 2):                       # los dos branches del currículum, fuera del cronómetro
        params, state, l, _ = un_paso(s, params, state)
        l.block_until_ready()

    t0 = time.time()
    for s in range(3, n_pasos + 3):
        params, state, l, _ = un_paso(s, params, state)
    l.block_until_ready()
    return (time.time() - t0) / n_pasos


def main():
    ap = argparse.ArgumentParser(description="Costo por paso de un escalón de R-BANDA")
    ap.add_argument("--escalon", default="E-b", choices=[e["nombre"] for e in ESCALONES])
    ap.add_argument("--pasos", type=int, default=20, help="pasos cronometrados (default 20)")
    args = ap.parse_args()

    esc = next(e for e in ESCALONES if e["nombre"] == args.escalon)
    dev = jax.devices()[0]
    tope_h = TOPE_SEG_POR_ESCALON / 3600

    print(f"=== Costo por paso · escalón {esc['nombre']} · {dev.platform.upper()} "
          f"{getattr(dev, 'device_kind', '')} ===")
    # Ojo con T: el dry-run de calibrar_rbanda rotula 3·L+2, pero el padding efectivo —el que
    # manda en el costo, porque la atención va como T²— es 4·L+2.
    print(f"NK={esc['NK']} VOCAB={esc['NK'] + 69} L_max={esc['L_max']} "
          f"T_pad={4 * esc['L_max'] + 2} · condición {COND} · batch {BATCH}\n")

    s_paso = medir(esc, args.pasos)
    print(f"s/paso medido: {s_paso:.3f}\n")

    ref = S_PASO_REAL_CPU.get(esc["nombre"])
    if ref:
        print(f"referencia CPU (corrida real del 2026-08-06): {ref:.2f} s/paso "
              f"→ esta máquina va {ref / s_paso:.1f}× más rápido\n")

    n_sem = len(SEMILLAS)
    print(f"proyección para {n_sem} semillas (tope congelado: {tope_h:.0f} h por escalón)")
    print(f"  {'pasos hasta converger':>24} {'total':>9}   veredicto")
    for pasos, etiqueta in ((1000, "1000 (como E-a)"), (1500, "1500 (dry-run)"),
                            (TOPE_PASOS, f"{TOPE_PASOS} (tope)")):
        h = n_sem * pasos * s_paso / 3600
        print(f"  {etiqueta:>24} {h:>7.1f} h   {'entra' if h <= tope_h else 'NO entra — corta por tope'}")

    print("\nLa bisección sobre L vive en la evaluación y es barata (en E-a fue < 1 min).")
    print("Si ni el escenario de 1000 pasos entra, no es cuestión de insistir: hace falta")
    print("presupuesto adicional declarado (R4) o más GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
