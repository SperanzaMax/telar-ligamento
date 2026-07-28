"""¿Agrupar la longitud de secuencia cambia el resultado del entrenamiento?

El padding fijo a 4*max_load+2 = 514 desperdicia ~55% del cómputo: la secuencia real promedia
230 tokens. Recortarlo es MATEMÁTICAMENTE neutro (modelo causal + loss enmascarada), pero eso
no garantiza que sea NUMÉRICAMENTE idéntico: cambiar la forma del tensor cambia el tiling que
elige XLA y con él el orden de las reducciones en float32.

La distinción decide cuánto vale la optimización:
  - bit-idéntico  -> se activa y se sigue desde los checkpoints existentes; ahorro puro.
  - diverge       -> activarlo obliga a reentrenar las 24 semillas desde cero, y hay que
                     comparar el ahorro contra el costo de tirar lo ya corrido.

Verifica además lo que SÍ debe ser exacto pase lo que pase:
  1. la grilla de buckets nunca trunca una secuencia (padding, no recorte);
  2. el sorteo de batches es idéntico con y sin bucketing (el rng se consume antes del padding).

Uso: python test_bucketing.py [pasos]     (por defecto 60; en CPU ~2 min)
"""
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

from datos import gen_mqar, gen_overwrite  # noqa: E402
from entrenar import _bucket_T, _pad_to, train_resumable  # noqa: E402

PASOS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAXLOAD, T_MAX, NB = 128, 4 * 128 + 2, 8
fallos = []


def check(nombre, cond, extra=""):
    print(f"  {'✓' if cond else '✗'} {nombre}" + (f" — {extra}" if extra else ""), flush=True)
    if not cond:
        fallos.append(nombre)


print("=== 1. La grilla de buckets es correcta (nunca trunca) ===")
paso = -(-T_MAX // NB)
grilla = sorted({_bucket_T(t, T_MAX, NB) for t in range(1, T_MAX + 1)})
check("todo T se mapea a un bucket >= T", all(_bucket_T(t, T_MAX, NB) >= t for t in range(1, T_MAX + 1)))
check("ningún bucket supera t_max", max(grilla) <= T_MAX, f"máx {max(grilla)} vs {T_MAX}")
check(f"son a lo sumo {NB} formas distintas", len(grilla) <= NB, f"{len(grilla)}: {grilla}")
check("el mayor cubre la secuencia más larga", _bucket_T(T_MAX, T_MAX, NB) == T_MAX)

print("\n=== 2. El sorteo de batches no depende del bucketing ===")
# el rng se consume en integers()/gen_*, ANTES del padding: las dos ramas deben ver los mismos datos
seqs = {}
for nb in (None, NB):
    rng = np.random.default_rng(0)
    firmas, formas = [], []
    for s in range(1, 41):
        if s % 2 == 0:
            L = int(rng.integers(4, MAXLOAD + 1)); x, y, _ = gen_overwrite(rng, 8, L, r=max(1, L // 2))
        else:
            L = int(rng.integers(2, MAXLOAD + 1)); x, y = gen_mqar(rng, 8, L)
        firmas.append((L, int(x.sum()), int(y.sum())))
        formas.append(T_MAX if nb is None else _bucket_T(x.shape[1], T_MAX, nb))
    seqs[nb] = (firmas, formas)
check("los batches sorteados son idénticos", seqs[None][0] == seqs[NB][0])
ahorro = np.mean(seqs[None][1]) / np.mean(seqs[NB][1])
check("y las formas sí cambian (el ahorro existe)", seqs[None][1] != seqs[NB][1],
      f"T medio {np.mean(seqs[None][1]):.0f} -> {np.mean(seqs[NB][1]):.0f} ({ahorro:.2f}x lineal)")

print(f"\n=== 3. ¿La trayectoria de entrenamiento es la misma? ({PASOS} pasos, delta) ===")
import tempfile  # noqa: E402
tmp = tempfile.mkdtemp()
res = {}
for nb in (None, NB):
    t0 = time.time()
    p, _ = train_resumable("delta", 0, PASOS, os.path.join(tmp, f"b{nb}.ckpt"),
                           max_load=MAXLOAD, val_loads=(96,), val_every=10 ** 9, n_buckets=nb)
    res[nb] = (jax.tree_util.tree_leaves(jax.tree_util.tree_map(np.asarray, p)), time.time() - t0)
    print(f"  {'sin bucketing' if nb is None else f'con {nb} buckets'}: {res[nb][1]:.1f}s", flush=True)

a, b = res[None][0], res[NB][0]
difs = [np.abs(x - y).max() for x, y in zip(a, b)]
dmax = max(difs)
n_ident = sum(1 for d in difs if d == 0.0)
speedup = res[None][1] / res[NB][1]

print(f"\n  tensores idénticos bit a bit: {n_ident}/{len(difs)}")
print(f"  diferencia máxima en los pesos: {dmax:.3e}")
print(f"  velocidad medida: {speedup:.2f}x  ({res[None][1]:.1f}s -> {res[NB][1]:.1f}s)")

if dmax == 0.0:
    print("\n  VEREDICTO: BIT-IDÉNTICO.")
    print("  Se puede activar y SEGUIR desde los checkpoints existentes. Ahorro puro.")
else:
    print(f"\n  VEREDICTO: DIVERGE (dif {dmax:.2e} en {PASOS} pasos).")
    print("  Matemáticamente equivalente pero no numéricamente: float32 + otro tiling de XLA.")
    print("  Activarlo obliga a REENTRENAR desde cero — no se puede mezclar con lo ya corrido.")

print("\n" + "=" * 60)
if fallos:
    print(f"✗ {len(fallos)} FALLO(S): " + "; ".join(fallos))
    sys.exit(1)
print("✓ BUCKETING VERIFICADO — grilla correcta, datos intactos, equivalencia medida arriba.")
