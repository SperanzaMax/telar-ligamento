"""SONDEO EXPLORATORIO — ¿está cerca la caída de `softmax` fuera del techo?

NO es range-finding de E-005 ni elige régimen: solo informa la redacción de una enmienda futura.
Reusa los checkpoints de E-a que ya están entrenados y evalúa en cargas MAYORES a las que vio
entrenando. No reentrena nada: cuesta minutos.

Dos sesgos, y los dos empujan la accuracy HACIA ABAJO:

  1. Extrapolación de longitud. El modelo entrenó con L ≤ 128; evaluarlo en L > 128 mezcla
     capacidad con generalización de longitud.
  2. Menos distractores. E-005 §3.1 pide L ≤ NK/2; acá NK se queda en 256 (es el que fija los
     embeddings del checkpoint), así que pasando L = 128 el pool de claves libres se achica.

Por eso el sondeo tiene un uso asimétrico, y solo uno es válido:

  · Si la accuracy NO cae → evidencia FUERTE de que E-b (L_max = 256) tampoco alcanza la banda,
    porque un entrenamiento propio en ese régimen rendiría todavía MÁS que esta extrapolación.
  · Si la accuracy SÍ cae → NO concluye nada: puede ser el sesgo y no la capacidad.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import jax
import jax.numpy as jnp
import numpy as np

BANDA = (0.50, 0.80)
CARGAS = [128, 160, 192, 224, 256]     # 128 = la última medida de E-a, para anclar
CKPT = "resultados/calibracion/ckpt/calib_E-a_s{}.pkl"
SEMILLAS = (0, 1)


def main():
    from datos import hacer_vocab
    from entrenar import eval_capacity

    voc = hacer_vocab(256, 64)      # el mismo vocabulario con el que se entrenó E-a
    print(__doc__)
    print(f"cargas: {CARGAS} · NK=256 · k=1 · condición softmax · semillas {SEMILLAS}\n")

    por_semilla = {}
    for s in SEMILLAS:
        ruta = CKPT.format(s)
        if not os.path.exists(ruta):
            print(f"falta {ruta} — se omite la semilla {s}")
            continue
        with open(ruta, "rb") as f:
            ck = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        print(f"[s{s}] checkpoint del paso {ck['step']}")
        cap = eval_capacity(params, "softmax", loads=CARGAS, seed=1234, reps=4,
                            topk=(1,), voc=voc)
        por_semilla[s] = {L: float(cap[L][1]) for L in CARGAS}
        for L in CARGAS:
            print(f"  L={L:4d}  acc@1={por_semilla[s][L]:.4f}")
        print()

    if not por_semilla:
        print("sin checkpoints: nada que sondear")
        return 1

    print("=== promedio sobre semillas ===")
    prom = {L: float(np.mean([por_semilla[s][L] for s in por_semilla])) for L in CARGAS}
    en_banda = []
    for L in CARGAS:
        marca = ""
        if BANDA[0] <= prom[L] <= BANDA[1]:
            marca = "  ← DENTRO de la banda"
            en_banda.append(L)
        elif prom[L] < BANDA[0]:
            marca = "  ← por debajo de la banda"
        print(f"  L={L:4d}  acc@1={prom[L]:.4f}{marca}")

    peor = min(prom.values())
    print("\n--- lectura (exploratoria, sin veredicto) ---")
    if peor > 0.95:
        print(f"La accuracy NO cae: la peor celda es {peor:.4f}, todavía en el techo.")
        print("Como los dos sesgos del sondeo empujan hacia abajo, esto es evidencia FUERTE de que")
        print("E-b (L_max=256) TAMPOCO alcanzaría la banda: subir la carga por este eje no rinde,")
        print("y conviene mirar el eje de la capacidad (d menor) antes de pagar horas de E-b.")
    elif en_banda:
        print(f"Hay caída y toca la banda en L={en_banda}. NO concluye: puede ser el sesgo del")
        print("sondeo y no la capacidad. Solo dice que el eje de la carga MERECE la enmienda.")
    else:
        print(f"Hay caída (peor celda {peor:.4f}) pero sin quedar en la banda. Igual de ambiguo:")
        print("el sesgo del sondeo basta para explicarla.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
