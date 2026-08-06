"""SONDEO EXPLORATORIO del eje de la CAPACIDAD (d) — un punto por corrida.

NO es range-finding de E-005, NO elige régimen y NO emite veredictos: sirve para redactar con
criterio una enmienda futura, en vez de elegir el espacio candidato a ciegas.

Pregunta: ¿a qué d sale `softmax` del techo y entra en la banda [0,50 · 0,80]?

Por qué un proceso por punto: cambiar d cambia la forma de todos los parámetros. Corriéndolos
por separado, ninguna compilación previa de JAX puede quedar reusada entre puntos, y de paso el
pico de memoria es el de un solo modelo.

Cómo NO leerlo mal: si un punto llega al tope de pasos SIN converger, su accuracy baja puede ser
falta de entrenamiento y no falta de capacidad. Esos puntos salen rotulados NO INTERPRETABLE.

Uso: python sondeo_d.py --d 32 [--lmax 64] [--pasos 1000] [--seed 0]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

BANDA = (0.50, 0.80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True, help="d_model del punto (múltiplo de 4 cabezas)")
    ap.add_argument("--lmax", type=int, default=64, help="carga máxima de entrenamiento")
    ap.add_argument("--pasos", type=int, default=1000, help="tope de pasos (corta por convergencia)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buckets", type=int, default=4,
                    help="agrupa la longitud en vez de paddear siempre al máximo: mismo "
                         "entrenamiento (el rng se consume antes del padding), bastante más barato")
    ap.add_argument("--salida", default="resultados/sondeo_d")
    args = ap.parse_args()

    # --- el barrido de d vive acá, sin tocar modelos.py (artefacto de una campaña cerrada) ---
    import modelos
    if args.d % 4:
        sys.exit("d debe ser múltiplo de H=4")
    modelos.D = args.d
    modelos.DH = args.d // modelos.H
    modelos.FFN_HID = 3 * args.d          # se mantiene la expansión 3 de §5

    from datos import hacer_vocab
    from entrenar import train_resumable, eval_capacity, converged, count_params

    voc = hacer_vocab(128, 64)            # layout de E-001; L_max=64 respeta L ≤ NK/2
    grilla = [g for g in (8, 16, 32, 48, 64, 96, 128) if g <= args.lmax]

    os.makedirs(args.salida, exist_ok=True)
    ckpt = os.path.join(args.salida, f"d{args.d}_L{args.lmax}_s{args.seed}.pkl")

    print(f"=== punto d={args.d} (DH={modelos.DH}, FFN={modelos.FFN_HID}) · "
          f"L_max={args.lmax} · seed={args.seed} · tope {args.pasos} pasos ===", flush=True)

    t0 = time.time()
    params, val_hist = train_resumable(
        "softmax", args.seed, args.pasos, ckpt,
        max_load=args.lmax, voc=voc,
        parar_al_converger=True, val_every=250,
        val_loads=tuple(grilla[-3:]), n_buckets=args.buckets,
    )
    mins = (time.time() - t0) / 60
    paso_final = val_hist[-1]["step"] if val_hist else args.pasos
    conv = converged(val_hist, paso_final, window=250, tol=0.005)
    interpretable = bool(conv) or paso_final < args.pasos

    print(f"  parámetros : {count_params(params):,}")
    print(f"  entrenado  : {paso_final} pasos en {mins:.1f} min "
          f"({'convergió' if conv else 'TOPE sin converger'})", flush=True)

    cap = eval_capacity(params, "softmax", loads=grilla, seed=1234, reps=4, topk=(1,), voc=voc)
    medidas = {L: float(cap[L][1]) for L in grilla}
    en_banda = []
    for L in grilla:
        a = medidas[L]
        marca = ""
        if BANDA[0] <= a <= BANDA[1]:
            marca = "  ← EN BANDA"
            en_banda.append(L)
        elif a < BANDA[0]:
            marca = "  ← por debajo"
        print(f"  L={L:4d}  acc@1={a:.4f}{marca}")

    if not interpretable:
        print("  ⚠ NO INTERPRETABLE: llegó al tope sin converger — la caída puede ser "
              "sub-entrenamiento, no capacidad.")
    else:
        peor = min(medidas.values())
        if peor > 0.95:
            print(f"  → sigue EN EL TECHO (peor celda {peor:.4f}): d={args.d} no alcanza.")
        elif en_banda:
            print(f"  → toca la banda en L={en_banda}: candidato para la enmienda.")
        else:
            print(f"  → cayó de más (peor celda {peor:.4f}): pasado de rosca, d demasiado chico.")

    reg = {"tipo": "sondeo exploratorio — no elige régimen", "d": args.d, "DH": modelos.DH,
           "L_max": args.lmax, "seed": args.seed, "pasos": paso_final, "tope": args.pasos,
           "convergio": conv, "interpretable": interpretable, "minutos": round(mins, 2),
           "params": int(count_params(params)), "medidas": medidas, "en_banda": en_banda}
    with open(os.path.join(args.salida, f"punto_d{args.d}_L{args.lmax}_s{args.seed}.json"), "w") as f:
        json.dump(reg, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
