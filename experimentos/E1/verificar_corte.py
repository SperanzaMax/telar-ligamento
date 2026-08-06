"""¿El corte por convergencia está declarando «convergió» dentro de una meseta?

Riesgo concreto del barrido de d: en MQAR es normal que la accuracy se quede pegada al azar un
buen rato y después despegue. Un corte con ventana corta puede dispararse ahí y hacernos leer
«no tiene capacidad» donde en realidad decía «faltaba entrenar». Es el mismo tipo de error que la
auditoría del 2026-07-27 encontró en el borrador del preprint: el tramo tratado como meseta no lo
era, y la ventana pesaba más que la tolerancia.

Esto reentrena UN punto SIN corte por convergencia, hasta el tope alto, y muestra la curva entera.

  · Si la accuracy nunca despega  → el colapso es real, el corte no mintió.
  · Si despega después del corte  → el criterio estaba mal calibrado y hay que rehacer el barrido
    con ventana más larga. Los puntos ya medidos quedarían NO concluyentes.

Uso: python verificar_corte.py --d 8 [--lmax 64] [--pasos 3000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--lmax", type=int, default=64)
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="softmax", help="condición a entrenar (softmax, mix22, ...)")
    args = ap.parse_args()

    import modelos
    modelos.D = args.d
    modelos.DH = args.d // modelos.H
    modelos.FFN_HID = 3 * args.d

    from datos import hacer_vocab
    from entrenar import train_resumable

    voc = hacer_vocab(128, 64)
    ckpt = f"/tmp/verificar_corte_{args.cond}_d{args.d}_s{args.seed}.pkl"
    if os.path.exists(ckpt):
        os.remove(ckpt)      # sin reanudar: interesa la curva desde cero

    print(f"=== {args.cond} · d={args.d} · L_max={args.lmax} · {args.pasos} pasos SIN corte ===",
          flush=True)
    _, val_hist = train_resumable(
        args.cond, args.seed, args.pasos, ckpt,
        max_load=args.lmax, voc=voc,
        parar_al_converger=False, val_every=250, n_buckets=4,
        val_loads=tuple(L for L in (8, 16, 24, 32, 48, 64) if L <= args.lmax)[-3:],
    )

    print("\n paso   val_acc")
    for h in val_hist:
        print(f"{h['step']:>5}   {h['val_acc']:.4f}")

    accs = [h["val_acc"] for h in val_hist]
    if not accs:
        print("sin historial")
        return 1
    pico, final = max(accs), accs[-1]
    despego = pico > 3 * max(accs[0], 1e-9) and pico > 0.10
    print("\n--- lectura ---")
    if despego:
        paso_pico = val_hist[accs.index(pico)]["step"]
        print(f"DESPEGÓ: llegó a {pico:.4f} en el paso {paso_pico}. El corte por convergencia se")
        print("disparó dentro de una meseta → el barrido con ventana 250 NO es concluyente y hay")
        print("que rehacerlo con ventana más larga.")
    else:
        print(f"NO despega: pico {pico:.4f}, final {final:.4f} en {args.pasos} pasos.")
        print("El colapso es real y el corte no mintió: el barrido se puede leer como está.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
