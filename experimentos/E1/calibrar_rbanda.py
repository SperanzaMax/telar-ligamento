"""Calibración de régimen para E2–E4 — ejecuta R-BANDA (enmienda E-005, SHA 3e6572d2…).

Su ÚNICO producto es la elección de régimen. NO emite veredictos sobre ninguna predicción,
ni de E1 ni de E2–E4: es range-finding, y así queda rotulado en el JSON de salida.

Lo que dice la regla congelada, y que este script se limita a ejecutar:

    El régimen de E2–E4 será la menor extensión del espacio candidato tal que `softmax`,
    evaluada en k=1, caiga dentro de [0,50 · 0,80] en al menos TRES cargas de la franja
    superior de la grilla extendida.

Cuatro ramas de salida, evaluadas en orden (E-005 §5):
    R1  E-a cumple la banda                          → régimen = E-a
    R2  E-a no cumple y E-b sí                       → régimen = E-b
    R3  ninguno cumple, con la bisección CERRADA     → frontera inalcanzable
    R4  el tope se agota con la bisección ABIERTA    → decisión SUSPENDIDA, no resuelta

Por qué un solo entrenamiento por semilla y escalón: en E1 se entrena una vez con cargas
sorteadas hasta `max_load` y después se evalúa la grilla entera. Evaluar es barato y no
reentrena, así que la bisección sobre L vive en la EVALUACIÓN. Un escalón cuesta
2 entrenamientos (las semillas 0 y 1), no 2 por cada carga probada.

Uso:
    python experimentos/E1/calibrar_rbanda.py                 # E-a, y E-b si hace falta
    python experimentos/E1/calibrar_rbanda.py --escalon E-a   # solo un escalón
    python experimentos/E1/calibrar_rbanda.py --dry-run       # plan y costo, sin entrenar
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np

# --- Constantes congeladas por E-005. NO se tocan sin enmienda. ---------------------------------
BANDA = (0.50, 0.80)          # §3: techo 0,80 = 1 − 10·R11; piso 0,50 = el cruce que persigue la bisección
CARGAS_EN_BANDA = 3           # §3: "al menos tres cargas de la franja superior"
K_EVAL = 1                    # §3: la celda más exigente
SEMILLAS = (0, 1)             # §4, por R3 del protocolo: no se eligen, no se sustituyen
TOPE_PASOS = 2500             # §4: tope de la parada por convergencia
TOPE_SEG_POR_ESCALON = 3 * 3600   # §4: 6 h repartidas en dos mitades declaradas
CRUCE_BISECCION = 0.5         # §4: la bisección busca el cruce de 0,5
COND = "softmax"              # §4: la calibración corre SOLO la condición de referencia

# §3.1 — espacio candidato enumerado. Se recorre EN ORDEN y se corta en el primero que cumpla.
ESCALONES = [
    {"nombre": "E-a", "NK": 256, "L_max": 128,
     "grilla": [8, 16, 32, 64, 96, 128]},
    {"nombre": "E-b", "NK": 512, "L_max": 256,
     "grilla": [8, 16, 32, 64, 96, 128, 192, 256]},
]

SHA_E005 = "3e6572d2121840dc2fb9262d23a025e1bb131c845b4da75a5d57f1afecef4f0f"

# Medido en la estación de trabajo el 2026-08-05 (3 núcleos, batch 64, softmax). Solo alimenta
# el dry-run: no interviene en ninguna decisión.
S_PASO_MEDIDO = {"E-a": 1.815, "E-b": 5.865}


def en_banda(acc):
    return BANDA[0] <= acc <= BANDA[1]


def _cargas_franja_superior(medidas):
    """Las tres cargas más altas medidas, ordenadas de mayor a menor."""
    return sorted(medidas, reverse=True)[:CARGAS_EN_BANDA]


def cumple_banda(medidas):
    """§3: ¿hay al menos 3 cargas de la franja superior dentro de la banda?

    `medidas` es {L: acc@1 promediada sobre semillas}. La franja superior se toma sobre las
    cargas efectivamente medidas, que incluyen las intermedias que propuso la bisección
    (§3, "los tres puntos que satisfacen la banda pasan a integrar la grilla del régimen").
    """
    dentro = sorted([L for L, a in medidas.items() if en_banda(a)], reverse=True)
    return len(dentro) >= CARGAS_EN_BANDA, dentro[:CARGAS_EN_BANDA]


def proponer_carga(medidas, L_max):
    """Bisección adaptativa: siguiente L a evaluar, apuntando al cruce de 0,5.

    Devuelve None si el intervalo ya no se puede partir — ahí la bisección está CERRADA y la
    distinción entre R3 y R4 queda decidida.
    """
    if not medidas:
        return L_max
    ordenadas = sorted(medidas.items())
    # par de cargas contiguas entre las que la accuracy cruza el objetivo
    for (L1, a1), (L2, a2) in zip(ordenadas, ordenadas[1:]):
        if (a1 - CRUCE_BISECCION) * (a2 - CRUCE_BISECCION) <= 0 and L2 - L1 > 1:
            medio = (L1 + L2) // 2
            if medio not in medidas:
                return medio
    # sin cruce todavía: si todo está por encima del objetivo, empujar hacia la carga máxima
    L_alto, a_alto = ordenadas[-1]
    if a_alto > CRUCE_BISECCION and L_alto < L_max:
        return L_max
    # todo por debajo: refinar hacia abajo, entre la menor medida y la carga más chica útil
    L_bajo, a_bajo = ordenadas[0]
    if a_bajo < CRUCE_BISECCION and L_bajo > 8:
        medio = max(8, L_bajo // 2)
        return medio if medio not in medidas else None
    return None


def entrenar_referencia(escalon, seed, dir_ckpt, verbose=True):
    """Entrena `softmax` en el régimen del escalón, con parada por convergencia y tope 2500."""
    from datos import hacer_vocab
    from entrenar import train_resumable
    voc = hacer_vocab(escalon["NK"], 64)
    ckpt = os.path.join(dir_ckpt, f"calib_{escalon['nombre']}_s{seed}.pkl")
    t0 = time.time()
    params, val_hist = train_resumable(
        COND, seed, TOPE_PASOS, ckpt,
        max_load=escalon["L_max"], voc=voc,
        parar_al_converger=True, val_every=500,
        val_loads=tuple(escalon["grilla"][-3:]),
    )
    paso_final = val_hist[-1]["step"] if val_hist else TOPE_PASOS
    if verbose:
        print(f"  [{escalon['nombre']} s{seed}] entrenada hasta {paso_final} pasos "
              f"en {(time.time()-t0)/60:.1f} min", flush=True)
    return params, voc, paso_final, time.time() - t0


def medir(params, voc, cargas, seed_eval=1234, reps=4):
    from entrenar import eval_capacity
    cap = eval_capacity(params, COND, loads=list(cargas), seed=seed_eval, reps=reps,
                        topk=(K_EVAL,), voc=voc)
    return {L: cap[L][K_EVAL] for L in cargas}


def correr_escalon(escalon, dir_ckpt, tope_seg=TOPE_SEG_POR_ESCALON, verbose=True):
    """Devuelve el registro del escalón: medidas, si cumple la banda, y si la bisección cerró."""
    t0 = time.time()
    if verbose:
        print(f"\n=== escalón {escalon['nombre']} · NK={escalon['NK']} · "
              f"L_max={escalon['L_max']} · VOCAB={escalon['NK']+69} ===", flush=True)

    modelos, pasos = [], {}
    for seed in SEMILLAS:
        params, voc, paso, _ = entrenar_referencia(escalon, seed, dir_ckpt, verbose)
        modelos.append((params, voc))
        pasos[seed] = paso
        if time.time() - t0 > tope_seg:
            return {"escalon": escalon["nombre"], "medidas": {}, "cumple": False,
                    "biseccion_cerrada": False, "motivo": "tope agotado entrenando",
                    "pasos": pasos, "seg": time.time() - t0}

    medidas, por_semilla = {}, {s: {} for s in SEMILLAS}
    pendientes = list(escalon["grilla"])
    biseccion_cerrada = False

    while True:
        if time.time() - t0 > tope_seg:
            break
        if not pendientes:
            prox = proponer_carga(medidas, escalon["L_max"])
            if prox is None:
                biseccion_cerrada = True
                break
            pendientes = [prox]
        L = pendientes.pop(0)
        if L in medidas:
            continue
        accs = []
        for (params, voc), seed in zip(modelos, SEMILLAS):
            a = medir(params, voc, [L])[L]
            por_semilla[seed][L] = a
            accs.append(a)
        medidas[L] = float(np.mean(accs))
        if verbose:
            marca = "◀ EN BANDA" if en_banda(medidas[L]) else ""
            print(f"  L={L:4d}  acc@1={medidas[L]:.4f}  {marca}", flush=True)
        ok, _ = cumple_banda(medidas)
        if ok and not pendientes:
            biseccion_cerrada = True
            break

    ok, cargas_ok = cumple_banda(medidas)
    return {"escalon": escalon["nombre"], "NK": escalon["NK"], "L_max": escalon["L_max"],
            "medidas": {str(k): v for k, v in medidas.items()},
            "por_semilla": {str(s): {str(k): v for k, v in d.items()} for s, d in por_semilla.items()},
            "cumple": ok, "cargas_en_banda": cargas_ok,
            "biseccion_cerrada": biseccion_cerrada, "pasos": pasos,
            "seg": time.time() - t0}


def decidir(registros):
    """§5 — las cuatro ramas, en orden, exhaustivas y mutuamente excluyentes."""
    for reg in registros:
        if reg["cumple"]:
            rama = "R1" if reg["escalon"] == "E-a" else "R2"
            return rama, reg["escalon"], (
                f"{rama}: el escalón {reg['escalon']} cumple la banda en las cargas "
                f"{reg['cargas_en_banda']}. Régimen de E2-E4 = {reg['escalon']} "
                f"(NK={reg['NK']}, L_max={reg['L_max']}). Se corren P2.1, P2.2 y P2.3 con sus "
                f"umbrales intactos; la casilla roja de la auditoría se levanta.")
    corridos = {r["escalon"] for r in registros}
    completo = corridos >= {e["nombre"] for e in ESCALONES}
    if completo and all(r["biseccion_cerrada"] for r in registros):
        return "R3", None, (
            "R3 — FRONTERA INALCANZABLE: ningún escalón cumple la banda y la bisección cerró en "
            "ambos. E2-E4 se re-alcanzan a contrastes con al menos un brazo fuera del techo (los 5 "
            "verdes y P2.1 con cota inferior declarada) y P2.2 QUEDA SIN CORRER — se registra como "
            "NO CORRIDA, nunca como confirmada. Cualquier claim de frontera pasa a una campaña "
            "futura con d menor, como rediseño documentado.")
    return "R4", None, (
        "R4 — DECISIÓN SUSPENDIDA: el tope se agotó con la bisección ABIERTA. No se elige régimen "
        "y E2 NO arranca. Esto NO es un resultado sobre la física del modelo: es presupuesto "
        "insuficiente, y el escalón E-b cuesta ~4x por paso que E-a. Se retoma con presupuesto "
        "adicional declarado, continuando la bisección desde donde quedó — no se reinicia ni se "
        "reinterpreta. Leer R4 como R3 seria convertir una restriccion de credito en un hallazgo.")


def main():
    ap = argparse.ArgumentParser(description="Calibración de régimen (R-BANDA, E-005)")
    ap.add_argument("--escalon", choices=["E-a", "E-b"], help="correr solo este escalón")
    ap.add_argument("--dry-run", action="store_true", help="mostrar el plan sin entrenar")
    ap.add_argument("--tope-min", type=float, default=TOPE_SEG_POR_ESCALON / 60,
                    help="tope por escalón en minutos (default 180 = las 3 h de E-005 §4)")
    ap.add_argument("--salida", default="resultados/calibracion")
    ap.add_argument("--ckpt", default=None, help="directorio de checkpoints (default: <salida>/ckpt)")
    args = ap.parse_args()

    escalones = [e for e in ESCALONES if not args.escalon or e["nombre"] == args.escalon]

    print("=== CALIBRACIÓN DE RÉGIMEN · R-BANDA (enmienda E-005) ===")
    print(f"banda [{BANDA[0]:.2f} · {BANDA[1]:.2f}] · {CARGAS_EN_BANDA} cargas de la franja superior "
          f"· k={K_EVAL} · condición {COND} · semillas {SEMILLAS}")
    print(f"parada por convergencia, tope {TOPE_PASOS} pasos · tope {args.tope_min:.0f} min por escalón")
    print("range-finding SIN VEREDICTOS: su único producto es la elección de régimen.\n")

    if args.dry_run:
        for e in escalones:
            T = 3 * e["L_max"] + 2
            s_paso = S_PASO_MEDIDO[e["nombre"]]
            h = 2 * 1500 * s_paso / 3600
            cabe = "entra" if h <= args.tope_min / 60 else "NO ENTRA en el tope"
            print(f"{e['nombre']}: NK={e['NK']} VOCAB={e['NK']+69} L_max={e['L_max']} T={T} "
                  f"→ {s_paso:.2f} s/paso · 2 semillas × ~1500 pasos ≈ "
                  f"{h:.1f} h (tope {args.tope_min/60:.1f} h) — {cabe}")
        print("\ns/paso medidos en la estación de trabajo el 2026-08-05, 3 núcleos (taskset 0-2),")
        print("batch 64, condición softmax. Con 4 núcleos E-a daba 1,84: el cuarto núcleo casi no")
        print("aporta a este tamaño de modelo, así que acotar la CPU no cuesta tiempo de corrida.")
        print("\n(dry-run: no se entrenó nada)")
        return 0

    os.makedirs(args.salida, exist_ok=True)
    dir_ckpt = args.ckpt or os.path.join(args.salida, "ckpt")
    os.makedirs(dir_ckpt, exist_ok=True)

    registros = []
    for e in escalones:
        reg = correr_escalon(e, dir_ckpt, tope_seg=args.tope_min * 60)
        registros.append(reg)
        if reg["cumple"]:
            print(f"\n  {e['nombre']} CUMPLE la banda → no se evalúa el escalón siguiente "
                  f"(«menor extensión primero», §3)")
            break

    rama, regimen, texto = decidir(registros)
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)

    salida = {
        "tipo": "range-finding sin veredictos",
        "enmienda": "E-005", "sha_enmienda": SHA_E005,
        "banda": list(BANDA), "k": K_EVAL, "condicion": COND, "semillas": list(SEMILLAS),
        "tope_pasos": TOPE_PASOS, "parada": "por convergencia",
        "escalones": registros, "rama": rama, "regimen_elegido": regimen, "veredicto": texto,
        "hardware": _describir_hardware(),
    }
    ruta = os.path.join(args.salida, "calibracion_rbanda.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nregistro en {ruta}")
    return 0


def _describir_hardware():
    """E-005 §4: el hardware es indistinto pero se DECLARA en el informe."""
    try:
        import jax
        d = jax.devices()[0]
        return {"platform": d.platform, "device_kind": getattr(d, "device_kind", "?")}
    except Exception:
        return {"platform": "?", "device_kind": "?"}


if __name__ == "__main__":
    sys.exit(main())
