"""Emite la rama de E-005 §5 sobre la escalera COMPLETA, cuando los escalones corrieron aparte.

Por qué hace falta. La enmienda contempla que E-a y E-b se corran en momentos distintos (§4 reparte
el tope en dos mitades declaradas, y R4 existe justamente para el caso «se acabó el presupuesto en
el escalón que se estaba explorando»). Pero `calibrar_rbanda.py --escalon E-b` sólo tiene en la mano
el registro de E-b, y `decidir()` no puede declarar R3 sin ver los dos: R3 exige «bisección cerrada
en **ambos** escalones». El resultado es que cada corrida aislada devuelve R4 aunque la escalera ya
esté completa.

Este script no decide nada por su cuenta: junta los registros que ya existen y llama a la **misma**
`decidir()` del runner, en el orden congelado (E-a antes que E-b). Ninguna constante se toca.

Uso:
    python experimentos/E1/decidir_escalera.py \
        resultados/calibracion/calibracion_rbanda.json \
        resultados/calibracion/calibracion_rbanda_E-b.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrar_rbanda import ESCALONES, decidir  # noqa: E402

ORDEN = [e["nombre"] for e in ESCALONES]        # §3.1: el orden es fijo, no depende de los datos


def cargar(rutas):
    """Devuelve los registros de escalón de todos los JSON, deduplicados y en el orden congelado."""
    por_nombre = {}
    for ruta in rutas:
        with open(ruta) as fh:
            d = json.load(fh)
        for reg in d.get("escalones", []):
            nombre = reg["escalon"]
            if nombre in por_nombre:
                raise SystemExit(f"El escalón {nombre} aparece en más de un archivo: "
                                 "no se puede elegir cuál vale sin una regla, y no la hay.")
            por_nombre[nombre] = (reg, ruta, d.get("hardware"))
    return por_nombre


def main():
    rutas = sys.argv[1:]
    if not rutas:
        raise SystemExit(__doc__)

    por_nombre = cargar(rutas)
    faltan = [n for n in ORDEN if n not in por_nombre]

    print("=== ESCALERA DE R-BANDA (E-005 §3.1) ===\n")
    for nombre in ORDEN:
        if nombre not in por_nombre:
            print(f"  {nombre}: NO CORRIDO")
            continue
        reg, ruta, hw = por_nombre[nombre]
        peor = min(reg["medidas"].values()) if reg["medidas"] else float("nan")
        print(f"  {nombre}: NK={reg['NK']} L_max={reg['L_max']} · "
              f"cumple={reg['cumple']} · bisección_cerrada={reg['biseccion_cerrada']}")
        print(f"        cargas medidas {sorted(int(L) for L in reg['medidas'])}")
        print(f"        peor celda {peor:.4f} · en banda {reg['cargas_en_banda'] or '—'}")
        print(f"        {reg['seg'] / 60:.1f} min · {(hw or {}).get('device_kind', '?')} · {ruta}")
    print()

    if faltan:
        print(f"ESCALERA INCOMPLETA (falta {', '.join(faltan)}): la rama que salga NO es final.")
        print("R3 exige bisección cerrada en AMBOS escalones.\n")

    registros = [por_nombre[n][0] for n in ORDEN if n in por_nombre]
    rama, regimen, texto = decidir(registros)

    print("=" * 78)
    print(texto)
    print("=" * 78)
    print(f"\nrama={rama} · régimen elegido={regimen}")

    # Hardware heterogéneo entre escalones: legítimo acá y sólo acá. Se declara, no se corrige.
    hws = {n: (por_nombre[n][2] or {}).get("device_kind", "?") for n in ORDEN if n in por_nombre}
    if len(set(hws.values())) > 1:
        print(f"\nHardware heterogéneo entre escalones: {hws}")
        print("Permitido por E-005 §4 («hardware indistinto, declarado»), porque la calibración no")
        print("emite veredictos. La homogeneidad sigue siendo obligatoria DENTRO de E2-E4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
