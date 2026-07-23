"""Gate de integridad del pre-registro: verifica que los artefactos congelados no cambiaron.

Recorre `protocolo/FREEZE*.md`, extrae de cada companion el par (**Archivo:**, **SHA-256:**) y
re-calcula el hash del artefacto referido. Sale con código 1 si alguno no coincide o falta.

Uso:
    python experimentos/verificar_anclas.py            # verifica todos los freezes presentes
    python experimentos/verificar_anclas.py --requiere PREREG_SEGUIMIENTO
                                                       # además EXIGE que exista un freeze cuyo
                                                       # nombre contenga ese patrón (bloquea E1 si
                                                       # el prereg de seguimiento no está congelado)
"""
import hashlib
import os
import re
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROTO = os.path.join(BASE, "protocolo")

RE_ARCHIVO = re.compile(r"\*\*Archivo:\*\*\s*`([^`]+)`")
RE_SHA = re.compile(r"\*\*SHA-256:\*\*\s*`([0-9a-f]{64})`")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verificar():
    """Devuelve (ok, filas) con una fila (freeze, artefacto, estado) por ancla encontrada."""
    filas, ok = [], True
    freezes = sorted(f for f in os.listdir(PROTO) if f.startswith("FREEZE") and f.endswith(".md"))
    for fz in freezes:
        texto = open(os.path.join(PROTO, fz), encoding="utf-8").read()
        archivo, esperado = RE_ARCHIVO.search(texto), RE_SHA.search(texto)
        if not (archivo and esperado):
            filas.append((fz, "—", "SIN ANCLA LEGIBLE"))
            ok = False
            continue
        ruta = os.path.join(BASE, archivo.group(1))
        if not os.path.exists(ruta):
            filas.append((fz, archivo.group(1), "ARTEFACTO AUSENTE"))
            ok = False
            continue
        real = sha256(ruta)
        coincide = real == esperado.group(1)
        filas.append((fz, archivo.group(1), "OK" if coincide else f"ALTERADO ({real[:12]}…)"))
        ok &= coincide
    return ok, filas


def main():
    ok, filas = verificar()
    print("=== Anclas de pre-registro ===")
    for fz, art, estado in filas:
        marca = "✓" if estado == "OK" else "✗"
        print(f" {marca} {fz:<45} → {art:<45} {estado}")
    if "--requiere" in sys.argv:
        patron = sys.argv[sys.argv.index("--requiere") + 1]
        if not any(patron in fz for fz, _, est in filas if est == "OK"):
            print(f"\n✗ BLOQUEADO: no hay freeze verificado que contenga «{patron}».")
            return 1
        print(f" ✓ requisito «{patron}» presente y verificado")
    print("\n" + ("✓ Todas las anclas coinciden — habilitado a correr." if ok
                  else "✗ HAY ANCLAS ROTAS — no correr hasta resolver."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
