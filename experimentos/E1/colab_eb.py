"""Escalón E-b de la calibración R-BANDA, para correr por `colab` CLI en vez de por navegador.

Mismo contenido que `notebooks/calibracion_eb_colab.ipynb`, con una diferencia estructural: el
notebook persiste en Google Drive, y **`colab drivemount` no es ejecutable por un agente** (pide
TTY y cuelga). Acá el estado vive en el disco de la VM y se baja con `colab download`.

Consecuencia que hay que tener presente: **si la VM se pierde, se pierde el checkpoint**. Por eso
el paso `correr` baja los .pkl apenas termina, y conviene no dejar la sesión sin recoger.

Uso (desde la máquina local, con la sesión ya creada):

    colab new -s calib-eb --gpu T4
    colab exec -s calib-eb -f experimentos/E1/colab_eb.py --timeout 600     # (setup por defecto)
    PASO=bench   colab exec -s calib-eb -f experimentos/E1/colab_eb.py --timeout 900
    PASO=correr  colab exec -s calib-eb -f experimentos/E1/colab_eb.py --timeout 11000
    PASO=diagnostico colab exec -s calib-eb -f experimentos/E1/colab_eb.py --timeout 300
    colab download -s calib-eb /content/salida/calibracion_rbanda.json ./
    colab stop -s calib-eb

El paso se pasa por la variable de entorno PASO porque `colab exec` no reenvía argv al archivo.
El kernel PERSISTE entre invocaciones, así que los pasos comparten estado.

NO toca ninguna constante congelada. Es range-finding: no emite veredictos.
"""
import os
import subprocess
import sys

REPO_URL = "https://github.com/SperanzaMax/telar-ligamento"
REPO = "/content/telar-ligamento"
SALIDA = "/content/salida"
CKPT = os.path.join(SALIDA, "ckpt")
PASO = os.environ.get("PASO", "setup")


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.rstrip(), flush=True)
    if r.stderr.strip():
        print(r.stderr.rstrip(), file=sys.stderr, flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FALLO ({r.returncode}): {cmd}")
    return r


def setup():
    """Clona el codigo pre-registrado, instala dependencias y pasa el gate de anclas."""
    print("=== 1 · HARDWARE (se DECLARA, no se exige: E-005 §4 dice hardware indistinto) ===")
    sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", check=False)

    print("\n=== 2 · CODIGO PRE-REGISTRADO ===")
    if os.path.isdir(REPO):
        sh(f"cd {REPO} && git pull -q && git log --oneline -1")
    else:
        sh(f"git clone -q {REPO_URL} {REPO} && cd {REPO} && git log --oneline -1")

    print("\n=== 3 · DEPENDENCIAS ===")
    sh("pip -q install optax", check=False)

    print("\n=== 4 · GATE DE PRE-REGISTRO (bloquea si algun congelado no coincide) ===")
    sh(f"cd {REPO} && python experimentos/verificar_anclas.py --requiere E-005")

    os.makedirs(CKPT, exist_ok=True)
    print(f"\nsalida -> {SALIDA}\ncheckpoints -> {CKPT}")
    print("\nSETUP OK. Siguiente: PASO=bench")


def bench():
    """Costo real en ESTA GPU. E-a primero: de el tenemos el numero real y hace de control."""
    print("=== COSTO POR PASO EN ESTA GPU ===")
    print("E-a corrio de verdad el 2026-08-06 en CPU a 2,60 s/paso: sirve de control del bench")
    print("y da el factor GPU/CPU, que es lo que decide si E-b entra en el tope de 3 h.\n")
    sh(f"cd {REPO} && python experimentos/E1/bench_calibracion.py --escalon E-a --pasos 20")
    print("\n" + "=" * 78 + "\n")
    sh(f"cd {REPO} && python experimentos/E1/bench_calibracion.py --escalon E-b --pasos 20")
    print("\nSi la proyeccion dice que ni el escenario de 1000 pasos entra en 3 h, PARA ACA:")
    print("no es cuestion de insistir, hace falta presupuesto adicional declarado (R4).")


def correr():
    """La corrida. Parametros congelados, sin tocar ninguna constante."""
    print("=== CALIBRACION · ESCALON E-b ===")
    print("banda [0,50 · 0,80] · k=1 · softmax · semillas (0,1) · parada por convergencia")
    print("tope 2500 pasos · tope 180 min. NADA de esto se toca.\n")
    log = os.path.join(SALIDA, "corrida_E-b.log")
    sh(f"cd {REPO} && python experimentos/E1/calibrar_rbanda.py "
       f"--escalon E-b --salida {SALIDA} --ckpt {CKPT} 2>&1 | tee -a {log}", check=False)
    print("\n=== ARCHIVOS PARA BAJAR ===")
    sh(f"ls -la {SALIDA} {CKPT}", check=False)
    print("\nBajalos YA (la VM no persiste):")
    print(f"  colab download -s <sesion> {SALIDA}/calibracion_rbanda.json ./")
    print(f"  colab download -s <sesion> {log} ./")
    for s in (0, 1):
        print(f"  colab download -s <sesion> {CKPT}/calib_E-b_s{s}.pkl ./")


def diagnostico():
    """El corte por convergencia, cayo dentro de una meseta? Rotula, NO cambia el veredicto."""
    import pickle
    print("=== DIAGNOSTICO DEL CORTE (D-006(c)) ===")
    print("La calibracion corta con ventana 500 / tol 0,005, y ese criterio se dispara adentro")
    print("de mesetas: en d=8 una meseta de 750 pasos fue leida como convergencia. E-a no estaba")
    print("expuesto porque corto en el techo; E-b es el escalon dificil.\n")
    for s in (0, 1):
        ruta = os.path.join(CKPT, f"calib_E-b_s{s}.pkl")
        if not os.path.exists(ruta):
            print(f"  falta {ruta} (la corrida no llego a esta semilla)")
            continue
        with open(ruta, "rb") as fh:
            vh = pickle.load(fh).get("val_hist", [])
        if not vh:
            print(f"  semilla {s}: sin val_hist")
            continue
        pasos = [h["step"] for h in vh]
        vals = [h["val_acc"] for h in vh]
        pico = max(vals)
        print(f"\n=== semilla {s} · corte en {pasos[-1]} pasos ===")
        print("  curva:", "  ".join(f"{p}:{v:.3f}" for p, v in zip(pasos, vals)))
        print(f"  pico {pico:.4f} @ {pasos[vals.index(pico)]}   final {vals[-1]:.4f}")

        banderas = []
        if vals[-1] < 0.95:
            banderas.append(f"corto lejos del techo (val {vals[-1]:.3f} < 0,95): "
                            "meseta-y-despegue POSIBLE")
        else:
            print("  ok: corto en el techo — sin lugar donde despegar, como E-a")
        if len(vals) >= 3 and (vals[-1] - vals[-3]) > 0.02:
            banderas.append(f"venia subiendo fuerte (+{vals[-1] - vals[-3]:.3f} "
                            "en los ultimos 1000 pasos)")
        if vals[-1] < pico - 0.05:
            banderas.append(f"DIVERGENCIA: final {vals[-1]:.3f} < pico {pico:.3f} — INESTABLE")

        if banderas:
            print("  BANDERAS:")
            for b in banderas:
                print("   -", b)
            print("  -> registrar como desviacion antes de usar el numero para elegir regimen.")
        else:
            print("  sin banderas: el corte es creible.")


PASOS = {"setup": setup, "bench": bench, "correr": correr, "diagnostico": diagnostico}

if PASO not in PASOS:
    raise SystemExit(f"PASO desconocido: {PASO}. Opciones: {', '.join(PASOS)}")
print(f"### PASO = {PASO} ###\n")
PASOS[PASO]()
