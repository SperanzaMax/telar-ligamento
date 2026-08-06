"""Corre un comando largo vigilando la temperatura del paquete de CPU.

Bajar hilos y `nice` reparten el calor, no lo quitan: si la máquina no disipa, una corrida de
horas sube igual. Esto sí lo controla — cuando el paquete pasa el umbral de pausa, SUSPENDE el
proceso (SIGSTOP a todo su grupo) hasta que baja al umbral de reanudación.

El trabajo no se pierde: el runner de calibración guarda checkpoint y una pausa solo lo demora.

Uso:
    python guardian_termico.py --pausa 80 --reanuda 70 -- <comando...>

Salida: además de lo que imprima el comando, deja un resumen con la temperatura máxima
alcanzada, cuántas veces pausó y cuánto tiempo estuvo detenido.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time

RE_PKG = re.compile(r"Package id \d+:\s*\n\s*temp\d+_input:\s*([\d.]+)")


def temp_paquete():
    """Temperatura del paquete de CPU en °C, o None si no se puede leer."""
    try:
        out = subprocess.run(["sensors", "-u"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = RE_PKG.search(out)
    if m:
        return float(m.group(1))
    # fallback: el máximo de los cores, si el paquete no aparece
    vals = [float(v) for v in re.findall(r"temp\d+_input:\s*([\d.]+)", out)]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description="Vigila la temperatura y pausa el proceso si sube")
    ap.add_argument("--pausa", type=float, default=80.0, help="°C a los que se suspende (default 80)")
    ap.add_argument("--reanuda", type=float, default=70.0, help="°C a los que se reanuda (default 70)")
    ap.add_argument("--intervalo", type=float, default=5.0, help="segundos entre lecturas")
    ap.add_argument("--abortar", type=float, default=95.0,
                    help="°C a los que se mata el proceso en vez de pausarlo (default 95)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- seguido del comando a correr")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("falta el comando (usar: guardian_termico.py [opciones] -- comando...)")
    if not (args.reanuda < args.pausa < args.abortar):
        ap.error("debe cumplirse reanuda < pausa < abortar")

    t_ini = temp_paquete()
    if t_ini is None:
        print("AVISO: no se puede leer la temperatura (¿falta lm-sensors?). Se corre SIN guardián.",
              flush=True)
    else:
        print(f"guardián térmico: pausa a {args.pausa:.0f}°C · reanuda a {args.reanuda:.0f}°C · "
              f"aborta a {args.abortar:.0f}°C · ahora {t_ini:.1f}°C", flush=True)

    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = os.getpgid(proc.pid)

    detenido, t_max = False, t_ini or 0.0
    pausas, seg_detenido, t_pausa = 0, 0.0, None
    try:
        while proc.poll() is None:
            time.sleep(args.intervalo)
            t = temp_paquete()
            if t is None:
                continue
            t_max = max(t_max, t)
            if not detenido and t >= args.abortar:
                print(f"\n⚠ {t:.1f}°C ≥ {args.abortar:.0f}°C — ABORTANDO la corrida", flush=True)
                os.killpg(pgid, signal.SIGTERM)
                break
            if not detenido and t >= args.pausa:
                os.killpg(pgid, signal.SIGSTOP)
                detenido, pausas, t_pausa = True, pausas + 1, time.time()
                print(f"\n⏸ {t:.1f}°C ≥ {args.pausa:.0f}°C — PAUSADO (van {pausas})", flush=True)
            elif detenido and t <= args.reanuda:
                os.killpg(pgid, signal.SIGCONT)
                seg_detenido += time.time() - t_pausa
                detenido = False
                print(f"▶ {t:.1f}°C ≤ {args.reanuda:.0f}°C — reanudado "
                      f"(detenido {seg_detenido/60:.1f} min en total)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrumpido: terminando el proceso hijo", flush=True)
        if detenido:
            os.killpg(pgid, signal.SIGCONT)
        os.killpg(pgid, signal.SIGTERM)
    finally:
        if detenido and proc.poll() is None:
            os.killpg(pgid, signal.SIGCONT)
            seg_detenido += time.time() - t_pausa

    rc = proc.wait()
    print(f"\n--- guardián térmico ---\n"
          f"temperatura máxima : {t_max:.1f}°C\n"
          f"pausas             : {pausas}\n"
          f"tiempo detenido    : {seg_detenido/60:.1f} min\n"
          f"salida del comando : {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
