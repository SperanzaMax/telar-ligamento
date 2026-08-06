#!/usr/bin/env bash
# Barrido exploratorio del eje de la capacidad: 5 puntos chicos, uno por proceso.
#
# Premisa de Maxi: no saturar la máquina ni en esfuerzo ni en temperatura, aunque tarde más.
# Por eso: 2 de los 4 hilos lógicos (no 3), nice 19 (la prioridad más baja), techo térmico de
# 70 °C con pausa automática, y un minuto de enfriamiento entre punto y punto.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/.venv-ligamento/bin/python}"
# El aviso por Telegram es opcional: el repo es publico (GitHub + DOI Zenodo), asi que el
# token NUNCA va en claro. Se lee de ~/.config/ligamento/telegram.secret (dos lineas: token
# y chat_id) o de las variables TG_TOKEN / TG_CHAT. Sin eso, tg() no hace nada y el barrido
# corre igual.
SECRETO="${TG_SECRET:-$HOME/.config/ligamento/telegram.secret}"
if [[ -z "${TG_TOKEN:-}" && -r "$SECRETO" ]]; then
  TG_TOKEN="$(sed -n 1p "$SECRETO")"
  TG_CHAT="$(sed -n 2p "$SECRETO")"
fi
TOKEN="${TG_TOKEN:-}"
CHAT="${TG_CHAT:-}"
PUNTOS="${PUNTOS:-8 12 16 24 32}"
LMAX="${LMAX:-64}"
PASOS="${PASOS:-1000}"
LOG="$REPO/resultados/sondeo_d/barrido_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO"
mkdir -p "$REPO/resultados/sondeo_d"

tg() { [[ -n "$TOKEN" && -n "$CHAT" ]] || return 0
       curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
        -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null; }
temp() { sensors -u coretemp-isa-0000 2>/dev/null \
           | grep -oE 'temp[0-9]+_input: [0-9.]+' | grep -oE '[0-9.]+$' | sort -rn | head -1; }

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2"
export PYTHONPATH="$REPO/src"

{
echo "=== barrido exploratorio del eje d ==="
echo "puntos d : $PUNTOS · L_max=$LMAX · tope $PASOS pasos · seed 0"
echo "recursos : 2 de 4 hilos (taskset 0-1) · nice 19 · techo 70 °C · 60 s de enfriado entre puntos"
echo "ancla ya medida: d=64 a L=64 dio acc@1 = 1.0000 en la calibración de hoy (no se recorre)"
echo
} | tee "$LOG"

tg "🔬 Arranca el barrido del eje d: puntos $PUNTOS, a L_max=$LMAX. Con 2 hilos de 4 y nice 19, para que la máquina ni se entere. Te aviso punto por punto."

i=0
for d in $PUNTOS; do
  i=$((i+1))
  t0=$(date +%s)
  echo "--- punto $i/$(echo $PUNTOS | wc -w): d=$d · $(date +%H:%M:%S) · $(temp) °C ---" | tee -a "$LOG"

  taskset -c 0-1 nice -n 19 "$PY" experimentos/E1/guardian_termico.py \
      --pausa 70 --reanuda 63 --intervalo 3 -- \
      "$PY" experimentos/E1/sondeo_d.py --d "$d" --lmax "$LMAX" --pasos "$PASOS" 2>&1 | tee -a "$LOG"

  mins=$(( ($(date +%s) - t0) / 60 ))
  RES="$(grep -aE "^  L=|^  →|NO INTERPRETABLE|entrenado" "$LOG" | tail -n 12)"
  tg "📍 Punto d=$d listo (${mins} min, $(temp) °C):

$RES"

  echo "  (enfriando 60 s · $(temp) °C)" | tee -a "$LOG"
  sleep 60
done

echo | tee -a "$LOG"
echo "=== resumen del barrido ===" | tee -a "$LOG"
"$PY" - "$REPO/resultados/sondeo_d" <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, os, sys
carpeta = sys.argv[1]
filas = []
for f in sorted(glob.glob(os.path.join(carpeta, "punto_d*.json"))):
    r = json.load(open(f))
    filas.append(r)
filas.sort(key=lambda r: r["d"])
print(f"{'d':>4} {'params':>9} {'pasos':>6} {'conv':>5}  peor celda   en banda")
for r in filas:
    peor = min(r["medidas"].values()) if r["medidas"] else float("nan")
    conv = "sí" if r["convergio"] else ("—" if r["interpretable"] else "NO")
    banda = ",".join(str(L) for L in r["en_banda"]) or "-"
    print(f"{r['d']:>4} {r['params']:>9,} {r['pasos']:>6} {conv:>5}  {peor:>10.4f}   {banda}")
print("\nd=64 (ancla de la calibración de hoy, L=64): acc@1 = 1.0000 — techo")
cand = [r for r in filas if r["en_banda"] and r["interpretable"]]
if cand:
    print("\nCandidatos con celdas en la banda:", ", ".join(f"d={r['d']}" for r in cand))
else:
    print("\nNingún punto tocó la banda: hay que mover el rango del barrido.")
PYEOF

tg "✅ Barrido terminado. Máxima temperatura del tramo: $(temp) °C. Te dejo el resumen en Claude Code."
