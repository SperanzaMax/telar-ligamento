#!/usr/bin/env bash
# Lanza la calibración de R-BANDA (E-005) en la estación de trabajo, acotada en CPU y temperatura.
#
#   ./experimentos/E1/lanzar_calibracion.sh            # escalón E-a (el que entra en su tope)
#   ./experimentos/E1/lanzar_calibracion.sh E-b        # escalón E-b (ver AVISO abajo)
#
# Límite de CPU: 3 de los 4 núcleos (75 %, el escalón más alto por debajo del 80 % pedido).
# El límite es del kernel (`taskset`), no una sugerencia: el proceso no puede usar el 4º núcleo.
# Aun así el calor no se controla con hilos, así que encima corre el guardián térmico, que
# SUSPENDE el proceso a 80 °C y lo reanuda a 70 °C. Una pausa demora la corrida, no la pierde:
# el runner deja checkpoint.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/.venv-ligamento/bin/python}"
ESCALON="${1:-E-a}"
NUCLEOS="${NUCLEOS:-0-2}"          # 3 de 4
PAUSA_C="${PAUSA_C:-80}"
REANUDA_C="${REANUDA_C:-70}"
LOG="$REPO/resultados/calibracion/calibracion_${ESCALON}_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO"
mkdir -p "$(dirname "$LOG")"

[ -x "$PY" ] || { echo "No existe el intérprete $PY (crear venv con jax[cpu] optax numpy)"; exit 1; }

echo "=== calibración R-BANDA · escalón $ESCALON ==="
echo "intérprete : $PY"
echo "núcleos    : $NUCLEOS (de $(nproc)) · hilos de BLAS/OMP fijados a 3"
echo "térmica    : pausa ${PAUSA_C}°C · reanuda ${REANUDA_C}°C · aborta 95°C"
echo "log        : $LOG"
echo

# Gate de anclas: no correr si algún artefacto congelado cambió.
"$PY" experimentos/verificar_anclas.py --requiere E-005 || {
  echo "ANCLAS ROTAS — no se corre."; exit 1; }

# Hilos: 3 en todos los backends que JAX/numpy puedan usar por debajo.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=3"
export PYTHONPATH="$REPO/src"

exec taskset -c "$NUCLEOS" nice -n 10 \
  "$PY" experimentos/E1/guardian_termico.py --pausa "$PAUSA_C" --reanuda "$REANUDA_C" -- \
  "$PY" experimentos/E1/calibrar_rbanda.py --escalon "$ESCALON" 2>&1 | tee "$LOG"
