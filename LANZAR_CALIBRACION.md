# Calibración de régimen — cómo lanzarla

Todo está preparado. **Un solo comando**, desde la raíz del repo:

```bash
./experimentos/E1/lanzar_calibracion.sh
```

Eso corre el escalón **E-a**. No hay nada que editar ni ninguna perilla que mover.

---

## Qué hace, en orden

1. **Verifica las anclas** (`--requiere E-005`). Si algún artefacto congelado cambió, **no corre**.
2. **Acota la CPU a 3 de los 4 núcleos** con `taskset` — es un límite del kernel, no una sugerencia:
   el proceso no puede tocar el cuarto núcleo. Más `nice -n 10` y los hilos de BLAS/OMP/XLA en 3.
3. **Levanta el guardián térmico**: a **80 °C suspende** el proceso, a **70 °C lo reanuda**, y a
   **95 °C aborta**. Probado: pausa y reanuda sin perder el proceso.
4. Corre la calibración y **guarda todo en un log** con fecha en `resultados/calibracion/`.

Una pausa térmica **demora la corrida, no la pierde**: el runner deja checkpoint.

## Cuánto tarda

| escalón | s/paso (3 núcleos) | 2 semillas × ~1500 pasos | tope de E-005 §4 |
|---|---|---|---|
| **E-a** (NK 256, L ≤ 128) | 1,81 | **≈ 1,5 h** | 3 h — entra cómodo |
| E-b (NK 512, L ≤ 256) | 5,87 | ≈ 4,9 h | 3 h — **no entra** |

Acotar a 3 núcleos **no cuesta tiempo**: con 4 daba 1,84 s/paso. A este tamaño de modelo el cuarto
núcleo casi no aportaba.

## Qué esperar al terminar

El runner imprime una de las cuatro ramas de E-005 §5 y deja
`resultados/calibracion/calibracion_rbanda.json`, rotulado *range-finding sin veredictos*:

- **R1** — E-a cumple la banda → **régimen de E2–E4 = E-a**. Es el desenlace bueno: se levanta la
  casilla roja de la auditoría y P2.2 pasa a ser corrible.
- **R2** — hace falta E-b. Ver el aviso de abajo antes de lanzarlo.
- **R3** — ningún escalón cumple con la bisección cerrada: frontera inalcanzable. **P2.2 se registra
  como NO CORRIDA, nunca como confirmada.**
- **R4** — el tope se agotó con la bisección abierta. **No es un resultado**: es presupuesto
  insuficiente, y la decisión queda suspendida.

## Aviso sobre E-b

En esta máquina E-b necesita ≈ 4,9 h contra un tope de 3 h, así que **agotaría el presupuesto y
caería en R4 por presupuesto, no por física**. Si E-a no cumple, las opciones son correr E-b en
Colab T4 (donde entra) o declarar presupuesto adicional, que es lo que R4 prevé. **No lo decidas
mirando el resultado de E-a: la rama ya está escrita, solo hay que elegir dónde pagar las horas.**

## Si algo se ve raro

| síntoma | qué hacer |
|---|---|
| «ANCLAS ROTAS» | **no correr**. Algún artefacto congelado cambió; avisar antes de seguir |
| pausas térmicas muy seguidas | la máquina no está disipando: bajar a 2 núcleos con `NUCLEOS=0-1 ./experimentos/E1/lanzar_calibracion.sh` |
| querés ver el plan sin gastar nada | `PYTHONPATH=src ~/.venv-ligamento/bin/python experimentos/E1/calibrar_rbanda.py --dry-run` |
| cortar a mitad | Ctrl-C: el guardián termina el proceso hijo de forma limpia y el checkpoint queda |

## Lo que la calibración NO decide

- **No emite veredictos.** Su único producto es la elección de régimen (E-005 §4).
- **No garantiza que P2.2 se pueda correr.** Calibra sobre la tarea de recuperación de E1; que T3
  (la tarea de E2) tenga rango dinámico en el régimen elegido es otra pregunta (E-005 §7).
- **No decide sobre qué cabezas corre E2.** Sigue abierto: `softmax`, como está congelado, o
  `mix13`. Conviene resolverlo **antes** de calibrar, porque la banda se evalúa sobre `softmax`.
