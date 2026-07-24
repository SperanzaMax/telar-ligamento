# Cómo continuar la campaña E1 (una sesión por vez)

> Estado al 2026-07-23: prereg de seguimiento v1.1 **congelado** (`0b93a36f…`, tag firmado, timestamp GitHub
> 22:42:47Z) → **E1 habilitado a correr**. Campaña en curso en **Colab free / Tesla T4** (0.90 s/paso medidos).

## La idea en una línea

Todo el estado vive en **Google Drive** (`MyDrive/ligamento_e1`), no en la PC ni en la sesión de Colab.
Apagar la máquina no pierde nada: la campaña está fraccionada en **unidades atómicas** (una semilla, un
bloque de 2500 pasos, ~40 min) que siempre terminan escribiendo checkpoint.

## Cada vez que retomás

1. Abrir el notebook: [colab.research.google.com](https://colab.research.google.com) →
   `Archivo → Abrir cuaderno → GitHub` → `SperanzaMax/telar-ligamento` → `notebooks/e1_colab.ipynb`.
2. `Entorno de ejecución → Cambiar tipo de entorno → GPU` (la VM es nueva cada vez).
3. Correr las celdas **1 a 7** (segundos: instalan optax, clonan el repo actualizado, montan Drive,
   verifican las anclas del pre-registro).
4. Correr la celda **8**. Lee el estado desde Drive y sigue exactamente donde quedó.

Para ver cuánto falta **sin gastar GPU**: celda **8b** (unidades pendientes y sesiones restantes).
Para regenerar el informe en cualquier momento: celda **9**.

## Cuándo cortar

El mejor momento para cerrar es cuando el runner imprime **«sesión terminada»**: corta en frontera de unidad
y no se pierde nada. Si Colab mata la sesión de golpe, se pierde **solo la unidad en curso** (hasta ~40 min);
todo lo anterior queda en Drive.

No hace falta quedarse mirando. Pegando el token del bot en la celda 7, avisa por Telegram al terminar cada
unidad y al cerrar la sesión.

## Qué chequear al retomar

- **Que den GPU y no CPU** (celda 1). Colab free tiene límite diario; si se agota, al día siguiente puede
  tocar CPU. La celda 2 corta con un `assert` — **no forzarlo**: en CPU esto es ~20× más lento.
- **Que Drive tenga los archivos** (celda 10: detalle por semilla + checkpoints presentes). Si aparecieran
  vacíos, **no relanzar**: avisar primero, es el único escenario que implicaría rehacer trabajo.
- Si toca otra GPU, no hay que tocar nada: el planificador se recalibra solo con los tiempos reales.

## Configuración actual (celda 8)

```
CONDS=delta,softmax,mix22      # dan TODOS los veredictos pre-registrados
N_SEEDS=8
PRESUPUESTO_MIN=210            # 3.5 h, con margen antes del corte de Colab
MODO=sesion
```

`mix31,mix13` (C4) son **exploratorias**: no entran en ningún veredicto del prereg. Se corren en una segunda
tanda, agregándolas a `CONDS` cuando las tres principales estén listas.

## Cuánto falta

Con las 3 condiciones en T4: **~47 h de cómputo ≈ 14 sesiones** de 3.5 h. A dos sesiones por día, ~1 semana.

## Carta guardada (no aplicada)

**Bucketing del padding**: el loop de entrenamiento paddea siempre a 514 tokens aunque la secuencia real
promedia ~230 → ~2.5× de ganancia (47 h → ~19 h). Es matemáticamente equivalente (padding a la derecha,
atención causal, el scan de delta emite las salidas reales antes de tocar el relleno), pero **exige reiniciar
la campaña desde cero** para no mezclar implementaciones entre condiciones, y sería un cambio post-freeze a
documentar en `desviaciones.md`. Cuanto más avance la campaña, más caro es cambiar de opinión.

## Cuando termine

La celda 8b dirá **«CAMPAÑA COMPLETA»** y el informe se emite solo: `E1_informe.md` en Drive, con PS-1 (doble
tabla + regla de discordancia del Anexo B), PS-2, PS-4 (i/ii/iii), PS-5 (con control por paso de parada) y
P1.1/P1.2/P1.3 del protocolo madre. Ese informe es la entrada del análisis final, que se revisa a mano.
