# Cómo continuar la campaña E1 (una sesión por vez)

> **Estado al 2026-08-01 · lo que falta correr.** La **fase A está CERRADA** (24/24 unidades). Queda la
> **fase B, sólo para `mix22`**: extenderla de 2500 a `N_common` = 10000. **15,66 h ≈ 5 sesiones.**
> `softmax` quedó **exceptuada** por la enmienda E-003′ (congelada, `ed8709c2…`, release 21:48:36Z),
> lo que ahorra ~14 h. `delta` ya está en el tope y no se toca.
>
> **En el notebook eso es una sola línea: `FASE = 'B'` en la celda 8.** Ya viene puesta.

## Lo único que cambió respecto de la vez pasada

Antes había que sacar a mano `%env FASE_MAX=A` para largar la fase B, y eso se prestaba a un error
que ya pasó dos veces: al sacarlo arrancaba la fase B **entera**, `softmax` incluida (~30 h en vez de
~16). Ahora hay **una sola perilla** arriba de la celda 8:

```python
FASE = 'B'    # 'A' = freno, no extiende nada · 'B' = extiende mix22 (lo que hay que correr)
```

La celda arma el resto sola e imprime qué va a hacer antes de empezar. No hay que tocar nada más.

## La idea en una línea

Todo el estado vive en **Google Drive** (`MyDrive/ligamento_e1`), no en la PC ni en la sesión de Colab.
Apagar la máquina no pierde nada: la campaña está fraccionada en **unidades atómicas** (una semilla, un
bloque de 2500 pasos, ~40 min) que siempre terminan escribiendo checkpoint.

## Cada vez que retomás

1. Abrir el notebook: [colab.research.google.com](https://colab.research.google.com) →
   `Archivo → Abrir cuaderno → GitHub` → `SperanzaMax/telar-ligamento` → `notebooks/e1_colab.ipynb`.
2. `Entorno de ejecución → Cambiar tipo de entorno → GPU` (la VM es nueva cada vez).
3. Correr las celdas **1 a 7** (segundos: instalan optax, clonan el repo actualizado, montan Drive,
   verifican las anclas del pre-registro). La celda 1 **frena si no te tocó una T4** y la celda 6
   frena si falta alguna ancla congelada, incluida la enmienda E-003′.
4. Correr la celda **8**. Debe imprimir, antes de arrancar:

   ```
   FASE = B | se extiende: mix22 | freno: sin freno
   ```

   Si dice `se extiende: TODAS`, **pará**: se perdió la línea `FASE_B_CONDS` y estaría por gastar
   ~14 h de más en `softmax`. Si dice `freno: A`, la perilla quedó en `'A'` y no va a correr nada.

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

---

## Qué esperar de esta tanda (fase B de `mix22`)

Son **24 unidades** (8 semillas × 3 bloques de 2500 pasos), **15,66 h**, ~5 sesiones de 3,5 h.

Al terminar, el informe deja de decir «B1-ter — NO APLICA» y pasa a dar su veredicto real. **Ese
veredicto se aplica solo, sin que nadie decida nada:**

- Si `mix22` **no** se degrada → la tabla primaria queda con las tres condiciones bien rotuladas y
  PS-1 conserva su veredicto. Es el desenlace esperable: `mix22` está en 1,0000 y no puede mejorar.
- Si `mix22` **cae 0,0200 o más** en L96 o L128 → **PS-1 pasa automáticamente a «no concluyente por
  sensibilidad al presupuesto»**. No es un error ni algo a discutir: es el criterio que se congeló
  el 1 de agosto, antes de que estos datos existieran, y por eso vale.

Cualquiera de los dos desenlaces es un resultado publicable. El que no valdría nada es decidir la
regla después de ver el número.

## Si algo se ve raro

| síntoma | qué hacer |
|---|---|
| la celda 1 corta: «te tocó» otra GPU | cerrar la sesión y volver a pedir entorno hasta que salga **T4**. No forzar: mezclar GPUs mete el hardware adentro de la comparación |
| la celda 6 dice «BLOQUEADO» o «ANCLAS ROTAS» | **no correr**. Algún artefacto congelado cambió; avisar antes de seguir |
| la celda 8 imprime `se extiende: TODAS` | parar: falta `FASE_B_CONDS=mix22`, gastaría ~14 h de más |
| Colab mata la sesión de golpe | se pierde **sólo la unidad en curso** (~40 min). Volver a abrir y correr 1→8 |
| Drive aparece vacío en la celda 10 | **no relanzar**: avisar primero. Relanzar sobre un Drive vacío reentrena desde cero |
