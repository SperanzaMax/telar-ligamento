# Cómo continuar la campaña E1 (una sesión por vez)

> ## ✅ CAMPAÑA E1 COMPLETA — 2026-08-03. No queda GPU por correr.
>
> Fase A cerrada (24/24) **y fase B cerrada**: `mix22` 8/8 a `N_common` = 10000. Las tres condiciones
> con N homogéneo — `delta` 8/8 @10000, `mix22` 8/8 @10000, `softmax` 8/8 @2500 (exceptuada por la
> enmienda E-003′, `ed8709c2…`). Los 48 JSON versionados en `resultados/E1/`, commit `5442cac`.
>
> **Veredictos:** PS-1 **CONFIRMA** (dif +0,0792 · IC95 [+0,0747, +0,0838] · 4,0× el margen R11) en las
> dos tablas, concordantes · **B1-ter: sin degradación material** (caída apareada +0,0000 contra umbral
> 0,0200) → no sobrescribe a PS-1 · PS-2 f=1,001 · PS-4 confirma sus tres partes · **PS-5 no
> concluyente** (T2 no evaluable a L96, fallback cross-carga a L32) · P1.2 +0,0139 ✓ · P1.3 ✓.
>
> **Limitación que va al paper:** la peor celda de toda la grilla de `mix22` es 0,999512 → su capacidad
> está **censurada por el techo**, no medida. PS-1 es **cota inferior**, y su enunciado correcto es sobre
> `delta`: «delta sola no alcanza a estas cargas; la hibridación restituye el techo».
>
> **Lo único que queda del lado del notebook:** correr la **celda 9** una vez para emitir el informe
> definitivo (el de `resultados/E1/E1_informe.md` lo regeneró `regenerar_informe_local.py`, que se
> auto-rotula «parcial» por venir sin JAX). No hace falta GPU.
>
> ---
>
> ## ✅ `mix13` TAMBIÉN CORRIÓ — E-004 cerrada 2026-08-05. **No queda GPU pendiente en E1.**
>
> 8/8 semillas en 3 sesiones de Colab T4 (2026-08-04/05), integridad uniforme (T4, 2500 pasos,
> `converged=true`, 193 493 params). JSON en `resultados/E1_mix13/`, commits `27450f6` + `f1f1bcf`,
> **pusheados**.
>
> **Veredicto: FILA T** — `evaluar_e004.py` ejecutando la tabla congelada el 2026-08-03 (SHA
> `6662724c…`, antes del primer dato):
>
> ```
> peor celda de la grilla : déficit 0,000122 — 0 de 144 celdas sobre el umbral 0,0020
> dif apareada @L96       : +0,0792 · IC95 [+0,0748, +0,0837] · 8/8 semillas > R11
> ►► FILA T
> ```
>
> Qué fijó eso, sin que nadie lo decidiera después: **1 de 4 cabezas softmax ya restituye el techo**
> → el claim sobre `mix22` baja a **suficiencia, no necesidad**; **PS-3 se queda sin
> dosis-respuesta** (los tres puntos hibridados saturan); y la censura por techo ahora cubre las
> tres condiciones → **PS-1 sigue siendo cota inferior**. Detalle en `CIERRE_E1_20260803.md` §5.
>
> **`mix13` deja de ser bloqueante: el paper de E1 se puede redactar.**
>
> ---
>
> ## ▶ Lo único que queda del lado del notebook (sin GPU)
>
> **Celda 9 con la perilla de la celda 7a en `PLAN = 'E1_B'`**, una vez. Emite el informe definitivo
> de E1.
>
> ⚠️ **Si dejás `PLAN = 'MIX13'`, la celda 9 vuelve a correr `evaluar_e004.py` en vez del informe de
> E1.** Hay que mover la perilla. El informe que hay hoy en `resultados/E1/E1_informe.md` se
> auto-rotula «parcial» porque lo generó `regenerar_informe_local.py` sin JAX.
>
> Lo demás que falta es **escritura, no cómputo**: paper de E1, nota de backends v3. Y del lado de
> E2-E4, antes de tocar GPU: congelar **R-BANDA** y el **range-finding de T5-b** (2 semillas).
>
> ---
>
> Lo de abajo queda como **registro del procedimiento** de la campaña, ya ejecutado.

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
FASE = 'B'                     # la perilla: extiende mix22 hasta N_common
CONDS=delta,softmax,mix22      # las tres siguen declaradas; la fase B sólo toca mix22
N_SEEDS=8
PRESUPUESTO_MIN=210            # 3.5 h, con margen antes del corte de Colab
MODO=sesion
```

`mix31,mix13` (C4) son **exploratorias**: no entran en ningún veredicto del prereg y corren en **run
separado que no comparte `N_common`**. `mix13` es la única que puede caer del techo —`softmax` y
`mix22` están las dos en 1,000— así que es la única que puede dar dosis-respuesta. ~5,1 h, después
de esta tanda.

## Cuánto falta

**15,66 h ≈ 5 sesiones** de 3,5 h (la fase B de `mix22`). A dos sesiones por día, unos 3 días.

*(El «~47 h ≈ 14 sesiones» de la versión anterior era la campaña entera desde el 23 de julio; la fase
A ya se pagó.)*

## Carta descartada: bucketing del padding

**Medido y NO sirve para E1.** El loop paddea a 514 tokens con secuencia real ~230, y el bucketing
baja el cómputo 1,83× — pero **diverge**: 0 de 69 tensores idénticos, diferencia 1,9·10⁻³ en 40 pasos.
Obligaría a reentrenar las 24 semillas desde cero, así que el ahorro neto real es 21-29 %, no el 61 %
que se estimó al principio. Queda para **E2-E4, que arrancan de cero**. Implementado y apagado por
defecto (`_bucket_T` en `entrenar.py`, `n_buckets=None`).

## Cuando termine

La celda 8b dirá **«CAMPAÑA COMPLETA»** y el informe se emite solo: `E1_informe.md` en Drive, con PS-1
(doble tabla + regla de discordancia B3 + **el veredicto de B1-ter**), PS-2, PS-4 (i/ii/iii), PS-5
(con control por paso de parada) y P1.1/P1.2/P1.3 del protocolo madre.

Ese informe es la entrada del análisis final. **Lo que no hay que hacer es retocarlo a mano:** los
veredictos que puede mover un humano —B3 y B1-ter— están automatizados a propósito, porque después
del 1 de agosto no queda una capa de revisión independiente en el proyecto.

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
