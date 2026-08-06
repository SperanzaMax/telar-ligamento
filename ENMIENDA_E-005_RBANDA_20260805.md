# Enmienda E-005 — R-BANDA: elección mecánica del régimen de E2–E4

**Fecha:** 2026-08-05 · **Estatuto:** enmienda **pre-datos con hash**, NO pre-registro (el
pre-registro del programa es `PREREG_SEGUIMIENTO_C3vsC2_v1.1.md`, SHA `0b93a36f…`). Se lee sobre el
protocolo madre (`2f8ebb82…`), el prereg v1.1 y las enmiendas E-003′ (`ed8709c2…`) y E-004
(`6662724c…`), sin modificar ninguno.

---

## 1. Por qué

E1 cerró con las cuatro condiciones no degradadas —`softmax`, `mix22`, `mix13`— **en el techo**:
0 de 144 celdas por debajo de 0,99 en cada una, y `mix13` cerrando en fila T el 2026-08-05. La
censura por techo, que en el cierre del 3-ago cubría dos condiciones, hoy cubre las tres hibridadas.

La auditoría del 2026-08-03 (`AUDITORIA_CENSURA_E2E4_20260803.md`) proyectó esa censura sobre los
nueve veredictos pre-registrados de E2–E4 y devolvió **5 verdes, 2 amarillos, 1 rojo**. El rojo es
**P2.2**, y es del tipo peor: una **equivalencia** entre dos rutas que tienen las dos acceso pleno al
contexto. Si ambos brazos saturan, D «empata» con B dentro de cualquier margen y P2.2 **se confirma
sin haber medido nada** — no un vacío, sino un vacío que se lee como confirmación.

De ahí que el régimen de E2–E4 —vocabulario y grilla de cargas— tenga que **elegirse**, y que esa
elección tenga que quedar fijada antes de que exista el dato que la determina. Sin regla previa, mirar
dónde cae el borde y después decidir el régimen sería juicio post-exposición sobre los contrastes
nuevos.

**Principio que ordena esto, ya declarado en la auditoría §0 y que esta enmienda instrumenta:**

> Las **reglas de veredicto** no pueden condicionarse en datos vistos; las **decisiones de
> instrumento** sí pueden y deben.

R-BANDA no toca ningún veredicto. Hace **mecánica** una decisión de instrumento.

## 2. Qué congela esta enmienda

El artefacto congelado es **R-BANDA** (§3), su **protocolo de calibración** (§4), sus **cuatro ramas de
salida** (§5) y las **dos métricas de resguardo** (§6). Nada más.

**No modifica** el protocolo madre, el prereg v1.1, E-003′ ni E-004. **No reformula ningún veredicto
de E2–E4**: los umbrales de P2.1, P2.2, P2.3, P3.1–P3.3 y P4.1–P4.3 quedan exactamente como están
congelados. No reabre ningún veredicto de E1.

## 3. R-BANDA — la regla

> **R-BANDA.** El régimen de E2–E4 será la **menor extensión** del espacio candidato de §3.1 tal que
> la condición de referencia `softmax`, evaluada en `k = 1`, caiga dentro de la banda
> **[0,50 · 0,80]** en al menos **tres cargas de la franja superior** de la grilla extendida.
>
> Entre extensiones que cumplan la banda se elige la de **menor pool de claves**; si empatan, la de
> **menor carga máxima**.

**Precisiones que hacen la regla ejecutable, y por qué cada una:**

- **Condición de referencia: `softmax`.** La redacción preliminar de la auditoría (§5) la llamaba «la
  condición más fuerte de E1». Tras E-004 ese rótulo ya no distingue: `softmax`, `mix22` y `mix13`
  son indistinguibles entre sí en el techo. La justificación correcta es otra y es la que se congela:
  **`softmax` es la condición sobre la que efectivamente corren E2–E4** (auditoría §1 — E2–E4 no
  varían la regla de escritura de las cabezas). Además es conservadora: si la condición de referencia
  tiene rango dinámico en un régimen, las demás —que nunca la superaron— también lo tienen.
- **`k = 1`.** Es la celda más exigente de la grilla, verificado en los datos de E1: en las tres
  condiciones hibridadas los únicos déficits no nulos aparecen en `k = 1` (p. ej. `mix13`, seed 2,
  L32, k=1, déficit 0,000122). Evaluar la banda en `k = 4` o `k = 16` la haría trivialmente
  incumplible.
- **Franja superior:** las **tres cargas más altas** de la grilla del régimen evaluado. La grilla
  **no** está limitada a las cargas de E1: la bisección puede proponer cargas intermedias, y **los
  tres puntos que satisfacen la banda pasan a integrar la grilla del régimen**. Esto es lo que hace
  alcanzable una banda de 30 puntos con `L` discreto; sin ello, el ancho de la banda competiría contra
  el espaciado de la grilla y la regla podría fallar por discretización en vez de por física.
- **Los dos extremos de la banda se eligen acá, y se declara así.** La redacción preliminar de la
  auditoría proponía **[0,30 · 0,98]**, con los dos extremos heredados —0,98 era el corte de censura
  del 2026-08-03— pero ese techo **no cumple la función de la regla**: con `R11 = 0,0200`, una
  referencia calibrada en 0,975 deja **1,25 veces el margen efectivo** de aire antes de la saturación,
  de modo que la banda se daría por cumplida en un régimen que sigue midiendo contra el techo. Los
  extremos que se congelan son:
  - **Techo 0,80 = 1 − 10·R11.** Deja **diez veces el margen efectivo** de aire. No es una cifra de
    gusto: es una función de una constante ya congelada en el prereg.
  - **Piso 0,50.** Es el punto que la bisección ya persigue (§4), de modo que el instrumento busca
    exactamente lo que la regla premia. La redacción preliminar tenía las dos cosas descoordinadas:
    bisecaba al cruce de 0,5 y lo aceptaba como interior de una banda que llegaba hasta 0,30.

  **Estatuto de esta elección, declarado sin atenuar:** estos dos números se fijan el **2026-08-05**,
  con E1 completo a la vista, y por lo tanto **no** heredan la anterioridad de los de la auditoría. Lo
  que los hace oponibles es que preceden a **todo dato de calibración** —§8 lo certifica— y que el
  techo se deriva de `R11` en vez de elegirse a ojo. La legitimidad no viene de que la cifra sea
  vieja; viene de que precede al dato que juzga.

### 3.1 Espacio candidato — enumerado, para que «menor extensión» no sea ambiguo

El régimen queda determinado por el par **(NK, L_max)**: pool de claves y carga máxima de la grilla.
`NV = 64` no se toca —la capacidad se mide en claves, no en valores— y `VOCAB = NK + NV + 5`.

| escalón | NK | VOCAB | L_max | grilla de cargas |
|---|---|---|---|---|
| **E-a** | 256 | 325 | 128 | 8, 16, 32, 64, 96, 128 |
| **E-b** | 512 | 581 | 256 | 8, 16, 32, 64, 96, 128, 192, 256 |

**Restricción de no degeneración: `L_max ≤ NK / 2`.** Es la razón, independiente de la censura, por
la que la celda tope del régimen viejo mide mal: con NK = 128 y L = 128 **todas** las claves del pool
están en juego y no queda ni un distractor sin usar. Ambos escalones la cumplen con igualdad.

**El espacio se recorre en orden y se corta en el primer escalón que cumpla la banda.** Ese orden
—E-a antes que E-b— es exactamente el «menor vocabulario primero» de la regla.

## 4. Protocolo de calibración

- **Condiciones: solo `softmax`.** La banda se evalúa sobre `softmax` y nada más, de modo que correr
  las otras condiciones cerradas —`delta`, `mix22`, `mix13`— gastaría presupuesto en puntos que **no
  entran en la decisión**. Si al cerrar la bisección sobrara presupuesto dentro del tope, pueden
  registrarse como referencia descriptiva; **no** son requisito y su ausencia no bloquea nada.
  Cualquier condición elegible debe estar **cerrada** (veredicto confirmatorio emitido): la
  calibración nunca toca una condición cuyo veredicto siga abierto.
- **2 semillas** —las **0 y 1**, por R3 del protocolo; no se eligen, no se sustituyen y no se
  descartan—, con **parada por convergencia y tope N = 2500** — no 2500 pasos fijos. El criterio de parada es el propio del pipeline, el que ya emite
  `converged` y `paso_conv_propio` en los JSON de E1; hoy se **registra** pero no corta, y para la
  calibración **corta**.

  Por qué no un N fijo más chico, que sería lo obvio: en E1 `softmax` converge en 1000 pasos y toca
  el techo de validación en 500, de modo que 2500 es ≈ 2,5× lo necesario **en el régimen fácil**. Pero
  `delta` —la condición que sí sufre— converge entre 1500 y 3500. **Cuanto más cuesta la tarea, más
  tarde converge**, y la calibración va exactamente a un régimen donde la referencia sufre. Un N fijo
  recortado mediría *no convergió* y lo haría pasar por *la tarea es difícil*: se elegiría un régimen
  que parece caer en la banda por falta de entrenamiento, y E2 lo encontraría más fácil de lo
  calibrado. La parada adaptativa se abarata sola donde se puede y se extiende donde hace falta.

  **Condición de transferencia, sin la cual la calibración no calibra nada:** las campañas de E2–E4
  corren en el régimen elegido **con el mismo criterio de parada y el mismo tope**. Calibrar con una
  regla de parada y correr con otra mide una cosa y aplica otra.
- **Declarada `range-finding sin veredictos`.** No emite ninguna afirmación sobre ninguna predicción,
  ni de E1 ni de E2–E4. Su único producto es la elección de régimen.
- **Hardware indistinto, declarado.** A diferencia de las campañas, la calibración **no** exige Tesla
  T4 ni homogeneidad de hardware: no emite veredictos, y la accuracy de una condición no depende de la
  máquina donde se entrena. Puede correr en **CPU local**, y se declara en el informe de calibración
  qué máquina la produjo. La homogeneidad de hardware sigue siendo obligatoria **dentro de** cada
  campaña de E2–E4, que es donde mezclarla metería el hardware adentro de la comparación.

  Medido en la CPU de la estación de trabajo el 2026-08-05 (4 núcleos), con el código del repo y
  `batch = 64`: **1,84 s/paso a L = 128** (T = 386), 1,17 a L = 96, 0,90 a L = 64. Es del orden de
  1,5–2× el costo de la T4, no 20×: con 193 493 parámetros el cuello es el `scan` secuencial y
  matrices demasiado chicas para que la GPU rinda. **La calibración entera entra en una corrida
  nocturna local y no consume crédito de Colab**, que queda reservado para las campañas.
- **Tope duro de 6 h de cómputo**, repartido en dos mitades declaradas: **hasta 3 h en el escalón E-a**;
  si E-a no cumple la banda al agotarlas, las **3 h restantes** van al escalón E-b. La búsqueda
  dentro de cada escalón es por **bisección adaptativa sobre L al cruce de 0,5** de `softmax`, y se
  corta **al agotar el tope, no al completar una grilla**.
- **Los dos escalones no cuestan lo mismo, y el reparto en mitades iguales no lo compensa.** La
  secuencia es `T = 3L + 2`, y la condición de referencia es `softmax`, cuyo costo va con `T²`: entre
  L_max 128 (T = 386) y L_max 256 (T = 770) el costo por paso crece **≈ 4×**. Las 3 h de E-b compran
  del orden de **cuatro veces menos puntos de bisección** que las 3 h de E-a. Se declara acá para que
  la rama R3 no se lea como «la frontera es inalcanzable» cuando lo que pasó fue que **el escalón caro
  se quedó sin presupuesto**: si E-b agota su mitad **sin haber cerrado la bisección**, el informe de
  calibración lo registra como *tope agotado en bisección abierta*, distinto de *banda incumplida con
  bisección cerrada*. Sólo el segundo dispara R3 por física.
- **Cada escalón entrena de cero.** Cambiar NK cambia VOCAB y por lo tanto el embedding: la
  calibración **no** reusa los checkpoints de E1 ni los de un escalón en el otro. Las 6 h son
  entrenamiento nuevo, no evaluación. Está declarado para que el presupuesto no se subestime.
- **El régimen definitivo es el escalón con el que se calibró.** No se «baja» a un NK menor después de
  haber calibrado en uno mayor: un NK menor tiene menos distractores, es **más fácil**, y empujaría el
  régimen de vuelta hacia el techo — es decir, en contra de todo el propósito de la calibración.

**Cortafuegos, explícitos:** (i) el régimen se elige con **condiciones cerradas**, neutral respecto de
los contrastes nuevos de E2–E4; (ii) la banda y sus dos extremos están fijados antes de ver dónde cae
el borde; (iii) el range-finding no emite veredictos; (iv) el orden de escalones es fijo y no depende
del resultado.

## 5. Las cuatro ramas de salida — pre-escritas

Se evalúan **en orden**; la primera que se cumple decide. Son exhaustivas y mutuamente excluyentes:
R4 cubre el caso en que el presupuesto termina antes que la búsqueda, que de otro modo caería
indebidamente en R3.

| # | condición al cerrar la calibración | consecuencia mecánica |
|---|---|---|
| **R1** | **E-a cumple la banda** | Régimen de E2–E4 = **E-a** (NK 256, L_max 128). Se corren P2.1, P2.2 y P2.3 con sus umbrales intactos. La casilla roja de la auditoría se levanta. |
| **R2** | **E-a no cumple y E-b sí** | Régimen de E2–E4 = **E-b** (NK 512, L_max 256). Igual que R1, con el costo de cómputo por token que corresponda al régimen mayor, declarado en el paper. |
| **R3** | **Ninguno cumple la banda, con la bisección cerrada en ambos escalones** | **Frontera inalcanzable:** la frontera de la hibridación excede lo medible con el presupuesto de calibración a este `d_model`. Consecuencias: (a) E2–E4 se re-alcanzan a contrastes con **al menos un brazo fuera del techo** — es decir, corren los 5 verdes y P2.1 con cota inferior declarada, y **P2.2 queda sin correr**; (b) cualquier claim de frontera pasa a una campaña futura con `d` menor, **como rediseño documentado, no como parche**, porque achicar `d` rompe el «misma arquitectura» que hoy ancla la comparabilidad entre campañas. |
| **R4** | **El tope se agota con la bisección abierta** en el escalón que se estaba explorando | **Decisión suspendida, no resuelta.** No se elige régimen y **E2 no arranca**. La calibración se retoma con presupuesto adicional declarado, continuando la bisección desde donde quedó — no se reinicia ni se reinterpreta. Esta rama existe porque el escalón E-b cuesta ≈ 4× por paso (§4) y un presupuesto insuficiente **no es un resultado sobre la física del modelo**. Leer R4 como R3 sería convertir una restricción de crédito en un hallazgo. |

**Prohibición que la rama R3 hace explícita, y que es el motivo por el que esta enmienda existe:**
bajo R3, **P2.2 se registra como NO CORRIDA, nunca como confirmada.** Correrla en un régimen saturado
produce un renglón que no puede fallar; omitir esta cláusula dejaría abierta la lectura de que el
empate observado la confirma.

## 6. Dos resguardos que no cuestan GPU

Ambos se pre-registran acá, para E2–E4, y se aplican **cualquiera sea la rama**:

- **Margen logit como métrica secundaria.** En el techo la exactitud queda ciega, pero los márgenes
  siguen ordenando condiciones. Se registra el margen logit en todas las campañas de E2–E4. **Cuesta
  logging, no cómputo.** Las primarias siguen siendo las de exactitud: la comparabilidad manda, y una
  secundaria no puede rescatar un veredicto que la primaria no sostiene.
- **Celdas ancla.** Cada campaña del régimen nuevo incluye **un par de cargas del régimen viejo**
  (NK = 128) para comparabilidad directa con E1. Sin ellas, un cambio de régimen corta la serie.

## 7. Alcance de la regla — la limitación que R-BANDA no resuelve

R-BANDA calibra sobre la **tarea de recuperación de E1**, porque es donde viven las condiciones con
veredicto cerrado. Que esa tarea tenga rango dinámico en el régimen elegido **no garantiza** que lo
tengan **T3** (polisémica, E2) ni **T5-b** (emparejamiento cruzado, E4), que son tareas distintas con
su propio azar y su propia dificultad.

Consecuencia declarada: el **range-finding de T5-b con 2 semillas** sigue siendo necesario y
**no** queda cubierto por esta enmienda. Es la única casilla del triage que un dato barato puede
mover —resuelve P4.1 y P4.3 a la vez— y conserva su lugar en la agenda.

## 8. Prueba de que la calibración no ha corrido

Requisito de legitimidad: la regla se fija antes del dato que la determina. Verificado al momento de
congelar esta enmienda:

```
$ find . -iname "*calib*" -o -iname "*rbanda*" -o -iname "*vocab256*" -o -iname "*vocab512*"
                                                   → (vacío, fuera de __pycache__)
$ git log --all --diff-filter=A -- '*calib*' '*banda*'
                                                   → (vacío)
```

`src/datos.py` sigue en `NK = 128`, `NV = 64`, `VOCAB = 197` por la enmienda E-001: **el vocabulario
extendido no existe todavía ni como código**. El estado del programa en este momento es E1 completa y
cerrada (incluida `mix13` 8/8, fila T) y E2 **no iniciada**.

## 9. Estatuto epistémico — declarado, no minimizado

**Enmienda pre-datos con hash, NO pre-registro.** Se redacta con la **capa ciega ya perdida** sobre
E1: el ejecutor vio todos los resultados de la campaña confirmatoria, incluidos PS-1 y la fila T de
E-004.

Lo que la hace oponible no es ceguera —no la hay— sino que **la elección de régimen queda
completamente determinada por el dato de calibración**, y que ese dato se produce sobre condiciones
cuyos veredictos ya están cerrados, neutral respecto de los contrastes que E2–E4 van a medir. Ninguna
de las cuatro ramas favorece al ejecutor: R1 y R2 obligan a correr P2.2 con el umbral congelado —que
puede fallar—, y R3 le prohíbe explícitamente cobrar P2.2 como confirmada.

**Origen:** paso 1 prescrito por `AUDITORIA_CENSURA_E2E4_20260803.md` §5, redactado allí como
propuesta a congelar «antes de la primera hora de calibración». Esta enmienda es ese congelamiento,
con cuatro precisiones que la propuesta preliminar dejaba abiertas y que sin resolver la hacían
inejecutable: la **condición de referencia** tras E-004 (§3), el **`k`** de evaluación (§3), el
**espacio candidato enumerado** (§3.1) y el **reparto del tope de 6 h entre escalones** (§4).

---

## Anclas previas de este expediente

| artefacto | SHA-256 | tag |
|---|---|---|
| protocolo madre v1.0 | `2f8ebb82…` | `ligamento-v1.0-freeze` |
| prereg de seguimiento v1.1 | `0b93a36f…` | `ligamento-prereg-seguimiento-v1.1` |
| enmienda E-003′ | `ed8709c2…` | `ligamento-enmienda-e003p` |
| enmienda E-004 | `6662724c…` | `ligamento-enmienda-e004` |
| **enmienda E-005 (este documento)** | ver `protocolo/FREEZE_ENMIENDA_E-005.md` | `ligamento-enmienda-e005` |
