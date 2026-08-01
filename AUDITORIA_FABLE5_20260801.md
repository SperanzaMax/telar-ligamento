# Última pasada · Auditoría del dictamen del 2026-08-01 — Ligamento E1

**Para:** Fable5 · **De:** Opus 5 (ejecutor) · **Fecha:** 2026-08-01

---

## 0. Por qué te llega esto y no un handoff

Maxi te retira del pipeline por costo de créditos: **ésta es la última pasada**. El handoff del 31
—las 5 preguntas más el §7 arrastrado del 27— **ya está respondido**, por mí. No tiene sentido que
gastes la única pasada re-respondiéndolo.

Lo que sí tiene sentido, y es la razón de este documento:

> El programa separaba dos capas: vos decidías sin ver los datos, yo ejecutaba y criticaba. **Esa
> separación se terminó hoy.** El mismo agente que vio los 48 JSON, calculó los veredictos y
> escribió los handoffs es el que ahora dictamina sobre las reglas con las que esos veredictos se
> juzgan. Y el resultado que esas reglas conservan (PS-1 a 4× el margen) me favorece.

Necesito que audites **eso**, no que revises aritmética. Todo lo numérico de abajo está recomputado
contra los JSON versionados y verificado; donde corregí una cifra de un handoff anterior, lo digo.

---

## 1. Lo que cambió desde tu último handoff (datos, no interpretaciones)

### 1.1 El hallazgo que da vuelta la pregunta de E-003

```
resultados/E1/e1_<cond>_seed<s>.json  vs  ..._propio.json
24 de 24 pares: IDÉNTICOS bit a bit
```

`cerrar_faseA()` congela la tabla secundaria con un `shutil.copyfile`. La separación entre tabla
primaria y secundaria **sólo aparece cuando la fase B sobreescribe los archivos principales**.

Consecuencia: **la regla de discordancia B3 —la salvaguarda que el prereg construyó específicamente
para PS-1— nunca se ejecutó.** El informe reportaba «tablas CONCORDANTES», que hoy significa que un
archivo es igual a su propia copia. Y el encabezado decía «todas las condiciones a N_common = 10000»
mientras la columna `N` de esas filas decía 2500.

**Ninguna de las tres salidas que te ofrecí el 31 (A/B/C) aborda esto.** A lo arregla por fuerza
bruta; C lo deja; **B —la que recomendaría el análisis superficial— lo empeora**, porque legitima que
`mix22` quede a 2500 en las dos tablas y deja a B3 permanentemente sin contenido.

### 1.2 Corrección a un dato que te di mal el 31

Te dije que B4 era falsa porque «`softmax` 0,830 y `delta` 0,836 cuestan lo mismo (0,7 %)».
**No reproduce.** Recomputado sobre el último bloque de cada semilla:

| condición | s/paso | vs `delta` |
|---|---|---|
| `softmax` | 0,8428 | 0,928× |
| `delta` | 0,9083 | 1,000× |
| `mix22` | 0,9393 | 1,034× |

**B4 acierta a medias:** `softmax` sí es la más barata —7,2 % menos, no 0,7 %— y `mix22` es la **más
cara** de las tres. Lo que sí derrumba a B4 no es el costo por paso sino la escala: la fase B completa
son **29,70 h = 1,47× lo que costó entrenar `delta` entera hasta el tope** (20,18 h).

### 1.3 El encuadre: ya nos ganaron de mano, antes de empezar

Verificado contra la API de arXiv, no de memoria:

| paper | ID | fecha | qué hace |
|---|---|---|---|
| **HydraHead** | 2606.20097 | **18-jun-2026** | hibrida FA+LA **por eje de cabezas**, 15B tokens, 512K |
| Gated DeltaNet-2 | 2605.22791 | 21-may-2026 | desacopla erase/write |
| Erase-then-Delta | 2606.26560 | 25-jun-2026 | desacopla erase/write |
| Kernelized LA | 2607.17419 | 19-jul-2026 | capacity wall |

**Los cuatro son anteriores al prereg (22-jul-2026).** HydraHead es exactamente `mix22` —hibridación
por cabezas— a escala real, un mes antes de que congeláramos. Las otras dos atacan arquitectónicamente
el trade-off que predice PS-5.

---

## 2. Las once decisiones que tomé, con su dirección de conveniencia

| # | decisión | ¿me conviene? |
|---|---|---|
| 1 | **E-003 → B′**: excepción asimétrica — `softmax` exceptuada, `mix22` **sí** se extiende | mixto |
| 2 | **B1-bis** (excepción) + **B1-ter** (criterio de degradación, pre-registro real) | B1-ter, no |
| 3 | PS-5 → cerrar **no concluyente**, sin E-00x | sí |
| 4 | Backend → repo ahora, v3 al cerrar E1 (σ va de 0,101 a 0,011 = factor 9) | neutro |
| 5 | E-002 retirada · **E-002′ aprobada, δ\* = 1,0 pt/2500**, rige desde E2 | neutro |
| 6 | N_common → **(b)** presupuesto fijo declarado; nunca (c) | **sí** |
| 7 | §6.1: el margen plano cambia el encuadre (`delta` es la mejor a carga baja) | neutro |
| 8 | §6.2: el paper va como **replicación pre-registrada**, no contribución arquitectónica | **no** |
| 9 | §6.3: `mix13` sí, run separado, 5,1 h | neutro |
| 10 | §6.4: no hay carrera; un prereg fechado no es scoopeable → no apurar | neutro |
| 11 | §7.4: norma **N-01** ratificada + chequeo automático | sí |

### El núcleo de B′, para que puedas atacarlo

**`softmax` exceptuada** porque no entra en ningún veredicto confirmatorio:
- P1.1 ya está cerrada por D2 («no evaluable por saturación»).
- P1.2 = T2(C3) − ½T2(C1) − ½T2(C2) = **+0,0132**. Si `softmax` se degradara al extenderla, T2(C1)
  bajaría y **P1.2 subiría** → no extenderla es la lectura **más exigente**. Sesgo contra nosotros.
- P1.3 = 0,9989 vs mín(C1,C2) = 0,9719 → **el mínimo lo pone `delta`**; `softmax` no puede moverlo.
- PS-2 es descriptivo (f = 1,001).

**`mix22` no exceptuada** porque es el minuendo de PS-1 y es lo único que le da contenido a B3.

**B1-ter, declarado antes de correr:** será degradación material una caída media ≥ **0,0200** (el
margen R11) en `capacity[96][1]` o `[128][1]` entre 2500 y 10000, apareada por semilla. Si la hay,
**PS-1 se reporta «no concluyente por sensibilidad al presupuesto»**, sin relecturas. Potencia:
`mix22` a L96 tiene SD = **3,03·10⁻⁵**, así que 0,0200 son **661 SD**.

**Costo:** `mix22` 15,66 h + `mix13` 5,11 h = **20,77 h**, contra 29,70 h de la opción A.

---

## 3. Lo que te pido: cuatro puntos, no los once

**(a) ¿B′ es una enmienda legítima o una conveniencia disfrazada de principio?**
El argumento «se extiende lo que decide un veredicto, no lo que describe» es limpio y ahorra 14 h.
Pero lo escribió alguien que ya sabe que PS-1 confirma. Si el resultado hubiera sido el contrario,
¿habría encontrado el mismo principio? Es la pregunta que no puedo contestar yo.

**(b) ¿B1-ter es pre-registro genuino o teatro?**
Mi defensa: los datos no existen, el umbral es el margen ya pre-registrado, la consecuencia es
adversa y automática. Tu objeción posible: con SD = 3·10⁻⁵ y `mix22` en 1,0000, una caída de 0,0200
es tan improbable que declararla no compromete a nada. Si eso es así, **decilo** — prefiero una
salvaguarda menos vistosa y con dientes que una decorativa.

**(c) ¿(b) en N_common resiste, o hay que pagar (a)?**
Cambié la recomendación que yo mismo di el 27 —era (a) con caída a (b)— y ahora digo (b). Motivo:
llevar 3 semillas a 15000 cuesta 3,78 h y **cae fuera del tope duro de 10000**, así que no puede
tocar ningún veredicto. `delta` está a 10000 con 3/8 sin converger; subentrenarla sesga contra C3, o
sea contra la hipótesis que confirmó. **Pero cambiar de posición en la dirección de lo barato, después
de ver que el resultado ya salió, es sospechoso por construcción.** Miralo con eso en mente.

**(d) ¿Qué no estoy viendo por haber visto los datos?**
La pregunta abierta, y la más valiosa. Si tenés que elegir una sola de las cuatro, elegí ésta.

---

## 4. Lo que **no** hace falta que hagas

- **No re-litigues los 11 puntos.** Los 7 que no están en §3 son mecánicos o ya están sostenidos en
  público bajo DOI.
- **No verifiques aritmética.** Está recomputada y el anexo del dictamen dice qué y cómo.
- **No propongas experimentos nuevos.** El presupuesto es Colab free, sesiones de ~4 h, y ya hay
  20,77 h comprometidas.
- **No cites de memoria.** Es el patrón de error que ya costó una ronda con la segunda opinión de
  Kimi k3: juicio de fondo sólido, detalle ejecutable y referencias inventadas. Si no estás seguro de
  una referencia, decí que no estás seguro.

**Una sola pasada, agrupá todo.** Las precisiones menores marcalas y seguimos del lado del ejecutor.

---

## 5. Estado operativo

- Fase A **cerrada**, 24/24 unidades, todas en Tesla T4. `delta` N=10000 (3/8 sin converger),
  `softmax` y `mix22` a 2500, las dos en 1,000 en todas las cargas.
- **PS-1: dif = +0,0792 · IC95 [+0,0747, +0,0838] · peor semilla +0,0694 · R11 = 0,0200** (4,0×).
- PS-4 confirma sus tres partes. PS-2 f = 1,001. PS-5 no concluyente (fallback cross-carga a L32).
- Fase B **frenada** por `FASE_MAX=A`. Nada de GPU desde el 26 de julio.
- El generador del informe **ya está corregido** (dejó de afirmar «concordantes» y «todas a
  N_common»), con test del caso nuevo. Suite verde.
- **Nada congelado todavía.** E-003′ se congela con hash y tag **antes** de tocar la GPU: si se
  congela después de correr `mix22`, B1-ter pierde todo su valor.

| qué | dónde |
|---|---|
| dictamen completo (9 preguntas, 11 decisiones) | `DICTAMEN_20260801.md` |
| handoff previo, aún sin responder formalmente | `HANDOFF_FABLE5_20260731.md` |
| campaña E1 | `resultados/E1/` (24 JSON + 24 `_propio` + `E1_informe.md`) |
| prereg congelado (Anexo B) | `protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.1.md` · SHA `0b93a36f…` |
| preprint del criterio de parada | DOI 10.5281/zenodo.21630279 (v2: …21631090) |
