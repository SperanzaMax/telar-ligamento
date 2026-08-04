# Congelamiento · Enmienda E-004 — promoción de `mix13` (2026-08-03)

Companion de anclaje. El artefacto congelado es el documento de enmienda; este archivo registra su
hash, el procedimiento de verificación y las condiciones bajo las que se emitió.

- **Archivo:** `ENMIENDA_E-004_mix13_20260803.md`
- **SHA-256:** `6662724cff8bcee36a8076ad2a04df0a56cd6c14857968162e7abee42c11c684`
- **Tamaño:** 6 492 bytes

Verificación:

```bash
sha256sum ENMIENDA_E-004_mix13_20260803.md
python3 experimentos/verificar_anclas.py --requiere E-004
```

## Qué congela

La **tabla de interpretación de `mix13`** (§4 del documento): el mapeo de cada resultado posible a lo
que el paper puede afirmar, en tres filas exhaustivas y mutuamente excluyentes evaluadas en orden.
La tabla congelada **es** la promoción de `mix13` de exploratoria a confirmatoria: sin ella, correr
`mix13` y leerla después sería juicio post-exposición.

**No modifica** el protocolo madre (`2f8ebb82…`), el prereg v1.1 (`0b93a36f…`) ni la enmienda E-003′
(`ed8709c2…`). No reabre ningún veredicto de E1: PS-1 quedó cerrado por el prereg y por B1-ter.

## Momento — lo que le da valor

Se congela **antes de que exista un solo dato de `mix13`**. Verificado al momento de emitir:

```
find . -name "*mix13*" -o -name "*mix31*"     → vacío (fuera de __pycache__)
git log --all --diff-filter=A -- '*mix13*'    → vacío
```

`KIND_RULES["mix13"]` existe en `src/modelos.py:33` desde el 2026-07-22, pero la condición nunca fue
agendada: el planificador corrió siempre con `CONDS=delta,softmax,mix22`. El estado de la campaña en
este momento es E1 **completa y cerrada** (`delta` 8/8 @10000, `mix22` 8/8 @10000, `softmax` 8/8
@2500) y `mix13` **0/8**.

Si `mix13` hubiera corrido antes de este congelamiento, la promoción quedaría vedada y correspondería
la Salida B (§6 del documento): degradar el claim, no leer el dato después.

## Estatuto epistémico — declarado, no minimizado

**Enmienda pre-datos con hash, NO pre-registro.** Se redacta con la capa ciega perdida sobre E1: el
ejecutor vio todos los resultados de la campaña confirmatoria, incluido PS-1. Lo que la hace oponible
no es ceguera —no la hay— sino que **la consecuencia de cada resultado posible queda fijada antes de
que el resultado exista**, y que ninguna de las tres filas favorece al ejecutor: la fila T degrada el
claim de `mix22` a suficiencia-no-necesidad, la fila D lo restringe a «≥ 2 cabezas», y solo la fila B
conserva la lectura actual.

Origen: respuesta de la capa de razonamiento (Fable 5) del 2026-08-03 a la pregunta 1 de
`CIERRE_E1_20260803.md`, que identificó la inconsistencia entre «`mix13` es lo único que acota el
claim» y «`mix13` es prioridad media», y prescribió la promoción por enmienda pre-datos con tabla de
interpretación congelada — el patrón de B1-ter.

## Anclas previas de este expediente

| artefacto | SHA-256 | tag |
|---|---|---|
| protocolo madre v1.0 | `2f8ebb82…` | `ligamento-v1.0-freeze` |
| prereg de seguimiento v1.1 | `0b93a36f…` | `ligamento-prereg-seguimiento-v1.1` |
| enmienda E-003′ | `ed8709c2…` | `ligamento-enmienda-e003p` |
| **enmienda E-004** | **`6662724c…`** | `ligamento-enmienda-e004` |
