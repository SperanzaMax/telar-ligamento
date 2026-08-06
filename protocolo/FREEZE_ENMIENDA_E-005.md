# Congelamiento · Enmienda E-005 — R-BANDA (2026-08-05)

Companion de anclaje. El artefacto congelado es el documento de enmienda; este archivo registra su
hash, el procedimiento de verificación y las condiciones bajo las que se emitió.

- **Archivo:** `ENMIENDA_E-005_RBANDA_20260805.md`
- **SHA-256:** `3e6572d2121840dc2fb9262d23a025e1bb131c845b4da75a5d57f1afecef4f0f`
- **Tamaño:** 18102 bytes

Verificación:

```bash
sha256sum ENMIENDA_E-005_RBANDA_20260805.md
python3 experimentos/verificar_anclas.py --requiere E-005
```

## Ancla pública — timestamps server-side de GitHub

| evento | timestamp (UTC) |
|---|---|
| push a `main` | **2026-08-06T00:10:10Z** |
| release publicada | *(pendiente — falta el tag firmado)* |

- Commit: `d6af405` · Tag firmado previsto: `ligamento-enmienda-e005`
- Release prevista: https://github.com/SperanzaMax/telar-ligamento/releases/tag/ligamento-enmienda-e005

El timestamp del push lo devuelve la API de eventos de GitHub
(`repos/SperanzaMax/telar-ligamento/events`, `PushEvent` con `head = d6af405`), no el autor.

**Estos timestamps los pone el servidor de GitHub, no el autor.** Deben preceder a la **primera hora
de calibración**, que al momento de este congelamiento no ha corrido nunca. Sin ellos la regla sería
una fecha escrita por quien ya podría haber visto dónde cae el borde.

> ⚠️ **El push ya está anclado (2026-08-06T00:10:10Z), pero falta el tag firmado.** El
> ancla local (SHA-256) protege contra alteración posterior, pero no es oponible a terceros: eso lo da
> el timestamp del servidor.

## Qué congela

**R-BANDA** y su instrumentación: la regla de elección de régimen (§3), el espacio candidato
enumerado (§3.1), el protocolo de calibración con su tope repartido (§4), las cuatro ramas de salida
pre-escritas (§5) y los dos resguardos sin costo de GPU —margen logit secundario y celdas ancla— (§6).

**No modifica** el protocolo madre (`2f8ebb82…`), el prereg v1.1 (`0b93a36f…`), la enmienda E-003′
(`ed8709c2…`) ni la E-004 (`6662724c…`). **No reformula ningún veredicto**: los umbrales de P2.1,
P2.2, P2.3, P3.1–P3.3 y P4.1–P4.3 quedan intactos, y ningún veredicto de E1 se reabre.

Lo único que fija es **el orden y el régimen** en que se corre E2–E4 — decisión de instrumento, no de
hipótesis.

## Momento — lo que le da valor

Se congela **antes de que exista un solo dato de calibración**, y antes de que el vocabulario
extendido exista siquiera como código. Verificado al momento de emitir:

```
find . -iname "*calib*" -o -iname "*rbanda*" -o -iname "*vocab256*" -o -iname "*vocab512*"
                                              → vacío (fuera de __pycache__)
git log --all --diff-filter=A -- '*calib*' '*banda*'
                                              → vacío
src/datos.py                                  → NK=128, NV=64, VOCAB=197 (E-001, sin tocar)
```

Estado del programa en este momento: **E1 completa y cerrada** (`delta` 8/8 @10000, `mix22` 8/8
@10000, `softmax` 8/8 @2500, `mix13` 8/8 @2500 → fila T) y **E2 no iniciada**.

## Estatuto epistémico — declarado, no minimizado

**Enmienda pre-datos con hash, NO pre-registro.** Redactada con la capa ciega ya perdida sobre E1: el
ejecutor vio todos los resultados de la campaña confirmatoria, incluidos PS-1 y la fila T.

Lo que la hace oponible es que **la elección de régimen queda determinada por el dato de
calibración**, producido sobre condiciones ya cerradas y neutral respecto de los contrastes nuevos.
Ninguna de las cuatro ramas favorece al ejecutor: R1 y R2 obligan a correr P2.2 con su umbral congelado
—que puede fallar—, y R3 prohíbe explícitamente registrar P2.2 como confirmada.

Origen: paso 1 prescrito por `AUDITORIA_CENSURA_E2E4_20260803.md` §5, que dejó la regla como
propuesta de redacción «a congelar antes de la primera hora de calibración». Este congelamiento
resuelve las indeterminaciones que la volvían inejecutable —condición de referencia tras
E-004, `k` de evaluación, espacio candidato enumerado y reparto del tope de 6 h— y corrige los dos
extremos de la banda, que en la redacción preliminar no cumplían su función (§3 del documento).

## Anclas previas de este expediente

| artefacto | SHA-256 | tag |
|---|---|---|
| protocolo madre v1.0 | `2f8ebb82…` | `ligamento-v1.0-freeze` |
| prereg de seguimiento v1.1 | `0b93a36f…` | `ligamento-prereg-seguimiento-v1.1` |
| enmienda E-003′ | `ed8709c2…` | `ligamento-enmienda-e003p` |
| enmienda E-004 | `6662724c…` | `ligamento-enmienda-e004` |
| **enmienda E-005** | **`3e6572d2…`** | `ligamento-enmienda-e005` |
