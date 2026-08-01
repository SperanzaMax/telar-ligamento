# Congelamiento · Enmienda E-003′ al Anexo B (2026-08-01)

Companion de anclaje. El artefacto congelado es el documento de enmienda; este archivo registra su
hash, el procedimiento de verificación y las condiciones bajo las que se emitió.

- **Archivo:** `ENMIENDA_E-003p_20260801.md`
- **SHA-256:** `ed8709c2057af9808d156aceb6a42e5c19649bc749fabca22b1ab85121d2aad4`
- **Tamaño:** 11 191 bytes

Verificación:

```bash
sha256sum ENMIENDA_E-003p_20260801.md
python3 experimentos/verificar_anclas.py --requiere E-003p
```

## Ancla pública — timestamps server-side de GitHub (inatacables)

| evento | timestamp (UTC) |
|---|---|
| push a `main` | **2026-08-01T21:48:01Z** |
| release publicada | **2026-08-01T21:48:36Z** |

- Commit: `83a02a8` · Tag firmado: `ligamento-enmienda-e003p` (objeto anotado `9482b415…`,
  ED25519, *Good signature*)
- Release: https://github.com/SperanzaMax/telar-ligamento/releases/tag/ligamento-enmienda-e003p

**Estos timestamps los pone el servidor de GitHub, no el autor.** Preceden a la primera unidad de GPU
de la extensión de `mix22`, que al momento de este congelamiento no había corrido. Es lo que hace
oponible a terceros la declaración previa de B1-ter: sin ellos, el criterio sería una fecha escrita
por quien ya podría haber visto el resultado.

---

## Qué congela

Enmienda **E-003′** al Anexo B del prereg de seguimiento v1.1: **B1-bis** (excepción por saturación
sin consecuencia inferencial, con umbral ligado al margen y prueba de dominancia acotada) y
**B1-ter** (criterio de degradación material de C3 al extenderla hasta `N_common`).

**No modifica** el prereg v1.1 (`0b93a36f…`, verificado intacto el 2026-08-01) ni el protocolo madre
(`2f8ebb82…`). Se aplica **sobre** ellos y se lee junto a ellos.

## Momento — lo que le da valor

Se congela **antes** de que exista un solo dato de la extensión de `mix22`. En el momento del
congelamiento el estado de la campaña es:

| condición | N | fuente |
|---|---|---|
| `delta` | 10000 | fase A cerrada, 8/8 |
| `mix22` | **2500** | fase A cerrada, 8/8 — **la extensión NO corrió** |
| `softmax` | 2500 | fase A cerrada, 8/8 |

El informe agregado imprime en este momento: *«B1-ter — **NO APLICA**: la extensión no corrió (ambas
tablas a N=2500)»*. Ese renglón es la prueba de que el criterio se fijó antes del dato que juzga.

## Estatuto epistémico — declarado, no minimizado

**Esto es una enmienda pre-datos con hash, NO un pre-registro.** El pre-registro del programa es el
v1.1 y su SHA. La distinción se mantiene en el paper.

Se redactó con la **capa ciega ya perdida**: el ejecutor había visto todos los resultados de la fase
A, incluido PS-1. Auditada en una pasada por Fable5 el 2026-08-01, también sin ceguera, con el
criterio de **si las reglas prescriben lo mismo bajo el resultado invertido**. La auditoría encontró
que el borrador de B1-bis(i) era inejecutable —exigía `capacity = 1,0000` exacto, que `softmax` no
cumple en 12 de 48 celdas— y que la defensa de B1-ter por desviaciones estándar estaba mal
construida. Ambas correcciones están incorporadas en el artefacto congelado.

Registro completo del proceso, con la tabla de conveniencias declaradas por decisión:
`DICTAMEN_20260801.md` (§0 conflicto de rol, §10 post-auditoría).

## Ejecución automática

B1-ter no depende de juicio humano: `analisis_e1.veredicto_b1ter()` lo evalúa y el agregador imprime
su veredicto siempre, sobrescribiendo el de PS-1 si detecta degradación material. Tres ramas
cubiertas en `test_agregador_e1.py`. Es un fusible **más sensible que B3**: el veredicto de PS-1 se
caería con una caída de 0,0592 y B1-ter dispara con 0,0200 — factor **2,96×**.

## Anclas previas de este expediente

| artefacto | SHA-256 | tag |
|---|---|---|
| protocolo madre v1.0 | `2f8ebb82…` | `ligamento-v1.0-freeze` |
| prereg de seguimiento v1.1 | `0b93a36f…` | `ligamento-prereg-seguimiento-v1.1` |
| **enmienda E-003′** | **`ed8709c2…`** | `ligamento-enmienda-e003p` |
