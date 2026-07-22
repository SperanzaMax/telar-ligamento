# FREEZE — «Ligamento» Protocolo experimental v1.0

**Estado:** CONGELADO. Este archivo companion certifica el hash del protocolo pre-registrado.
Un archivo no puede contener su propio SHA-256, por eso el hash vive acá y no dentro del protocolo.

## Artefacto congelado
- **Archivo:** `protocolo/PROTOCOLO_v1.0.md`
- **SHA-256:** `2f8ebb829ddcaf9de0b4409b44567e84eff52bd5fa62bf1233836b166712f7f1`
- **Tamaño:** 40066 bytes · 298 líneas
- **Fecha de freeze (UTC):** 2026-07-22T13:19:30Z
- **Algoritmo:** `sha256sum` (coreutils)

## Verificación (reproducible)
```bash
cd telar-ligamento/protocolo
sha256sum PROTOCOLO_v1.0.md
# debe imprimir:
# 2f8ebb829ddcaf9de0b4409b44567e84eff52bd5fa62bf1233836b166712f7f1  PROTOCOLO_v1.0.md
```
Si el hash no coincide, el protocolo fue modificado después del freeze y el pre-registro queda invalidado.
Cualquier cambio de diseño posterior es un protocolo **nuevo** (v2.0…), no una edición de este.

## Procedencia del diseño
- **Diseño:** Fable 5 + Maxi (SperanzaMax). **Ejecutor:** Claude Opus 4.8 (Claude Code).
- **Revisión adversarial:** 3 rondas (umbrales v0.1→v0.2; lectura de ejecutor v0.2→v0.3;
  verificación de ejecutor v0.3→v0.3.1) + resolución de D1/D2 por consenso de los tres.
- **Decisiones de freeze (2026-07-22):**
  - **D1** — quitar el factor √2 de R11 (margen = máx(piso, 1.5×SD); apareo conservado; 3 veredictos como salvaguarda).
  - **D2** — carga de evaluación de E1 fijada empíricamente por saturación del baseline (T1 a L∈{8,16,32,64,96,128}).
  - Reconciliación hermana en TELAR-03 completada: `telar03/docs/reconciliation_softmax_L64.md`.

## Depósito
- **Depósito local:** este repo, `telar-ligamento/protocolo/` (protocolo + companion versionados juntos).

### Ancla pública de fecha (git tag firmado + GitHub) — REGISTRADO 2026-07-22
- **Repositorio:** https://github.com/SperanzaMax/telar-ligamento (público)
- **Commit del freeze:** `638cc89e9d6b099cd4634e87bb18fbdee66507da` (firmado, SSH ed25519, verificado `Good signature`)
- **Tag firmado:** `ligamento-v1.0-freeze` → https://github.com/SperanzaMax/telar-ligamento/releases/tag/ligamento-v1.0-freeze
- **Clave de firma:** ed25519 dedicada `telar_signing` (fingerprint `SHA256:ykmA9jBYXfxM5g7X8hyw29ZHp4mjbim+zbKYbEKX8OQ`, verificable vía `~/.config/git/allowed_signers`)
- **Timestamps server-side (los pone GitHub, fuera del control del autor = inatacables):**
  - repo `created_at`: `2026-07-22T13:48:33Z` · `pushed_at`: `2026-07-22T13:48:38Z`
  - release `published_at`: `2026-07-22T13:49:05Z`
- Estas fechas preceden a cualquier corrida principal (Fase 0 aún no ejecutada) → el reclamo de pre-registro
  queda sin flanco: el diseño estaba fijado y públicamente fechado antes de ver un solo resultado.

### DOI de Zenodo — REGISTRADO 2026-07-22
- **DOI:** [10.5281/zenodo.21495252](https://doi.org/10.5281/zenodo.21495252)
- **Record:** https://zenodo.org/records/21495252 (público, Open Access, licencia CC-BY-4.0)
- **Tipo:** Publication / Working paper · **Versión:** 1.0 · **Publicado:** 2026-07-22
- **Autor:** Speranza, Maximiliano Rodrigo (Independent Researcher)
- **Archivo depositado:** `telar-ligamento-v1.0-freeze.zip` (snapshot del freeze; contiene `PROTOCOLO_v1.0.md`
  con SHA-256 `2f8ebb82…` verificable). El protocolo **no se re-hasheó** al depositar: el hash y el tag git
  anclan la versión; Zenodo solo agrega el identificador citable. (El zip depositado es anterior a esta línea
  del companion, por construcción — un archivo no puede contener el DOI que se genera al publicarlo.)

## Regla de oro post-freeze
A partir de este archivo, **no se discute más diseño: se corre.** Orden: Fase 0 → E1 → E2 → E3 → E4.
Toda desviación operativa va a `desviaciones.md` con fecha y motivo, antes de mirar los resultados afectados.
