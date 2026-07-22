# FREEZE — Pre-registro de seguimiento C3-vs-C2 v1.0

**Estado:** CONGELADO. Ratificado por Maxi el 2026-07-22, **antes de correr una sola semilla de C2 (delta)**.

## Artefacto congelado
- **Archivo:** `protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.0.md`
- **SHA-256:** `8b85aed730d82004bc2eba6836404b52ccbdd8150d082e73e64a9b00f6f8a20a`
- **Tamaño:** 4076 bytes
- **Fecha de freeze (UTC):** 2026-07-22T22:26:29Z

## Doble ancla (integridad + fecha)
1. **Fecha del pre-registro (antes de datos):** el contenido de predicciones fue pusheado como borrador v0.1
   en el commit `8a6e690`, con **timestamp server-side de GitHub 2026-07-22T19:08:03Z** — antes de que el
   S0.9 midiera C2 (delta) a cargas nuevas (la campaña arrancó con softmax; delta aún no corrió al momento
   del freeze).
2. **Integridad del contenido:** el **cuerpo** del v1.0 (desde «NO modifica el protocolo…», que incluye
   Motivación, Método y las predicciones PS-1/PS-2/PS-3) es **byte-idéntico** al del borrador anclado.
   - SHA-256 del cuerpo (ambas versiones): `729f5b339ce5dd305e0c15deccee72d8b0fc030af70b3b794adb61b6881d9dac`
   - Único cambio v0.1→v1.0: el encabezado de estatus (BORRADOR → CONGELADO). Verificable:
     ```
     git show 8a6e690:protocolo/PREREG_SEGUIMIENTO_C3vsC2_v0.1.md | awk '/^\*\*NO modifica/{p=1} p' | sha256sum
     awk '/^\*\*NO modifica/{p=1} p' protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.0.md | sha256sum
     ```

## Verificación
```bash
cd telar-ligamento/protocolo
sha256sum PREREG_SEGUIMIENTO_C3vsC2_v1.0.md
# -> 8b85aed730d82004bc2eba6836404b52ccbdd8150d082e73e64a9b00f6f8a20a
```

## Alcance
Predicciones PS-1 (rescate C3>C2), PS-2 (posición de C3 entre piso y techo, descriptiva) y PS-3 (monotonía
del rescate) quedan pre-registradas e inmutables. Baseline de referencia: C2 (delta). Carga de evaluación:
menor L con delta acc@1<95%, instanciada en S0.9. **No modifica** el protocolo madre v1.0 (`2f8ebb82…`).
Habilitado a partir de aquí: correr C2 (delta) y mirar sus cargas nuevas.
