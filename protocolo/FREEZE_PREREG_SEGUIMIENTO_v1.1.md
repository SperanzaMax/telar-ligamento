# FREEZE — Pre-registro de seguimiento C3-vs-C2 v1.1

**Estado:** CONGELADO. Ratificado por Maxi (vía los veredictos O1–O4 de Fable 5) el 2026-07-23, **antes de
correr una sola semilla de E1**.

## Artefacto congelado
- **Archivo:** `protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.1.md`
- **SHA-256:** `0b93a36fdfb950f6187a05a015424263106d213ff8b08bbe070d676694458426`
- **Tamaño:** 14127 bytes
- **Fecha de freeze (UTC):** 2026-07-23T22:41:19Z

## Triple ancla (integridad + fecha + depósito)

1. **Herencia verificable del v1.0 congelado.** El bloque «## Motivación … PS-3» del v1.1 es **byte-idéntico**
   al del v1.0 (`8b85aed7…`, congelado 2026-07-22 antes de correr delta), que a su vez es byte-idéntico al
   borrador v0.1 anclado con timestamp server-side de GitHub (commit `8a6e690`, push 2026-07-22T19:08:03Z,
   **antes de los datos de C2**). Las predicciones PS-1/PS-2/PS-3 conservan, por lo tanto, su ancla original.
   - SHA-256 del cuerpo heredado (ambas versiones): `3fef46ed8c877cdd14916a5c312baef6762ce3f6d4468f0c1ef2affc6c436351`
   - Verificable:
     ```bash
     awk '/^## Motivación/{p=1} /^## Congelamiento/{p=0} p' protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.0.md | sha256sum
     awk '/^## Motivación/{p=1} /^---$/{p=0} p'             protocolo/PREREG_SEGUIMIENTO_C3vsC2_v1.1.md | sha256sum
     ```
2. **Integridad del contenido:** SHA-256 del archivo completo (arriba), más el commit del freeze y el **tag
   firmado** `ligamento-prereg-seguimiento-v1.1` en el repo público
   https://github.com/SperanzaMax/telar-ligamento (clave SSH ed25519 dedicada `~/.ssh/telar_signing`).
3. **Fecha server-side:** el timestamp del push del tag y del commit de freeze en GitHub, inatacable, precede
   a toda corrida de E1 (la campaña se lanza después de este freeze).

## Qué agrega el v1.1 sobre el v1.0

- **PS-4** (forma de la degradación: inicio como predicción de punto sobre `mediana(L₀)` con umbral 0.99;
  monotonía; pendiente creciente) y **PS-5** (anticorrelación capacidad↔correctabilidad), ambas con su
  **estatus epistémico declarado** (generadas viendo el snapshot @2500, confirmadas en la C2 convergida de E1).
- **Anexo A:** las tres condiciones de la nota de cierre de S0.9 (régimen único, márgenes desde C2 convergida,
  tabla @2500 rotulada snapshot).
- **Anexo B (resuelve O1):** `N_common`, checkpoint de convergencia propia, **doble reporte** y **regla de
  discordancia** — si las dos tablas discrepan, PS-1 es «no concluyente por sensibilidad al presupuesto».
- **Anexo C (resuelve O4):** T2 medida también en la carga de evaluación, con **fallback pre-registrado** a L32
  si queda pisada contra el suelo.
- **Precisión O5** en la regla de veredicto de PS-5: banda de retención del 50 % en la correlación parcial,
  porque el caso canónico del confound **atenúa** la parcial hacia cero en vez de invertirle el signo.

**No modifica** el protocolo madre (`PROTOCOLO_v1.0.md`, `2f8ebb82…`) ni el contenido de PS-1/PS-2/PS-3 del
v1.0. Lo único que el v1.1 agrega sobre esas predicciones es el **punto de medición** y una regla cuyo único
desenlace nuevo es «no concluyente»: **no puede confirmar nada que el v1.0 no confirmaría**.

## Verificación
```bash
cd telar-ligamento
python3 experimentos/verificar_anclas.py --requiere PREREG_SEGUIMIENTO_v1.1
python3 experimentos/E1/test_analisis_e1.py     # veredictos, incluidos los terceros estados
python3 experimentos/E1/test_agregador_e1.py    # doble tabla + regla de discordancia + fallback de T2
```

## Alcance
Habilitado a partir de aquí: **lanzar E1** (C1/C2/C3/C4, 8 semillas). Ninguna predicción se ajusta después de
ver datos. Los v1.0 y madre permanecen congelados e inalterados.

*Diseño: Maxi + Fable 5 (marco, veredictos O1–O4). Ejecución, lectura de ejecutor (O1–O4), precisión O5 e
implementación: Opus 4.8.*
