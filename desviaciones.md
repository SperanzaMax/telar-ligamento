# Desviaciones — «Ligamento» (registro operativo, R0.4)

Cada entrada: fecha, motivo, alcance. Se registra **antes** de mirar los resultados afectados.

---

## D-001 · 2026-07-22 · CONFLICTO DE ESPECIFICACIÓN (bloqueante de Fase 0, frenado por §0.5)

**Estado:** **RESUELTO 2026-07-22 por la enmienda E-001 (opción a, ratificada por Maxi).** Ver
`protocolo/FREEZE_v1.0.md` § «Enmiendas post-freeze → E-001». Resolución: vocab de T1–T4 = 128 claves +
64 valores + 5 especiales (BOS/SEP/PAD/CTX_A/CTX_B) = **197**; params de vocab 25 413, compartidos e
idénticos en toda condición → R2 intacto; +13 029 params uniformes vs. lectura literal ≤96. Causa
registrada con nombre: **defecto de redacción de D2 (Fable 5), detectado por el ejecutor en
implementación** antes de correr. El protocolo hasheado NO se tocó (SHA-256 `2f8ebb82…` intacto).

### Registro original (se conserva)

**Qué.** Tensión interna en el protocolo congelado v1.0 entre D2 y §4:
- **§6 / D2** extienden T1 a `L ∈ {8, 16, 32, 64, 96, 128}` y definen la carga de evaluación de
  P1.1 como la menor L donde C1 (softmax) cae < 95%.
- **§4** especifica `vocabulario ≤ 96 tokens` y T1 con **claves sin repetir** (MQAR estándar,
  port de TELAR-01 `gen_batch`: `argsort(random)[:, :n]` → n claves distintas de un pool de NK).

**Por qué es infactible bajo lectura literal.** Con claves únicas, L=128 requiere ≥128 símbolos de
clave distintos; L=96 requiere ≥96. Pero el vocab total (claves + valores + BOS/SEP/PAD) es ≤96.
TELAR-01 usaba NK=64 claves + NV=64 valores + 3 especiales = 131 y topaba justo a L=64. No hay forma
de muestrear 96 ni 128 claves únicas dentro de un vocab de 96 que además aloje valores y especiales.
→ Las dos cargas nuevas de D2 (96, 128) — las que importan para decidir si softmax satura — no se
pueden instanciar con la definición de tarea de §4.

**Origen probable.** El `vocab ≤ 96` viene del régimen L≤64 de v0.1/v0.2 (TELAR-01). D2 se agregó en
v0.3.1→v1.0 y extendió las cargas sin reconciliar §4. Inconsistencia introducida tarde; detectada al
implementar los generadores (antes de correr una sola semilla — el orden correcto).

**Por qué se frena y no se improvisa.** El protocolo está pre-registrado y firmado (tag
`ligamento-v1.0-freeze`). Elegir la resolución cambia el diseño de la tarea y podría sesgar P1.1;
§0.5 obliga a frenar y reportar. La resolución debe pre-registrarse como enmienda antes de correr.

**Opciones candidatas (para que decidan Maxi/Fable5 — el ejecutor NO elige):**
- **(a) Separar el espacio de claves del de valores** y relajar el «≤96» a lo que D2 exige:
  claves de un pool ≥128, valores en un rango propio (p.ej. ≤64), + especiales. Enmienda mínima,
  no toca ninguna predicción, solo el número de §4. Es la más limpia. Riesgo: un vocab de claves
  mayor cambia levemente la dificultad de embedding, pero igual para todas las condiciones de E1
  (no confunde P1.1). **Recomendación tentativa del ejecutor**, sujeta a ratificación.
- **(b) Claves compuestas (bigramas)**: cada clave = 2 tokens de un vocab chico → 96² claves. Cambia
  el layout de secuencia y los largos; más invasivo; interactúa con la igualación de largo (O4) y con
  el conteo de FLOPs. Menos preferible.
- **(c) Acotar L al máximo que el vocab permita** (no extender a 96/128). Rechaza de facto a D2:
  si el vocab corta L antes de que softmax caiga, la pregunta de D2 («¿satura el baseline?») queda
  sin responder. Va contra el objetivo mismo de D2. No recomendada.

**Alcance.** Bloquea la instanciación de T1 (E1) a L∈{96,128} y, por lo tanto, S0.9 (carga de
evaluación de E1) y la corrida de C1. NO bloquea T2 (correctabilidad, L≤64), ni T3/T4/T5, ni la
medición de ruido/márgenes de las otras familias (P2.x, P3.x, P4.x). Se puede avanzar con el resto de
la infraestructura mientras se decide D-001.
