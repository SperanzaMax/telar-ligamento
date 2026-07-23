# Desviaciones — «Ligamento» (registro operativo, R0.4)

Cada entrada: fecha, motivo, alcance. Se registra **antes** de mirar los resultados afectados.

---

## D-003 · 2026-07-22 · R5 (early stopping) no implementado en la campaña S0.9 — pasos fijos

**Qué.** El runner `fase0_s09.py` corrió con **STEPS=2500 fijos, sin el early stopping por validación** que
manda R5 («paciencia 2000, tope 20 000»). Detectado al analizar la curva de entrenamiento de `delta_seed0`:
train acc 0.8411 (step 2000) → 0.9353 (step 2500) — **delta aún subía fuerte, no convergió** a 2500. En cambio
softmax converge holgado antes de 2500 (queda como está).

**Consecuencia.** La eval por carga de delta a 2500 (L128 acc@1=0.755, carga de evaluación preliminar L96) es un
**snapshot no convergido = lower bound de la capacidad de delta**. No interpretable como el plateau real ni como
ancla del prereg C3-vs-C2.

**Estado:** corregida por D-004. La foto a 2500 se conserva como snapshot documentado, no como resultado principal.

## D-004 · 2026-07-22 · Corrección: extensión uniforme hasta convergencia (criterio explícito)

**Fechada ANTES de mirar la eval por carga de las semillas extendidas** (requisito de §0.4).

**Criterio de convergencia (pre-registrado acá):** una condición/semilla converge cuando la **mejora de accuracy
de validación es < 0.5 puntos en la ventana de los últimos 500 pasos**. Val-acc = media de acc@1 sobre las cargas
discriminantes {64, 96, 128} en un set de validación con semilla propia.

**Procedimiento (uniforme, nunca por semilla individual):**
1. Completar las 8 semillas de delta a 2500 (consistencia de la pasada actual). Softmax queda a 2500 (cumple el
   criterio).
2. Fase de extensión (`fase0_s09_extend.py`), en **bloques de +2500 pasos aplicados a TODAS las semillas de delta
   por igual**, hasta que **todas** cumplan el criterio o se alcance el **tope duro de 10 000 pasos**.
3. **Resultado principal = la tabla convergida** (todas al mismo N, `delta_conv_seed*.json`). La foto a 2500
   (`delta_seed*.json`) = snapshot documentado.

**Nota de implementación (registrada al construir el mecanismo).** El runner base `fase0_s09.py` guardó solo
métricas, **no los pesos**, así que el **primer bloque de la extensión re-entrena delta desde 0** (re-computa los
2500 ya hechos + 2500 nuevos); los bloques siguientes (5000→7500→…) **sí reanudan** desde checkpoint de pesos.
La reanudación se validó **determinista bit-a-bit** (`max|Δ| = 0.00e+00` reanudado vs. corrido), así que el
re-cómputo produce exactamente los mismos primeros 2500 que el snapshot base — sin inconsistencia, solo costo.
*(Lección: los runners de E1/TELAR-03 guardan checkpoint de pesos desde el arranque — ya en `train_resumable`.)*

**Motivo y costo real.** El 0.755@L128 y la carga de evaluación del prereg C3-vs-C2 solo son interpretables
anclados a números **convergidos**. Costo real (mayor que la estimación inicial de ~5 h, que asumía reanudación
desde 2500): **~10 h de T4 si delta converge a 5000 · ~15 h si a 7500** (por el re-cómputo del primer bloque).
Una noche larga, contra el costo de que la primera tabla real de Ligamento nazca con un asterisco en su cifra más
importante.

**Para el futuro (fuera de esta corrección):** la restauración plena de R5 (early stopping con paciencia, tope
10k) queda para E1 real y TELAR-03 Fase 2. Consenso de los tres.

---

## D-002 · 2026-07-22 · Presupuesto de parámetros por encima de lo declarado (observación, NO bloqueante)

**Qué.** La arquitectura §5 tal como está especificada (d_model=64, H=4, 4 bloques, 4 proyecciones D×D
por bloque, FFN hidden 192) produce un modelo softmax de **192 453 parámetros**, por encima del rango
«≈100k–150k» declarado en §3/§5. Desglose: bloques ≈167 040 (4 × ~41 760) + vocab (E-001) 25 413.
El grueso del exceso es la arquitectura misma (~167k sin vocab), no la enmienda E-001.

**Estado:** ABIERTO como observación; **no bloquea nada, no se corrige** (la arquitectura está congelada).
Es una inconsistencia pre-existente entre el presupuesto declarado y la arquitectura declarada, detectada al
instanciar el modelo. **R2 (params ±5% ENTRE condiciones) no se ve afectado**: es un criterio relativo, y
el embedding/FFN/proyecciones son comunes a C1..C4; la diferencia entre condiciones viene de las cabezas.
El presupuesto absoluto es una guía de §3, no un criterio de veredicto. Se documenta para trazabilidad y para
que el informe reporte el conteo real (S0.6 exige la tabla de params por condición de todos modos).

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
