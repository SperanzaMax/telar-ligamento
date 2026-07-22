# Verificación de v0.3 contra `lectura_ejecutor_v0.2.md`

Opus 4.8, 2026-07-22. Chequeo item por item de las 12 resoluciones. **Veredicto global:
12/12 direccionadas, calidad alta. APRUEBO el congelamiento con 2 ajustes antes del hash y
1 nota opcional.**

| Flag ejecutor | Resolución v0.3 | Veredicto |
|---|---|---|
| A1 R2/E1 FLOPs | params ±5%, FLOPs reportados, cláusula de sesgo conservador | ✅ coincide |
| A2 R2 rompe en E4 | control **S⅓** + regla de desambiguación (3 estados, incl. «confundida por cómputo») | ✅ **mejora** lo que propuse |
| A3 P2.1 ρ=0 | Spearman en ρ∈{0.25…1}, chequeo ρ=0 → exactitud total | ✅ coincide exacto |
| O1 umbral multimodal | percentil 95 del nulo permutado (+ cableado en S0.5) | ✅ coincide |
| O2 margen equivalencia | ×√2 + diferencias apareadas por semilla | ⚠️ ver ajuste 2 |
| O3 ratio P1.2 | contraste lineal L=C3−0.5C1−0.5C2≥0, IC apareado | ✅ **mejora** (elimina el cociente) |
| O4 ruta C largo sec. | k=8 posiciones reservadas, PAD neutro en A/B/D | ✅ coincide |
| T1 S0.7 determinismo | hilos XLA fijos + tolerancia (<1e-6 pérdida, 0 en tarea) | ✅ coincide |
| T2 MoD causalidad | router causal por umbral + pérdida de carga, capacidad realizada reportada | ✅ **mejora** |
| D concat C3 | RMSNorm por cabeza antes de concat + W_O compartida | ⚠️ ver ajuste 1 |
| D LR selección | por exactitud de validación (desempate por pérdida) | ✅ ok (val, no test; simétrico) |
| D test T3 | 50/50 por clave + Bayes-óptimo enmascarado 50±1% + modelo | ✅ **mejora** |

## Ajuste 1 (antes del hash) — RMSNorm por cabeza debe aplicarse en C1 y C2, no solo en C3
Este es el punto que Fable5 anticipó como el de mayor riesgo, y sí hay que tocarlo. La
RMSNorm-por-cabeza de C3 es la resolución correcta de mi flag de escala (en TELAR-01 la rama
delta ya normaliza la lectura con `rmsn`, la softmax no → al concatenar heterogéneas hay
mismatch de escala). **Pero si solo C3 la lleva, las cabezas softmax de C3 dejan de ser
idénticas a las de C1**, y eso confunde P1.1 (C3≈C1) y P1.2: la diferencia dejaría de ser
"solo el tipo de regla". Fix de una línea: **aplicar RMSNorm-por-cabeza uniformemente en las
tres condiciones (C1, C2, C3)**, de modo que lo único que varíe entre condiciones sea la regla
de escritura de cada cabeza. Con eso la comparación queda limpia y el flag cerrado.

## Ajuste 2 (confirmar intención) — √2 y apareo empujan la equivalencia en la MISMA dirección
La v0.2 tenía el margen mal escalado; el arreglo es legítimo, pero la justificación de "se
compensa solo" está invertida y conviene que Maxi decida el rigor buscado. En un test de
equivalencia (declarar A≈B si IC(A−B) ⊂ ±margen): un **margen más grande** (×√2) hace la
equivalencia **más fácil** de declarar, y el **apareo** (IC más angosto) también la hace **más
fácil**. No se cancelan: **ambos aflojan el veredicto de equivalencia.** "Más conservador"
aplica a un test de *diferencia*, no de equivalencia. No es un error fatal —el margen sigue
anclado al ruido— pero si la intención era volver las equivalencias (P1.1, P2.2, P4.3) *más
exigentes*, el √2 va en la dirección contraria. Opción A: dejarlo (SESOI anclado al ruido de
diferencia, con apareo por potencia) y documentar que la equivalencia es deliberadamente
generosa. Opción B: usar margen ×1 (SD de una condición) con apareo, para una equivalencia más
estricta. Cualquiera sirve; solo hay que elegirla a propósito, no por creer que se compensan.

## Nota opcional (no bloquea) — yardstick de A2(ii)
En la regla de desambiguación de E4, `(S − S⅓)` mide el dividendo de cómputo de pasar de ⅓ a
denso en **los 4 bloques**, pero M solo agrega denso en **2 de 4**. Así `(S − S⅓)` sobreestima
el dividendo real de M → la condición (ii) queda **conservadora** (más difícil de pasar), que
es la dirección segura. Se puede dejar como está; si se quiere afinar, usar medio dividendo.

## Conclusión
Con el **ajuste 1 aplicado** (RMSNorm uniforme en C1/C2/C3) y el **ajuste 2 decidido**
(dirección del rigor de equivalencia), doy el **visto bueno a v0.3 para congelar**. El resto
está verificado y en varios puntos mejora lo que yo tenía en mente.
