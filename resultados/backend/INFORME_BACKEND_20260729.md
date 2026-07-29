# Reproducibilidad entre backends · CPU vs Tesla T4 · delta · 8/8 semillas

**Fecha de cierre:** 2026-07-29 09:18 (corrida local, 2026-07-28 22:30 → 2026-07-29 09:18)
**Condición:** `delta`, 8 semillas (0-7), hasta 2500 pasos, evaluación cada 500.
**Referencia T4:** los `val_hist` de las corridas ya versionadas en `resultados/E1/`.
**Regenerar:** `MODO=informe python experimentos/backend/determinismo_cpu_vs_t4.py`

## Pregunta

Con una sola semilla se había medido que cambiar de dispositivo movía la métrica más que
cambiar de semilla. Con n=1 no se podía distinguir **sesgo sistemático del backend** (CPU
rinde consistentemente distinto que T4) de **simple dispersión** (el backend agrega ruido sin
dirección). Ocho semillas en cada backend responden eso con un t-test sobre las diferencias
firmadas.

## Resultado

```
cambiar el DISPOSITIVO (T4 -> CPU, misma semilla):
   |dif| media = 0.03312   máx = 0.22371   n = 40
cambiar la SEMILLA (8 semillas en T4):
   SD          = 0.01105

razón backend/semilla = 3.00x (media) · 20.25x (peor)
como fracción del margen R11 (0.0200): 166%

signo de (CPU - T4): 25 negativos de 40
media por semilla: -0.02611   t(7) = -1.62   p = 0.1501
-> sin sesgo detectable: es dispersión
```

## Cómo evolucionó al sumar semillas

| n | razón media | peor caso | % de R11 | t | p | veredicto |
|---|---|---|---|---|---|---|
| 1 | 1.16× | — | 64 % | — | — | no computable |
| 3 | 1.66× | 5.59× | 92 % | t(2) = −2.64 | 0.1182 | dispersión |
| 4 | 1.89× | 6.52× | 104 % | t(3) = −2.60 | 0.0802 | dispersión |
| 8 | **3.00×** | **20.25×** | **166 %** | t(7) = −1.62 | **0.1501** | **dispersión** |

**Advertencia metodológica sobre esta tabla.** Entre n=3 y n=4 el `p` bajó (0.1182 → 0.0802) y
eso invitaba a extrapolar que con más semillas cruzaría 0.05 y el veredicto pasaría a sesgo
sistemático. **No ocurrió: con las 8 semillas el `p` subió a 0.1501.** La caída era ruido de
muestra chica. Queda como recordatorio de que las filas intermedias de esta tabla no son
evidencia parcial de una tendencia, son estimaciones inestables.

## Lectura

1. **No hay sesgo direccional del backend.** 25 de 40 diferencias negativas y `p = 0.1501`: no
   se puede afirmar que CPU rinda sistemáticamente por debajo de T4. Es un resultado limpio,
   no un «casi».

2. **Sí hay dispersión, y es grande.** Cambiar de backend mete **3 veces** el ruido de cambiar
   de semilla, y en el peor caso **20 veces**. El `|dif|` máximo, 0.224, es un orden de
   magnitud por encima del margen R11 (0.020).

3. **Consecuencia para el reporte de resultados.** El campo reporta «media ± SD sobre N
   semillas». Esa SD no contiene el término de backend, que acá vale **166 % del margen de
   decisión entero**. Dos laboratorios que corran el mismo código con la misma semilla en
   dispositivos distintos pueden diferir más que el efecto que están midiendo.

4. **Por qué el protocolo exige T4.** Este resultado es la justificación cuantitativa del gate
   de hardware del notebook de E1: mezclar dispositivos dentro de una misma campaña introduce
   una fuente de variación mayor que la que el diseño considera decisiva.

## Alcance y límites

- Vale para `delta` hasta 2500 pasos. No se probó en `softmax` ni `mix22`, que saturan en
  1.000 y donde el efecto probablemente sea menor por efecto de techo.
- Un solo modelo de GPU (Tesla T4) contra un solo CPU (i3-7100, 4 hilos). No separa «CPU vs
  GPU» de «esta CPU vs esta GPU».
- La corrida de CPU se hizo con la frecuencia topeada al 75 % por temperatura. Eso cambia el
  tiempo por paso, no la aritmética: no afecta las métricas.
- `p = 0.1501` con n=8 **no prueba que no haya sesgo**; prueba que si lo hay, es chico frente a
  la dispersión. Descartarlo del todo pediría bastante más potencia.
