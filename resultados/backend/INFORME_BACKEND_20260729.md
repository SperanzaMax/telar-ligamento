# Reproducibilidad entre backends · CPU vs Tesla T4 · delta · 8/8 semillas

**Fecha de cierre:** 2026-07-29 09:18 (corrida local, 2026-07-28 22:30 → 2026-07-29 09:18)
**Corregido:** 2026-07-29 tarde — ver «Corrección» más abajo. **La primera versión de este
informe reportaba 3.00× y ese número estaba mal construido.**
**Condición:** `delta`, 8 semillas (0-7), hasta 2500 pasos, evaluación cada 500 (40 comparaciones).
**Referencia T4:** los `val_hist` de las corridas ya versionadas en `resultados/E1/`.
**Regenerar:** `MODO=informe python experimentos/backend/determinismo_cpu_vs_t4.py`

## Pregunta

Con una sola semilla se había medido que cambiar de dispositivo movía la métrica más que
cambiar de semilla. Con n=1 no se podía distinguir **sesgo sistemático del backend** de **simple
dispersión**. Ocho semillas en cada backend responden eso.

## Resultado

```
cambiar el DISPOSITIVO (T4 -> CPU, misma semilla):
   |dif| media = 0.03312   máx = 0.22371   n = 40

pareado por paso (la SD entre semillas NO es constante):
     paso  |dif| backend   SD semillas   razón
      500        0.03532       0.10056   0.35x
     1000        0.03621       0.07271   0.50x
     1500        0.04424       0.03838   1.15x
     2000        0.02749       0.00954   2.88x
     2500        0.02236       0.01380   1.62x

razón backend/semilla = 1.30x  (media de las razones por paso)

signo de (CPU - T4): 25 negativos de 40
media por semilla: -0.02611   t(7) = -1.62   p = 0.1501
-> sin sesgo detectable: es dispersión
```

## Corrección (2026-07-29)

La primera versión comparaba el `|dif|` medio de backend contra una constante,
`SIGMA_SEMILLA = 0.01105`, que es la **SD entre semillas a 7500 pasos**. Pero las diferencias de
backend se miden en el tramo **500–2500**, y ahí la SD entre semillas es otra cosa:

| paso | 500 | 1000 | 1500 | 2000 | 2500 | 5000 | 7500 |
|---|---|---|---|---|---|---|---|
| SD entre semillas (T4) | 0.101 | 0.073 | 0.038 | 0.010 | 0.014 | 0.009 | 0.011 |

Es decir: comparaba **ruido del transitorio temprano contra la SD del régimen tardío**. Eso
inflaba la razón de **1.30× a 3.00×**, y el «peor caso 20.25×» era el mismo artefacto — el
`|dif|` máximo de 0.224 cae en el **paso 1500 de la semilla 7**, donde la SD entre semillas vale
0.038, no 0.011.

El script quedó corregido: ahora calcula la SD paso por paso desde las corridas de T4 y compara
cada paso contra la SD de ese paso.

## Lectura, ya corregida

1. **No hay sesgo direccional del backend.** 25 de 40 diferencias negativas, `t(7) = -1.62`,
   `p = 0.1501`. Esta conclusión **no cambió** con la corrección: no depende de la SD de
   referencia, sale de las diferencias firmadas.

2. **El efecto de backend es del mismo orden que el de la semilla, no un múltiplo de él.**
   1.30× de media. Sigue siendo un término que el «media ± SD sobre N semillas» no contiene,
   pero es una afirmación mucho más modesta que la que se reportó primero.

3. **La razón depende fuerte del paso** (0.35× a 500, 2.88× a 2000). Con 5 puntos, el promedio
   de razones es inestable; no conviene citar un solo número sin la tabla.

4. **`|dif|` media = 166 % del margen R11 (0.0200).** Este cociente es correcto en términos
   absolutos, pero R11 se aplica a comparaciones **en N_common** (7500/10000), no en el
   transitorio. Comparar contra él una dispersión medida en 500–2500 es, de nuevo, mezclar
   regímenes.

## Alcance y límites

- **El límite principal:** se midió hasta 2500 pasos, que es **régimen transitorio**. Ahí las
  curvas suben rápido y cualquier perturbación numérica se amplifica. Para afirmar algo sobre
  el régimen donde se toman las decisiones haría falta medir a 7500–10000, que en esta CPU son
  ~26 h más.
- Vale para `delta`. No se probó en `softmax` ni `mix22`, que saturan en 1.000.
- Un solo modelo de GPU (Tesla T4) contra una sola CPU (i3-7100). No separa «CPU vs GPU» de
  «esta CPU vs esta GPU».
- La corrida de CPU se hizo con la frecuencia topeada al 75 % por temperatura. Cambia el tiempo
  por paso, no la aritmética.
- `p = 0.1501` con n=8 **no prueba** que no haya sesgo; prueba que si lo hay, es chico frente a
  la dispersión.

## Qué queda en pie para el protocolo

El gate de T4 del notebook de E1 sigue justificado: aun a 1.30×, mezclar dispositivos agrega
una fuente de variación que el diseño no contempla, y el peor caso puntual (0.224 de diferencia
en un checkpoint) es grande en términos absolutos. Lo que **no** se puede afirmar con estos
datos es que el backend domine sobre la semilla.

---

## ACTUALIZACIÓN 2026-07-30: las 8 semillas hasta 7500 pasos

La versión anterior medía hasta 2500, que es **régimen transitorio**, y ese límite estaba anotado
como el principal. Se extendió a 7500 (28 h de CPU) y **el resultado cambia de signo**.

```
                     |dif| backend    SD semillas    razón
transitorio  500-2500     0.03312       —            1.30x
TARDÍO      3000-7500     0.00604       —            0.53x
                          caída: 5.5x

global (n=120): razón 0.81x · 64 negativos de 120
t(7) = -1.42 · p = 0.1979  ->  sin sesgo detectable
```

**En el régimen donde se toman las decisiones, cambiar de backend mete la MITAD del ruido que
cambiar la semilla**, y el `|dif|` medio vale el **30 % del margen R11**, no el 166 % que se
reportó primero. Las trayectorias **convergen**: arrancan separadas por la aritmética distinta de
cada dispositivo y se juntan a medida que el entrenamiento avanza.

### Lo que esto significa

1. **El efecto grande era del transitorio.** Medir reproducibilidad en checkpoints tempranos
   sobreestima el problema en un orden de magnitud. Ese es el aporte metodológico transferible:
   quien mida a 2500 pasos y publique «el hardware mete tanto ruido como la semilla» está
   reportando un artefacto del régimen, no una propiedad del sistema.
2. **La ausencia de sesgo se sostuvo** en las tres muestras (n=3, 4 y 8) y en los dos horizontes.
   Es la conclusión más robusta del experimento.
3. **El gate de T4 sigue siendo buena práctica**, pero por prolijidad, no porque el backend
   domine: a 0.53× no domina nada.

### Control que salió gratis

seed0 no tenía checkpoint y se reentrenó desde cero. Dio `val_acc = 0.5182` en el paso 500,
idéntico al `0.51822` de la corrida original: **la CPU es determinista consigo misma**. Las
diferencias medidas vienen del cambio de backend, no de ruido de ejecución.
