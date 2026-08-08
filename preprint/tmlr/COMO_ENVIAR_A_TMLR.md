# Envío del paper de E1 a TMLR

## Lo que hay acá

Un solo fuente, `e1_hibridacion_tmlr.tex`, produce **las dos versiones**:

| versión | cómo | qué imprime |
|---|---|---|
| **envío** (la que se sube) | como está | «Anonymous authors / Paper under double-blind review», encabezado «Under review as submission to TMLR», repo anónimo, sin agradecimientos |
| **final** (si es aceptado) | destapar `\usepackage[accepted]{tmlr}` y poner `\camerareadytrue` | autor, e-mail, afiliación, GitHub real, DOI de Zenodo, ORCID, agradecimientos |

```
latexmk -pdf e1_hibridacion_tmlr.tex
```

Compila limpio, sin referencias indefinidas. 6 páginas.

## La restricción que manda: la cuota

Desde el **1-jul-2026** TMLR aplica cuota anual por autor (regla armónica generalizada,
N₁=2). **Autor único ⇒ 2 envíos por año.** Cuentan *todos* los envíos: aceptados,
rechazados, retirados y **desk rejects**. No hay segunda oportunidad barata: un envío
mal preparado gasta la mitad del año.

## Pasos

1. ~~**Repo anónimo.**~~ **HECHO 2026-08-08:**
   <https://anonymous.4open.science/r/hybrid-heads-mqar-2072> — ya está puesto en el `.tex`.
   - El ID **no** se llama `telar-ligamento` a propósito: ese nombre, googleado, lleva
     derecho al repo real.
   - Términos redactados: nombre, apellido, usuario de GitHub, «Maxi», ORCID, e-mail, UTN,
     los seis identificadores de Zenodo, el nombre del repo y el del tag firmado.
   - **`Display PDFs` desactivado**: los PDF de `preprint/` llevan el nombre y el ORCID
     adentro y la redacción de términos no los alcanza. Verificado: devuelven HTTP 403.
   - Expira el **2027-02-08**. Si la revisión se estira, hay que renovarlo.
2. **Crear cuenta en OpenReview** con el perfil completo (TMLR lo exige para asignar
   Action Editor).
4. **Sugerir Action Editor.** El envío pide recomendar uno; conviene elegir a alguien de
   atención eficiente / arquitecturas recurrentes.
5. **Subir el PDF anónimo** en <https://openreview.net/group?id=TMLR>.

## Lo que decide un revisor de TMLR

No novedad ni impacto. Dos preguntas:

1. **¿Los claims están sostenidos por evidencia precisa, convincente y clara?** — la más
   importante.
2. ¿Le interesaría a alguien de la audiencia de TMLR?

El paper está construido para la primera: el efecto declarado como cota inferior, la
predicción registrada como *no corrida*, el «ledger of conveniences». Ese es el argumento.

## Lo que el espejo anónimo NO puede tapar

El `README.md` y varios documentos del expediente imprimen el **SHA-256 completo del
protocolo congelado** (`2f8ebb829d…`), y el paper cita su forma truncada. Ese hash es el
argumento del pre-registro, así que no se puede quitar; pero buscado en Google lleva al
repo público y de ahí al autor.

Esto **no es un defecto del espejo, es inherente**: el ancla pública que prueba la
anterioridad es, necesariamente, pública. El doble ciego de TMLR no exige que
desanonimizar sea imposible — exige que el autor no se identifique y que el revisor no
salga a buscarlo. La sección *Open Practices* está redactada justo así: hashes verificables
contra la copia anónima, y anclas públicas declaradas como existentes, a revelar en la
aceptación.

Queda además todo el expediente en español (handoffs, dictámenes, cierres) visible en el
espejo. No identifica al autor, pero es ruido para un revisor. Si se quiere una superficie
más chica, la vía limpia es crear una rama `anon-submission` en GitHub con sólo `src/`,
`experimentos/E1/`, `resultados/`, `protocolo/` y un README en inglés, y anonimizar **esa**
rama en vez de `main`.

## Puntos abiertos (decisión de Maxi)

- **La autocita `speranza2026stopping`** aparece en las referencias con nombre y DOI de
  Zenodo. Citar trabajo propio en tercera persona está permitido bajo doble ciego y así
  está redactado («an internal stopping criterion … was later shown»), pero es lo único
  del PDF que apunta al autor. Alternativa: dejarla como está (recomendado — quitarla
  debilitaría el argumento) o anonimizarla como «(anonymized citation)».
- **Preprint público en paralelo.** TMLR lo permite explícitamente (arXiv, bioRxiv), pero
  debilita el doble ciego de hecho. Recomendación: esperar a la primera ronda.
- **NO subir este paper a Zenodo antes del envío.** Dos de los cinco rechazos de
  Preprints.org fueron literalmente por eso: «found to have been announced on another
  platform … multiple links and DOIs may cause problems with indexing services». La
  prioridad ya está anclada por el pre-registro congelado; el paper no necesita DOI propio.
