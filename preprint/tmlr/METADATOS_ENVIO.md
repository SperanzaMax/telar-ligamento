# Metadatos del envío a OpenReview — para copiar y pegar

## Título

```
One Softmax Head in Four Restores Ceiling Recall: A Pre-Registered Replication with a Censored Effect and an Unreachable Frontier
```

## Abstract

OpenReview renderiza LaTeX inline entre `$…$`, así que este texto se puede pegar tal cual.

```
Linear-attention architectures trade the quadratic cost of softmax attention for a fixed-size recurrent state, and pay for it with a capacity ceiling on associative recall. Hybridizing the two along the head axis is an established remedy. We report a pre-registered replication that quantifies it, and two results that the pre-registration forced us to state against our own interest. First, in a multi-query associative recall benchmark at $d_{model}=64$ with four heads, the delta-rule condition falls to $.921$ and $.820$ accuracy at loads 96 and 128, while every hybrid condition sits at $1.000$; the paired difference at the pre-registered evaluation load is $+.0792$, $95\%$ CI $[+.0747, +.0838]$, four times the pre-registered margin, positive in eight of eight seeds. Second, the minimum dose tested---one softmax head in four---already restores the ceiling (worst cell deficit $1.22\times10^{-4}$), so the claim about the two-head mixture is one of sufficiency, not necessity, and the cost of the remedy is bounded above at one quarter of the heads. Critically, the effect is censored: all hybrid conditions saturate, so $+.0792$ is a lower bound and the measured statement is about the delta rule---it falls below ceiling at loads where hybrids do not---not about how much better hybrids are. A pre-registered calibration then searched for a regime with dynamic range, and failed under its own rules: doubling the key pool and the load left the reference condition at $1.000$, so the frontier is unreachable at this scale and the equivalence prediction it would have served is recorded as not run, never as confirmed. We argue the durable contribution is the pre-registration itself, and report the ledger of conveniences it refused.
```

## Keywords

```
linear attention, associative recall, pre-registration, hybrid attention, ceiling effects, replication
```

## Action Editor sugerido

El formulario pide recomendar uno. Candidatos por afinidad, en orden:

1. **Nicolas Gontier** (ServiceNow Research) — «transformers rnns, natural language
   processing». Es el eje exacto del paper: recurrencia frente a atención.
2. **Serguei Barannikov** (CNRS / Institut de Mathématiques de Jussieu) — «attention
   mechanism, large language models, transformer models».
3. **Andrew Lampinen** (Anthropic) — «language models, memory, episodic memory». Buen
   encaje conceptual con capacidad de memoria; considerar si se prefiere a alguien de
   industria.

Verificar en <https://jmlr.org/tmlr/editorial-board.html> que sigan activos antes de elegir:
la lista rota.

## Lo demás del formulario

- **Autores**: se cargan igual (OpenReview los oculta a los revisores). No los omitas.
- **PDF**: `e1_hibridacion_tmlr.pdf` — la versión anónima, sin `[accepted]`.
- **Supplementary material** (opcional, ZIP): alternativa al espejo anónimo si se prefiere
  no depender de un servicio externo que expira.
- **Previously published**: **No**. Ojo con la política: TMLR *sí* acepta solapamiento con
  preprints y con venues no archivales; *no* acepta versiones extendidas de papers ya
  publicados en conferencia. Este caso es limpio: nunca fue publicado en ningún lado.
- **Conflictos de interés**: declarar los dominios de las instituciones con las que hubo
  vínculo. Como investigador independiente, es corto.
- **Sin deadline**: TMLR es *rolling submission*. Se envía cualquier día.

## Antes de darle a enviar

- [ ] El PDF es la versión **anónima** (dice «Anonymous authors», no tu nombre).
- [ ] El enlace de `anonymous.4open.science` abre y no muestra los PDF (403).
- [ ] El perfil de OpenReview está **activado** (puede demorar: ver abajo).

## La cuenta (la tiene que crear Maxi)

En <https://openreview.net/signup>. Dos cosas que traban a los independientes:

1. **«Education & Career History» es obligatorio** y hay que cargar al menos una entrada.
   Como investigador independiente: posición *Independent Researcher*, sin dominio
   institucional. No inventes una afiliación — además contradiría la nota de autor del
   paper, que declara explícitamente que la UTN no auspició el trabajo.
2. **Un perfil con e-mail no institucional suele pasar por moderación manual**, y eso puede
   demorar días. Conviene crear la cuenta **ya**, aunque el envío se haga después. Cargar
   el **ORCID 0009-0005-0413-8554** ayuda: valida identidad y evita perfiles duplicados.
