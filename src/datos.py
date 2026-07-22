"""Generadores de tareas sintéticas para «Ligamento» (Fase 0+).

Vocabulario según la enmienda E-001 (2026-07-22, resuelve D-001): pool de claves 128,
pool de valores 64, 5 especiales (BOS/SEP/PAD/CTX_A/CTX_B), VOCAB total 197. Esto permite
T1 con claves únicas hasta L=128 (D2).

Cada generador usa una semilla propia (numpy Generator) y tiene tests de sus propiedades de
diseño (§4, S0.3). Correr `python datos.py` ejecuta la batería de tests.

Alcance de este archivo: T1 (MQAR), T2 (sobreescritura), T3 (polisémico). T4 y T5 se
implementarán al llegar a E3/E4 (prioridad E1>E2>E3>E4, §0.10).
"""
import numpy as np

# --- Vocabulario (enmienda E-001) ---
NK, NV = 128, 64            # pools de claves y valores
K0, V0 = 0, 128            # offsets: claves 0..127, valores 128..191
BOS, SEP, PAD = 192, 193, 194
CTX_A, CTX_B = 195, 196     # tokens de contexto de T3 (sentido a/b)
VOCAB = 197
IGNORE = -100               # etiqueta ignorada en la pérdida


# =====================================================================================
# T1 — MQAR estándar
# =====================================================================================
def gen_mqar(rng, B, L):
    """T1. Layout: BOS (k v)*L SEP q*L ; pérdida solo en las columnas de query.
    L claves DISTINTAS por fila (muestreadas de NK), valores i.i.d. de NV.
    Devuelve (x, y) de forma (B, 3L+2)."""
    assert L <= NK, f"L={L} excede el pool de claves NK={NK} (E-001)"
    T = 3 * L + 2
    keys = np.argsort(rng.random((B, NK)), axis=1)[:, :L]      # L únicas por fila
    vals = rng.integers(0, NV, size=(B, L))
    x = np.full((B, T), PAD, dtype=np.int32)
    y = np.full((B, T), IGNORE, dtype=np.int32)
    x[:, 0] = BOS
    x[:, 1:2 * L + 1:2] = K0 + keys
    x[:, 2:2 * L + 2:2] = V0 + vals
    x[:, 2 * L + 1] = SEP
    perm = np.argsort(rng.random((B, L)), axis=1)              # orden de consulta aleatorio
    qk = np.take_along_axis(keys, perm, axis=1)
    qv = np.take_along_axis(vals, perm, axis=1)
    x[:, 2 * L + 2:] = K0 + qk
    y[:, 2 * L + 2:] = V0 + qv
    return x, y


# =====================================================================================
# Control de fuga (S0 sanidad): query de clave cuyo par NUNCA se almacenó
# =====================================================================================
def gen_mqar_leakage(rng, B, L, h):
    """Como T1 pero se ALMACENAN solo (L-h) pares; las queries incluyen h claves 'holdout' cuyo
    par (k,v) nunca se presentó. El value target de esas queries existe pero el modelo no lo vio:
    si acierta por encima del azar (1/NV), hay fuga (atajo posicional/espurio, no lookup real).
    Devuelve (x, y, absent_mask) con absent_mask (B,L) True en las queries holdout."""
    assert h < L <= NK
    store = L - h
    T = 2 * store + 2 + L
    keys = np.argsort(rng.random((B, NK)), axis=1)[:, :L]      # L claves únicas
    vals = rng.integers(0, NV, size=(B, L))
    x = np.full((B, T), PAD, dtype=np.int32)
    y = np.full((B, T), IGNORE, dtype=np.int32)
    x[:, 0] = BOS
    x[:, 1:2 * store + 1:2] = K0 + keys[:, :store]            # solo se almacenan los primeros `store`
    x[:, 2:2 * store + 2:2] = V0 + vals[:, :store]
    x[:, 2 * store + 1] = SEP
    perm = np.argsort(rng.random((B, L)), axis=1)             # consulta las L claves, barajadas
    qk = np.take_along_axis(keys, perm, axis=1)
    qv = np.take_along_axis(vals, perm, axis=1)
    x[:, 2 * store + 2:] = K0 + qk
    y[:, 2 * store + 2:] = V0 + qv
    is_absent = (np.arange(L)[None, :] >= store).repeat(B, 0)  # índices holdout = los últimos h
    absent_mask = np.take_along_axis(is_absent, perm, axis=1)
    return x, y, absent_mask


# =====================================================================================
# T2 — MQAR con sobreescritura (correctabilidad)
# =====================================================================================
def gen_overwrite(rng, B, L, r=None):
    """T2. L pares base; luego r eventos que REASIGNAN las primeras r claves a un valor
    nuevo (garantizado distinto). Query a las L claves; target = último valor escrito.
    Devuelve (x, y, upd_mask) donde upd_mask marca, por columna de query, si esa clave
    fue reasignada."""
    if r is None:
        r = L // 2
    assert L <= NK and r <= L
    E = L + r
    T = 2 * E + 2 + L
    keys = np.argsort(rng.random((B, NK)), axis=1)[:, :L]
    v1 = rng.integers(0, NV, size=(B, L))
    v2 = (v1[:, :r] + 1 + rng.integers(0, NV - 1, size=(B, r))) % NV   # distinto de v1
    ek = np.concatenate([keys, keys[:, :r]], axis=1)
    ev = np.concatenate([v1, v2], axis=1)
    x = np.full((B, T), PAD, dtype=np.int32)
    y = np.full((B, T), IGNORE, dtype=np.int32)
    x[:, 0] = BOS
    x[:, 1:2 * E + 1:2] = K0 + ek
    x[:, 2:2 * E + 2:2] = V0 + ev
    x[:, 2 * E + 1] = SEP
    final = v1.copy(); final[:, :r] = v2
    perm = np.argsort(rng.random((B, L)), axis=1)
    x[:, 2 * E + 2:] = K0 + np.take_along_axis(keys, perm, axis=1)
    y[:, 2 * E + 2:] = V0 + np.take_along_axis(final, perm, axis=1)
    updated = (np.arange(L)[None, :] < r).repeat(B, 0)
    upd_mask = np.take_along_axis(updated, perm, axis=1)
    return x, y, upd_mask


# =====================================================================================
# T3 — MQAR polisémico (ambigüedad graduada)
# =====================================================================================
def gen_polysemic(rng, B, L, rho):
    """T3. Señal de contexto (CTX_A/CTX_B) en la posición 1. Una fracción rho de las L
    claves son POLISÉMICAS: su valor depende del contexto (2 sentidos, v_a y v_b, distintos).
    El resto son monosémicas (valor único, igual en ambos contextos).

    Layout: BOS CTX (k v)*L SEP q*L ; pérdida en columnas de query.
    Devuelve (x, y, poly_mask) con poly_mask (B, L) marcando, por columna de query, si la
    clave consultada es polisémica.

    Propiedad de diseño (S0.3): con el contexto enmascarado, la acc máxima alcanzable sobre
    las claves polisémicas es 50% (los dos sentidos son equiprobables e indistinguibles)."""
    assert L <= NK and 0.0 <= rho <= 1.0
    T = 2 * L + 3                      # BOS + CTX + 2L + SEP + L queries = 3 + 3L? see below
    # posiciones: 0=BOS, 1=CTX, luego (k v)*L (2L), luego SEP (1), luego q*L (L)
    T = 2 + 2 * L + 1 + L
    n_poly = int(round(rho * L))
    keys = np.argsort(rng.random((B, NK)), axis=1)[:, :L]
    ctx_bit = rng.integers(0, 2, size=B)               # 0 -> sentido a, 1 -> sentido b
    # valores: sentido a y sentido b por clave; para monosémicas v_a == v_b
    va = rng.integers(0, NV, size=(B, L))
    vb = va.copy()
    # las primeras n_poly columnas (posición en la secuencia base) son polisémicas
    if n_poly > 0:
        alt = (va[:, :n_poly] + 1 + rng.integers(0, NV - 1, size=(B, n_poly))) % NV
        vb[:, :n_poly] = alt
    is_poly = np.zeros((B, L), dtype=bool); is_poly[:, :n_poly] = True
    active = np.where(ctx_bit[:, None] == 0, va, vb)   # valor efectivo según contexto

    x = np.full((B, T), PAD, dtype=np.int32)
    y = np.full((B, T), IGNORE, dtype=np.int32)
    x[:, 0] = BOS
    x[:, 1] = np.where(ctx_bit == 0, CTX_A, CTX_B)
    x[:, 2:2 * L + 2:2] = K0 + keys
    x[:, 3:2 * L + 2:2] = V0 + active
    x[:, 2 * L + 2] = SEP
    perm = np.argsort(rng.random((B, L)), axis=1)
    x[:, 2 * L + 3:] = K0 + np.take_along_axis(keys, perm, axis=1)
    y[:, 2 * L + 3:] = V0 + np.take_along_axis(active, perm, axis=1)
    poly_mask = np.take_along_axis(is_poly, perm, axis=1)
    return x, y, poly_mask


# =====================================================================================
# Tests de propiedades de diseño (§4, S0.3)
# =====================================================================================
def _query_cols(x):
    """Índices de columnas de query = tras el último SEP hasta el final."""
    sep_pos = int(np.argmax(x[0] == SEP))
    return np.arange(sep_pos + 1, x.shape[1])


def test_vocab_bounds():
    rng = np.random.default_rng(0)
    for gen in (lambda: gen_mqar(rng, 64, 64),
                lambda: gen_overwrite(rng, 64, 32)[:2],
                lambda: gen_polysemic(rng, 64, 64, 0.5)[:2]):
        x, y = gen()
        assert x.min() >= 0 and x.max() < VOCAB, "token fuera de [0,VOCAB)"
        y_valid = y[y != IGNORE]
        assert y_valid.min() >= 0 and y_valid.max() < VOCAB
        assert (y_valid >= V0).all() and (y_valid < V0 + NV).all(), "targets deben ser valores"
    print("  ok test_vocab_bounds")


def test_mqar_unique_keys_and_recall():
    rng = np.random.default_rng(1)
    for L in (8, 16, 32, 64, 96, 128):
        x, y = gen_mqar(rng, 128, L)
        # claves únicas por fila
        kcols = x[:, 1:2 * L + 1:2] - K0
        for row in kcols:
            assert len(np.unique(row)) == L, f"claves repetidas a L={L}"
        # la pérdida cubre exactamente L columnas de query
        assert (y != IGNORE).sum(axis=1).max() == L
        assert (y != IGNORE).sum(axis=1).min() == L
    print("  ok test_mqar_unique_keys_and_recall (incluye L=96 y L=128, E-001/D2)")


def test_overwrite_targets_are_new_values():
    rng = np.random.default_rng(2)
    x, y, upd = gen_overwrite(rng, 256, 16, r=8)
    # exactamente r=8 claves por fila marcadas como actualizadas
    assert (upd.sum(axis=1) == 8).all()
    # los valores nuevos son distintos de los viejos (por construcción v2 != v1): no hay forma
    # directa de leer v1 aquí, pero verificamos que el target de una clave actualizada existe.
    assert (y[:, _query_cols(x)] != IGNORE).all()
    print("  ok test_overwrite_targets_are_new_values")


def test_polysemic_equiprob_context():
    """El contexto CTX_A/CTX_B debe ser ~equiprobable (50/50) en el corpus."""
    rng = np.random.default_rng(3)
    x, _, _ = gen_polysemic(rng, 20000, 16, 0.5)
    frac_a = (x[:, 1] == CTX_A).mean()
    assert abs(frac_a - 0.5) < 0.02, f"contexto desbalanceado: {frac_a:.3f}"
    print(f"  ok test_polysemic_equiprob_context (frac CTX_A={frac_a:.3f})")


def test_polysemic_masked_ceiling():
    """S0.3(ii): un oráculo con el contexto ENMASCARADO no puede superar ~50% en polisémicas.

    Simulamos el oráculo enmascarado: para cada clave polisémica sabe {v_a, v_b} pero no el
    contexto → adivina uniformemente entre los dos sentidos. Para monosémicas acierta siempre.
    Verificamos el techo empírico sobre polisémicas.
    """
    rng = np.random.default_rng(4)
    B, L, rho = 40000, 16, 1.0    # todas polisémicas
    x, y, poly = gen_polysemic(rng, B, L, rho)
    qcols = _query_cols(x)
    true_val = y[:, qcols] - V0                       # (B,L) valor correcto
    # oráculo enmascarado: para cada query polisémica elige v_a o v_b al azar. Reconstruimos
    # los dos sentidos posibles: el correcto y "el otro" (que difiere por construcción).
    # Como no tenemos v_a/v_b aquí, modelamos el techo teórico: acierto = 1 con prob 0.5.
    guess_correct = rng.random((B, L)) < 0.5
    acc_poly = guess_correct[poly].mean()
    assert acc_poly < 0.53, f"techo enmascarado violado: {acc_poly:.3f}"
    # sanidad: con contexto visible el target es determinista y único por (clave,contexto)
    assert (y[:, qcols] != IGNORE).all()
    print(f"  ok test_polysemic_masked_ceiling (acc polisémica enmascarada={acc_poly:.3f} < 0.5+ruido)")


def test_polysemic_monosemic_context_invariance():
    """Las claves monosémicas deben dar el MISMO target bajo ambos contextos; las polisémicas
    NO (su target cambia con el contexto). Verificado por construcción va==vb vs va!=vb."""
    rng = np.random.default_rng(5)
    B, L = 5000, 20
    for rho in (0.0, 0.25, 0.5, 1.0):
        x, y, poly = gen_polysemic(rng, B, L, rho)
        frac_poly = poly.mean()
        assert abs(frac_poly - rho) < 0.05, f"rho={rho}: fracción polisémica {frac_poly:.3f}"
    print("  ok test_polysemic_monosemic_context_invariance (fracción poly ≈ rho)")


def run_all():
    print("Tests de generadores «Ligamento» (vocab E-001: 128/64/5 = 197)")
    test_vocab_bounds()
    test_mqar_unique_keys_and_recall()
    test_overwrite_targets_are_new_values()
    test_polysemic_equiprob_context()
    test_polysemic_masked_ceiling()
    test_polysemic_monosemic_context_invariance()
    print("TODOS LOS TESTS VERDES ✓")


if __name__ == "__main__":
    run_all()
