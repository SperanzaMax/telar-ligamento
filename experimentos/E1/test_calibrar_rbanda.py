"""Tests de la calibración de R-BANDA (E-005). Sin GPU y sin entrenar: la lógica de decisión
se prueba con medidas sintéticas, que es donde puede haber error de interpretación de la regla.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import calibrar_rbanda as C


def _reg(nombre, medidas, cerrada=True, nk=256, lmax=128):
    ok, cargas = C.cumple_banda(medidas)
    return {"escalon": nombre, "NK": nk, "L_max": lmax, "medidas": medidas,
            "cumple": ok, "cargas_en_banda": cargas, "biseccion_cerrada": cerrada}


def test_banda_es_la_congelada():
    assert C.BANDA == (0.50, 0.80), "la banda no puede cambiar sin enmienda"
    assert C.CARGAS_EN_BANDA == 3 and C.K_EVAL == 1
    assert C.SEMILLAS == (0, 1) and C.TOPE_PASOS == 2500
    assert C.COND == "softmax"
    assert C.en_banda(0.50) and C.en_banda(0.80), "los extremos son inclusivos"
    assert not C.en_banda(0.81) and not C.en_banda(0.49)
    print("  ok test_banda_es_la_congelada")


def test_techo_no_cuenta_como_banda():
    """El caso que motivó toda la enmienda: saturación NO puede leerse como banda cumplida."""
    medidas = {8: 1.0, 16: 1.0, 32: 1.0, 64: 1.0, 96: 1.0, 128: 0.9995}
    ok, _ = C.cumple_banda(medidas)
    assert not ok, "una grilla en el techo jamás debe cumplir la banda"
    print("  ok test_techo_no_cuenta_como_banda")


def test_banda_con_la_vieja_habria_pasado():
    """Con [0,30 · 0,98] una grilla casi saturada pasaba; con [0,50 · 0,80] no. Es el punto de §3."""
    medidas = {64: 0.975, 96: 0.972, 128: 0.968}
    assert all(0.30 <= a <= 0.98 for a in medidas.values()), "la banda vieja los aceptaba"
    ok, _ = C.cumple_banda(medidas)
    assert not ok, "la banda nueva debe rechazar un régimen que sigue midiendo contra el techo"
    print("  ok test_banda_con_la_vieja_habria_pasado")


def test_tres_cargas_exactas():
    assert C.cumple_banda({32: 0.79, 64: 0.65, 96: 0.55})[0]
    assert not C.cumple_banda({32: 0.79, 64: 0.65, 96: 0.45})[0], "dos en banda no alcanzan"
    ok, cargas = C.cumple_banda({16: 0.90, 32: 0.78, 64: 0.66, 96: 0.54, 128: 0.30})
    assert ok and cargas == [96, 64, 32], f"debe tomar las 3 MÁS ALTAS en banda, dio {cargas}"
    print("  ok test_tres_cargas_exactas")


def test_ramas_en_orden():
    ea_ok = _reg("E-a", {32: 0.78, 64: 0.66, 96: 0.55})
    rama, reg, _ = C.decidir([ea_ok])
    assert rama == "R1" and reg == "E-a"

    ea_no = _reg("E-a", {96: 0.99, 128: 0.985})
    eb_ok = _reg("E-b", {128: 0.79, 192: 0.64, 256: 0.52}, nk=512, lmax=256)
    rama, reg, _ = C.decidir([ea_no, eb_ok])
    assert rama == "R2" and reg == "E-b"

    eb_no = _reg("E-b", {128: 0.99, 192: 0.99, 256: 0.985}, nk=512, lmax=256)
    rama, reg, txt = C.decidir([ea_no, eb_no])
    assert rama == "R3" and reg is None
    assert "NO CORRIDA" in txt, "R3 debe prohibir leer P2.2 como confirmada"
    print("  ok test_ramas_en_orden")


def test_r4_no_se_confunde_con_r3():
    """La distinción que R4 existe para proteger: tope agotado ≠ frontera inalcanzable."""
    ea_no = _reg("E-a", {96: 0.99, 128: 0.985}, cerrada=True)
    eb_abierta = _reg("E-b", {128: 0.99}, cerrada=False, nk=512, lmax=256)
    rama, _, txt = C.decidir([ea_no, eb_abierta])
    assert rama == "R4", f"bisección abierta debe dar R4, dio {rama}"
    assert "SUSPENDIDA" in txt and "presupuesto insuficiente" in txt

    # un solo escalón corrido tampoco puede declarar frontera inalcanzable
    rama, _, _ = C.decidir([_reg("E-a", {96: 0.99}, cerrada=True)])
    assert rama == "R4", "sin correr E-b no se puede concluir R3"
    print("  ok test_r4_no_se_confunde_con_r3")


def test_biseccion_apunta_al_cruce():
    assert C.proponer_carga({}, 128) == 128, "arranca por la carga máxima"
    # cruce de 0,5 entre 64 y 128 → parte al medio
    assert C.proponer_carga({64: 0.90, 128: 0.20}, 128) == 96
    # todo por encima del cruce y ya en L_max → no hay nada más que partir
    assert C.proponer_carga({128: 0.99}, 128) is None
    # refina hasta agotar el intervalo entero, sin repetir cargas
    medidas, vistas = {64: 0.9, 128: 0.2}, set()
    for _ in range(40):
        p = C.proponer_carga(medidas, 128)
        if p is None:
            break
        assert p not in vistas, f"propuso {p} dos veces"
        vistas.add(p)
        medidas[p] = 0.9 if p < 96 else 0.2
    else:
        raise AssertionError("la bisección no termina")
    print(f"  ok test_biseccion_apunta_al_cruce (cerró tras {len(vistas)} refinamientos)")


def test_espacio_candidato_es_el_de_la_enmienda():
    assert [e["nombre"] for e in C.ESCALONES] == ["E-a", "E-b"], "orden: menor extensión primero"
    for e in C.ESCALONES:
        assert e["L_max"] <= e["NK"] // 2, f"{e['nombre']} viola L_max <= NK/2 (§3.1)"
        assert e["grilla"][-1] == e["L_max"]
    assert C.ESCALONES[0]["NK"] == 256 and C.ESCALONES[1]["NK"] == 512
    print("  ok test_espacio_candidato_es_el_de_la_enmienda")


def test_corte_por_convergencia_apagado_por_defecto():
    """El corte NO debe alterar las campañas de E1: default apagado, y el criterio es el de D-004."""
    import inspect
    from entrenar import train_resumable, converged
    firma = inspect.signature(train_resumable)
    assert firma.parameters["parar_al_converger"].default is False
    assert firma.parameters["voc"].default is not None
    # el criterio que usa el corte es el mismo `converged` de D-004, sin reinterpretación
    vh = [{"step": 500, "val_acc": 0.90}, {"step": 1000, "val_acc": 0.902}]
    assert converged(vh, 1000, window=500, tol=0.005) is True, "mejora 0.002 < 0.005 → convergió"
    vh = [{"step": 500, "val_acc": 0.80}, {"step": 1000, "val_acc": 0.90}]
    assert converged(vh, 1000, window=500, tol=0.005) is False, "mejora 0.10 → no convergió"
    assert converged(vh, 1500, window=500, tol=0.005) is None, "sin dato → None, no False"
    print("  ok test_corte_por_convergencia_apagado_por_defecto")


def run_all():
    print("Tests de calibración R-BANDA (E-005)")
    test_banda_es_la_congelada()
    test_techo_no_cuenta_como_banda()
    test_banda_con_la_vieja_habria_pasado()
    test_tres_cargas_exactas()
    test_ramas_en_orden()
    test_r4_no_se_confunde_con_r3()
    test_biseccion_apunta_al_cruce()
    test_espacio_candidato_es_el_de_la_enmienda()
    test_corte_por_convergencia_apagado_por_defecto()
    print("TODOS LOS TESTS VERDES ✓")


if __name__ == "__main__":
    run_all()
