"""Test de integración del flujo por sesiones de E1, sin JAX ni GPU.

Simula la campaña completa encadenando sesiones (entrenamiento instantáneo vía stubs) y verifica que:
  - cada sesión respeta el presupuesto y corta limpia,
  - la siguiente retoma exactamente donde quedó la anterior,
  - la fase A cierra en el bloque correcto y congela la tabla secundaria,
  - la fase B extiende todas las condiciones a N_common,
  - al terminar se emite el informe una sola vez.

Uso: python3 experimentos/E1/test_sesion_e1.py
"""
import json
import os
import shutil
import sys
import tempfile
import types

AQUI = os.path.dirname(os.path.abspath(__file__))
LOADS = [8, 16, 32, 64, 96, 128]
fallos = []

# convergencia simulada: delta necesita 3 bloques, softmax 1, mix22 2
CONVERGE_EN = {"delta": 7500, "softmax": 2500, "mix22": 5000}


def check(nombre, cond, extra=""):
    print(f"  {'✓' if cond else '✗'} {nombre}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        fallos.append(nombre)


def _stubs():
    jax = types.ModuleType("jax")
    jax.config = types.SimpleNamespace(update=lambda *a, **k: None)
    jax.devices = lambda: [types.SimpleNamespace(platform="cpu", device_kind="stub")]
    sys.modules["jax"] = jax

    entrenar = types.ModuleType("entrenar")

    def train_resumable(cond, seed, target, ckpt, **kw):
        open(ckpt, "w").write("stub")                       # el checkpoint existe, como en la corrida real
        vh = [{"step": s, "val_acc": 0.5 + 0.0001 * s} for s in range(500, target + 1, 500)]
        return {"cond": cond, "seed": seed, "steps": target}, vh

    def converged(vh, target):
        return None if not vh else target >= CONVERGE_EN[_CONTEXTO["cond"]]

    def eval_capacity(params, cond, loads, seed, reps):
        base = {8: 1.0, 16: 1.0, 32: .999, 64: .974, 96: .887, 128: .773}
        if cond != "delta":
            base = {L: min(1.0, v + .05) for L, v in base.items()}
        return {L: {1: base[L], 4: min(1.0, base[L] + .1), 16: min(1.0, base[L] + .2)} for L in loads}

    def eval_overwrite(params, cond, L, seed):
        return {"delta": .89, "softmax": 1.0, "mix22": .93}[cond] - (0.3 if L > 32 else 0)

    for n, f in (("train_resumable", train_resumable), ("converged", converged),
                 ("eval_capacity", eval_capacity), ("eval_overwrite", eval_overwrite)):
        setattr(entrenar, n, f)
    sys.modules["entrenar"] = entrenar

    modelos = types.ModuleType("modelos")
    modelos.count_params = lambda p: 192453
    sys.modules["modelos"] = modelos
    s09 = types.ModuleType("fase0_s09")
    s09.device_info = lambda: {"device_kind": "stub"}
    s09.notify = lambda *a, **k: None
    sys.modules["fase0_s09"] = s09


_CONTEXTO = {"cond": "delta"}


def correr_sesion(d, presupuesto_min, modo="sesion", n_seeds=2):
    os.environ.update({"RESULTS_DIR": d, "CONDS": "delta,softmax,mix22",
                       "N_SEEDS": str(n_seeds), "PRESUPUESTO_MIN": str(presupuesto_min), "MODO": modo})
    for m in ("e1_runner", "analisis_e1", "planificador"):
        sys.modules.pop(m, None)
    sys.path.insert(0, AQUI)
    import e1_runner

    # el stub de `converged` necesita saber qué condición se está entrenando
    orig = e1_runner.entrenar_unidad

    def envuelto(cond, seed, target, previos):
        _CONTEXTO["cond"] = cond
        return orig(cond, seed, target, previos)

    e1_runner.entrenar_unidad = envuelto
    e1_runner.main()
    return e1_runner


def estado_de(d, n_seeds=2):
    sys.path.insert(0, AQUI)
    import planificador as pl
    return pl.leer_estado(d, ["delta", "softmax", "mix22"], n_seeds)


_stubs()
d = tempfile.mkdtemp()
N = 2

print("=== Sesión 1 (presupuesto para 2 unidades) ===")
correr_sesion(d, presupuesto_min=95)          # ~2 unidades de 2745 s
est = estado_de(d)
hechas = sum(1 for c in est for s in est[c]["semillas"])
check("hizo exactamente 2 unidades", hechas == 2, f"{hechas} semillas con JSON")
check("empezó por delta (fija N_common y la carga de evaluación)",
      len(est["delta"]["semillas"]) == 2 and not est["softmax"]["semillas"])
check("dejó checkpoints en Drive", len([f for f in os.listdir(d) if f.endswith(".ckpt")]) == 2)

print("\n=== Sesiones siguientes hasta terminar la fase A ===")
for i in range(2, 12):
    correr_sesion(d, presupuesto_min=95)
    est = estado_de(d)
    if all(est[c]["faseA_cerrada"] for c in ("delta", "softmax", "mix22")):
        break
check(f"la fase A cerró tras {i} sesiones", all(est[c]["faseA_cerrada"] for c in est), i)
check("delta cerró en 7500 (necesitó 3 bloques)", est["delta"]["N_final"] == 7500, est["delta"]["N_final"])
check("softmax cerró en 2500 (converge enseguida)", est["softmax"]["N_final"] == 2500, est["softmax"]["N_final"])
check("mix22 cerró en 5000", est["mix22"]["N_final"] == 5000, est["mix22"]["N_final"])
prop = [f for f in os.listdir(d) if f.endswith("_propio.json")]
check("congeló la tabla secundaria de las 3 condiciones", len(prop) == 3 * N, len(prop))
sec = json.load(open(os.path.join(d, "e1_softmax_seed0_propio.json")))
check("la secundaria de softmax quedó en SU convergencia (2500), no en la de delta",
      sec["steps"] == 2500, sec["steps"])

print("\n=== Fase B: extensión a N_common ===")
for j in range(20):
    correr_sesion(d, presupuesto_min=95)
    if os.path.exists(os.path.join(d, "E1_informe.md")):
        break
est = estado_de(d)
check("todas las condiciones llegaron a N_common = 7500",
      all(est[c]["semillas"][s]["steps"] == 7500 for c in est for s in range(N)),
      {c: [est[c]["semillas"][s]["steps"] for s in range(N)] for c in est})
check("la tabla secundaria NO se pisó al extender (sigue en su convergencia propia)",
      json.load(open(os.path.join(d, "e1_softmax_seed0_propio.json")))["steps"] == 2500)
check("se emitió el informe", os.path.exists(os.path.join(d, "E1_informe.md")))

inf = open(os.path.join(d, "E1_informe.md"), encoding="utf-8").read()
check("el informe declara N_common = 7500", "N_common = 7500" in inf)
check("trae las dos tablas", "Tabla PRIMARIA" in inf and "Tabla SECUNDARIA" in inf)
check("trae los veredictos pre-registrados",
      all(k in inf for k in ("## PS-1", "## PS-2", "## PS-4", "## PS-5", "**P1.2**")))

print("\n=== Sesión extra sobre campaña terminada (no debe romper ni re-entrenar) ===")
antes = {f: os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".json")}
correr_sesion(d, presupuesto_min=95)
despues = {f: os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".json")}
check("no re-entrena nada ya hecho", antes == despues)
shutil.rmtree(d)

print("\n" + "=" * 60)
if fallos:
    print(f"✗ {len(fallos)} FALLO(S): " + "; ".join(fallos))
    sys.exit(1)
print("✓ FLUJO POR SESIONES VERIFICADO — encadena, retoma, cierra fases y emite el informe.")
