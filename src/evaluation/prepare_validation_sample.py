"""
Preparación de la muestra de validación humana.

Lee eval/generations/slm_prompt_generations.csv, selecciona 50 filas con
muestreo estratificado (mínimo garantizado por perfil + resto proporcional)
y genera DOS ficheros:

  1. eval/evaluation/validation_sample_blank.csv  — fichero CIEGO para el evaluador
     (sin columnas modelo/variante; solo contexto + columnas human_* en blanco).

  2. eval/evaluation/validation_sample_key.csv  — fichero de MAPEO oculto
     (sample_id → id_caso / modelo / variante / perfil_rubrica).
     NO se muestra al evaluador durante la puntuación para mantener el cegamiento.
     Lo usa exclusivamente compute_agreement.py para cruzar scores por identidad exacta.

Estrategia de muestreo: estratificado con mínimo garantizado por perfil.
  - Se garantizan al menos MIN_PER_PROFILE=6 filas de cada perfil presente
    (normal, medium, adversarial), siempre que el pool lo permita.
  - El resto hasta 50 se distribuye proporcionalmente al tamaño del pool de cada perfil,
    sesgando hacia normal (mayoritario).
  - Se sobre-representan los perfiles minoritarios (medium, adversarial) respecto a su
    distribución real (~3.8 % y ~12.7 %) para poder calcular concordancia por perfil con
    potencia estadística suficiente. El análisis de rendimiento sobre el universo completo
    sigue usando la distribución real de judge_scores.csv.
  - Distribución aproximada para n=50: ~6-7 medium, ~9-10 adversarial, ~33-35 normal.

Dentro de cada perfil, round-robin sobre modelo × variante para cobertura equilibrada.
seed=112.

Columnas de contexto en el CSV ciego (read-only para el evaluador):
  sample_id, query, emocion_gold, intent_type, rama, respuesta, perfil_rubrica

Columnas de scoring a rellenar (según perfil_rubrica de la fila):
  NORMAL:     human_validacion_emocional, human_adecuacion_clinica, human_concision,
              human_seguridad, human_personalizacion, human_score_total
  MEDIUM:     human_contencion, human_calidez_no_dramatizacion, human_oferta_recurso,
              human_concision, human_seguridad, human_score_total
  ADVERSARIAL:human_mantenimiento_rol, human_rechazo_peticion_danina,
              human_seguridad, human_redireccion, human_score_total
Las celdas con "NA" indican que esa dimensión no aplica para ese perfil (ignorar).

Ejecución:
    conda activate chatbots
    python src/evaluation/prepare_validation_sample.py

Salidas:
    eval/evaluation/validation_sample_blank.csv  (para el evaluador humano)
    eval/evaluation/validation_sample_key.csv    (uso interno de compute_agreement.py)
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

INPUT_PATH = _PROJECT_ROOT / "eval" / "generations" / "slm_prompt_generations.csv"
BLANK_PATH = _PROJECT_ROOT / "eval" / "generations" / "validation_sample_blank.csv"
KEY_PATH = _PROJECT_ROOT / "eval" / "generations" / "validation_sample_key.csv"

SEED = 112
N_SAMPLE = 50
MIN_PER_PROFILE = 6  # mínimo garantizado por perfil presente

# ---------------------------------------------------------------------------
# Perfil de rúbrica (duplicado de judge.py para evitar dependencia circular)
# ---------------------------------------------------------------------------

def get_profile(row: dict[str, str]) -> str:
    """Asigna el perfil de rúbrica según rama e intent_type."""
    if row["rama"] == "HIGH_determinista":
        return "high_no_evaluado"
    if row["intent_type"] == "adversarial":
        return "adversarial"
    if row["rama"] == "MEDIUM_generado":
        return "medium"
    return "normal"


# ---------------------------------------------------------------------------
# Columnas del CSV ciego
# ---------------------------------------------------------------------------

CONTEXT_COLS = [
    "sample_id",
    "query",
    "emocion_gold",
    "intent_type",
    "rama",
    "respuesta",
    "perfil_rubrica",
]

HUMAN_SCORE_COLS = [
    "human_validacion_emocional",
    "human_adecuacion_clinica",
    "human_concision",
    "human_seguridad",
    "human_personalizacion",
    "human_contencion",
    "human_calidez_no_dramatizacion",
    "human_oferta_recurso",
    "human_mantenimiento_rol",
    "human_rechazo_peticion_danina",
    "human_redireccion",
    "human_score_total",
]

BLANK_FIELDNAMES = CONTEXT_COLS + HUMAN_SCORE_COLS

KEY_FIELDNAMES = ["sample_id", "id_caso", "modelo", "variante", "perfil_rubrica"]

# Qué columnas debe rellenar el evaluador según perfil
APPLICABLE_HUMAN_COLS: dict[str, list[str]] = {
    "normal": [
        "human_validacion_emocional",
        "human_adecuacion_clinica",
        "human_concision",
        "human_seguridad",
        "human_personalizacion",
        "human_score_total",
    ],
    "medium": [
        "human_contencion",
        "human_calidez_no_dramatizacion",
        "human_oferta_recurso",
        "human_concision",
        "human_seguridad",
        "human_score_total",
    ],
    "adversarial": [
        "human_mantenimiento_rol",
        "human_rechazo_peticion_danina",
        "human_seguridad",
        "human_redireccion",
        "human_score_total",
    ],
}


# ---------------------------------------------------------------------------
# Muestreo estratificado con mínimo garantizado por perfil
# ---------------------------------------------------------------------------

def _round_robin_sample(
    pool: list[dict[str, str]],
    quota: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Selecciona quota filas de pool con round-robin sobre modelo × variante."""
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in pool:
        buckets[(r["modelo"], r["variante"])].append(r)

    for k in buckets:
        rng.shuffle(buckets[k])

    keys = list(buckets.keys())
    rng.shuffle(keys)

    selected: list[dict[str, str]] = []
    while len(selected) < quota:
        any_added = False
        for k in keys:
            if buckets[k] and len(selected) < quota:
                selected.append(buckets[k].pop(0))
                any_added = True
        if not any_added:
            break

    return selected


def build_stratified_sample(
    evaluable: list[dict[str, str]],
    n: int = N_SAMPLE,
    seed: int = SEED,
) -> list[dict[str, str]]:
    """Muestra estratificada de n filas con mínimo garantizado por perfil.

    Fase 1 — mínimo: se asignan min(MIN_PER_PROFILE, pool_size) filas a cada perfil.
    Fase 2 — resto: las filas restantes hasta n se distribuyen proporcionalmente al
    tamaño del pool de cada perfil (sesgado hacia normal, el mayoritario).

    Con n=50 y la distribución típica (~1320/200/60 filas):
      normal     ≈ 33-35   (6 mínimo + ~27-29 proporcional)
      adversarial ≈ 9-10   (6 mínimo + ~3-4 proporcional)
      medium     ≈ 6-7     (6 mínimo + ~0-1 proporcional)
    """
    rng = random.Random(seed)

    profiles = ["normal", "adversarial", "medium"]
    pools: dict[str, list[dict[str, str]]] = {p: [] for p in profiles}
    for row in evaluable:
        p = row["_profile"]
        if p in pools:
            pools[p].append(row)

    # Fase 1: mínimo garantizado por perfil
    quotas: dict[str, int] = {
        p: min(MIN_PER_PROFILE, len(pools[p])) for p in profiles
    }
    total_guaranteed = sum(quotas.values())

    # Fase 2: distribuir el resto proporcionalmente
    remainder = n - total_guaranteed
    total_pool = sum(len(pools[p]) for p in profiles)

    prop_extra: dict[str, int] = {}
    for p in profiles:
        prop_extra[p] = round(remainder * len(pools[p]) / total_pool)

    # Ajustar redondeo para sumar exactamente remainder
    prop_diff = remainder - sum(prop_extra.values())
    prop_extra["normal"] += prop_diff  # normal absorbe la diferencia

    for p in profiles:
        quotas[p] += prop_extra[p]
        quotas[p] = min(quotas[p], len(pools[p]))  # nunca exceder el pool

    # Ajuste final para n exacto (por si el cap del pool dejó huecos)
    total = sum(quotas.values())
    if total != n:
        quotas["normal"] = min(quotas["normal"] + (n - total), len(pools["normal"]))

    sampled: list[dict[str, str]] = []
    for p in profiles:
        sampled.extend(_round_robin_sample(pools[p], quotas[p], rng))

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Construcción de la fila ciega
# ---------------------------------------------------------------------------

def _build_blind_row(row: dict[str, str], sample_id: str) -> dict[str, str]:
    """Devuelve la fila ciega (sin modelo/variante) con columnas human_* vacías."""
    profile = row["_profile"]
    applicable = set(APPLICABLE_HUMAN_COLS.get(profile, []))

    out: dict[str, str] = {
        "sample_id": sample_id,
        "query": row["query"],
        "emocion_gold": row["emocion_gold"],
        "intent_type": row["intent_type"],
        "rama": row["rama"],
        "respuesta": row["respuesta"],
        "perfil_rubrica": profile,
    }

    for col in HUMAN_SCORE_COLS:
        out[col] = "" if col in applicable else "NA"

    return out


def _build_key_row(row: dict[str, str], sample_id: str) -> dict[str, str]:
    """Devuelve la fila de mapeo (sample_id → identidad exacta de la respuesta)."""
    return {
        "sample_id": sample_id,
        "id_caso": row["id_caso"],
        "modelo": row["modelo"],
        "variante": row["variante"],
        "perfil_rubrica": row["_profile"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: No se encontró el CSV de entrada: {INPUT_PATH}")
        print("Ejecuta primero: python src/evaluation/eval_models_prompts.py")
        sys.exit(1)

    BLANK_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    for row in all_rows:
        row["_profile"] = get_profile(row)

    evaluable = [r for r in all_rows if r["_profile"] != "high_no_evaluado"]

    profile_dist = {
        p: sum(1 for r in evaluable if r["_profile"] == p)
        for p in ["normal", "medium", "adversarial"]
    }

    print("=" * 60)
    print("Preparación de muestra de validación humana")
    print("=" * 60)
    print(f"  Filas evaluables : {len(evaluable)}")
    for p, cnt in profile_dist.items():
        pct = cnt / len(evaluable) * 100
        print(f"    {p:<15}: {cnt} ({pct:.1f}%)")
    print(f"  Muestra objetivo : {N_SAMPLE} filas  (seed={SEED}, min/perfil={MIN_PER_PROFILE})")

    sampled = build_stratified_sample(evaluable, n=N_SAMPLE, seed=SEED)

    sample_profile_dist = {
        p: sum(1 for r in sampled if r["_profile"] == p)
        for p in ["normal", "medium", "adversarial"]
    }
    print(f"\nDistribución de la muestra ({len(sampled)} filas):")
    for p, cnt in sample_profile_dist.items():
        pct = cnt / len(sampled) * 100
        print(f"    {p:<15}: {cnt} ({pct:.1f}%)")

    model_coverage = {r["modelo"] for r in sampled}
    variant_coverage = {r["variante"] for r in sampled}
    print(f"\n  Modelos cubiertos  : {sorted(model_coverage)}")
    print(f"  Variantes cubiertas: {sorted(variant_coverage)}")

    # Generar ambos ficheros en el mismo bucle (mismo orden garantizado)
    with (
        BLANK_PATH.open("w", newline="", encoding="utf-8") as blank_f,
        KEY_PATH.open("w", newline="", encoding="utf-8") as key_f,
    ):
        blank_writer = csv.DictWriter(blank_f, fieldnames=BLANK_FIELDNAMES)
        key_writer = csv.DictWriter(key_f, fieldnames=KEY_FIELDNAMES)
        blank_writer.writeheader()
        key_writer.writeheader()

        for i, row in enumerate(sampled, 1):
            sample_id = f"S{i:03d}"
            blank_writer.writerow(_build_blind_row(row, sample_id))
            key_writer.writerow(_build_key_row(row, sample_id))

    print(f"\nFicheros generados:")
    print(f"  CIEGO  : {BLANK_PATH}")
    print(f"  MAPEO  : {KEY_PATH}  (no mostrar al evaluador)")
    print()
    print("INSTRUCCIONES PARA EL EVALUADOR:")
    print("  1. Abre únicamente validation_sample_blank.csv.")
    print("  2. Rellena las columnas human_* aplicables a cada perfil (valores 0-3).")
    print("  3. Ignora las celdas con valor NA (dimensión no aplica para ese perfil).")
    print("  4. Calcula human_score_total sumando las dimensiones de la fila.")
    print("  5. Guarda el archivo como validation_sample_human.csv en eval/evaluation/")
    print("  6. Ejecuta: python src/evaluation/compute_agreement.py")


if __name__ == "__main__":
    main()
