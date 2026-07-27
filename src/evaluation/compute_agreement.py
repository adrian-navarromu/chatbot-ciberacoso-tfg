"""
Concordancia juez LLM vs evaluador humano.

Lee judge_scores.csv, validation_sample_human.csv (relleno por el evaluador)
y validation_sample_key.csv (mapeo oculto generado por prepare_validation_sample.py),
los cruza por identidad exacta (id_caso, modelo, variante) y calcula métricas de
acuerdo interevaluador.

Cruce de tablas (Cambio 1 — clave única):
  sample_id (human) → key.csv → (id_caso, modelo, variante) → judge_scores.csv
  Esto garantiza comparar el score del juez sobre la MISMA respuesta que evaluó el
  humano. El cruce por query es incorrecto: hay 20 filas con la misma query en
  judge_scores (5 modelos × 4 variantes) y sin el mapeo se elige la incorrecta.

Métricas calculadas:
  - Spearman ρ y Pearson r  sobre score_normalizado (continuo 0-1)  [métrica principal]
  - Cohen's κ ponderado cuadrático por dimensión (0-3)              [métrica principal]
  - Cohen's κ ponderado cuadrático sobre score_total (0-15)         [secundario]
  - Tabla de los 10 mayores desacuerdos

Prerequisitos (en orden de ejecución):
  1. python src/generations/eval_models_prompts.py  →  slm_prompt_generations.csv
  2. python src/generations/prepare_validation_sample.py  →  validation_sample_blank.csv
                                                             validation_sample_key.csv
  3. Evaluador humano rellena blank y guarda como validation_sample_human.csv
  4. python src/generations/judge.py  →  judge_scores.csv
  5. python src/generations/compute_agreement.py  →  agreement_report.txt

Salida: results/generations/agreement_report.txt (+ consola)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import numpy as np
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score
except ImportError as exc:
    print(f"ERROR: dependencia no encontrada — {exc}")
    print("Instala con: conda install -c conda-forge scipy scikit-learn numpy")
    sys.exit(1)

JUDGE_PATH = _PROJECT_ROOT / "eval" / "generations" / "judge_scores.csv"
HUMAN_PATH = _PROJECT_ROOT / "eval" / "generations" / "validation_sample_human.csv"
KEY_PATH = _PROJECT_ROOT / "eval" / "generations" / "validation_sample_key.csv"
REPORT_PATH = _PROJECT_ROOT / "eval" / "generations" / "agreement_report.txt"

DIMS_BY_PROFILE: dict[str, list[str]] = {
    "normal": ["validacion_emocional", "adecuacion_clinica", "concision", "seguridad", "personalizacion"],
    "medium": ["contencion", "calidez_no_dramatizacion", "oferta_recurso", "concision", "seguridad"],
    "adversarial": ["mantenimiento_rol", "rechazo_peticion_danina", "seguridad", "redireccion"],
}

# score_max por perfil para normalizar el total humano
SCORE_MAX_BY_PROFILE: dict[str, int] = {"normal": 15, "medium": 15, "adversarial": 12}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int_dim(val: str) -> int | None:
    """Convierte a entero validando rango de dimensión individual: 0-3."""
    try:
        v = int(val)
        return v if 0 <= v <= 3 else None
    except (ValueError, TypeError):
        return None


def _safe_int_total(val: str) -> int | None:
    """Convierte a entero validando rango de score_total: 0-15."""
    try:
        v = int(val)
        return v if 0 <= v <= 15 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Cruce de tablas por identidad exacta (id_caso, modelo, variante)
# ---------------------------------------------------------------------------

def _join_by_key(
    judge_rows: list[dict[str, str]],
    human_rows: list[dict[str, str]],
    key_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Cruza human ↔ judge por (id_caso, modelo, variante) usando el fichero de mapeo.

    Garantiza que cada fila humana se cruza con la respuesta EXACTA que el humano
    evaluó, no con otra respuesta a la misma query de otro modelo o variante.

    Proceso:
      1. Indexa key.csv por sample_id → (id_caso, modelo, variante)
      2. Indexa judge_scores.csv por (id_caso, modelo, variante)  [clave única]
      3. Para cada fila humana: sample_id → mapeo → judge_row exacta
    """
    # Índice del mapeo: sample_id → fila key
    key_index: dict[str, dict[str, str]] = {k["sample_id"]: k for k in key_rows}

    # Índice del juez: (id_caso, modelo, variante) → fila judge
    judge_index: dict[tuple[str, str, str], dict[str, str]] = {}
    for jr in judge_rows:
        if jr.get("evaluado_por_juez", "").lower() != "true":
            continue
        jkey = (jr["id_caso"], jr["modelo"], jr["variante"])
        judge_index[jkey] = jr

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    unmatched: list[str] = []

    for hr in human_rows:
        sid = hr.get("sample_id", "?")
        mapping = key_index.get(sid)
        if mapping is None:
            unmatched.append(f"{sid} (no encontrado en validation_sample_key.csv)")
            continue

        jkey = (mapping["id_caso"], mapping["modelo"], mapping["variante"])
        jr = judge_index.get(jkey)
        if jr is None:
            unmatched.append(f"{sid} → {jkey} (no encontrado en judge_scores.csv)")
            continue

        pairs.append((jr, hr))

    if unmatched:
        print(f"  AVISO: {len(unmatched)} filas sin match exacto:")
        for u in unmatched[:5]:
            print(f"    {u}")
        if len(unmatched) > 5:
            print(f"    ... y {len(unmatched) - 5} más")

    return pairs


# ---------------------------------------------------------------------------
# Cálculo de métricas
# ---------------------------------------------------------------------------

def _spearman_pearson(
    judge_vals: list[float],
    human_vals: list[float],
) -> tuple[float, float, float, float]:
    """Devuelve (spearman_rho, spearman_p, pearson_r, pearson_p)."""
    arr_j = np.array(judge_vals)
    arr_h = np.array(human_vals)
    sp_rho, sp_p = stats.spearmanr(arr_j, arr_h)
    pe_r, pe_p = stats.pearsonr(arr_j, arr_h)
    return float(sp_rho), float(sp_p), float(pe_r), float(pe_p)


def _kappa_weighted(y_judge: list[int], y_human: list[int]) -> float | None:
    """Cohen's κ ponderado cuadrático para escala ordinal."""
    if len(y_judge) < 2 or len(set(y_judge + y_human)) < 2:
        return None
    return float(cohen_kappa_score(y_judge, y_human, weights="quadratic"))


def _interpret_kappa(k: float) -> str:
    if k > 0.80:
        return "casi perfecto (>0.80)"
    if k > 0.60:
        return "sustancial (0.61-0.80)"
    if k > 0.40:
        return "moderado (0.41-0.60)"
    if k > 0.20:
        return "leve (0.21-0.40)"
    return "pobre (≤0.20)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    required = {JUDGE_PATH: "judge.py", HUMAN_PATH: "rellenar validation_sample_blank.csv", KEY_PATH: "prepare_validation_sample.py"}
    for path, hint in required.items():
        if not path.exists():
            print(f"ERROR: No se encontró: {path}")
            print(f"  Prerequisito: {hint}")
            sys.exit(1)

    print("=" * 60)
    print("Concordancia juez LLM vs evaluador humano")
    print("=" * 60)

    judge_rows = _load_csv(JUDGE_PATH)
    human_rows = _load_csv(HUMAN_PATH)
    key_rows = _load_csv(KEY_PATH)

    print(f"  Filas judge_scores.csv       : {len(judge_rows)}")
    print(f"  Filas validation_sample_human: {len(human_rows)}")
    print(f"  Filas validation_sample_key  : {len(key_rows)}")

    pairs = _join_by_key(judge_rows, human_rows, key_rows)
    print(f"  Pares enlazados para análisis: {len(pairs)}")

    if len(pairs) < 5:
        print("\nERROR: menos de 5 pares para calcular métricas.")
        print("Comprueba que validation_sample_human.csv usa los mismos")
        print("sample_ids que validation_sample_blank.csv.")
        sys.exit(1)

    lines: list[str] = [
        "CONCORDANCIA JUEZ LLM vs EVALUADOR HUMANO",
        "=" * 60,
        f"Pares analizados: {len(pairs)}",
        "",
    ]

    # --- Correlación sobre score_normalizado (métrica principal) ---
    judge_norm: list[float] = []
    human_norm: list[float] = []

    for jr, hr in pairs:
        jn = _safe_float(jr.get("score_normalizado", ""))
        ht = _safe_int_total(hr.get("human_score_total", ""))
        profile = jr.get("perfil_rubrica", "")
        s_max = SCORE_MAX_BY_PROFILE.get(profile, 0)

        if jn is not None and ht is not None and s_max > 0:
            judge_norm.append(jn)
            human_norm.append(ht / s_max)

    if len(judge_norm) >= 5:
        sp_rho, sp_p, pe_r, pe_p = _spearman_pearson(judge_norm, human_norm)
        block = [
            "CORRELACIÓN GLOBAL (score_normalizado, 0-1)  [métrica principal]",
            f"  n                    : {len(judge_norm)}",
            f"  Spearman ρ           : {sp_rho:+.4f}  (p={sp_p:.4f})",
            f"  Pearson  r           : {pe_r:+.4f}  (p={pe_p:.4f})",
        ]
        print()
        for ln in block:
            print(f"  {ln}")
        lines.extend(block)
        lines.append("")

    # --- Cohen's κ sobre score_total (secundario) ---
    judge_total_int: list[int] = []
    human_total_int: list[int] = []

    for jr, hr in pairs:
        jt = _safe_int_total(jr.get("score_total", ""))
        ht = _safe_int_total(hr.get("human_score_total", ""))
        if jt is not None and ht is not None:
            judge_total_int.append(jt)
            human_total_int.append(ht)

    if len(judge_total_int) >= 5:
        kappa_total = _kappa_weighted(judge_total_int, human_total_int)
        if kappa_total is not None:
            block_k = [
                "COHEN'S κ PONDERADO (score_total, 0-15)  [secundario]",
                f"  n                    : {len(judge_total_int)}",
                f"  κ cuadrático total   : {kappa_total:+.4f}",
                f"  Interpretación       : {_interpret_kappa(kappa_total)}",
                "  Nota: con n=50 y 16 niveles posibles, el κ del total puede ser",
                "        inestable. Confiar principalmente en Spearman/Pearson y κ por dim.",
            ]
            print()
            for ln in block_k:
                print(f"  {ln}")
            lines.extend(block_k)
            lines.append("")

    # --- Cohen's κ por dimensión (métrica principal) ---
    dim_kappas: list[tuple[str, int, float]] = []

    for profile, dims in DIMS_BY_PROFILE.items():
        profile_pairs = [(jr, hr) for jr, hr in pairs if jr.get("perfil_rubrica") == profile]
        if not profile_pairs:
            continue

        for dim in dims:
            judge_col = f"{dim}_score"
            human_col = f"human_{dim}"
            yj: list[int] = []
            yh: list[int] = []
            for jr, hr in profile_pairs:
                j_val = _safe_int_dim(jr.get(judge_col, ""))
                h_val = _safe_int_dim(hr.get(human_col, ""))
                if j_val is not None and h_val is not None:
                    yj.append(j_val)
                    yh.append(h_val)
            if len(yj) >= 4:
                k = _kappa_weighted(yj, yh)
                if k is not None:
                    dim_kappas.append((f"{profile}/{dim}", len(yj), k))

    if dim_kappas:
        dim_kappas.sort(key=lambda x: x[2], reverse=True)
        block_dim = ["COHEN'S κ PONDERADO POR DIMENSIÓN (0-3)  [métrica principal, descendente]"]
        block_dim.append(f"  {'Perfil/Dimensión':<42}  n   κ cuadrático  Interpretación")
        block_dim.append("  " + "-" * 72)
        for dim_label, n_dim, k_val in dim_kappas:
            block_dim.append(
                f"  {dim_label:<42}  {n_dim:<4} {k_val:+.4f}        {_interpret_kappa(k_val)}"
            )
        print()
        for ln in block_dim:
            print(f"  {ln}" if not ln.startswith("  ") else ln)
        lines.extend(block_dim)
        lines.append("")

    # --- Top 10 desacuerdos ---
    disagreements: list[tuple[float, str, str, str]] = []
    for jr, hr in pairs:
        jn = _safe_float(jr.get("score_normalizado", ""))
        ht = _safe_int_total(hr.get("human_score_total", ""))
        profile = jr.get("perfil_rubrica", "")
        s_max = SCORE_MAX_BY_PROFILE.get(profile, 0)
        if jn is not None and ht is not None and s_max > 0:
            hn = ht / s_max
            diff = abs(jn - hn)
            sid = hr.get("sample_id", "?")
            query_short = jr.get("query", "")[:60]
            disagreements.append((diff, sid, profile, query_short))

    if disagreements:
        disagreements.sort(reverse=True)
        block_dis = [
            "TOP 10 DESACUERDOS (|score_norm_juez - score_norm_humano|)",
            f"  {'sample_id':<10}  {'perfil':<12}  {'|diff|':<8}  query",
            "  " + "-" * 72,
        ]
        for diff, sid, profile, q_short in disagreements[:10]:
            block_dis.append(f"  {sid:<10}  {profile:<12}  {diff:.3f}     {q_short!r}")
        print()
        for ln in block_dis:
            print(f"  {ln}" if not ln.startswith("  ") else ln)
        lines.extend(block_dis)
        lines.append("")

    # --- Guardar informe ---
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Informe guardado en: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
