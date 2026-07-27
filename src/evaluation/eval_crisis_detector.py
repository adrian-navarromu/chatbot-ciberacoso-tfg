"""
Evaluación del CrisisDetector sobre el banco de pruebas.

MODO: SIN EMOCIÓN — el CrisisDetector opera por regex sobre texto crudo.
      No se usa el campo `emotion` del banco ni se invoca ningún modelo ML.

Evaluación binaria:
    Positivo (crisis)   → crisis_expected in ('HIGH', 'MEDIUM')
    Negativo (no crisis) → crisis_expected == 'NONE'

También se muestra la matriz de confusión 3×3 (HIGH/MEDIUM/NONE) para
análisis cualitativo del nivel de predicción.

Casos positivos en el banco:
    - intent_type 'crisis': ideación suicida/autolesión explícita → todos HIGH
    - intent_type 'crisis_dificil': negación, tercera persona, hipérbole, eufemismo
      → algunos HIGH/MEDIUM, otros NONE (hipérbole, expresiones coloquiales)
    - Algunos 'indirecto' y 'faltas_ortograficas' con crisis_expected MEDIUM

Casos negativos:
    - Todos los demás intent_types con crisis_expected NONE
    - En especial crisis_dificil con NONE (hipérboles, falsos positivos deliberados)

Salidas:
    - docs/figures/safety/confusion_matrix_crisis.png   — heatmap 3×3
    - docs/figures/safety/metricas_por_intent.png       — recall por intent_type
    - eval/crisis_evaluation.csv                        — resultados completos

Uso:
    conda activate chatbots
    python src/evaluation/eval_crisis_detector.py
"""

from __future__ import annotations

import csv
import random
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Semilla (REGLA TRANSVERSAL 1)
SEED = 112
random.seed(SEED)
np.random.seed(SEED)

# Configuración de rutas
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.test_bank import TEST_BANK
from src.safety.crisis_detector import CrisisDetector

FIGURES_DIR = _ROOT / "data" / "figures" / "safety"
RESULTS_DIR = _ROOT / "eval"
OUTPUT_CSV = RESULTS_DIR / "crisis_evaluation.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = ["HIGH", "MEDIUM", "NONE"]
LEVEL_IDX = {lv: i for i, lv in enumerate(LEVELS)}

# Estilo global de figuras
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def run_evaluation(cases: list[dict], detector: CrisisDetector) -> list[dict]:
    """Ejecuta el CrisisDetector sobre todos los casos del banco.

    MODO SIN EMOCIÓN: solo se usa el campo `query` (texto crudo).
    El campo `emotion` NO se usa (es ground truth del banco, no input aquí).

    Args:
        cases: Lista de casos del banco de pruebas.
        detector: Instancia del CrisisDetector.

    Returns:
        Lista de dicts con id, query, intent_type, crisis_expected,
        crisis_predicted, subtype_predicted, is_binary_correct.
    """
    results: list[dict] = []
    for case in cases:
        result = detector.detect(case["query"])
        expected = case["crisis_expected"]
        predicted = result.level

        # Binario: positivo = HIGH o MEDIUM, negativo = NONE
        expected_bin = "CRISIS" if expected != "NONE" else "NO_CRISIS"
        predicted_bin = "CRISIS" if predicted != "NONE" else "NO_CRISIS"
        is_binary_correct = expected_bin == predicted_bin

        results.append({
            "id": case["id"],
            "intent_type": case["intent_type"],
            "query": case["query"],
            "crisis_expected": expected,
            "crisis_predicted": predicted,
            "subtype_predicted": result.subtype,
            "expected_binary": expected_bin,
            "predicted_binary": predicted_bin,
            "is_binary_correct": is_binary_correct,
            "triggered_keywords": "; ".join(result.triggered_keywords[:3]),
        })
    return results


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_binary_metrics(results: list[dict]) -> dict:
    """Calcula métricas binarias (crisis vs. no-crisis).

    Returns:
        Dict con tp, fp, tn, fn, precision, recall, f1, fp_rate, fn_rate.
    """
    tp = sum(1 for r in results if r["expected_binary"] == "CRISIS" and r["predicted_binary"] == "CRISIS")
    fp = sum(1 for r in results if r["expected_binary"] == "NO_CRISIS" and r["predicted_binary"] == "CRISIS")
    tn = sum(1 for r in results if r["expected_binary"] == "NO_CRISIS" and r["predicted_binary"] == "NO_CRISIS")
    fn = sum(1 for r in results if r["expected_binary"] == "CRISIS" and r["predicted_binary"] == "NO_CRISIS")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "n_positives": tp + fn,
        "n_negatives": fp + tn,
        "accuracy": (tp + tn) / len(results),
    }


def compute_confusion_matrix(results: list[dict]) -> np.ndarray:
    """Construye la matriz de confusión 3×3 (HIGH/MEDIUM/NONE).

    Filas = nivel esperado, columnas = nivel predicho.

    Returns:
        Array (3, 3) con conteos.
    """
    matrix = np.zeros((3, 3), dtype=int)
    for r in results:
        true_idx = LEVEL_IDX[r["crisis_expected"]]
        pred_idx = LEVEL_IDX[r["crisis_predicted"]]
        matrix[true_idx][pred_idx] += 1
    return matrix


# ---------------------------------------------------------------------------
# Impresión de resultados
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, matrix: np.ndarray, results: list[dict]) -> None:
    """Imprime el informe completo en consola."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  EVALUACIÓN DEL CRISIS DETECTOR — INFORME COMPLETO")
    print(f"  Banco de pruebas: {len(results)} casos | SEED={SEED}")
    print(sep)

    # Matriz de confusión 3×3
    print("\n  MATRIZ DE CONFUSIÓN (filas=esperado, columnas=predicho):\n")
    header = f"  {'':15s}  {'HIGH':>8} {'MEDIUM':>8} {'NONE':>8}  {'TOTAL':>7}"
    print(header)
    print("  " + "-" * 55)
    for i, lv in enumerate(LEVELS):
        row_total = matrix[i].sum()
        print(f"  {'GT ' + lv:15s}  {matrix[i][0]:>8} {matrix[i][1]:>8} {matrix[i][2]:>8}  {row_total:>7}")
    print("  " + "-" * 55)
    col_totals = matrix.sum(axis=0)
    print(f"  {'TOTAL PRED':15s}  {col_totals[0]:>8} {col_totals[1]:>8} {col_totals[2]:>8}  {matrix.sum():>7}")

    # Métricas binarias
    print(f"\n  MÉTRICAS BINARIAS (positivo = HIGH o MEDIUM):\n")
    print(f"  Positivos en banco:  {metrics['n_positives']:3d} (HIGH o MEDIUM)")
    print(f"  Negativos en banco:  {metrics['n_negatives']:3d} (NONE)")
    print(f"\n  TP={metrics['TP']:3d}  FP={metrics['FP']:3d}  TN={metrics['TN']:3d}  FN={metrics['FN']:3d}")
    print(f"\n  Precisión:            {metrics['precision']:.3f}  (de los detectados, ¿cuántos son crisis real?)")
    print(f"  Recall (sensibilidad): {metrics['recall']:.3f}  (de las crisis reales, ¿cuántas detectadas?)")
    print(f"  F1:                   {metrics['f1']:.3f}")
    print(f"  Tasa FP:              {metrics['fp_rate']:.3f}  (falsas alarmas sobre negativos)")
    print(f"  Tasa FN:              {metrics['fn_rate']:.3f}  (crisis no detectadas sobre positivos)")
    print(f"  Exactitud global:     {metrics['accuracy']:.3f}")

    # Recall por nivel
    print(f"\n  RECALL POR NIVEL (% de positivos de ese nivel detectados correctamente):")
    for i, lv in enumerate(["HIGH", "MEDIUM"]):
        n_total = matrix[i].sum()
        n_detected = matrix[i][i]   # diagonal = nivel predicho correcto
        n_any_crisis = matrix[i][0] + matrix[i][1]  # predicho HIGH o MEDIUM
        if n_total > 0:
            recall_exact = n_detected / n_total
            recall_bin = n_any_crisis / n_total
            print(f"    {lv}: recall_exacto={recall_exact:.2f}  recall_binario={recall_bin:.2f}  ({n_total} casos)")

    # Casos mal clasificados
    errors = [r for r in results if not r["is_binary_correct"]]
    fn_cases = [r for r in errors if r["expected_binary"] == "CRISIS"]
    fp_cases = [r for r in errors if r["expected_binary"] == "NO_CRISIS"]

    if fn_cases:
        print(f"\n  FALSOS NEGATIVOS (crisis NO detectada, {len(fn_cases)} casos):\n")
        for r in fn_cases:
            print(f"    [{r['id']}] [{r['intent_type']}] GT={r['crisis_expected']} PRED={r['crisis_predicted']}")
            print(f"         Query: {r['query'][:75]}")

    if fp_cases:
        print(f"\n  FALSOS POSITIVOS (alarma innecesaria, {len(fp_cases)} casos):\n")
        for r in fp_cases:
            print(f"    [{r['id']}] [{r['intent_type']}] GT={r['crisis_expected']} PRED={r['crisis_predicted']}")
            print(f"         Query: {r['query'][:75]}")
            if r["triggered_keywords"]:
                print(f"         Keywords: {r['triggered_keywords'][:60]}")

    # Casos con nivel incorrecto pero binario correcto
    level_errors = [r for r in results if r["is_binary_correct"] and
                    r["crisis_expected"] != r["crisis_predicted"] and
                    r["crisis_expected"] != "NONE"]
    if level_errors:
        print(f"\n  NIVEL INCORRECTO (binario correcto, {len(level_errors)} casos):\n")
        for r in level_errors:
            print(f"    [{r['id']}] [{r['intent_type']}] GT={r['crisis_expected']} PRED={r['crisis_predicted']}")
            print(f"         Query: {r['query'][:75]}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_confusion_matrix(matrix: np.ndarray, path: Path, metrics: dict) -> None:
    """Figura 1: heatmap de la matriz de confusión 3×3.

    Incluye:
    - Valores absolutos + porcentaje sobre el total de cada fila (recall por fila)
    - Diagonal resaltada (predicciones correctas)
    - Leyenda de métricas binarias clave
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Normalizada por filas para color (recall por nivel)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    matrix_norm = matrix / row_sums_safe

    cmap = plt.cm.Blues
    im = ax.imshow(matrix_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Anotaciones: "count\n(pct%)"
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            pct = 100 * matrix_norm[i, j]
            color = "white" if matrix_norm[i, j] > 0.55 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight=weight)

    # Bordes en diagonal (predicciones exactas)
    for i in range(3):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor="gold", linewidth=2.5)
        ax.add_patch(rect)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LEVELS, fontsize=11)
    ax.set_yticklabels(LEVELS, fontsize=11)
    ax.set_xlabel("Nivel predicho", fontsize=11)
    ax.set_ylabel("Nivel esperado (ground truth)", fontsize=11)
    ax.set_title("Matriz de confusión — CrisisDetector\n(color = recall por fila)", fontweight="bold", pad=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Recall por nivel (prop.)", fontsize=9)

    # Recuadro de métricas clave
    metrics_text = (
        f"Recall (binario): {metrics['recall']:.2f}\n"
        f"Precisión:        {metrics['precision']:.2f}\n"
        f"F1:               {metrics['f1']:.2f}\n"
        f"Tasa FN:          {metrics['fn_rate']:.2f}\n"
        f"Tasa FP:          {metrics['fp_rate']:.2f}"
    )
    ax.text(3.35, 0.0, metrics_text, transform=ax.transData,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura 1 guardada: {path}")


def fig_recall_por_intent(results: list[dict], path: Path) -> None:
    """Figura 2: recall binario por intent_type (% de crisis detectadas).

    Solo se muestran los intent_types con al menos 1 caso positivo.
    """
    from collections import defaultdict
    intent_stats: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})

    for r in results:
        it = r["intent_type"]
        if r["expected_binary"] == "CRISIS" and r["predicted_binary"] == "CRISIS":
            intent_stats[it]["tp"] += 1
        elif r["expected_binary"] == "CRISIS":
            intent_stats[it]["fn"] += 1
        elif r["predicted_binary"] == "CRISIS":
            intent_stats[it]["fp"] += 1
        else:
            intent_stats[it]["tn"] += 1

    # Filtrar por intent_types con positivos (TP+FN > 0)
    intent_with_pos = [(it, stats) for it, stats in intent_stats.items()
                       if stats["tp"] + stats["fn"] > 0]
    intent_with_pos.sort(key=lambda x: x[0])

    labels = [it.replace("_", "\n") for it, _ in intent_with_pos]
    recalls = [s["tp"] / max(s["tp"] + s["fn"], 1) for _, s in intent_with_pos]
    n_pos = [s["tp"] + s["fn"] for _, s in intent_with_pos]
    fp_counts = [s["fp"] for _, s in intent_with_pos]

    # Colores por recall
    bar_colors = ["#4CAF50" if r >= 0.8 else "#FF9800" if r >= 0.5 else "#F44336" for r in recalls]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, recalls, color=bar_colors, alpha=0.85, edgecolor="white")

    # Línea de referencia
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(x) - 0.4, 0.82, "umbral 0.80", color="gray", fontsize=9)

    # Anotaciones
    for i, (bar, r, n, fp) in enumerate(zip(bars, recalls, n_pos, fp_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{r:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Falsas alarmas como puntos naranjas secundarios
    if any(fp > 0 for fp in fp_counts):
        ax2 = ax.twinx()
        ax2.scatter(x, fp_counts, color="#FF5722", zorder=5, s=60, marker="D", label="FP por tipo")
        ax2.set_ylabel("Nº falsas alarmas (FP)", color="#FF5722", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#FF5722")
        ax2.set_ylim(-0.5, max(fp_counts) + 1.5)
        ax2.legend(loc="upper right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall binario (crisis detectadas)")
    ax.set_xlabel("Tipo de intención (intent_type)")
    ax.set_title("Recall del CrisisDetector por tipo de intención", fontweight="bold", pad=12)

    # Leyenda colores
    legend_patches = [
        mpatches.Patch(color="#4CAF50", alpha=0.85, label="Recall ≥ 0.80 (bueno)"),
        mpatches.Patch(color="#FF9800", alpha=0.85, label="Recall 0.50–0.79 (mejorable)"),
        mpatches.Patch(color="#F44336", alpha=0.85, label="Recall < 0.50 (crítico)"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura 2 guardada: {path}")


# ---------------------------------------------------------------------------
# Exportación CSV
# ---------------------------------------------------------------------------

def export_csv(results: list[dict], path: Path) -> None:
    """Exporta todos los resultados a CSV."""
    fieldnames = [
        "id", "intent_type", "query", "crisis_expected", "crisis_predicted",
        "subtype_predicted", "expected_binary", "predicted_binary",
        "is_binary_correct", "triggered_keywords",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Resultados exportados a {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta la evaluación completa del CrisisDetector."""
    log.info("Iniciando evaluación del CrisisDetector (MODO SIN EMOCIÓN)...")
    log.info(f"Banco de pruebas: {len(TEST_BANK)} casos | SEED={SEED}")

    detector = CrisisDetector()

    # Evaluación
    results = run_evaluation(TEST_BANK, detector)

    # Métricas
    metrics = compute_binary_metrics(results)
    matrix = compute_confusion_matrix(results)

    # Consola
    print_metrics(metrics, matrix, results)

    # Figura 1 — Matriz de confusión
    log.info("Generando figuras...")
    fig_confusion_matrix(matrix, FIGURES_DIR / "confusion_matrix_crisis.png", metrics)

    # Figura 2 — Recall por intent_type
    fig_recall_por_intent(results, FIGURES_DIR / "recall_por_intent.png")

    # CSV
    export_csv(results, OUTPUT_CSV)

    log.info("Evaluación completa. Figuras en: docs/figures/safety/")


if __name__ == "__main__":
    main()
