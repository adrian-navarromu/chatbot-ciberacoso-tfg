"""
Análisis descriptivo del corpus RAG.

Genera tabla estadística por pilar y figuras para la memoria del TFG.
Modo: sin GPU. Ejecutable directamente.

Figuras generadas en docs/figures/rag/:
  - chunks_por_pilar.png      — barras horizontales de nº chunks por pilar
  - longitudes_boxplot.png    — boxplot de longitudes (palabras) por pilar
  - fuentes_por_pilar.png     — matriz presencia fuente × pilar (heatmap)

CSV generado en results/rag_experiments/:
  - corpus_analysis.csv

Uso:
    conda activate chatbots
    python src/rag/experiments/corpus_analysis.py
"""

from __future__ import annotations

import random
import re
import sys
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sin ventana de escritorio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Semilla (REGLA TRANSVERSAL 1)
SEED = 112
random.seed(SEED)
np.random.seed(SEED)

# Configuración de rutas
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.experiments.utils import CORPUS_DIR, load_chunks_from_markdown

FIGURES_DIR = _ROOT / "data" / "figures" / "rag"
RESULTS_DIR = _ROOT / "eval" / "rag_experiments"
OUTPUT_CSV = RESULTS_DIR / "corpus_analysis.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Nombres completos de los pilares
PILLAR_NAMES: dict[int, str] = {
    1: "P1 — Protocolos de actuación",
    2: "P2 — TCC y regulación emocional",
    3: "P3 — Esperanza y contranarrativa",
    4: "P4 — PAP y recursos de crisis",
    5: "P5 — Autodefensa digital",
}

PILLAR_COLORS: dict[int, str] = {
    1: "#2196F3",   # azul
    2: "#4CAF50",   # verde
    3: "#FF9800",   # naranja
    4: "#F44336",   # rojo
    5: "#9C27B0",   # púrpura
}

# Estilo global de figuras para TFG
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Normalización de fuentes bibliográficas
# ---------------------------------------------------------------------------

# Tabla de normalización: variantes → nombre canónico
_SOURCE_CANON: dict[str, str] = {
    "Guía Clínica Ciberacoso SEMA-Red.es (2015)": "SEMA-Red.es Guía Clínica (2015)",
    "Guía Clínica SEMA-Red.es (2015)": "SEMA-Red.es Guía Clínica (2015)",
    "Guía Clínica SEMA (2015)": "SEMA-Red.es Guía Clínica (2015)",
    "Guía Actuación Acoso Escolar Madrid (2015-2016)": "Guía Actuación Madrid (2015-2016)",
    "Guía Madrid (2015-2016)": "Guía Actuación Madrid (2015-2016)",
    "Guía Actuación Madrid (2015-2016)": "Guía Actuación Madrid (2015-2016)",
    "Manual RedesConCorazón (2019)": "RedesConCorazón (2019)",
    "Hendricks et al. TF-CBT Workbook (2019)": "Hendricks TF-CBT Workbook (2019)",
    "SEPSM TCC (2022)": "SEPSM Protocolo TCC (2022)",
    "PNUD/Integra Guía PAP (2022)": "PNUD/Integra Guía PAP (2022)",
    "Guía PAP PNUD (2022)": "PNUD/Integra Guía PAP (2022)",
    "O'Hara, Hope-focused Therapy (2025)": "O'Hara Hope-focused Therapy (2025)",
    "O'Hara (2025)": "O'Hara Hope-focused Therapy (2025)",
    "Snyder, Hope Theory (1994/2002)": "Snyder Hope Theory (1994/2002)",
    "Future of Free Speech / Dangerous Speech Project": "DSP Manual Contranarrativas",
    "Future of Free Speech / DSP": "DSP Manual Contranarrativas",
    "Autodefensa Digital (2025)": "Autodefensa Digital (2025)",
    "INCIBE": "INCIBE Recursos Menores",
    "PantallasAmigas pantallasamigas.net": "PantallasAmigas",
    "PantallasAmigas": "PantallasAmigas",
    "Meta/Instagram Safety": "Guías Plataformas (Meta/TikTok/Discord/Roblox)",
    "TikTok Safety Center (tiktok.com/safety/es)": "Guías Plataformas (Meta/TikTok/Discord/Roblox)",
    "Discord Support": "Guías Plataformas (Meta/TikTok/Discord/Roblox)",
    "Roblox Help Center (support.roblox.com)": "Guías Plataformas (Meta/TikTok/Discord/Roblox)",
    "AEPD": "AEPD Canal Joven",
}


def _normalize_source(raw: str) -> set[str]:
    """Extrae nombres canónicos de fuentes desde un campo source del corpus."""
    result: set[str] = set()
    for part in raw.split(";"):
        part = part.strip()
        # Eliminar referencias a páginas/secciones
        part = re.sub(r",?\s*(?:pp?\.|cap\.|módulo|bloque|sección|Anexo)\s*[\w\s,.\-–]+$", "", part).strip()
        # Eliminar sufijos tras "—"
        part = re.sub(r"\s*—\s*(?:ficha|TIPAR|esperanza|Agency|Pathway)\s+.*$", "", part).strip()
        # Eliminar URLs
        part = re.sub(r"\s+\w[\w.-]+\.\w{2,}/\S*", "", part).strip()
        # Eliminar referencias email
        part = re.sub(r"\s+\S+@\S+", "", part).strip()
        # Intentar canonicalizar
        canon = None
        for key, val in _SOURCE_CANON.items():
            if key.lower() in part.lower() or part.lower() in key.lower():
                canon = val
                break
        result.add(canon if canon else part)
    return result


# ---------------------------------------------------------------------------
# Cálculo de estadísticas del corpus
# ---------------------------------------------------------------------------

def compute_stats(docs: list) -> pd.DataFrame:
    """Calcula estadísticas por pilar y devuelve un DataFrame.

    Columnas: Pilar, Fuente(s) original(es), Nº fuentes, Nº chunks,
              Media palabras, Mediana palabras, Desv. típica.

    Args:
        docs: Lista de Document de LangChain con page_content y metadata.

    Returns:
        DataFrame con una fila por pilar + fila de totales.
    """
    pillar_data: dict[int, dict] = {}
    for p in sorted(PILLAR_NAMES.keys()):
        pillar_data[p] = {"chunks": [], "sources": set()}

    for doc in docs:
        p = doc.metadata["pillar"]
        word_count = len(doc.page_content.split())
        pillar_data[p]["chunks"].append(word_count)
        pillar_data[p]["sources"].update(_normalize_source(doc.metadata.get("source", "")))

    rows: list[dict] = []
    for p in sorted(PILLAR_NAMES.keys()):
        wc = np.array(pillar_data[p]["chunks"])
        srcs = sorted(pillar_data[p]["sources"])
        rows.append({
            "Pilar": PILLAR_NAMES[p],
            "Fuente(s) original(es)": "; ".join(srcs),
            "Nº fuentes": len(srcs),
            "Nº chunks": len(wc),
            "Media (palabras)": round(float(np.mean(wc)), 1),
            "Mediana (palabras)": round(float(np.median(wc)), 1),
            "Desv. típica": round(float(np.std(wc)), 1),
        })

    # Fila de totales
    all_wc = np.array([w for p in pillar_data.values() for w in p["chunks"]])
    all_srcs = set()
    for p in pillar_data.values():
        all_srcs.update(p["sources"])
    rows.append({
        "Pilar": "TOTAL",
        "Fuente(s) original(es)": f"({len(all_srcs)} fuentes distintas)",
        "Nº fuentes": len(all_srcs),
        "Nº chunks": int(np.sum([len(p["chunks"]) for p in pillar_data.values()])),
        "Media (palabras)": round(float(np.mean(all_wc)), 1),
        "Mediana (palabras)": round(float(np.median(all_wc)), 1),
        "Desv. típica": round(float(np.std(all_wc)), 1),
    })

    df = pd.DataFrame(rows)
    return df, pillar_data


def print_table(df: pd.DataFrame) -> None:
    """Imprime la tabla en consola de forma legible."""
    sep = "=" * 100
    print(f"\n{sep}")
    print("  ANÁLISIS DESCRIPTIVO DEL CORPUS RAG — ESTADÍSTICAS POR PILAR")
    print(sep)
    # Versión simplificada sin fuentes (demasiado largas)
    display = df[["Pilar", "Nº fuentes", "Nº chunks", "Media (palabras)", "Mediana (palabras)", "Desv. típica"]].copy()
    print(display.to_string(index=False))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_chunks_por_pilar(pillar_data: dict, path: Path) -> None:
    """Figura 1: barras horizontales de nº chunks por pilar."""
    pilares = sorted(PILLAR_NAMES.keys())
    nombres = [PILLAR_NAMES[p] for p in pilares]
    counts = [len(pillar_data[p]["chunks"]) for p in pilares]
    colors = [PILLAR_COLORS[p] for p in pilares]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(nombres, counts, color=colors, edgecolor="white", linewidth=0.8)

    # Anotaciones
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=11, fontweight="bold")

    ax.set_xlabel("Número de chunks terapéuticos")
    ax.set_title("Distribución de chunks por pilar del corpus RAG", fontweight="bold", pad=12)
    ax.set_xlim(0, max(counts) + 2)
    ax.set_xticks(range(0, max(counts) + 2, 2))
    ax.invert_yaxis()
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura 1 guardada: {path}")


def fig_longitudes_boxplot(pillar_data: dict, path: Path) -> None:
    """Figura 2: boxplot + puntos jittered de longitudes por pilar."""
    pilares = sorted(PILLAR_NAMES.keys())
    data = [pillar_data[p]["chunks"] for p in pilares]
    labels = [PILLAR_NAMES[p] for p in pilares]
    colors = [PILLAR_COLORS[p] for p in pilares]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Boxplots
    bp = ax.boxplot(data, patch_artist=True, vert=True,
                    widths=0.5, medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set(color="gray", linewidth=1)

    # Puntos jittered
    rng = np.random.default_rng(SEED)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter([i + j for j in jitter], vals, color=color,
                   alpha=0.7, s=28, zorder=3, edgecolors="white", linewidths=0.5)

    ax.set_xticks(range(1, len(pilares) + 1))
    ax.set_xticklabels([f"P{p}" for p in pilares])
    ax.set_ylabel("Longitud del chunk (palabras)")
    ax.set_xlabel("Pilar RAG")
    ax.set_title("Distribución de longitudes de chunks por pilar", fontweight="bold", pad=12)

    # Leyenda
    legend_patches = [mpatches.Patch(facecolor=PILLAR_COLORS[p], alpha=0.8, label=PILLAR_NAMES[p])
                      for p in pilares]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura 2 guardada: {path}")


def fig_fuentes_por_pilar(pillar_data: dict, path: Path) -> None:
    """Figura 3: heatmap de presencia fuente × pilar."""
    pilares = sorted(PILLAR_NAMES.keys())

    # Recopilar todas las fuentes canónicas
    all_sources: set[str] = set()
    for p_data in pillar_data.values():
        all_sources.update(p_data["sources"])
    sources_sorted = sorted(all_sources)

    # Matriz de presencia (nº chunks por fuente-pilar)
    matrix = np.zeros((len(sources_sorted), len(pilares)), dtype=int)
    for j, p in enumerate(pilares):
        p_chunks = pillar_data[p]["chunks"]   # lista de word counts
        # Obtener fuentes de cada chunk — necesitamos re-leer con fuentes por chunk
        pass  # se rellena abajo con la versión extendida

    # Usamos la variable global docs (pasada al scope global)
    matrix_count = np.zeros((len(sources_sorted), len(pilares)), dtype=int)
    for doc in _GLOBAL_DOCS:
        p = doc.metadata["pillar"]
        j = pilares.index(p)
        for src in _normalize_source(doc.metadata.get("source", "")):
            if src in sources_sorted:
                i = sources_sorted.index(src)
                matrix_count[i, j] += 1

    # Abreviar nombres de fuentes para el eje Y
    def _abbrev(s: str) -> str:
        s = re.sub(r"\s*\(\d{4}(?:/\d{4})?\)", "", s).strip()
        s = re.sub(r"\s*\(s\.f\.\)", "", s).strip()
        return s[:38] + "…" if len(s) > 38 else s

    y_labels = [_abbrev(s) for s in sources_sorted]
    x_labels = [f"P{p}" for p in pilares]

    # Quitar filas con todos ceros (no deberían existir pero por si acaso)
    nonzero_rows = matrix_count.sum(axis=1) > 0
    matrix_count = matrix_count[nonzero_rows]
    y_labels_filt = [y for y, keep in zip(y_labels, nonzero_rows) if keep]

    fig, ax = plt.subplots(figsize=(7, max(4, len(y_labels_filt) * 0.45 + 1.5)))

    im = ax.imshow(matrix_count, cmap="Blues", aspect="auto", vmin=0)

    # Anotaciones en celdas
    for i in range(matrix_count.shape[0]):
        for j in range(matrix_count.shape[1]):
            val = matrix_count[i, j]
            if val > 0:
                color = "white" if val >= matrix_count.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(pilares)))
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.set_yticks(range(len(y_labels_filt)))
    ax.set_yticklabels(y_labels_filt, fontsize=9)
    ax.set_title("Contribución de fuentes bibliográficas por pilar\n(nº de chunks aportados)", fontweight="bold", pad=12)
    ax.set_xlabel("Pilar RAG")
    ax.set_ylabel("Fuente bibliográfica")

    plt.colorbar(im, ax=ax, label="Nº chunks aportados", shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura 3 guardada: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_GLOBAL_DOCS: list = []   # compartido para la figura de fuentes


def main() -> None:
    """Ejecuta el análisis completo del corpus RAG."""
    global _GLOBAL_DOCS
    log.info("Cargando corpus RAG...")
    docs = load_chunks_from_markdown(CORPUS_DIR)
    _GLOBAL_DOCS = docs
    log.info(f"  {len(docs)} chunks cargados desde {CORPUS_DIR}.")

    # Estadísticas
    log.info("Calculando estadísticas por pilar...")
    df, pillar_data = compute_stats(docs)

    # Consola
    print_table(df)

    # CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"Tabla exportada a {OUTPUT_CSV}")

    # Figura 1 — Chunks por pilar
    log.info("Generando figuras...")
    fig_chunks_por_pilar(pillar_data, FIGURES_DIR / "chunks_por_pilar.png")

    # Figura 2 — Boxplot longitudes
    fig_longitudes_boxplot(pillar_data, FIGURES_DIR / "longitudes_boxplot.png")

    # Figura 3 — Fuentes × Pilar
    fig_fuentes_por_pilar(pillar_data, FIGURES_DIR / "fuentes_por_pilar.png")

    log.info("Análisis completo. Figuras en: docs/figures/rag/")


if __name__ == "__main__":
    main()
