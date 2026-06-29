"""
Utilidades compartidas para los experimentos de comparación RAG (Exp. 1-4).

Proporciona la fuente canónica de casos de evaluación, funciones de métricas,
carga del corpus, construcción de índice FAISS en memoria y guardado de JSON.

Utilidades principales:
  - ORACLE_ROUTING              : tabla de routing emocional
  - load_eval_cases()           : carga y filtra el banco (72 casos)
  - oracle_faiss_search()       : búsqueda numpy-cosine con filtro de pilar
  - make_oracle_result()        : construye resultado por consulta
  - compute_eval_metrics()      : métricas sobre el conjunto de evaluación completo
  - rag_figure_bar()            : figura comparativa de configs para TFG

Importar con:
    from src.rag.experiments.utils import (
        CORPUS_DIR, RESULTS_DIR,
        TEST_CASES, BM25_TEST_CASES,
        ORACLE_ROUTING,
        load_eval_cases, oracle_faiss_search, make_oracle_result,
        compute_eval_metrics, rag_figure_bar,
        compute_precision_at_k, compute_precision_at_1, compute_hit_at_any,
        compute_metrics_for_run, print_results_table,
        load_chunks_from_markdown,
        build_faiss_index_from_embeddings,
        save_results,
    )
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Union

import faiss
import numpy as np
import yaml
from langchain_core.documents import Document

# Configuración del logger para consistencia con el proyecto
log = logging.getLogger(__name__)

# Rutas canónicas - fuente única para todos los experimentos
CORPUS_DIR: Path = Path("data/rag_corpus")
RESULTS_DIR: Path = Path("eval/rag_experiments")
FIGURES_DIR: Path = Path("data/figures/rag")

# ---------------------------------------------------------------------------
# Routing emocional para experimentos RAG
# ---------------------------------------------------------------------------

# Tabla de routing emocional: emoción → pilares elegibles.
# disgust → [2,5,1]: P5 digital + P1 protocolo (sextorsión, exposición pública).
# sadness → [2,3]:  rama estable sin P4 (casos HIGH filtrados del banco).
ORACLE_ROUTING: dict[str, list[int] | None] = {
    "fear":    [2, 4, 1],
    "sadness": [2, 3],
    "anger":   [1, 2],
    "disgust": [2, 5, 1],
    "joy":     [3],
    "surprise": [2, 1],
    "others":  None,        # sin filtro → búsqueda global
}


def load_eval_cases() -> list[dict]:
    """Carga y filtra el banco de pruebas para experimentos RAG.

    Filtro aplicado:
      - Excluye expected_pillars == set()  (adversariales sin pilares: 7 casos)
      - Excluye crisis_expected == 'HIGH'  (activan failsafe PAP antes del RAG: 21 casos)

    Returns:
        Lista de 72 casos del conjunto de evaluación completo.
    """
    try:
        from src.evaluation.test_bank import TEST_BANK
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from src.evaluation.test_bank import TEST_BANK

    filtered = [
        c for c in TEST_BANK
        if c["expected_pillars"]               # excluye set() vacío
        and c["crisis_expected"] != "HIGH"     # excluye activación de failsafe PAP
    ]
    log.info(
        f"Banco de evaluación: {len(filtered)} casos "
        f"— filtrados {100 - len(filtered)} (7 sin pilares + 21 HIGH crisis)."
    )
    return filtered


def oracle_faiss_search(
    query_emb: np.ndarray,
    all_embs: np.ndarray,
    doc_pillars: list[int],
    pillar_filter: list[int] | None,
    top_k: int,
) -> list[int]:
    """Búsqueda coseno sobre subespacio filtrado por pilares (numpy, sin FAISS).

    Alternativa eficiente a reconstruir un índice FAISS por consulta. Aprovecha
    que doc_embs ya están normalizados: similitud coseno = producto escalar.

    Args:
        query_emb: Vector de consulta normalizado, shape (dim,).
        all_embs: Matriz de embeddings de corpus normalizados, shape (n, dim).
        doc_pillars: Lista de pilares de cada doc (mismo orden que all_embs).
        pillar_filter: Pilares elegibles, o None para búsqueda global.
        top_k: Número de documentos a devolver.

    Returns:
        Lista de índices (en all_embs) de los top_k más similares.
    """
    if pillar_filter is None:
        mask = np.ones(len(doc_pillars), dtype=bool)
    else:
        mask = np.array([p in pillar_filter for p in doc_pillars])
        if not mask.any():
            log.warning("Filtro de pilar vacío — fallback a búsqueda global.")
            mask = np.ones(len(doc_pillars), dtype=bool)

    filtered_global_idx = np.where(mask)[0]
    scores = all_embs[mask] @ query_emb    # dot product of normalized vecs = cosine
    top_local = np.argsort(scores)[::-1][:top_k]
    return [int(filtered_global_idx[i]) for i in top_local]


def make_oracle_result(
    tc: dict,
    retrieved_pillars: list[int],
    top_k: int,
    extra: dict | None = None,
) -> dict:
    """Crea el dict de métricas para una consulta del banco oracle.

    Incluye P@1, P@3, Hit@Any y los campos del banco (id, set, intent_type).

    Args:
        tc: Caso del banco (con id, query, emotion, expected_pillars, set, intent_type).
        retrieved_pillars: Lista de pilares recuperados (en orden de ranking).
        top_k: k usado en la evaluación (para P@k).
        extra: Dict adicional opcional (ej. pillar_filter_aplicado).

    Returns:
        Dict con métricas y metadatos del caso.
    """
    expected = tc["expected_pillars"]
    r: dict = {
        "id": tc["id"],
        "query": tc["query"],
        "set": tc["set"],
        "intent_type": tc["intent_type"],
        "emotion": tc.get("emotion", ""),
        "expected_pillars": sorted(expected),
        "retrieved_pillars": retrieved_pillars,
        "precision_at_1": compute_precision_at_1(retrieved_pillars, expected),
        "precision_at_3": compute_precision_at_k(retrieved_pillars, expected, top_k),
        "hit_at_any": compute_hit_at_any(retrieved_pillars, expected),
    }
    if extra:
        r.update(extra)
    return r


def compute_eval_metrics(results: list[dict]) -> dict:
    """Calcula métricas P@1, P@3, Hit@Any sobre el conjunto de evaluación completo.

    Args:
        results: Lista de dicts generados por make_oracle_result().

    Returns:
        Dict con n, precision_at_1_media, precision_at_3_media, hit_at_any_rate.
    """
    if not results:
        return {"n": 0, "precision_at_1_media": 0.0, "precision_at_3_media": 0.0, "hit_at_any_rate": 0.0}
    n = len(results)
    return {
        "n": n,
        "precision_at_1_media": round(sum(r["precision_at_1"] for r in results) / n, 4),
        "precision_at_3_media": round(sum(r["precision_at_3"] for r in results) / n, 4),
        "hit_at_any_rate": round(sum(1 for r in results if r["hit_at_any"]) / n, 4),
    }


def rag_figure_bar(
    config_names: list[str],
    config_metrics: dict[str, dict],
    baseline_name: str,
    title: str,
    out_path: Path,
    metric_keys: tuple[str, ...] = ("precision_at_1_media", "precision_at_3_media", "hit_at_any_rate"),
    metric_labels: tuple[str, ...] = ("P@1", "P@3", "Hit@Any"),
) -> None:
    """Genera figura de barras comparativa de configuraciones RAG para el TFG.

    Muestra P@1, P@3 y Hit@Any agrupados por configuración. Marca la baseline
    como referencia (◆) sin declarar ninguna configuración como ganadora.

    Args:
        config_names: Nombres de las configuraciones en orden de aparición.
        config_metrics: Dict config_name → métricas planas (de compute_eval_metrics).
        baseline_name: Nombre de la config de referencia V1 (marcada con ◆).
        title: Título de la figura.
        out_path: Path de salida (.png).
        metric_keys: Claves de métricas a mostrar.
        metric_labels: Etiquetas de las métricas.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib no disponible — figura no generada.")
        return

    METRIC_COLORS = ["#2196F3", "#4CAF50", "#FF9800"]
    n_configs = len(config_names)
    n_metrics = len(metric_keys)
    width = 0.22
    x = np.arange(n_configs)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(max(8, n_configs * 1.8), 5))

    for mi, (mkey, mlabel, mcolor) in enumerate(zip(metric_keys, metric_labels, METRIC_COLORS)):
        vals = []
        for cname in config_names:
            m = config_metrics.get(cname, {})
            vals.append(m.get(mkey, 0.0))
        bars = ax.bar(x + mi * width, vals, width, label=mlabel, color=mcolor, alpha=0.82, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Solo la baseline se marca como referencia (◆), sin etiqueta de ganador
    for ci, cname in enumerate(config_names):
        if cname == baseline_name:
            center = x[ci] + (n_metrics - 1) * width / 2
            ax.axvspan(x[ci] - 0.15, x[ci] + n_metrics * width - 0.05, alpha=0.05, color="gray", zorder=0)
            ax.text(center, 1.02, "◆ Ref. V1", ha="center", va="bottom", fontsize=8,
                    color="gray", transform=ax.get_xaxis_transform())

    ax.set_xticks(x + (n_metrics - 1) * width / 2)
    tick_labels = [f"◆ {c}" if c == baseline_name else c for c in config_names]
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Métrica (escala 0–1)")
    n_sample = list(config_metrics.values())[0].get("n", "?")
    ax.set_title(f"{title}\n(n={n_sample})", fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.axhline(y=1.0, color="lightgray", linestyle=":", linewidth=0.8)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura guardada: {out_path}")

# Casos de test canónicos - fuente única para todos los experimentos
TEST_CASES: list[dict] = [
    {
        "id": "Q01",
        "label": "Q1",
        "query": "tengo miedo de ir al instituto porque me amenazan por internet",
        "emotion": "fear",
        "expected_pillars": {1, 2},
        "set": "standard",
    },
    {
        "id": "Q02",
        "label": "Q2",
        "query": "me han publicado fotos mías sin permiso y no sé qué hacer",
        "emotion": "fear",
        "expected_pillars": {1, 5},
        "set": "standard",
    },
    {
        "id": "Q03",
        "label": "Q3",
        "query": "no puedo parar de llorar me siento muy mal y no sé por qué",
        "emotion": "sadness",
        "expected_pillars": {2},
        "set": "standard",
    },
    {
        "id": "Q04",
        "label": "Q4",
        "query": "todo el mundo me odia en clase y en los grupos de whatsapp me ignoran",
        "emotion": "sadness",
        "expected_pillars": {2},
        "set": "standard",
    },
    {
        "id": "Q05",
        "label": "Q5",
        "query": "quiero denunciar al que me acosa pero no sé cómo hacerlo",
        "emotion": "anger",
        "expected_pillars": {1},
        "set": "standard",
    },
    {
        "id": "Q06",
        "label": "Q6",
        "query": "cómo bloqueo a alguien en instagram que me está insultando",
        "emotion": "others",
        "expected_pillars": {5},
        "set": "standard",
    },
    {
        "id": "Q07",
        "label": "Q7",
        "query": "todo lo que hago sale mal y no sirvo para nada",
        "emotion": "sadness",
        "expected_pillars": {3},
        "set": "standard",
    },
    {
        "id": "Q08",
        "label": "Q8",
        "query": "me van a pegar si no hago lo que dicen me da muchísimo miedo",
        "emotion": "fear",
        "expected_pillars": {4},
        "set": "standard",
    },
    {
        "id": "Q09",
        "label": "Q9",
        "query": "ya no quiero seguir viviendo nadie me va a echar de menos",
        "emotion": "sadness",
        "expected_pillars": {4},
        "set": "standard",
    },
    {
        "id": "Q10",
        "label": "Q10",
        "query": "me da mucha vergüenza lo que pasó y no puedo contárselo a nadie",
        "emotion": "disgust",
        "expected_pillars": {2},
        "set": "standard",
    },
]

# Set de pruebas Lexical-Search (BM25). Evalúa la capacidad del sistema para recuperar entidades exactas, contrarrestando la debilidad de los modelos densos (embeddings) con acrónimos.
BM25_TEST_CASES: list[dict] = [
    {
        "id": "BM25_01",
        "label": "BM25_1",
        "query": "cómo bloqueo en instagram",
        "emotion": "others",
        "expected_pillars": {5},
        "set": "bm25",
        "note": "Nombre de plataforma exacto --> pillar5_digital",
    },
    {
        "id": "BM25_02",
        "label": "BM25_2",
        "query": "teléfono 024",
        "emotion": "fear",
        "expected_pillars": {4},
        "set": "bm25",
        "note": "Número de línea de crisis --> pillar4_pap",
    },
    {
        "id": "BM25_03",
        "label": "BM25_3",
        "query": "ANAR 900 202 010",
        "emotion": "fear",
        "expected_pillars": {3, 4},
        "set": "bm25",
        "note": "Número exacto ANAR --> pillar4_pap y pillar3_hope",
    },
    {
        "id": "BM25_04",
        "label": "BM25_4",
        "query": "discord bloquear usuario",
        "emotion": "others",
        "expected_pillars": {5},
        "set": "bm25",
        "note": "Plataforma Discord --> pillar5_digital",
    },
    {
        "id": "BM25_05",
        "label": "BM25_5",
        "query": "AEPD canal joven",
        "emotion": "others",
        "expected_pillars": {1, 4, 5},
        "set": "bm25",
        "note": "Sigla organismo --> pillar1, pillar4 y pillar5",
    },
    {
        "id": "BM25_06",
        "label": "BM25_6",
        "query": "llamar al 017 incibe",
        "emotion": "others",
        "expected_pillars": {1, 4, 5},
        "set": "bm25",
        "note": "Número y sigla clave en ciberseguridad (INCIBE)",
    },
    {
        "id": "BM25_07",
        "label": "BM25_7",
        "query": "me están haciendo sexting",
        "emotion": "fear",
        "expected_pillars": {1},
        "set": "bm25",
        "note": "Término técnico de ciberacoso",
    },
    {
        "id": "BM25_08",
        "label": "BM25_8",
        "query": "hacer capturas de pantalla notario",
        "emotion": "others",
        "expected_pillars": {1},
        "set": "bm25",
        "note": "Recolección de pruebas exactas --> pillar1_action",
    }
]


# Funciones de métricas
def compute_precision_at_k(retrieved: list[int], expected: set[int], k: int) -> float:
    """Fracción de los primeros k resultados que pertenecen a un pilar esperado.

    Args:
        retrieved: Lista de pilares recuperados (en orden de ranking).
        expected: Conjunto de pilares correctos para la consulta.
        k: Número de resultados a considerar.

    Returns:
        Valor entre 0.0 y 1.0, redondeado a 4 decimales.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for p in retrieved[:k] if p in expected)
    return round(hits / k, 4)


def compute_precision_at_1(retrieved: list[int], expected: set[int]) -> float:
    """1.0 si el primer resultado pertenece a un pilar esperado, 0.0 si no.

    Args:
        retrieved: Lista de pilares recuperados.
        expected: Conjunto de pilares correctos para la consulta.

    Returns:
        1.0 o 0.0.
    """
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] in expected else 0.0


def compute_hit_at_any(retrieved: list[int], expected: set[int]) -> bool:
    """True si al menos un resultado recuperado pertenece a un pilar esperado.

    Args:
        retrieved: Lista de pilares recuperados.
        expected: Conjunto de pilares correctos para la consulta.

    Returns:
        bool.
    """
    return any(p in expected for p in retrieved)


def compute_metrics_for_run(results: list[dict]) -> dict:
    """Calcula métricas agregadas para una lista de resultados por consulta.

    Args:
        results: Lista de dicts con al menos las claves 'precision_at_3' y
                 'hit_at_any'. Si los dicts incluyen 'precision_at_1',
                 también se calcula su media.

    Returns:
        Dict con 'precision_at_3_media', 'hit_at_any_rate' y, si está disponible
        en los resultados, 'precision_at_1_media'.
    """
    n = len(results)
    if n == 0:
        return {"precision_at_3_media": 0.0, "hit_at_any_rate": 0.0}

    p3 = round(sum(r["precision_at_3"] for r in results) / n, 4)
    hit = round(sum(1 for r in results if r["hit_at_any"]) / n, 4)
    out: dict = {"precision_at_3_media": p3, "hit_at_any_rate": hit}

    if "precision_at_1" in results[0]:
        p1 = round(sum(r["precision_at_1"] for r in results) / n, 4)
        out["precision_at_1_media"] = p1

    return out


# Presentación de resultados
def print_results_table(config_name: str, results: list[dict], show_emotion: bool = False) -> None:
    """Imprime tabla de resultados por consulta en formato de consola.

    Detecta automáticamente si los resultados incluyen 'precision_at_1'
    y añade esa columna a la tabla en ese caso.

    Args:
        config_name: Nombre de la configuración (aparece en la cabecera).
        results: Lista de dicts con métricas por consulta.
        show_emotion: Si True, incluye la columna de emoción detectada.
    """
    if not results:
        return

    has_p1 = "precision_at_1" in results[0]
    n = len(results)
    p3 = round(sum(r["precision_at_3"] for r in results) / n, 4)
    hit = round(sum(1 for r in results if r["hit_at_any"]) / n, 4)

    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {config_name}")
    print(sep)

    if show_emotion and has_p1:
        print(
            f"  {'Consulta':<8} {'Emoción':<10} {'Esperado':<12}"
            f" {'Recuperado':<16} {'P@1':>5} {'P@3':>6} {'Hit':>5}"
        )
        print(
            f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*5} {'-'*6} {'-'*5}"
        )
        for r in results:
            exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
            ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
            hit_sym = "V" if r["hit_at_any"] else "X"
            print(
                f"  {r['label']:<8} {r.get('emotion',''):<10} {exp_str:<12}"
                f" {ret_str:<16} {r.get('precision_at_1',0):>5.0f}"
                f" {r['precision_at_3']:>6.2f} {hit_sym:>5}"
            )
        print(
            f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*5} {'-'*6} {'-'*5}"
        )
        p1 = round(sum(r.get("precision_at_1", 0) for r in results) / n, 4)
        print(
            f"  {'MEDIA':<8} {'':<10} {'':<12} {'':<16}"
            f" {p1:>5.2f} {p3:>6.2f} {hit:>5.2f}"
        )

    elif show_emotion:
        print(
            f"  {'Consulta':<8} {'Emoción':<10} {'Esperado':<12}"
            f" {'Recuperado':<16} {'P@3':>6} {'Hit':>5}"
        )
        print(f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*6} {'-'*5}")
        for r in results:
            exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
            ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
            hit_sym = "V" if r["hit_at_any"] else "X"
            print(
                f"  {r['label']:<8} {r.get('emotion',''):<10} {exp_str:<12}"
                f" {ret_str:<16} {r['precision_at_3']:>6.2f} {hit_sym:>5}"
            )
        print(f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*6} {'-'*5}")
        print(
            f"  {'MEDIA':<8} {'':<10} {'':<12} {'':<16}"
            f" {p3:>6.2f} {hit:>5.2f}"
        )

    else:
        print(
            f"  {'Consulta':<8} {'Esperado':<12} {'Recuperado':<18}"
            f" {'P@3':>6} {'Hit':>5}"
        )
        print(f"  {'-'*7} {'-'*11} {'-'*17} {'-'*6} {'-'*5}")
        for r in results:
            exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
            ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
            hit_sym = "V" if r["hit_at_any"] else "X"
            print(
                f"  {r['label']:<8} {exp_str:<12} {ret_str:<18}"
                f" {r['precision_at_3']:>6.2f} {hit_sym:>5}"
            )
        print(f"  {'-'*7} {'-'*11} {'-'*17} {'-'*6} {'-'*5}")
        print(
            f"  {'MEDIA':<8} {'':<12} {'':<18} {p3:>6.2f} {hit:>5.2f}"
        )

    print(sep)


# Índice FAISS en memoria
def build_faiss_index_from_embeddings(embeddings: Union[list[list[float]], np.ndarray]) -> faiss.IndexFlatIP:
    """Construye un índice FAISS IndexFlatIP en memoria a partir de vectores precalculados.

    IndexFlatIP con vectores normalizados equivale a similitud coseno exacta.
    No persiste en disco; reproducible en cada ejecución.

    Args:
        embeddings: Lista de vectores normalizados o array numpy (n × dim).

    Returns:
        Índice FAISS listo para búsqueda.
    """
    arr = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


# Guardado de resultados
def save_results(filename: str, output: dict) -> Path:
    """Guarda resultados en JSON dentro de RESULTS_DIR y confirma en consola.

    Args:
        filename: Nombre del archivo (ej. 'chunking_results.json').
        output: Diccionario serializable a JSON.

    Returns:
        Path del archivo guardado.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Resultados de evaluación guardados en {out_path}")
    return out_path


# Carga del corpus
def load_chunks_from_markdown(corpus_dir: Path) -> list[Document]:
    """Parsea los archivos .md del corpus RAG y devuelve Documents de LangChain.

    Fuente canónica de carga del corpus para todos los experimentos.
    Cada chunk está delimitado por bloques '---' con front matter YAML.
    Solo se cargan chunks con los campos 'id' y 'pillar' presentes en el YAML.

    Los metadatos se serializan compatibles con ChromaDB:
    - emotions (list) --> JSON string
    - therapeutic_technique (None) --> ""

    Args:
        corpus_dir: Directorio con los archivos .md del corpus RAG.

    Returns:
        Lista de Document de LangChain con page_content y metadatos completos.
    """
    documents: list[Document] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        parts = [p for p in parts if p.strip()]
        for i in range(0, len(parts) - 1, 2):
            try:
                meta: dict = yaml.safe_load(parts[i]) or {}
            except yaml.YAMLError:
                continue
            if "id" not in meta or "pillar" not in meta:
                continue
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": str(meta["id"]),
                        "pillar": int(meta["pillar"]),
                        "title": str(meta.get("title", "")),
                        "source": str(meta.get("source", "")),
                        "emotions": json.dumps(
                            list(meta.get("emotions") or []), ensure_ascii=False
                        ),
                        "trigger": str(meta.get("trigger", "conditional")),
                        "therapeutic_technique": meta.get("therapeutic_technique") or "",
                        "audience": str(meta.get("audience", "adolescente")),
                    },
                )
            )
    return documents
