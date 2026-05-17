"""
Experimento 2: Comparación de modelos de embeddings para RAG de ciberacoso.

Usa el chunking manual terapéutico (Experimento 1) y FAISS en memoria para
aislar el impacto exclusivo del modelo de embeddings, desacoplado de BM25
y ChromaDB que se evalúan en experimentos posteriores.

Uso:
    python src/rag/experiments/embedding_comparison.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Garantiza que la raíz del proyecto está en sys.path al ejecutar el script directamente.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.experiments.utils import (  # noqa: E402
    CORPUS_DIR,
    TEST_CASES,
    build_faiss_index_from_embeddings,
    compute_hit_at_any,
    compute_metrics_for_run,
    compute_precision_at_k,
    load_chunks_from_markdown,
    save_results,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TOP_K = 3


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuración de un modelo de embeddings para el experimento."""

    short_name: str
    model_id: str
    dim: int
    requires_e5_prefix: bool = False
    description: str = ""

    def __str__(self) -> str:
        return self.short_name


MODELS: list[ModelConfig] = [
    ModelConfig(
        short_name="paraphrase-multilingual-mpnet",
        model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        dim=768,
        description="V1 baseline · dim=768 · ~278M params · arquitectura más antigua",
    ),
    ModelConfig(
        short_name="multilingual-e5-large",
        model_id="intfloat/multilingual-e5-large",
        dim=1024,
        requires_e5_prefix=True,
        description="dim=1024 · ~560M params · resultados CON prefijos obligatorios query:/passage: (Wang et al. 2024)",
    ),
    ModelConfig(
        short_name="BAAI/bge-m3",
        model_id="BAAI/bge-m3",
        dim=1024,
        description="dim=1024 · ~568M params · Apache 2.0 · estado del arte multilingual 2024",
    ),
    ModelConfig(
        short_name="paraphrase-spanish-distilroberta",
        model_id="hackathon-pln-es/paraphrase-spanish-distilroberta",
        dim=768,
        description="dim=768 · español nativo · Knowledge Distillation (Reimers & Gurevych, EMNLP 2020) · teacher=mpnet · student=BERTIN",
    ),
]


# ---------------------------------------------------------------------------
# Wrapper para multilingual-e5-large
# ---------------------------------------------------------------------------


class E5LargeEmbeddings:
    """Wrapper para multilingual-e5-large que añade los prefijos
    obligatorios según la documentación oficial del modelo.
    Queries: prefijo 'query: '
    Documentos: prefijo 'passage: '
    Referencia: Wang et al. (2024) Multilingual E5 Text Embeddings.
    """

    def __init__(self) -> None:
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Codifica documentos añadiendo el prefijo 'passage: '."""
        prefixed = [f"passage: {t}" for t in texts]
        return (
            self.model.encode(prefixed, normalize_embeddings=True, batch_size=32)
            .astype(np.float32)
            .tolist()
        )

    def embed_query(self, text: str) -> list[float]:
        """Codifica una query añadiendo el prefijo 'query: '."""
        return (
            self.model.encode(
                [f"query: {text}"], normalize_embeddings=True
            )[0]
            .astype(np.float32)
            .tolist()
        )

    def embed_queries_timed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        """Codifica queries con prefijo y devuelve embeddings + latencia media en ms."""
        prefixed = [f"query: {t}" for t in texts]
        latencies: list[float] = []
        embeddings: list[list[float]] = []
        for q in prefixed:
            t0 = time.perf_counter()
            emb = self.model.encode([q], normalize_embeddings=True)[0]
            latencies.append((time.perf_counter() - t0) * 1000)
            embeddings.append(emb.astype(np.float32).tolist())
        return embeddings, float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Carga de embeddings
# ---------------------------------------------------------------------------


def load_embeddings(config: ModelConfig) -> Any:
    """Carga el modelo de embeddings apropiado para la configuración dada.

    Para multilingual-e5-large utiliza E5LargeEmbeddings para garantizar
    los prefijos query:/passage: requeridos por el modelo. Para los demás
    modelos utiliza HuggingFaceEmbeddings de LangChain.

    Args:
        config: Configuración del modelo a cargar.

    Returns:
        Objeto con interfaz embed_documents / embed_query.
    """
    if config.requires_e5_prefix:
        print(f"    Usando E5LargeEmbeddings (prefijos query:/passage: obligatorios)")
        return E5LargeEmbeddings()

    return HuggingFaceEmbeddings(
        model_name=config.model_id,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


# ---------------------------------------------------------------------------
# Medición de latencia de embeddings
# ---------------------------------------------------------------------------


def measure_query_latency(
    embeddings_obj: Any,
    queries: list[str],
) -> tuple[list[list[float]], float]:
    """Mide la latencia media de embed_query para una lista de consultas.

    Args:
        embeddings_obj: Objeto con método embed_query(text) -> list[float].
        queries: Lista de textos de consulta.

    Returns:
        Tupla (lista de embeddings, latencia_media_ms).
    """
    if isinstance(embeddings_obj, E5LargeEmbeddings):
        return embeddings_obj.embed_queries_timed(queries)

    latencies: list[float] = []
    query_embeddings: list[list[float]] = []
    for q in queries:
        t0 = time.perf_counter()
        emb = embeddings_obj.embed_query(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        query_embeddings.append(emb)
    return query_embeddings, float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Evaluación por modelo
# ---------------------------------------------------------------------------


def evaluate_model(
    config: ModelConfig,
    docs: list,
) -> dict:
    """Evalúa un modelo de embeddings sobre las 10 consultas estándar.

    Pasos:
        1. Cargar el modelo (HuggingFaceEmbeddings o wrapper E5).
        2. Codificar los documentos e indexar en FAISS in-memory.
        3. Medir latencia media de embed_query sobre las 10 queries.
        4. Recuperar top-3 por consulta y calcular precision@3 y hit@any.

    Args:
        config: Configuración del modelo (id, dim, prefijos, etc.).
        docs: Documents del corpus (chunking manual terapéutico, ganador Exp. 1).

    Returns:
        Dict con métricas y resultados por consulta.
    """
    print(f"\n  [{config.short_name}]")
    print(f"    Cargando modelo: {config.model_id}")
    embeddings_obj = load_embeddings(config)

    # Codificar corpus
    print(f"    Codificando {len(docs)} chunks...")
    t_corpus_start = time.perf_counter()
    doc_embeddings = embeddings_obj.embed_documents([doc.page_content for doc in docs])
    t_corpus = (time.perf_counter() - t_corpus_start) * 1000
    print(f"    Corpus codificado en {t_corpus:.0f} ms.")

    # Construir índice FAISS in-memory
    index = build_faiss_index_from_embeddings(doc_embeddings)

    # Medir latencia de query + obtener embeddings de las 10 consultas
    print(f"    Midiendo latencia de embedding en 10 queries...")
    queries = [tc["query"] for tc in TEST_CASES]
    query_embeddings, latencia_media_ms = measure_query_latency(embeddings_obj, queries)

    # Evaluación: precision@3 y hit@any
    query_arr = np.array(query_embeddings, dtype=np.float32)
    _, all_indices = index.search(query_arr, TOP_K)

    resultados_por_consulta: list[dict] = []
    for tc, indices_row in zip(TEST_CASES, all_indices):
        retrieved_pillars = [
            docs[idx].metadata["pillar"] for idx in indices_row if 0 <= idx < len(docs)
        ]
        expected = tc["expected_pillars"]
        resultados_por_consulta.append(
            {
                "label": tc["label"],
                "query": tc["query"],
                "expected_pillars": sorted(expected),
                "retrieved_pillars": retrieved_pillars,
                "precision_at_3": compute_precision_at_k(retrieved_pillars, expected, TOP_K),
                "hit_at_any": compute_hit_at_any(retrieved_pillars, expected),
            }
        )

    m = compute_metrics_for_run(resultados_por_consulta)
    prec_mean = m["precision_at_3_media"]
    hit_rate = m["hit_at_any_rate"]
    print(f"    precision@3={prec_mean:.4f}  hit@any={hit_rate:.4f}  latencia={latencia_media_ms:.1f} ms")

    return {
        "short_name": config.short_name,
        "model_id": config.model_id,
        "dim": config.dim,
        "description": config.description,
        "n_chunks": len(docs),
        "precision_at_3_media": prec_mean,
        "hit_at_any_rate": hit_rate,
        "latencia_media_ms": round(latencia_media_ms, 2),
        "resultados_por_consulta": resultados_por_consulta,
    }


# ---------------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------------


def _print_results_table(model_results: list[dict]) -> None:
    """Imprime la tabla comparativa de los modelos evaluados en consola."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  TABLA COMPARATIVA — MODELOS DE EMBEDDINGS")
    print(sep)
    print(
        f"  {'Modelo':<35} {'dim':>5} {'precision@3':>12} {'hit@any':>8} {'latencia_ms':>12}"
    )
    print(f"  {'-'*34} {'-'*5} {'-'*12} {'-'*8} {'-'*12}")
    for r in model_results:
        print(
            f"  {r['short_name']:<35} {r['dim']:>5}"
            f" {r['precision_at_3_media']:>12.4f}"
            f" {r['hit_at_any_rate']:>8.4f}"
            f" {r['latencia_media_ms']:>12.1f}"
        )
    print(f"{sep}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Ejecuta el experimento comparativo de embeddings y guarda resultados en JSON."""
    print("Cargando corpus con chunking manual terapéutico (ganador Exp. 1)...")
    docs = load_chunks_from_markdown(CORPUS_DIR)
    print(f"{len(docs)} chunks cargados.\n")

    all_results: list[dict] = []
    for config in MODELS:
        result = evaluate_model(config, docs)
        all_results.append(result)

    # ── Tabla comparativa ──────────────────────────────────────────────────
    _print_results_table(all_results)

    # ── Guardar JSON ───────────────────────────────────────────────────────
    prec_values = [r["precision_at_3_media"] for r in all_results]
    max_diff = round(max(prec_values) - min(prec_values), 4)

    output: dict = {
        "experimento": "embedding_comparison",
        "chunking": "manual_terapeutico",
        "top_k": TOP_K,
        "max_diff_precision_at_3": max_diff,
        "modelos_evaluados": all_results,
    }

    save_results("embedding_results.json", output)


if __name__ == "__main__":
    main()
