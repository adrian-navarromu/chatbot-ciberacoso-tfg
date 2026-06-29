"""
Experimento 3: Comparación de bases vectoriales FAISS vs ChromaDB.

Banco de evaluación: 72 casos (expected_pillars ≠ ∅, crisis_expected ≠ HIGH).
La emoción del banco se inyecta directamente al routing (sin clasificador externo).

FIX: se reemplaza el acceso al cliente nativo de ChromaDB
(collection.query con where=$in era inestable) por LangChain Chroma
con similarity_search_with_relevance_scores, idéntico al Exp.4 que funciona.

Configuraciones evaluadas (conjunto completo de evaluación):
    1. FAISS global sin routing               (Ref. V1 — reconstruido sobre corpus actual)
    2. ChromaDB + mpnet, sin routing
    3. ChromaDB + mpnet, con routing oracle
    4. ChromaDB + distilroberta, sin routing
    5. ChromaDB + distilroberta, con routing oracle
    6. ChromaDB + bge-m3, sin routing
    7. ChromaDB + bge-m3, con routing oracle

Requiere GPU (3 modelos SentenceTransformer).

Uso:
    conda activate chatbots
    python src/rag/experiments/vectorstore_comparison.py
"""

from __future__ import annotations

import sys
import random
import logging
from pathlib import Path

import chromadb
import numpy as np
import torch
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 112
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.experiments.utils import (
    CORPUS_DIR, FIGURES_DIR,
    ORACLE_ROUTING,
    load_eval_cases, oracle_faiss_search, make_oracle_result,
    compute_eval_metrics, rag_figure_bar,
    load_chunks_from_markdown, save_results,
)

EMBEDDING_MODEL_V1            = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL_DISTILROBERTA = "hackathon-pln-es/paraphrase-spanish-distilroberta"
EMBEDDING_MODEL_BGE            = "BAAI/bge-m3"
TOP_K = 3

# Contador global para nombres únicos de colección (evita colisión en singleton)
_COL_COUNTER = 0


def _next_col_name(prefix: str) -> str:
    global _COL_COUNTER
    _COL_COUNTER += 1
    return f"{prefix}_{_COL_COUNTER}"


# ---------------------------------------------------------------------------
# Adaptador de embeddings LangChain
# ---------------------------------------------------------------------------

class _STEmbeddingsAdapter(Embeddings):
    """Adaptador LangChain sobre SentenceTransformer (para similarity_search)."""

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


# ---------------------------------------------------------------------------
# Construcción de índices
# ---------------------------------------------------------------------------

def build_faiss_embs(docs: list[Document], model: SentenceTransformer) -> np.ndarray:
    """Codifica el corpus y devuelve la matriz de embeddings (para FAISS numpy)."""
    log.info(f"  Codificando {len(docs)} chunks...")
    return model.encode(
        [d.page_content for d in docs],
        normalize_embeddings=True, batch_size=32, show_progress_bar=False,
    ).astype(np.float32)


def build_langchain_chroma(
    docs: list[Document],
    doc_embs: np.ndarray,
    adapter: _STEmbeddingsAdapter,
    col_name: str,
) -> Chroma:
    """Construye un Chroma LangChain efímero con embeddings pre-computados.

    FIX vs. cliente nativo: usa LangChain Chroma (mismo mecanismo que Exp.4).
    similarity_search_with_relevance_scores con filter={'pillar': {'$in': [...]}}
    funciona correctamente con esta interfaz.

    Args:
        docs: Documents del corpus.
        doc_embs: Embeddings pre-computados, shape (n, dim).
        adapter: Adaptador LangChain para embedding de queries en tiempo de búsqueda.
        col_name: Nombre único de la colección ChromaDB efímera.
    """
    client = chromadb.EphemeralClient()   # cliente aislado, sin singleton
    collection = client.create_collection(
        col_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    collection.add(
        ids=[d.metadata["chunk_id"] for d in docs],
        embeddings=doc_embs.tolist(),
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
    )
    return Chroma(
        client=client,
        collection_name=col_name,
        embedding_function=adapter,
    )


# ---------------------------------------------------------------------------
# Evaluación — 2 variantes (FAISS y ChromaDB)
# ---------------------------------------------------------------------------

def evaluate_faiss_global(
    docs: list[Document],
    doc_embs: np.ndarray,
    model: SentenceTransformer,
    cases: list[dict],
) -> list[dict]:
    """FAISS global SIN routing (Ref. V1). Reconstruido sobre corpus actual."""
    doc_pillars = [d.metadata.get("pillar", 0) for d in docs]
    results: list[dict] = []
    for tc in cases:
        q_emb = model.encode([tc["query"]], normalize_embeddings=True).astype(np.float32)[0]
        global_idx = oracle_faiss_search(q_emb, doc_embs, doc_pillars, None, TOP_K)
        retrieved_pillars = [docs[i].metadata.get("pillar", 0) for i in global_idx]
        results.append(make_oracle_result(tc, retrieved_pillars, TOP_K))
    return results


def evaluate_chroma_oracle(
    docs: list[Document],
    doc_embs: np.ndarray,
    adapter: _STEmbeddingsAdapter,
    cases: list[dict],
    use_routing: bool,
    col_prefix: str,
) -> list[dict]:
    """ChromaDB LangChain con o sin routing oracle.

    FIX: usa similarity_search_with_relevance_scores (mismo path que Exp.4),
    que aplica correctamente filter={'pillar': {'$in': pillar_filter}}.
    No usa collection.query del cliente nativo (inestable con $in en algunos casos).
    """
    chroma = build_langchain_chroma(docs, doc_embs, adapter, _next_col_name(col_prefix))
    results: list[dict] = []
    for tc in cases:
        pillar_filter = ORACLE_ROUTING.get(tc.get("emotion", "others")) if use_routing else None
        search_kwargs: dict = {"k": TOP_K}
        if pillar_filter:
            search_kwargs["filter"] = {"pillar": {"$in": pillar_filter}}
        try:
            sem_results = chroma.similarity_search_with_relevance_scores(tc["query"], **search_kwargs)
            retrieved_pillars = [doc.metadata.get("pillar", 0) for doc, _ in sem_results]
        except Exception as exc:
            log.warning(f"  Chroma query error para '{tc['id']}': {exc}")
            retrieved_pillars = []
        results.append(make_oracle_result(tc, retrieved_pillars, TOP_K,
                                          extra={"pillar_filter": pillar_filter}))
    return results


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

def _print_comparison_table(config_rows: list[tuple[str, dict]]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  TABLA COMPARATIVA — BASES VECTORIALES (banco=72)")
    print(sep)
    print(f"  {'Configuración':<52} {'P@1':>6} {'P@3':>6} {'Hit':>6}")
    print(f"  {'-'*51} {'-'*6} {'-'*6} {'-'*6}")
    for name, m in config_rows:
        tag = " [Ref.V1]" if "FAISS" in name and "routing" not in name.lower() else ""
        print(f"  {name[:52] + tag:<52}"
              f" {m.get('precision_at_1_media', 0):>6.3f}"
              f" {m.get('precision_at_3_media', 0):>6.3f}"
              f" {m.get('hit_at_any_rate', 0):>6.3f}")
    print(sep + "\n")


def _figure(config_names: list[str], all_metrics: dict) -> None:
    rag_figure_bar(
        config_names=config_names,
        config_metrics=all_metrics,
        baseline_name="FAISS global\n(Ref. V1)",
        title="Exp.3 — Vectorstore: FAISS vs ChromaDB",
        out_path=FIGURES_DIR / "exp3_vectorstore.png",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta Exp.3 sobre el conjunto de evaluación completo."""
    cases = load_eval_cases()
    log.info(f"Banco de evaluación: {len(cases)} casos.")

    docs = load_chunks_from_markdown(CORPUS_DIR)
    log.info(f"Corpus: {len(docs)} chunks.")

    all_metrics: dict[str, dict] = {}
    results_store: dict[str, list[dict]] = {}

    # ---- 1. FAISS global (Ref. V1) ----
    log.info("[1/7] FAISS global + mpnet (Ref. V1)...")
    model_mpnet = SentenceTransformer(EMBEDDING_MODEL_V1)
    adapter_mpnet = _STEmbeddingsAdapter(model_mpnet)
    doc_embs_mpnet = build_faiss_embs(docs, model_mpnet)
    res_faiss = evaluate_faiss_global(docs, doc_embs_mpnet, model_mpnet, cases)
    all_metrics["FAISS global\n(Ref. V1)"] = compute_eval_metrics(res_faiss)
    results_store["faiss_global"] = res_faiss

    # ---- 2-3. ChromaDB + mpnet ----
    log.info("[2/7] ChromaDB + mpnet, sin routing...")
    res_mpnet_nf = evaluate_chroma_oracle(docs, doc_embs_mpnet, adapter_mpnet, cases, False, "mpnet_nf")
    all_metrics["ChromaDB+mpnet\nsin routing"] = compute_eval_metrics(res_mpnet_nf)
    results_store["chromadb_mpnet_nf"] = res_mpnet_nf

    log.info("[3/7] ChromaDB + mpnet, con routing...")
    res_mpnet_rt = evaluate_chroma_oracle(docs, doc_embs_mpnet, adapter_mpnet, cases, True, "mpnet_rt")
    all_metrics["ChromaDB+mpnet\n+routing"] = compute_eval_metrics(res_mpnet_rt)
    results_store["chromadb_mpnet_routing"] = res_mpnet_rt
    del model_mpnet, doc_embs_mpnet

    # ---- 4-5. ChromaDB + distilroberta ----
    log.info("[4/7] ChromaDB + distilroberta, sin routing...")
    model_dr = SentenceTransformer(EMBEDDING_MODEL_DISTILROBERTA)
    adapter_dr = _STEmbeddingsAdapter(model_dr)
    doc_embs_dr = build_faiss_embs(docs, model_dr)
    res_dr_nf = evaluate_chroma_oracle(docs, doc_embs_dr, adapter_dr, cases, False, "dr_nf")
    all_metrics["ChromaDB+distil\nsin routing"] = compute_eval_metrics(res_dr_nf)
    results_store["chromadb_distil_nf"] = res_dr_nf

    log.info("[5/7] ChromaDB + distilroberta, con routing...")
    res_dr_rt = evaluate_chroma_oracle(docs, doc_embs_dr, adapter_dr, cases, True, "dr_rt")
    all_metrics["ChromaDB+distil\n+routing"] = compute_eval_metrics(res_dr_rt)
    results_store["chromadb_distil_routing"] = res_dr_rt
    del model_dr, doc_embs_dr

    # ---- 6-7. ChromaDB + bge-m3 ----
    log.info("[6/7] ChromaDB + bge-m3, sin routing...")
    model_bge = SentenceTransformer(EMBEDDING_MODEL_BGE)
    adapter_bge = _STEmbeddingsAdapter(model_bge)
    doc_embs_bge = build_faiss_embs(docs, model_bge)
    res_bge_nf = evaluate_chroma_oracle(docs, doc_embs_bge, adapter_bge, cases, False, "bge_nf")
    all_metrics["ChromaDB+bge\nsin routing"] = compute_eval_metrics(res_bge_nf)
    results_store["chromadb_bge_nf"] = res_bge_nf

    log.info("[7/7] ChromaDB + bge-m3, con routing...")
    res_bge_rt = evaluate_chroma_oracle(docs, doc_embs_bge, adapter_bge, cases, True, "bge_rt")
    all_metrics["ChromaDB+bge\n+routing"] = compute_eval_metrics(res_bge_rt)
    results_store["chromadb_bge_routing"] = res_bge_rt
    del model_bge, doc_embs_bge

    # ---- Presentación ----
    ordered = [(k, all_metrics[k]) for k in all_metrics]
    _print_comparison_table(ordered)

    config_names = list(all_metrics.keys())
    _figure(config_names, all_metrics)

    # ---- JSON ----
    routing_table = {k: v for k, v in ORACLE_ROUTING.items()}

    def _entry(key: str, desc: str, model_id: str, routing: bool, is_ref: bool = False) -> dict:
        m = all_metrics[key]
        d: dict = {
            "descripcion": desc,
            "embedding_model": model_id,
            "usa_routing": routing,
            "es_referencia_v1": is_ref,
            **m,
            "resultados_por_consulta": results_store[
                key.replace("\n", "_").replace("+", "_").replace(" ", "_").lower()
            ],
        }
        if routing:
            d["routing_emocional"] = routing_table
        return d

    output: dict = {
        "experimento": "vectorstore_comparison",
        "evaluacion": "completa",
        "seed": SEED,
        "top_k": TOP_K,
        "banco": {"n_total": len(cases)},
        "fix": "LangChain Chroma similarity_search_with_relevance_scores (no native collection.query)",
        "modelos_chroma": [EMBEDDING_MODEL_V1, EMBEDDING_MODEL_DISTILROBERTA, EMBEDDING_MODEL_BGE],
        "routing_emocional": routing_table,
        "faiss_global": {
            "descripcion": "FAISS numpy global SIN routing (Ref. V1, corpus actual)",
            "embedding_model": EMBEDDING_MODEL_V1,
            "es_referencia_v1": True,
            **all_metrics["FAISS global\n(Ref. V1)"],
            "resultados_por_consulta": results_store["faiss_global"],
        },
        "chromadb_mpnet_nf": {
            "descripcion": "ChromaDB sem. pura, mpnet, sin routing",
            "embedding_model": EMBEDDING_MODEL_V1,
            **all_metrics["ChromaDB+mpnet\nsin routing"],
            "resultados_por_consulta": results_store["chromadb_mpnet_nf"],
        },
        "chromadb_mpnet_routing": {
            "descripcion": "ChromaDB + routing emocional, mpnet",
            "embedding_model": EMBEDDING_MODEL_V1,
            "routing_emocional": routing_table,
            **all_metrics["ChromaDB+mpnet\n+routing"],
            "resultados_por_consulta": results_store["chromadb_mpnet_routing"],
        },
        "chromadb_distil_nf": {
            "descripcion": "ChromaDB sem. pura, distilroberta, sin routing",
            "embedding_model": EMBEDDING_MODEL_DISTILROBERTA,
            **all_metrics["ChromaDB+distil\nsin routing"],
            "resultados_por_consulta": results_store["chromadb_distil_nf"],
        },
        "chromadb_distil_routing": {
            "descripcion": "ChromaDB + routing emocional, distilroberta",
            "embedding_model": EMBEDDING_MODEL_DISTILROBERTA,
            "routing_emocional": routing_table,
            **all_metrics["ChromaDB+distil\n+routing"],
            "resultados_por_consulta": results_store["chromadb_distil_routing"],
        },
        "chromadb_bge_nf": {
            "descripcion": "ChromaDB sem. pura, bge-m3, sin routing",
            "embedding_model": EMBEDDING_MODEL_BGE,
            **all_metrics["ChromaDB+bge\nsin routing"],
            "resultados_por_consulta": results_store["chromadb_bge_nf"],
        },
        "chromadb_bge_routing": {
            "descripcion": "ChromaDB + routing emocional, bge-m3",
            "embedding_model": EMBEDDING_MODEL_BGE,
            "routing_emocional": routing_table,
            **all_metrics["ChromaDB+bge\n+routing"],
            "resultados_por_consulta": results_store["chromadb_bge_routing"],
        },
    }
    save_results("vectorstore_results.json", output)
    log.info("Exp.3 completado. Figura: exp3_vectorstore.png")


if __name__ == "__main__":
    main()
