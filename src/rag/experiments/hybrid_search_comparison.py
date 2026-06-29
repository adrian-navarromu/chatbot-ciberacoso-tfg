"""
Experimento 4: Diseño factorial 2×2 sobre el método de búsqueda RAG.

Banco de evaluación: 72 casos (expected_pillars ≠ ∅, crisis_expected ≠ HIGH).
La emoción del banco se inyecta directamente al routing (sin clasificador externo).
Fusión: combinación lineal ponderada de scores normalizados min-max (NO RRF).
    score_final = 0.4 * bm25_norm + 0.6 * sem_norm

Factores del 2×2 (todas las configs evaluadas sobre el conjunto completo):
    F1 BM25:    sin BM25 (semántico puro) vs con BM25 (fusión min-max)
    F2 Routing: sin routing (global) vs con routing oracle (ORACLE_ROUTING)

Configuraciones:
    A — Semántico puro, sin routing  (≈ Ref. V1 — punto de referencia)
    B — Híbrido min-max, sin routing
    C — Semántico puro, con routing oracle
    D — Híbrido min-max, con routing oracle

NOTA: no se declara ninguna configuración como ganadora.
      Los efectos factoriales se reportan para que el investigador decida.

Requiere GPU (SentenceTransformer embeddings, distilroberta).

Uso:
    conda activate chatbots
    python src/rag/experiments/hybrid_search_comparison.py
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
from rank_bm25 import BM25Okapi
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
    load_eval_cases, make_oracle_result,
    compute_eval_metrics, rag_figure_bar,
    load_chunks_from_markdown, save_results,
)

EMBEDDING_MODEL = "hackathon-pln-es/paraphrase-spanish-distilroberta"
TOP_K = 3
BM25_WEIGHT: float = 0.4
SEM_WEIGHT:  float = 0.6


# ---------------------------------------------------------------------------
# Fusión min-max ponderada
# ---------------------------------------------------------------------------

def _minmax_normalize(scores: list[float]) -> list[float]:
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _manual_hybrid(
    query: str,
    bm25_docs: list[Document],
    chroma: Chroma,
    chroma_filter: dict | None,
    top_k: int = TOP_K,
) -> list[Document]:
    """Fusión lineal min-max de BM25 y semántico (no RRF)."""
    n = len(bm25_docs)
    bm25_model = BM25Okapi([d.page_content.split() for d in bm25_docs])
    bm25_raw = list(bm25_model.get_scores(query.split()))

    search_kwargs: dict = {"k": n}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    sem_results = chroma.similarity_search_with_relevance_scores(query, **search_kwargs)
    sem_map: dict[str, float] = {doc.metadata["chunk_id"]: score for doc, score in sem_results}
    sem_raw = [sem_map.get(d.metadata["chunk_id"], 0.0) for d in bm25_docs]

    scored = [
        (doc, BM25_WEIGHT * b + SEM_WEIGHT * s)
        for doc, b, s in zip(bm25_docs, _minmax_normalize(bm25_raw), _minmax_normalize(sem_raw))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# Adaptador + ChromaDB LangChain
# ---------------------------------------------------------------------------

class _STEmbeddingsAdapter(Embeddings):
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def build_chroma_in_memory(documents: list[Document], embedding_adapter: _STEmbeddingsAdapter) -> Chroma:
    """Construye colección ChromaDB LangChain efímera (EphemeralClient, no singleton)."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        "exp4_hybrid",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    texts = [d.page_content for d in documents]
    embeddings = embedding_adapter.embed_documents(texts)
    collection.add(
        ids=[d.metadata["chunk_id"] for d in documents],
        embeddings=embeddings,
        documents=texts,
        metadatas=[d.metadata for d in documents],
    )
    return Chroma(
        client=client,
        collection_name="exp4_hybrid",
        embedding_function=embedding_adapter,
    )


# ---------------------------------------------------------------------------
# Evaluación — 4 configuraciones
# ---------------------------------------------------------------------------

def _retriever_result(tc: dict, docs: list[Document], top_k: int, extra: dict | None = None) -> dict:
    retrieved_pillars = [d.metadata.get("pillar") for d in docs[:top_k]]
    return make_oracle_result(tc, retrieved_pillars, top_k, extra)


def evaluate_config_a(chroma: Chroma, cases: list[dict], top_k: int = TOP_K) -> list[dict]:
    """A: semántico puro, sin routing (≈ Ref. V1)."""
    retriever = chroma.as_retriever(search_kwargs={"k": top_k})
    return [_retriever_result(tc, retriever.invoke(tc["query"]), top_k) for tc in cases]


def evaluate_config_b(documents: list[Document], chroma: Chroma, cases: list[dict], top_k: int = TOP_K) -> list[dict]:
    """B: híbrido BM25+sem (min-max), sin routing."""
    return [
        _retriever_result(tc, _manual_hybrid(tc["query"], documents, chroma, None, top_k), top_k)
        for tc in cases
    ]


def evaluate_config_c(chroma: Chroma, cases: list[dict], top_k: int = TOP_K) -> list[dict]:
    """C: semántico puro + pre-filtro oracle routing."""
    results: list[dict] = []
    for tc in cases:
        pillar_filter = ORACLE_ROUTING.get(tc.get("emotion", "others"))
        if pillar_filter is None:
            retriever = chroma.as_retriever(search_kwargs={"k": top_k})
        else:
            retriever = chroma.as_retriever(
                search_kwargs={"k": top_k, "filter": {"pillar": {"$in": pillar_filter}}}
            )
        results.append(_retriever_result(tc, retriever.invoke(tc["query"]), top_k,
                                          extra={"pillar_filter": pillar_filter}))
    return results


def evaluate_config_d(chroma: Chroma, documents: list[Document], cases: list[dict], top_k: int = TOP_K) -> list[dict]:
    """D: BM25 dinámico + ChromaDB, ambos en subespacio filtrado oracle."""
    results: list[dict] = []
    for tc in cases:
        pillar_filter = ORACLE_ROUTING.get(tc.get("emotion", "others"))
        if pillar_filter is None:
            bm25_docs, chroma_filter = documents, None
        else:
            bm25_docs = [d for d in documents if d.metadata.get("pillar") in pillar_filter]
            if not bm25_docs:
                bm25_docs = documents
            chroma_filter = {"pillar": {"$in": pillar_filter}}
        results.append(_retriever_result(
            tc, _manual_hybrid(tc["query"], bm25_docs, chroma, chroma_filter, top_k), top_k,
            extra={"pillar_filter": pillar_filter},
        ))
    return results


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict]) -> tuple[float, float, float]:
    if not results:
        return 0.0, 0.0, 0.0
    n = len(results)
    p1  = round(sum(r.get("precision_at_1", 0.0) for r in results) / n, 4)
    p3  = round(sum(r["precision_at_3"] for r in results) / n, 4)
    hit = round(sum(1 for r in results if r["hit_at_any"]) / n, 4)
    return p1, p3, hit


def _print_factorial_summary(cfg_metrics: dict[str, dict], model_name: str) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  DISEÑO FACTORIAL 2×2 — BM25 × ROUTING (banco=72)  [{model_name}]")
    print(f"  Factor 1: BM25 on/off   Factor 2: routing on/off")
    print(sep)
    print(f"  {'Config':<44} {'P@1':>6} {'P@3':>6} {'Hit':>6}")
    print(f"  {'-'*43} {'-'*6} {'-'*6} {'-'*6}")
    for cfg_name, m in cfg_metrics.items():
        tag = " [Ref.V1]" if cfg_name == "A" else ""
        print(f"  {cfg_name + tag:<44}"
              f" {m.get('precision_at_1_media', 0):>6.3f}"
              f" {m.get('precision_at_3_media', 0):>6.3f}"
              f" {m.get('hit_at_any_rate', 0):>6.3f}")
    print()
    # Efectos factoriales
    a = cfg_metrics["A"]; b = cfg_metrics["B"]
    c = cfg_metrics["C"]; d = cfg_metrics["D"]
    sign = lambda x: f"{x:+.3f}"
    print(f"  EFECTOS FACTORIALES (P@3) — solo descriptivos:")
    print(f"    Efecto BM25 sin routing (B−A):  ΔP@3 = {sign(b['precision_at_3_media']-a['precision_at_3_media'])}")
    print(f"    Efecto routing sin BM25 (C−A):  ΔP@3 = {sign(c['precision_at_3_media']-a['precision_at_3_media'])}")
    print(f"    BM25 con routing (D−C):         ΔP@3 = {sign(d['precision_at_3_media']-c['precision_at_3_media'])}")
    print(f"    Efecto total (D−A):             ΔP@3 = {sign(d['precision_at_3_media']-a['precision_at_3_media'])}")
    print(sep + "\n")


def _figure(cfg_metrics: dict[str, dict]) -> None:
    label_map = {
        "A\n(sem global)":     "A",
        "B\n(híbrido global)": "B",
        "C\n(sem+routing)":    "C",
        "D\n(híbrido+routing)":"D",
    }
    config_metrics = {lbl: cfg_metrics[k] for lbl, k in label_map.items()}
    rag_figure_bar(
        config_names=list(label_map.keys()),
        config_metrics=config_metrics,
        baseline_name="A\n(sem global)",
        title="Exp.4 — Factorial 2×2 BM25 × routing",
        out_path=FIGURES_DIR / "exp4_hybrid.png",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Exp.4: diseño factorial 2×2. Sin declaración de ganador."""
    cases = load_eval_cases()
    log.info(f"Banco de evaluación: {len(cases)} casos.")

    log.info(f"Cargando {EMBEDDING_MODEL}...")
    embedding_adapter = _STEmbeddingsAdapter(EMBEDDING_MODEL)

    log.info("Construyendo ChromaDB en memoria (EphemeralClient)...")
    documents = load_chunks_from_markdown(CORPUS_DIR)
    chroma = build_chroma_in_memory(documents, embedding_adapter)
    log.info(f"Corpus: {len(documents)} chunks indexados.")

    log.info("[A] Semántico puro, sin routing...")
    res_a = evaluate_config_a(chroma, cases)

    log.info("[B] Híbrido BM25+sem (min-max), sin routing...")
    res_b = evaluate_config_b(documents, chroma, cases)

    log.info("[C] Semántico puro, con routing...")
    res_c = evaluate_config_c(chroma, cases)

    log.info("[D] Híbrido BM25+sem + routing...")
    res_d = evaluate_config_d(chroma, documents, cases)

    cfg_metrics = {
        "A": compute_eval_metrics(res_a),
        "B": compute_eval_metrics(res_b),
        "C": compute_eval_metrics(res_c),
        "D": compute_eval_metrics(res_d),
    }

    _print_factorial_summary(cfg_metrics, EMBEDDING_MODEL)
    _figure(cfg_metrics)

    routing_table = {k: v for k, v in ORACLE_ROUTING.items()}

    def _mk_entry(cfg: str, res: list[dict], desc: str, is_ref: bool = False) -> dict:
        m = cfg_metrics[cfg]
        return {
            "descripcion": desc,
            "es_referencia_v1": is_ref,
            **m,
            "resultados_por_consulta": res,
        }

    output: dict = {
        "experimento": "hybrid_search_comparison",
        "evaluacion": "completa",
        "seed": SEED,
        "diseno": "factorial_2x2_bm25_x_routing",
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "bm25_weight": BM25_WEIGHT,
        "sem_weight":  SEM_WEIGHT,
        "fusion": "min-max ponderada (NO RRF)",
        "banco": {"n_total": len(cases)},
        "routing_emocional": routing_table,
        "efectos_factoriales": {
            "bm25_sin_routing_B_minus_A":    round(cfg_metrics["B"]["precision_at_3_media"] - cfg_metrics["A"]["precision_at_3_media"], 4),
            "routing_sin_bm25_C_minus_A":    round(cfg_metrics["C"]["precision_at_3_media"] - cfg_metrics["A"]["precision_at_3_media"], 4),
            "bm25_con_routing_D_minus_C":    round(cfg_metrics["D"]["precision_at_3_media"] - cfg_metrics["C"]["precision_at_3_media"], 4),
            "efecto_total_D_minus_A":        round(cfg_metrics["D"]["precision_at_3_media"] - cfg_metrics["A"]["precision_at_3_media"], 4),
        },
        "configuracion_a": _mk_entry("A", res_a, "ChromaDB sem. puro, sin BM25, sin routing (≈ Ref. V1)", is_ref=True),
        "configuracion_b": _mk_entry("B", res_b, "Fusión min-max BM25+ChromaDB, corpus completo, sin routing"),
        "configuracion_c": _mk_entry("C", res_c, "ChromaDB sem. con pre-filtro de routing emocional"),
        "configuracion_d": _mk_entry("D", res_d, "Fusión min-max BM25+ChromaDB + routing emocional"),
    }
    save_results("hybrid_search_results.json", output)
    log.info("Exp.4 completado. Figura: exp4_hybrid.png")


if __name__ == "__main__":
    main()
