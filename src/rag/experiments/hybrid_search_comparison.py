"""
Experimento 4: Diseño factorial 2x2 sobre el método de búsqueda RAG.

Factores evaluados:
  - BM25: sin BM25 (ChromaDB semántico puro) vs con BM25 (EnsembleRetriever RRF)
  - Routing: sin routing (búsqueda global) vs con routing (filtro por pilar)

Implementación crítica de Config D:
    El routing se aplica ANTES de instanciar el BM25Retriever, no como post-filtro.
    Para cada consulta se seleccionan los Documents cuyo pillar pertenece a la lista
    de routing, se instancia BM25Retriever sobre ese subconjunto y se construye el
    EnsembleRetriever con el ChromaDB filtrado por el mismo criterio. Esto garantiza
    que BM25 no accede a chunks fuera del pilar emocional relevante.

Conjunto de evaluación:
  - 10 consultas estándar: miden rendimiento en consultas emocionales.
  - 5 consultas BM25: términos técnicos exactos (teléfonos, plataformas, siglas).

Uso:
    python src/rag/experiments/hybrid_search_comparison.py
"""

from __future__ import annotations

import sys
import random
import logging
from pathlib import Path
from typing import Callable

import chromadb
import numpy as np
import torch
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Semilla global de reproducibilidad (REGLA TRANSVERSAL 1)
SEED = 112
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuración de rutas
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rank_bm25 import BM25Okapi

from src.rag.experiments.utils import CORPUS_DIR, BM25_TEST_CASES, TEST_CASES, build_faiss_index_from_embeddings, compute_hit_at_any, compute_metrics_for_run, compute_precision_at_1, compute_precision_at_k, load_chunks_from_markdown, save_results


# Constantes
# Modelo ganador del Exp. 3 (Configurado manualmente tras análisis)
EMBEDDING_MODEL_WINNER = "hackathon-pln-es/paraphrase-spanish-distilroberta"
TOP_K = 3
# Pesos fusión: combinación lineal ponderada de scores normalizados (NO RRF)
BM25_WEIGHT: float = 0.4
SEM_WEIGHT: float = 0.6

# Routing determinista por emoción --> pilares (MODO ORÁCULO, sin tendencia).
# En producción (enriched_retriever.py), sadness+estable usa [2,3,4] con P4 penalizado.
# Aquí, oracle mode sin tendencia usa rama estable sin P4 para limpiar la evaluación.
# disgust incluye P5 desde Tarea 1a: muchos casos de asco implican exposición digital.
PILLAR_ROUTING: dict[str, list[int] | None] = {
    "fear": [2, 4, 1],
    "sadness": [2, 3],      # oracle mode: sin tendencia → rama estable sin P4
    "anger": [1, 2],
    "disgust": [2, 5, 1],   # P5 digital + P1 protocolo: exposición pública y sextorsión
    "joy": [3],
    "surprise": [2, 1],
    "others": None,         # sin filtro --> búsqueda global
}

# Conjunto completo: 10 consultas estándar + 5 BM25 (importadas de utils)
ALL_QUERIES: list[dict] = TEST_CASES + BM25_TEST_CASES


# Fusión min-max ponderada (reemplaza EnsembleRetriever RRF)
def _minmax_normalize(scores: list[float]) -> list[float]:
    """Normalización min-max a [0,1]. Si max==min devuelve lista de 1.0."""
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
    """Fusión lineal ponderada de BM25 y semántico con normalización min-max.

    Reemplaza EnsembleRetriever (RRF) por combinación lineal ponderada:
        score = BM25_WEIGHT * bm25_norm + SEM_WEIGHT * sem_norm
    Cada canal se normaliza min-max a [0,1] independientemente.
    Si max==min en un canal, todos los scores de ese canal = 1.0.

    Args:
        query: Texto de la consulta.
        bm25_docs: Subconjunto de documentos para el índice BM25 (puede ser global).
        chroma: Vectorstore ChromaDB en memoria.
        chroma_filter: Filtro WHERE para ChromaDB, o None para búsqueda global.
        top_k: Número de documentos a devolver.

    Returns:
        Lista de top_k Documents ordenados por score combinado descendente.
    """
    n = len(bm25_docs)

    # Scores BM25 sobre el subconjunto
    tokenized = [d.page_content.split() for d in bm25_docs]
    bm25_model = BM25Okapi(tokenized)
    bm25_raw = list(bm25_model.get_scores(query.split()))

    # Scores semánticos de ChromaDB
    search_kwargs: dict = {"k": n}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    sem_results = chroma.similarity_search_with_relevance_scores(query, **search_kwargs)
    sem_map: dict[str, float] = {doc.metadata["chunk_id"]: score for doc, score in sem_results}
    sem_raw = [sem_map.get(d.metadata["chunk_id"], 0.0) for d in bm25_docs]

    # Normalización + combinación
    bm25_norm = _minmax_normalize(bm25_raw)
    sem_norm = _minmax_normalize(sem_raw)

    scored = [
        (doc, BM25_WEIGHT * b + SEM_WEIGHT * s)
        for doc, b, s in zip(bm25_docs, bm25_norm, sem_norm)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# Adaptador de embeddings (LangChain Embeddings sobre SentenceTransformer)
class _STEmbeddingsAdapter(Embeddings):
    """Adaptador LangChain para SentenceTransformer con vectores normalizados."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return (
            self._model.encode(texts, normalize_embeddings=True, batch_size=32)
            .tolist()
        )

    def embed_query(self, text: str) -> list[float]:
        return (
            self._model.encode([text], normalize_embeddings=True)[0]
            .tolist()
        )


# Selección del modelo ganador
def resolve_winner_model() -> str:
    """Devuelve el modelo elegido manualmente tras analizar el Exp. 3.

    Ajustar EMBEDDING_MODEL_WINNER tras revisar vectorstore_results.json.
    La decisión involucra múltiples dimensiones (p@1, p@3, Δrouting, hit@any
    y características arquitectónicas del modelo) y no se automatiza.

    Returns:
        Identificador HuggingFace del modelo configurado en EMBEDDING_MODEL_WINNER.
    """
    log.info(f"Modelo configurado para Exp. 4: {EMBEDDING_MODEL_WINNER}")
    return EMBEDDING_MODEL_WINNER


# Construcción de ChromaDB en memoria
def build_chroma_in_memory(documents: list[Document], embedding_adapter: _STEmbeddingsAdapter) -> Chroma:
    """Construye una colección ChromaDB en memoria con los documentos dados.

    Usa chromadb.Client() (efímero) para evitar escritura en disco durante
    el experimento. El experimento es reproducible pero no persiste.

    Args:
        documents: Lista de Document a indexar.
        embedding_adapter: Adaptador SentenceTransformer para LangChain.

    Returns:
        Vectorstore Chroma en memoria con todos los documentos indexados.
    """
    ephemeral_client = chromadb.Client()
    collection = ephemeral_client.create_collection(
        name="exp4_hybrid",
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
        client=ephemeral_client,
        collection_name="exp4_hybrid",
        embedding_function=embedding_adapter,
    )


# Evaluación — Config A y B (sin routing, retriever_fn uniforme por consulta)
def _make_result(tc: dict, docs: list[Document], top_k: int, extra: dict | None = None) -> dict:
    """Construye el dict de métricas para una consulta dado el top-k recuperado."""
    retrieved_pillars = [d.metadata.get("pillar") for d in docs[:top_k]]
    expected = tc["expected_pillars"]
    r: dict = {
        "label": tc["label"],
        "query": tc["query"],
        "set": tc.get("set", "standard"),
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


def evaluate_retriever(retriever_fn: Callable[[str], list[Document]], queries: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Evalúa un retriever estático (Configs A y B) sobre la lista de consultas.

    No aplica ningún filtro por emoción ni routing.

    Args:
        retriever_fn: Callable(query: str) -> list[Document].
        queries: Lista de dicts con 'label', 'query', 'expected_pillars', 'set'.
        top_k: Número de resultados a evaluar.

    Returns:
        Lista de dicts con métricas por consulta (incluye precision@1).
    """
    return [_make_result(tc, retriever_fn(tc["query"]), top_k) for tc in queries]


# Evaluación — Config C (semántica con pre-filtro WHERE pillar IN [...])
def evaluate_config_c(chroma: Chroma, queries: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Config C: búsqueda semántica con pre-filtro de metadatos en ChromaDB.

    Para cada consulta aplica WHERE pillar IN [lista_routing(emoción)] antes
    de la búsqueda vectorial. Si la emoción es 'others' o no tiene routing,
    realiza búsqueda global sin filtro.

    Args:
        chroma: Vectorstore ChromaDB en memoria.
        queries: Lista de consultas con campo 'emotion'.
        top_k: Número de resultados a recuperar.

    Returns:
        Lista de dicts con métricas por consulta (incluye precision@1).
    """
    results: list[dict] = []
    for tc in queries:
        pillar_filter = PILLAR_ROUTING.get(tc.get("emotion", "others"))

        if pillar_filter is None:
            retriever = chroma.as_retriever(search_kwargs={"k": top_k})
        else:
            retriever = chroma.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "filter": {"pillar": {"$in": pillar_filter}},
                }
            )

        docs = retriever.invoke(tc["query"])
        results.append(_make_result(tc, docs, top_k, {"pillar_filter_aplicado": pillar_filter}))
    return results


# Evaluación — Config D (BM25 dinámico sobre subconjunto filtrado + ChromaDB filtrado)
def evaluate_config_d(chroma: Chroma, documents: list[Document], queries: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Config D: BM25 dinámico + ChromaDB ambos sobre el subconjunto filtrado.

    Fusión lineal ponderada con normalización min-max (BM25_WEIGHT=0.4, SEM_WEIGHT=0.6).
    Reemplaza EnsembleRetriever (RRF) por combinación lineal ponderada de scores
    normalizados min-max (TAREA 3 original): score = 0.4*bm25_norm + 0.6*sem_norm.

    Para cada consulta:
      1. Filtra el corpus por pillar_filter según PILLAR_ROUTING(emotion).
      2. Llama a _manual_hybrid() sobre el subconjunto filtrado.

    Si la emoción es 'others' o el filtro está vacío, opera sobre el corpus completo.

    Args:
        chroma: Vectorstore ChromaDB en memoria (completo).
        documents: Lista completa de Documents del corpus.
        queries: Lista de consultas con campo 'emotion'.
        top_k: Número de resultados a recuperar.

    Returns:
        Lista de dicts con métricas por consulta (incluye precision@1).
    """
    results: list[dict] = []
    for tc in queries:
        pillar_filter = PILLAR_ROUTING.get(tc.get("emotion", "others"))

        if pillar_filter is None:
            bm25_docs = documents
            chroma_filter = None
        else:
            bm25_docs = [d for d in documents if d.metadata.get("pillar") in pillar_filter]
            if not bm25_docs:
                bm25_docs = documents
            chroma_filter = {"pillar": {"$in": pillar_filter}}

        docs = _manual_hybrid(tc["query"], bm25_docs, chroma, chroma_filter, top_k)
        results.append(_make_result(tc, docs, top_k, {"pillar_filter_aplicado": pillar_filter}))
    return results


# Presentación de resultados
def _aggregate(results: list[dict]) -> tuple[float, float, float]:
    """Devuelve (p@1_media, p@3_media, hit@any_rate) para una lista de resultados."""
    if not results:
        return 0.0, 0.0, 0.0
    n = len(results)
    p1 = round(sum(r.get("precision_at_1", 0.0) for r in results) / n, 4)
    m = compute_metrics_for_run(results)
    return p1, m["precision_at_3_media"], m["hit_at_any_rate"]


def _print_detail_table(config_name: str, results: list[dict]) -> None:
    """Imprime tabla detallada por consulta (incluye p@1), destacando las consultas BM25."""
    p1, prec, hit = _aggregate(results)
    sep = "=" * 86
    print(f"\n{sep}")
    print(f"  {config_name}")
    print(sep)
    print(
        f"  {'Consulta':<10} {'Set':<9} {'Esperado':<14} "
        f"{'Recuperado':<16} {'P@1':>5} {'P@3':>6} {'Hit':>5}"
    )
    print(f"  {'-'*9} {'-'*8} {'-'*13} {'-'*15} {'-'*5} {'-'*6} {'-'*5}")
    for r in results:
        exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
        ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
        hit_sym = "V" if r["hit_at_any"] else "X"
        marker = "<" if r["set"] == "bm25" else "  "
        print(
            f"  {r['label']:<10} {r['set']:<9} {exp_str:<14} "
            f"{ret_str:<16} {r.get('precision_at_1', 0):>5.2f}"
            f" {r['precision_at_3']:>6.2f} {hit_sym:>5}{marker}"
        )
    print(f"  {'-'*9} {'-'*8} {'-'*13} {'-'*15} {'-'*5} {'-'*6} {'-'*5}")
    print(
        f"  {'MEDIA':<10} {'ALL':<9} {'':<14} {'':<16}"
        f" {p1:>5.2f} {prec:>6.2f} {hit:>5.2f}"
    )
    print(f"{sep}")


def _print_comparison_summary(sem_all: list[dict], hyb_all: list[dict], sem_rout_all: list[dict], hyb_rout_all: list[dict], model_name: str) -> None:
    """Imprime el resumen comparativo en diseño factorial 2×2 (BM25 × routing)."""
    def _split(lst: list[dict]) -> tuple[list[dict], list[dict]]:
        return (
            [r for r in lst if r["set"] == "standard"],
            [r for r in lst if r["set"] == "bm25"],
        )

    a_std, a_bm = _split(sem_all)
    b_std, b_bm = _split(hyb_all)
    c_std, c_bm = _split(sem_rout_all)
    d_std, d_bm = _split(hyb_rout_all)

    p1_a_std, p_a_std, _ = _aggregate(a_std);  p1_a_bm, p_a_bm, _ = _aggregate(a_bm)
    p1_a_all, p_a_all, _ = _aggregate(sem_all)
    p1_b_std, p_b_std, _ = _aggregate(b_std);  p1_b_bm, p_b_bm, _ = _aggregate(b_bm)
    p1_b_all, p_b_all, _ = _aggregate(hyb_all)
    p1_c_std, p_c_std, _ = _aggregate(c_std);  p1_c_bm, p_c_bm, _ = _aggregate(c_bm)
    p1_c_all, p_c_all, _ = _aggregate(sem_rout_all)
    p1_d_std, p_d_std, _ = _aggregate(d_std);  p1_d_bm, p_d_bm, _ = _aggregate(d_bm)
    p1_d_all, p_d_all, _ = _aggregate(hyb_rout_all)

    sign = lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}"

    sep = "=" * 86
    print(f"\n{sep}")
    print(f"  RESUMEN COMPARATIVO - DISEÑO FACTORIAL 2x2  [{model_name.split('/')[-1]}]")
    print(f"  (factor 1: BM25 on/off, factor 2: routing on/off)")
    print(sep)
    print(f"  {'Configuración':<46} {'Q std P@3':>10} {'Q bm25 P@3':>11} {'Total P@3':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*11} {'-'*10}")
    print(f"  {'A - Sem. pura, sin routing':<46} {p_a_std:>10.4f} {p_a_bm:>11.4f} {p_a_all:>10.4f}")
    print(f"  {'B - Híbrido BM25, sin routing':<46} {p_b_std:>10.4f} {p_b_bm:>11.4f} {p_b_all:>10.4f}")
    print(f"  {'C - Sem. pura, con routing (pre-filtro)':<46} {p_c_std:>10.4f} {p_c_bm:>11.4f} {p_c_all:>10.4f}")
    print(f"  {'D - Híbrido BM25, con routing (dinámico)':<46} {p_d_std:>10.4f} {p_d_bm:>11.4f} {p_d_all:>10.4f}")
    print(f"  {'-'*45} {'-'*10} {'-'*11} {'-'*10}")
    print(f"  {'Delta BM25 sin routing (B-A)':<46} {sign(p_b_std-p_a_std):>10} {sign(p_b_bm-p_a_bm):>11} {sign(p_b_all-p_a_all):>10}")
    print(f"  {'Delta routing sin BM25 (C-A)':<46} {sign(p_c_std-p_a_std):>10} {sign(p_c_bm-p_a_bm):>11} {sign(p_c_all-p_a_all):>10}")
    print(f"  {'Delta BM25 con routing (D-C)':<46} {sign(p_d_std-p_c_std):>10} {sign(p_d_bm-p_c_bm):>11} {sign(p_d_all-p_c_all):>10}")
    print(f"  {'Delta routing con BM25 (D-B)':<46} {sign(p_d_std-p_b_std):>10} {sign(p_d_bm-p_b_bm):>11} {sign(p_d_all-p_b_all):>10}")
    print(f"  {'Delta total (D-A)':<46} {sign(p_d_std-p_a_std):>10} {sign(p_d_bm-p_a_bm):>11} {sign(p_d_all-p_a_all):>10}")
    print(f"{sep}")
    print()


# Main
def main() -> None:
    """Ejecuta el experimento factorial 2x2 de búsqueda y guarda resultados."""

    model_name = resolve_winner_model()

    # Cargar corpus
    log.info("Cargando corpus (chunking manual terapéutico)...")
    documents = load_chunks_from_markdown(CORPUS_DIR)

    # Cargar modelo de embeddings
    log.info(f"Cargando modelo de proyecciones latentes: {model_name}...")
    embedding_adapter = _STEmbeddingsAdapter(model_name)

    # Construir ChromaDB en memoria
    log.info("Construyendo ChromaDB en memoria...")
    chroma = build_chroma_in_memory(documents, embedding_adapter)

    # Config A: semántica pura
    log.info("[A] Configurando retriever semántico puro...")
    semantic_fn = lambda q: chroma.as_retriever(search_kwargs={"k": TOP_K}).invoke(q)

    # Config B: híbrido BM25 + semántico (corpus completo, sin routing)
    # Usa fusión lineal ponderada min-max (BM25_WEIGHT=0.4, SEM_WEIGHT=0.6), no RRF.
    log.info("[B] Configurando retriever híbrido estático (BM25 + ChromaDB, fusión min-max)...")
    hybrid_fn = lambda q: _manual_hybrid(q, documents, chroma, None, TOP_K)

    # Config C: semántica con pre-filtro WHERE pillar IN [...]
    log.info("[C] Configurando retriever semántico con pre-filtro de pilar...")

    # Config D: BM25 dinámico sobre subconjunto + ChromaDB filtrado
    log.info("[D] Configurando retriever híbrido con BM25 dinámico sobre subespacio...")

    # Evaluación
    log.info("Ejecutando batería de 15 consultas (10 Emocionales + 5 Léxicas)...")
    results_a = evaluate_retriever(semantic_fn, ALL_QUERIES)
    results_b = evaluate_retriever(hybrid_fn, ALL_QUERIES)
    results_c = evaluate_config_c(chroma, ALL_QUERIES)
    results_d = evaluate_config_d(chroma, documents, ALL_QUERIES)

    # Tablas individuales
    _print_detail_table(f"A - Sem. pura, sin routing  [{model_name}]", results_a)
    _print_detail_table(f"B - Híbrido BM25+ChromaDB [0.4/0.6], sin routing", results_b)
    _print_detail_table(f"C - Sem. pura + pre-filtro WHERE pillar IN [...]", results_c)
    _print_detail_table(f"D - Híbrido + BM25 dinámico sobre subconjunto filtrado", results_d)

    # Resumen comparativo
    _print_comparison_summary(results_a, results_b, results_c, results_d, model_name)

    # Preparar métricas agregadas
    def _agg_split(results: list[dict]) -> dict:
        std  = [r for r in results if r["set"] == "standard"]
        bm25 = [r for r in results if r["set"] == "bm25"]
        p1s, p3s, hs = _aggregate(std)
        p1b, p3b, hb = _aggregate(bm25)
        p1t, p3t, ht = _aggregate(results)
        return {
            "precision_at_1_standard":   p1s, "precision_at_3_standard":   p3s, "hit_at_any_standard":   hs,
            "precision_at_1_bm25":       p1b, "precision_at_3_bm25":       p3b, "hit_at_any_bm25":       hb,
            "precision_at_1_total":      p1t, "precision_at_3_total":       p3t, "hit_at_any_total":      ht,
        }

    agg_a = _agg_split(results_a)
    agg_b = _agg_split(results_b)
    agg_c = _agg_split(results_c)
    agg_d = _agg_split(results_d)

    def _delta(k: str, d1: dict, d2: dict) -> float:
        return round(d2[k] - d1[k], 4)

    # Guardar JSON
    output: dict = {
        "experimento": "hybrid_search_comparison",
        "diseno": "factorial_2x2_bm25_x_routing",
        "embedding_model": model_name,
        "top_k": TOP_K,
        "n_queries_standard": len(TEST_CASES),
        "n_queries_bm25": len(BM25_TEST_CASES),
        "pillar_routing": {k: v for k, v in PILLAR_ROUTING.items()},
        "bm25_queries": [
            {
                "label": q["label"], "query": q["query"],
                "expected_pillars": sorted(q["expected_pillars"]), "note": q["note"],
            }
            for q in BM25_TEST_CASES
        ],
        "configuracion_a_semantica": {
            "descripcion": "ChromaDB semántico puro, sin BM25, sin routing",
            **agg_a,
            "resultados_por_consulta": results_a,
        },
        "configuracion_b_hibrida": {
            "descripcion": "Fusión lineal min-max BM25+ChromaDB [0.4/0.6] sobre corpus completo, sin routing",
            "bm25_weight": BM25_WEIGHT, "semantic_weight": SEM_WEIGHT,
            **agg_b,
            "resultados_por_consulta": results_b,
        },
        "configuracion_c_semantica_routing": {
            "descripcion": "ChromaDB semántico con pre-filtro WHERE pillar IN [...] según emoción",
            "routing_table": {k: v for k, v in PILLAR_ROUTING.items()},
            **agg_c,
            "resultados_por_consulta": results_c,
        },
        "configuracion_d_hibrida_routing": {
            "descripcion": "BM25 dinámico sobre subconjunto filtrado + ChromaDB filtrado; EnsembleRetriever [0.4/0.6]",
            "bm25_weight": 0.4, "semantic_weight": 0.6,
            "routing_table": {k: v for k, v in PILLAR_ROUTING.items()},
            **agg_d,
            "resultados_por_consulta": results_d,
        },
        "delta_bm25_sin_routing": {
            "nota": "Efecto BM25 aislado (B-A)",
            "precision_at_3_standard": _delta("precision_at_3_standard", agg_a, agg_b),
            "precision_at_3_bm25_queries": _delta("precision_at_3_bm25",     agg_a, agg_b),
            "precision_at_3_total": _delta("precision_at_3_total",     agg_a, agg_b),
        },
        "delta_routing_sin_bm25": {
            "nota": "Efecto routing aislado (C-A)",
            "precision_at_3_standard": _delta("precision_at_3_standard", agg_a, agg_c),
            "precision_at_3_bm25_queries": _delta("precision_at_3_bm25",     agg_a, agg_c),
            "precision_at_3_total": _delta("precision_at_3_total",     agg_a, agg_c),
        },
        "delta_bm25_con_routing": {
            "nota": "Efecto BM25 sobre retriever con routing (D-C)",
            "precision_at_3_standard": _delta("precision_at_3_standard", agg_c, agg_d),
            "precision_at_3_bm25_queries": _delta("precision_at_3_bm25",     agg_c, agg_d),
            "precision_at_3_total": _delta("precision_at_3_total",     agg_c, agg_d),
        },
        "delta_routing_con_bm25": {
            "nota": "Efecto routing sobre retriever híbrido (D-B)",
            "precision_at_3_standard": _delta("precision_at_3_standard", agg_b, agg_d),
            "precision_at_3_bm25_queries": _delta("precision_at_3_bm25",     agg_b, agg_d),
            "precision_at_3_total": _delta("precision_at_3_total",     agg_b, agg_d),
        },
        "delta_total": {
            "nota": "Efecto combinado BM25+routing (D-A)",
            "precision_at_3_standard": _delta("precision_at_3_standard", agg_a, agg_d),
            "precision_at_3_bm25_queries": _delta("precision_at_3_bm25",     agg_a, agg_d),
            "precision_at_3_total": _delta("precision_at_3_total",     agg_a, agg_d),
        },
    }

    save_results("hybrid_search_results.json", output)


if __name__ == "__main__":
    main()
