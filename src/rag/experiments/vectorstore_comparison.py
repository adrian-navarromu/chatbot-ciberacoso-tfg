"""
Experimento 3: Comparación de bases vectoriales FAISS vs ChromaDB.

Justificación teórica:
    FAISS: índice vectorial puro, sin metadatos filtrables, construcción
     in-memory desde el corpus actual, ideal para millones de vectores con
     alta velocidad. ChromaDB: base vectorial con metadatos filtrables en
     SQLite, persistencia nativa, sin servidor, adecuada para corpus pequeños
     que requieren routing semántico por categoría. Milvus y Elasticsearch
     se descartan por overhead de configuración injustificado para 40 chunks.

Compara tres modelos de embeddings en ChromaDB (paraphrase-mpnet,
paraphrase-spanish-distilroberta, BAAI/bge-m3) con dos modos por modelo:
sin filtro de pilar y con routing determinista por emoción. FAISS in-memory
con paraphrase-mpnet actúa como baseline. El diseño aísla el efecto del
vectorstore y del modelo de embeddings sobre las mismas 10 consultas estándar.

Todos los índices se construyen en memoria desde data/rag_corpus/*.md en
tiempo de ejecución, garantizando coherencia con el corpus actual.

Uso:
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
from langchain_core.documents import Document
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

from src.rag.experiments.utils import CORPUS_DIR, TEST_CASES, build_faiss_index_from_embeddings, compute_metrics_for_run, compute_precision_at_1, compute_precision_at_k, compute_hit_at_any, load_chunks_from_markdown, print_results_table, save_results


# Constantes
EMBEDDING_MODEL_V1 = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL_DISTILROBERTA = "hackathon-pln-es/paraphrase-spanish-distilroberta"
EMBEDDING_MODEL_BGE = "BAAI/bge-m3"
TOP_K = 3

# Routing Semántico Determinista: Mapea la predicción del modelo de emociones directamente a subespacios 
# vectoriales (Pilares). Esto actúa como un Pre-filtro algorítmico, reduciendo 
# el espacio de búsqueda vectorial en ChromaDB y evitando colisiones semánticas.
# El módulo GRU no está añdido como filtro en este experimento
PILLAR_ROUTING: dict[str, list[int] | None] = {
    "fear": [2, 4, 1],
    "sadness": [2, 3],      # oracle mode: sin tendencia, usa rama estable sin P4
    "anger": [1, 2],
    "disgust": [2, 5, 1],   # P5 digital + P1 protocolo: exposición pública y sextorsión
    "joy": [3],
    "surprise": [2, 1],
    "others": None,
}


# Evaluación FAISS in-memory (baseline paraphrase-mpnet)
def evaluate_faiss_inmemory(docs: list[Document], model: SentenceTransformer) -> list[dict]:
    """Construye un índice FAISS en memoria desde el corpus actual y evalúa las 10 queries.

    El índice se construye en tiempo de ejecución desde data/rag_corpus/*.md,
    garantizando coherencia con cualquier modificación posterior de los chunks.
    No carga ningún artefacto de disco generado por V1.

    Args:
        docs: Documents cargados desde el corpus actual.
        model: SentenceTransformer para generar embeddings (paraphrase-mpnet).

    Returns:
        Lista de dicts con métricas por consulta.
    """
    log.info(f"[FAISS in-memory] Codificando {len(docs)} chunks (Baseline)...")
    doc_embeddings = model.encode(
        [doc.page_content for doc in docs],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype(np.float32)

    index = build_faiss_index_from_embeddings(doc_embeddings)
    log.debug(f"[FAISS] Índice construido: dim={doc_embeddings.shape[1]}")

    results: list[dict] = []
    for tc in TEST_CASES:
        query_vec = model.encode(
            [tc["query"]], normalize_embeddings=True
        ).astype(np.float32)
        _, indices = index.search(query_vec, TOP_K)

        retrieved_pillars = [
            docs[idx].metadata["pillar"]
            for idx in indices[0]
            if 0 <= idx < len(docs)
        ]
        expected = tc["expected_pillars"]
        results.append(
            {
                "label": tc["label"],
                "query": tc["query"],
                "emotion": tc["emotion"],
                "expected_pillars": sorted(expected),
                "retrieved_pillars": retrieved_pillars,
                "precision_at_1": compute_precision_at_1(retrieved_pillars, expected),
                "precision_at_3": compute_precision_at_k(retrieved_pillars, expected, TOP_K),
                "hit_at_any": compute_hit_at_any(retrieved_pillars, expected),
            }
        )
    return results


# ChromaDB — construcción de colección en memoria
def build_chroma_collection(docs: list[Document], embeddings: np.ndarray, name: str = "rag_exp") -> chromadb.Collection:
    """Crea una colección ChromaDB en memoria con los embeddings precalculados.

    Usa embedding_function=None para que ChromaDB no intente cargar ningún modelo
    propio; los embeddings se proporcionan explícitamente en add() y query().
    Los metadatos de Document ya están serializados compatibles con ChromaDB
    (emotions como JSON string, therapeutic_technique sin None).

    Args:
        docs: Lista de Document de LangChain con metadatos completos.
        embeddings: Array (n_chunks x dim) de vectores normalizados.
        name: Nombre único de la colección (ChromaDB comparte estado global).

    Returns:
        Colección ChromaDB en memoria con todos los chunks indexados.
    """
    client = chromadb.Client()
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None, # Forzamos inyección manual para aislar el modelo
    )
    collection.add(
        ids=[doc.metadata["chunk_id"] for doc in docs],
        embeddings=embeddings.tolist(),
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
    )
    return collection


# ChromaDB — evaluación sin filtro de pilar
def evaluate_chroma_no_filter(collection: chromadb.Collection, query_embeddings: np.ndarray) -> list[dict]:
    """Evalúa ChromaDB con búsqueda semántica pura, sin filtro por pilar.

    Args:
        collection: Colección ChromaDB con los chunks indexados.
        query_embeddings: Array (10 x dim) con los embeddings de las 10 queries.

    Returns:
        Lista de dicts con métricas por consulta.
    """
    results: list[dict] = []
    for tc, q_emb in zip(TEST_CASES, query_embeddings):
        try:
            res = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=TOP_K,
                include=["metadatas"],
            )
            retrieved_pillars = [
                m["pillar"] for m in res["metadatas"][0]
            ]
        except Exception as e:
            log.error(f"Error en consulta no-filter para {tc['label']}: {e}")
            retrieved_pillars = []

        expected = tc["expected_pillars"]
        results.append(
            {
                "label": tc["label"],
                "query": tc["query"],
                "emotion": tc["emotion"],
                "expected_pillars": sorted(expected),
                "retrieved_pillars": retrieved_pillars,
                "precision_at_1": compute_precision_at_1(retrieved_pillars, expected),
                "precision_at_3": compute_precision_at_k(retrieved_pillars, expected, TOP_K),
                "hit_at_any": compute_hit_at_any(retrieved_pillars, expected),
            }
        )
    return results


# ChromaDB — evaluación con routing por pilar
def _query_with_pillar_filter(collection: chromadb.Collection, query_embedding: list[float], pillar_filter: list[int], top_k: int) -> list[int]:
    """
    Motor de Búsqueda con Fallback.
    Intenta buscar restringiendo por los pilares dictados por la emoción. 
    Si la topología semántica del subespacio está vacía (retorna < 2 chunks), 
    relaja las restricciones y rellena el contexto buscando en todo el corpus global.

    Args:
        collection: Colección ChromaDB.
        query_embedding: Vector de la query (dim,).
        pillar_filter: Lista de pilares a incluir en el filtro.
        top_k: Número máximo de resultados a devolver.

    Returns:
        Lista de pilares de los chunks recuperados (hasta top_k).
    """
    where = {"pillar": {"$in": pillar_filter}}

    # Intento primario con routing estricto
    try:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas"],
        )
        filtered_pillars = [m["pillar"] for m in res["metadatas"][0]]
        filtered_ids = set(res["ids"][0])
    except Exception:
        filtered_pillars = []
        filtered_ids = set()

    # Fallback: completar hasta top_k con resultados sin filtro
    if len(filtered_pillars) < 2:
        try:
            res_all = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas"],
            )
            for doc_id, meta in zip(res_all["ids"][0], res_all["metadatas"][0]):
                if doc_id not in filtered_ids:
                    filtered_pillars.append(meta["pillar"])
                    filtered_ids.add(doc_id)
                if len(filtered_pillars) >= top_k:
                    break
        except Exception:
            pass

    return filtered_pillars[:top_k]


def evaluate_chroma_routing(collection: chromadb.Collection, query_embeddings: np.ndarray) -> list[dict]:
    """Evalúa ChromaDB con routing determinista por pilar según la emoción de la consulta.

    Aplica PILLAR_ROUTING para filtrar por pilar antes de la búsqueda vectorial.
    Para emotion='others' (sin filtro), el comportamiento es idéntico al modo
    sin filtro. El routing simula la señal del clasificador emocional en producción.

    Args:
        collection: Colección ChromaDB con los chunks indexados.
        query_embeddings: Array (10 x dim) con los embeddings de las 10 queries.

    Returns:
        Lista de dicts con métricas por consulta (incluyendo el pilar filtrado aplicado).
    """
    results: list[dict] = []
    for tc, q_emb in zip(TEST_CASES, query_embeddings):
        pillar_filter = PILLAR_ROUTING.get(tc["emotion"])

        if pillar_filter is None:
            # Sin filtro para 'others'
            try:
                res = collection.query(
                    query_embeddings=[q_emb.tolist()],
                    n_results=TOP_K,
                    include=["metadatas"],
                )
                retrieved_pillars = [m["pillar"] for m in res["metadatas"][0]]
            except Exception:
                retrieved_pillars = []
        else:
            # Aplicación del Pre-filtro condicionado
            retrieved_pillars = _query_with_pillar_filter(
                collection, q_emb.tolist(), pillar_filter, TOP_K
            )

        expected = tc["expected_pillars"]
        results.append(
            {
                "label": tc["label"],
                "query": tc["query"],
                "emotion": tc["emotion"],
                "pillar_filter_aplicado": pillar_filter,
                "expected_pillars": sorted(expected),
                "retrieved_pillars": retrieved_pillars,
                "precision_at_1": compute_precision_at_1(retrieved_pillars, expected),
                "precision_at_3": compute_precision_at_k(retrieved_pillars, expected, TOP_K),
                "hit_at_any": compute_hit_at_any(retrieved_pillars, expected),
            }
        )
    return results


# Presentación de resultados
def _aggregate(results: list[dict]) -> tuple[float, float, float]:
    """Devuelve (p@3, p@1, hit@any) usando compute_metrics_for_run."""
    m = compute_metrics_for_run(results)
    return m["precision_at_3_media"], m.get("precision_at_1_media", 0.0), m["hit_at_any_rate"]


def _print_comparison_table(configs: list[tuple[str, list[dict]]], routing_deltas: list[float | None] | None = None) -> None:
    """Imprime la tabla comparativa para N configuraciones.

    Args:
        configs: Lista de tuplas (nombre_config, lista_de_resultados).
        routing_deltas: delta routing = P@3_routing - P@3_sin_filtro por config.
                        None en posiciones sin routing (FAISS, sin filtro).
    """
    sep = "=" * 88
    print(f"\n{sep}")
    print("  TABLA COMPARATIVA - BASES VECTORIALES")
    print(sep)
    print(f"  {'Configuración':<44} {'p@1':>6} {'p@3':>8} {'hit@any':>8} {'delta_routing':>10}")
    print(f"  {'-'*43} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for i, (name, results) in enumerate(configs):
        prec3, prec1, hit = _aggregate(results)
        if routing_deltas is not None and i < len(routing_deltas) and routing_deltas[i] is not None:
            delta_str = f"{routing_deltas[i]:+.4f}"
        else:
            delta_str = "-"
        print(f"  {name:<44} {prec1:>6.4f} {prec3:>8.4f} {hit:>8.4f} {delta_str:>10}")
    print(f"{sep}\n")


# Main
def main() -> None:
    """Ejecuta el experimento comparativo de bases vectoriales y guarda resultados."""
    log.info("Iniciando Experimento 3: ChromaDB y Routing Determinista")

    # Cargar corpus completo
    chunks = load_chunks_from_markdown(CORPUS_DIR)
    log.info(f"Cargados {len(chunks)} chunks terapéuticos.")

    # FAISS in-memory baseline (paraphrase-mpnet, corpus actual)
    model_mpnet = SentenceTransformer(EMBEDDING_MODEL_V1)
    faiss_results = evaluate_faiss_inmemory(chunks, model_mpnet)

    # Helper para inyección manual de tensores
    def _encode(model: SentenceTransformer, label: str) -> tuple[np.ndarray, np.ndarray]:
        log.info(f"Proyectando espacio vectorial con {label}...")
        docs = model.encode(
            [c.page_content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)
        qs = model.encode(
            [tc["query"] for tc in TEST_CASES],
            normalize_embeddings=True,
        ).astype(np.float32)
        return docs, qs

    def _chroma_eval(docs: np.ndarray, qs: np.ndarray, col_name: str, label: str) -> tuple[list[dict], list[dict]]:
        col = build_chroma_collection(chunks, docs, name=col_name)
        log.debug(f"Colección Chroma '{col_name}' indexada.")
        nf = evaluate_chroma_no_filter(col, qs)
        rt = evaluate_chroma_routing(col, qs)
        return nf, rt

    # ChromaDB: paraphrase-mpnet
    docs_mpnet, qs_mpnet = _encode(model_mpnet, "paraphrase-mpnet")
    mpnet_nf, mpnet_rt = _chroma_eval(docs_mpnet, qs_mpnet, "rag_mpnet", "paraphrase-mpnet")

    # ChromaDB: paraphrase-spanish-distilroberta
    model_dr = SentenceTransformer(EMBEDDING_MODEL_DISTILROBERTA)
    docs_dr, qs_dr = _encode(model_dr, "paraphrase-spanish-distilroberta")
    dr_nf, dr_rt = _chroma_eval(docs_dr, qs_dr, "rag_distilroberta", "distilroberta")

    # ChromaDB: BAAI/bge-m3
    model_bge = SentenceTransformer(EMBEDDING_MODEL_BGE)
    docs_bge, qs_bge = _encode(model_bge, "bge-m3")
    bge_nf, bge_rt = _chroma_eval(docs_bge, qs_bge, "rag_bge", "bge-m3")

    # Tablas detalladas
    print_results_table("FAISS in-memory  (paraphrase-mpnet, corpus actual)", faiss_results, show_emotion=True)
    print_results_table("ChromaDB sin filtro  (paraphrase-mpnet)", mpnet_nf, show_emotion=True)
    print_results_table("ChromaDB sin filtro  (paraphrase-spanish-distilroberta)", dr_nf, show_emotion=True)
    print_results_table("ChromaDB sin filtro  (bge-m3)", bge_nf, show_emotion=True)
    print_results_table("ChromaDB + routing  (paraphrase-mpnet)", mpnet_rt, show_emotion=True)
    print_results_table("ChromaDB + routing  (paraphrase-spanish-distilroberta)", dr_rt, show_emotion=True)
    print_results_table("ChromaDB + routing  (bge-m3)", bge_rt, show_emotion=True)

    # Guardar JSON
    prec_f, prec1_f, hit_f = _aggregate(faiss_results)
    prec_mnf, prec1_mnf, hit_mnf = _aggregate(mpnet_nf)
    prec_dnf, prec1_dnf, hit_dnf = _aggregate(dr_nf)
    prec_bnf, prec1_bnf, hit_bnf = _aggregate(bge_nf)
    prec_mrt, prec1_mrt, hit_mrt = _aggregate(mpnet_rt)
    prec_drt, prec1_drt, hit_drt = _aggregate(dr_rt)
    prec_brt, prec1_brt, hit_brt = _aggregate(bge_rt)

    rdelta_mpnet = round(prec_mrt - prec_mnf, 4)
    rdelta_dr = round(prec_drt - prec_dnf, 4)
    rdelta_bge = round(prec_brt - prec_bnf, 4)

    # Tabla comparativa
    _print_comparison_table(
        [
            ("FAISS + paraphrase-mpnet (in-memory)", faiss_results),
            ("ChromaDB + paraphrase-mpnet", mpnet_nf),
            ("ChromaDB + paraphrase-spanish-distilroberta", dr_nf),
            ("ChromaDB + bge-m3", bge_nf),
            ("ChromaDB + paraphrase-mpnet + routing", mpnet_rt),
            ("ChromaDB + paraphrase-spanish-distilroberta + routing", dr_rt),
            ("ChromaDB + bge-m3 + routing", bge_rt),
        ],
        routing_deltas=[None, None, None, None, rdelta_mpnet, rdelta_dr, rdelta_bge],
    )

    # Serialización Estructurada JSON
    routing_table = {k: v for k, v in PILLAR_ROUTING.items()}

    def _entry(desc: str, model_id: str, prec3: float, prec1: float, hit: float, results: list[dict], routing: bool = False, routing_delta: float | None = None) -> dict:
        d: dict = {
            "descripcion": desc,
            "embedding_model": model_id,
            "precision_at_1_media": prec1,
            "precision_at_3_media": prec3,
            "hit_at_any_rate": hit,
            "resultados_por_consulta": results,
        }
        if routing:
            d["routing_table"] = routing_table
        if routing_delta is not None:
            d["routing_delta"] = routing_delta
        return d

    output: dict = {
        "experimento": "vectorstore_comparison",
        "top_k": TOP_K,
        "modelos_chroma": [EMBEDDING_MODEL_V1, EMBEDDING_MODEL_DISTILROBERTA, EMBEDDING_MODEL_BGE],
        "faiss_mpnet_inmemory": {
            "descripcion": "FAISS IndexFlatIP en memoria, construido desde el corpus actual, sin metadatos filtrables",
            "embedding_model": EMBEDDING_MODEL_V1,
            "precision_at_1_media": prec1_f,
            "precision_at_3_media": prec_f,
            "hit_at_any_rate": hit_f,
            "resultados_por_consulta": faiss_results,
        },
        "chromadb_mpnet_sin_filtro": _entry(
            "ChromaDB en memoria, búsqueda semántica pura, paraphrase-mpnet",
            EMBEDDING_MODEL_V1, prec_mnf, prec1_mnf, hit_mnf, mpnet_nf),
        "chromadb_distilroberta_sin_filtro": _entry(
            "ChromaDB en memoria, búsqueda semántica pura, paraphrase-spanish-distilroberta",
            EMBEDDING_MODEL_DISTILROBERTA, prec_dnf, prec1_dnf, hit_dnf, dr_nf),
        "chromadb_bge_sin_filtro": _entry(
            "ChromaDB en memoria, búsqueda semántica pura, BAAI/bge-m3",
            EMBEDDING_MODEL_BGE, prec_bnf, prec1_bnf, hit_bnf, bge_nf),
        "chromadb_mpnet_con_routing": _entry(
            "ChromaDB con filtro WHERE pillar IN [...], paraphrase-mpnet",
            EMBEDDING_MODEL_V1, prec_mrt, prec1_mrt, hit_mrt, mpnet_rt,
            routing=True, routing_delta=rdelta_mpnet),
        "chromadb_distilroberta_con_routing": _entry(
            "ChromaDB con filtro WHERE pillar IN [...], paraphrase-spanish-distilroberta",
            EMBEDDING_MODEL_DISTILROBERTA, prec_drt, prec1_drt, hit_drt, dr_rt,
            routing=True, routing_delta=rdelta_dr),
        "chromadb_bge_con_routing": _entry(
            "ChromaDB con filtro WHERE pillar IN [...], BAAI/bge-m3",
            EMBEDDING_MODEL_BGE, prec_brt, prec1_brt, hit_brt, bge_rt,
            routing=True, routing_delta=rdelta_bge),
        "tabla_comparativa": [
            {"config": "FAISS + paraphrase-mpnet (in-memory)", "precision_at_1": prec1_f,   "precision_at_3": prec_f,   "hit_at_any": hit_f,   "routing_delta": None},
            {"config": "ChromaDB + paraphrase-mpnet", "precision_at_1": prec1_mnf, "precision_at_3": prec_mnf, "hit_at_any": hit_mnf, "routing_delta": None},
            {"config": "ChromaDB + paraphrase-spanish-distilroberta", "precision_at_1": prec1_dnf, "precision_at_3": prec_dnf, "hit_at_any": hit_dnf, "routing_delta": None},
            {"config": "ChromaDB + bge-m3", "precision_at_1": prec1_bnf, "precision_at_3": prec_bnf, "hit_at_any": hit_bnf, "routing_delta": None},
            {"config": "ChromaDB + paraphrase-mpnet + routing", "precision_at_1": prec1_mrt, "precision_at_3": prec_mrt, "hit_at_any": hit_mrt, "routing_delta": rdelta_mpnet},
            {"config": "ChromaDB + paraphrase-spanish-distilroberta + routing", "precision_at_1": prec1_drt, "precision_at_3": prec_drt, "hit_at_any": hit_drt, "routing_delta": rdelta_dr},
            {"config": "ChromaDB + bge-m3 + routing", "precision_at_1": prec1_brt, "precision_at_3": prec_brt, "hit_at_any": hit_brt, "routing_delta": rdelta_bge},
        ],
    }

    save_results("vectorstore_results.json", output)


if __name__ == "__main__":
    main()
