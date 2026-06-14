"""
Recuperador híbrido V2: BM25 + ChromaDB con routing determinista por pilares.

Configuración ganadora (Experimentos RAG, Mayo 2025):
    - Embeddings: paraphrase-spanish-distilroberta - p@3=0.600, p@1=0.800
    - BM25 pesos: BM25_WEIGHT=0.4, SEM_WEIGHT=0.6
    - Routing: determinista por emoción + tendencia EmotionalMemoryTracker
    El routing mejora p@1 de 0.800 a 0.900 (distilroberta + routing)
    al prevenir que chunks de crisis (pilar 4) aparezcan en primera
    posición ante tristeza no crítica.

Implementa la Config D del diseño factorial 2x2 (Exp. 4): para cada consulta
instancia BM25 sobre el subconjunto de documentos del pilar emocional y aplica
el mismo filtro como WHERE pre-búsqueda en ChromaDB. Ambos canales operan en
el mismo subespacio filtrado antes de la FUSIÓN LINEAL PONDERADA con
normalización min-max — NO se usa RRF.

Fusión (REGLA TRANSVERSAL, Tarea 3):
    score_final = BM25_WEIGHT * bm25_norm + SEM_WEIGHT * sem_norm
    donde cada canal se normaliza min-max a [0,1] por separado.
    Si max == min en un canal, todos los scores de ese canal = 1.0
    (caso de corpus muy pequeño o query sin términos en común).

Penalización P4 blanda (Tarea 1b):
    En sadness + estable, P4 es elegible pero su score_final se
    multiplica por P4_SOFT_PENALTY=0.7 antes del ranking final.
    El CrisisDetector sigue siendo la barrera primaria de crisis aguda (HIGH):
    este mecanismo es solo una salvaguarda complementaria de cobertura
    para sesiones con deterioro leve sin keywords críticas explícitas.

Para 'others' (filtro None), el ensemble opera sobre el corpus completo,
comportamiento equivalente a Config B (híbrido sin routing).

Uso:
    from src.rag.enriched_retriever import EnrichedRetriever
    from langchain_core.documents import Document

    docs = [Document(page_content=c.content, metadata={...}) for c in chunks]
    retriever = EnrichedRetriever("data/vectorstore/chroma_v2", docs)
    results = retriever.retrieve_with_routing(query, emotion="sadness", trend="deterioro")
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Constantes
COLLECTION_NAME = "rag_ciberacoso_v2"
TOP_K = 3
BM25_WEIGHT: float = 0.4
SEM_WEIGHT: float = 0.6
# Factor de penalización P4 en sadness+estable: reduce score de chunks crisis
# para que solo emerjan si su relevancia es excepcionalmente alta.
# El CrisisDetector sigue siendo la barrera primaria para crisis aguda (HIGH).
P4_SOFT_PENALTY: float = 0.7


# Adaptador de embeddings
class _STEmbeddingsAdapter(Embeddings):
    """Adaptador LangChain para SentenceTransformer con normalización coseno.

    Permite usar cualquier modelo SentenceTransformer como embedding_function
    en LangChain Chroma, garantizando vectores normalizados equivalentes a
    similitud coseno en IndexFlatIP.

    Args:
        model_name: Identificador HuggingFace del modelo SentenceTransformer.
    """

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Codifica una lista de documentos con normalización L2."""
        return (self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist())

    def embed_query(self, text: str) -> list[float]:
        """Codifica una sola query con normalización L2."""
        return (self._model.encode([text], normalize_embeddings=True)[0].tolist())


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Normalización min-max a [0,1]. Si max==min, devuelve lista de 1.0."""
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# EnrichedRetriever
class EnrichedRetriever:
    """Recuperador híbrido Config D: BM25 dinámico + ChromaDB pre-filtrado por pilar.

    Arquitectura de recuperación (construida por consulta, Exp. 4 Config D):
        BM25 (0.4) - scores brutos con BM25Okapi sobre el subconjunto de docs
            del pilar emocional. Garantiza recuperación exacta de términos
            técnicos (024, ANAR, Instagram, AEPD) dentro del subespacio correcto.
        Semántico (0.6) - ChromaDB con WHERE pillar IN [lista_routing].
            Comprensión semántica del estado emocional del usuario.
        Fusión: combinación lineal ponderada de scores normalizados min-max.
            score_final = 0.4 * bm25_norm + 0.6 * sem_norm (no RRF).

    Para 'others' (filtro None), ambos canales operan sobre el corpus completo
    (equivalente a Config B: híbrido sin routing).

    Routing de pilares:
        fear:            [2, 4, 1]  regulación + crisis + protocolo
        sadness + mejora:[3, 2]     esperanza primero, luego TCC
        sadness + estable:[2, 3, 4] TCC + contranarrativa + crisis (P4 penalizado × 0.7)
        sadness + deteri.:[2, 4]    TCC + crisis (P4 SIN penalización: degradación activa)
        anger:           [1, 2]     protocolo + TCC
        disgust:         [2, 5, 1]  TCC + digital + protocolo (exposición pública y sextorsión)
        joy:             [3]        esperanza
        surprise:        [2, 1]     TCC + protocolo
        others:          None       búsqueda global

    Args:
        chroma_path: Ruta al directorio ChromaDB persistente
                     (data/vectorstore/chroma_v2/).
        documents: Lista de Document LangChain. Deben incluir metadata["pillar"]
                   y metadata["chunk_id"]. Se almacenan para BM25 filtrado por consulta.
        embedding_model_name: Modelo SentenceTransformer para los embeddings.
                               Ganador Exp. 2: hackathon-pln-es/paraphrase-spanish-distilroberta.

    Raises:
        FileNotFoundError: Si chroma_path no existe en disco.
    """

    def __init__(self, chroma_path: str, documents: list[Document], embedding_model_name: str = "hackathon-pln-es/paraphrase-spanish-distilroberta") -> None:
        chroma_dir = Path(chroma_path)
        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"Colección ChromaDB no encontrada en '{chroma_path}'. "
                "Ejecuta src/rag/document_ingestion_v2.py primero."
            )

        # Corpus almacenado para BM25 filtrado en cada retrieve()
        self._documents: list[Document] = list(documents)

        log.info(f"Inicializando EnrichedRetriever con {len(self._documents)} documentos.")
        embedding_adapter = _STEmbeddingsAdapter(embedding_model_name)

        # ChromaDB persistente: fuente semántica con pre-filtro WHERE en retrieve()
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self._chroma: Chroma = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_adapter,
        )

    # Routing
    def build_pillar_filter(self, emotion: str, trend: str) -> list[int] | None:
        """Routing determinista por pilar basado en emoción + tendencia EmotionalMemoryTracker.

        Devuelve la lista de pilares elegibles. La penalización de P4 en
        sadness+estable se aplica en retrieve() sobre los scores, no aquí.

        Args:
            emotion: Etiqueta de emoción del clasificador
                     (fear, sadness, anger, disgust, joy, surprise, others).
            trend: Tendencia del EmotionalMemoryTracker (mejora, deterioro, estable).

        Returns:
            Lista de pilares elegibles, o None para búsqueda global.
        """
        if emotion == "fear":
            return [2, 4, 1]
        if emotion == "sadness":
            if trend == "mejora":
                return [3, 2]
            if trend == "deterioro":
                return [2, 4]       # deterioro activo: P4 sin penalización
            return [2, 3, 4]        # estable: P4 elegible pero penalizado × 0.7 en retrieve()
        if emotion == "anger":
            return [1, 2]
        if emotion == "disgust":
            return [2, 5, 1]        # P5 digital + P1 protocolo: exposición pública, sextorsión
        if emotion == "joy":
            return [3]
        if emotion == "surprise":
            return [2, 1]
        return None                 # others → sin filtro

    def _p4_penalty(self, emotion: str, trend: str) -> float:
        """Devuelve el factor de penalización para chunks del pilar 4.

        Solo aplica 0.7 en sadness+estable. En el resto de configuraciones
        (incluido sadness+deterioro) devuelve 1.0 (sin penalización).
        """
        if emotion == "sadness" and trend == "estable":
            return P4_SOFT_PENALTY
        return 1.0

    # Recuperación
    def retrieve(self, query: str, emotion: str = "", trend: str = "", pillar_filter: list[int] | None = None, p4_penalty: float = 1.0) -> list[Document]:
        """Búsqueda híbrida Config D con fusión lineal ponderada de scores normalizados.

        Para cada consulta:
          1. Filtra el corpus por pillar_filter (o usa corpus completo si None).
          2. Computa scores BM25 (BM25Okapi) sobre el subconjunto.
          3. Recupera scores semánticos de ChromaDB (similarity_search_with_relevance_scores).
          4. Normaliza ambas listas min-max a [0,1] por separado.
          5. Aplica penalización a chunks de pilar 4 si p4_penalty < 1.0.
          6. Combina: score = BM25_WEIGHT * bm25_norm + SEM_WEIGHT * sem_norm.
          7. Ordena por score descendente y devuelve top-K.

        Args:
            query: Texto del mensaje del usuario.
            emotion: No usado directamente aquí (el filtro se pasa como pillar_filter).
            trend: No usado directamente aquí.
            pillar_filter: Lista de pilares elegibles, o None para búsqueda global.
            p4_penalty: Factor a aplicar a scores de chunks del pilar 4 (default 1.0 = sin penalización).

        Returns:
            Lista de hasta TOP_K Documents ordenados por score combinado descendente.
        """
        # Subconjunto de documentos para BM25 y filtro ChromaDB
        if pillar_filter is None:
            bm25_docs = self._documents
            chroma_filter = None
        else:
            bm25_docs = [
                d for d in self._documents
                if d.metadata.get("pillar") in pillar_filter
            ]
            if not bm25_docs:
                log.warning("Filtro vacío. Usando corpus global como fallback.")
                bm25_docs = self._documents
            chroma_filter = {"pillar": {"$in": pillar_filter}}

        n = len(bm25_docs)

        # BM25 scores sobre subconjunto
        tokenized_corpus = [d.page_content.split() for d in bm25_docs]
        bm25_model = BM25Okapi(tokenized_corpus)
        bm25_raw = list(bm25_model.get_scores(query.split()))

        # Scores semánticos de ChromaDB (cosine similarity en [0,1] para vectores normalizados)
        sem_results = self._chroma.similarity_search_with_relevance_scores(
            query,
            k=n,
            filter=chroma_filter,
        )
        sem_map: dict[str, float] = {doc.metadata["chunk_id"]: score for doc, score in sem_results}
        sem_raw = [sem_map.get(d.metadata["chunk_id"], 0.0) for d in bm25_docs]

        # Normalización min-max independiente por canal
        bm25_norm = _minmax_normalize(bm25_raw)
        sem_norm = _minmax_normalize(sem_raw)

        # Combinación ponderada + penalización P4
        scored: list[tuple[Document, float]] = []
        for doc, bm25_n, sem_n in zip(bm25_docs, bm25_norm, sem_norm):
            penalty = p4_penalty if doc.metadata.get("pillar") == 4 else 1.0
            score = (BM25_WEIGHT * bm25_n + SEM_WEIGHT * sem_n) * penalty
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:TOP_K]]

    def retrieve_with_routing(self, query: str, emotion: str, trend: str) -> list[Document]:
        """Calcula el filtro de pilar y la penalización P4, y delega en retrieve().

        Shortcut para el pipeline V2: recibe emoción y tendencia directamente
        del clasificador emocional + EmotionalMemoryTracker, computa el filtro de
        pilar (Config D) y la penalización P4 blanda, y delega en retrieve().

        Args:
            query: Texto del mensaje del usuario.
            emotion: Etiqueta clasificador emocional (fear, sadness, anger, disgust, joy,
                     surprise, others).
            trend: Tendencia EmotionalMemoryTracker (mejora, deterioro, estable).

        Returns:
            Lista de hasta 3 Document relevantes con routing por pilar.
        """
        pillar_filter = self.build_pillar_filter(emotion, trend)
        p4_penalty = self._p4_penalty(emotion, trend)
        log.info(
            f"Routing: emotion='{emotion}', trend='{trend}' "
            f"-> filter={pillar_filter}, p4_penalty={p4_penalty}"
        )
        return self.retrieve(query, emotion, trend, pillar_filter, p4_penalty)
