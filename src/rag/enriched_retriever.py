"""
Recuperador semántico V2 con routing determinista por pilares (Config C).

Configuración seleccionada (Experimentos RAG, Exp. 4 factorial 2×2):
    - Embeddings: paraphrase-spanish-distilroberta - P@3=0.838, P@1=0.903
    - Routing: determinista por emoción + tendencia EmotionalMemoryTracker
    El routing eleva P@1 de ~0.58 (global) a 0.903 (semántico + routing).
    El componente BM25 (Config D) tuvo efecto neutro/negativo sobre Config C,
    por lo que se descarta por parsimonia arquitectónica.

Implementa la Config C del diseño factorial 2×2 (Exp. 4): recuperación
semántica pura en ChromaDB con pre-filtro WHERE por pilar emocional.
No hay fusión ni canal léxico.

Penalización P4 blanda:
    En sadness + estable, P4 es elegible pero su score semántico se
    multiplica por P4_SOFT_PENALTY=0.7 antes del ranking final.
    El CrisisDetector sigue siendo la barrera primaria de crisis aguda (HIGH):
    este mecanismo es solo una salvaguarda complementaria de cobertura
    para sesiones con deterioro leve sin keywords críticas explícitas.

Para 'others' (filtro None), la búsqueda opera sobre el corpus completo.

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
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Constantes
COLLECTION_NAME = "rag_ciberacoso_v2"
TOP_K = 3
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


# EnrichedRetriever
class EnrichedRetriever:
    """Recuperador semántico Config C: ChromaDB pre-filtrado por pilar emocional.

    Arquitectura de recuperación (Exp. 4 Config C — ganadora):
        Semántico puro — ChromaDB con WHERE pillar IN [lista_routing].
        Comprensión semántica + pre-filtro por pilar emocional.
        No hay canal BM25 ni fusión (efecto neutro/negativo en Exp. 4).

    Para 'others' (filtro None), la búsqueda opera sobre el corpus completo.

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
                   y metadata["chunk_id"]. Se usan para el fallback de filtro vacío.
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

        # Corpus almacenado para el fallback de filtro vacío en retrieve()
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
        """Búsqueda semántica Config C con pre-filtro de pilar y penalización P4.

        Para cada consulta:
          1. Construye el filtro WHERE para ChromaDB a partir de pillar_filter.
          2. Recupera scores semánticos de ChromaDB (similarity_search_with_relevance_scores).
             Si el filtro produce resultado vacío, usa fallback al corpus global.
          3. Aplica penalización a chunks de pilar 4 si p4_penalty < 1.0.
          4. Ordena por score descendente y devuelve top-K.

        Args:
            query: Texto del mensaje del usuario.
            emotion: No usado directamente aquí (el filtro se pasa como pillar_filter).
            trend: No usado directamente aquí.
            pillar_filter: Lista de pilares elegibles, o None para búsqueda global.
            p4_penalty: Factor a aplicar a scores de chunks del pilar 4 (default 1.0 = sin penalización).

        Returns:
            Lista de hasta TOP_K Documents ordenados por score semántico descendente.
        """
        if pillar_filter is None:
            chroma_filter = None
        else:
            filtered_docs = [d for d in self._documents if d.metadata.get("pillar") in pillar_filter]
            if not filtered_docs:
                log.warning("Filtro vacío. Usando corpus global como fallback.")
                chroma_filter = None
            else:
                chroma_filter = {"pillar": {"$in": pillar_filter}}

        sem_results = self._chroma.similarity_search_with_relevance_scores(
            query, k=10, filter=chroma_filter,
        )

        scored: list[tuple[Document, float]] = []
        for doc, score in sem_results:
            penalty = p4_penalty if doc.metadata.get("pillar") == 4 else 1.0
            scored.append((doc, score * penalty))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:TOP_K]]

    def retrieve_with_routing(self, query: str, emotion: str, trend: str) -> list[Document]:
        """Calcula el filtro de pilar y la penalización P4, y delega en retrieve().

        Shortcut para el pipeline V2: recibe emoción y tendencia directamente
        del clasificador emocional + EmotionalMemoryTracker, computa el filtro de
        pilar (Config C) y la penalización P4 blanda, y delega en retrieve().

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
