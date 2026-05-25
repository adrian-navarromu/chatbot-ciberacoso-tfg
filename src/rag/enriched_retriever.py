"""
Recuperador híbrido V2: BM25 + ChromaDB con routing determinista por pilares.

Configuración ganadora (Experimentos RAG, Mayo 2025):
    - Embeddings: paraphrase-spanish-distilroberta - p@3=0.600, p@1=0.800
    - BM25 pesos: [0.4, 0.6] (léxico + semántico)
    - Routing: determinista por emoción BETO + tendencia GRU
    El routing mejora p@1 de 0.800 a 0.900 (distilroberta + routing)
    al prevenir que chunks de crisis (pilar 4) aparezcan en primera
    posición ante tristeza no crítica (Q3). El único miss del routing
    (Q9 - sadness sin pilar 4) está cubierto por el failsafe PAP.

Implementa la Config D del diseño factorial 2x2 (Exp. 4): para cada consulta
instancia BM25 sobre el subconjunto de documentos del pilar emocional y aplica
el mismo filtro como WHERE pre-búsqueda en ChromaDB. Ambos canales operan en
el mismo subespacio filtrado antes de la fusión RRF - sin post-filtro.

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
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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
    """Recuperador híbrido Config D: BM25 dinámico + ChromaDB pre-filtrado por pilar.

    Arquitectura de recuperación (construida por consulta, Exp. 4 Config D):
        BM25 (0.4) - instanciado en cada retrieve() sobre el subconjunto de
            documentos cuyo pilar pertenece al filtro de routing. Garantiza
            que BM25 no accede a chunks fuera del pilar emocional relevante.
            Ventaja: recuperación exacta de términos técnicos (024, ANAR,
            Instagram, AEPD) dentro del subespacio correcto.
        Semántico (0.6) - ChromaDB con WHERE pillar IN [lista_routing].
            El pre-filtro opera en la misma partición que BM25.
            Ventaja: comprensión semántica del estado emocional del usuario.
        Fusión: Reciprocal Rank Fusion (EnsembleRetriever, pesos [0.4, 0.6]).

    Para 'others' (filtro None), ambos canales operan sobre el corpus completo
    (equivalente a Config B: híbrido sin routing).

    P4 (crisis) se incluye solo en sadness + deterioro GRU: cuando el historial
    muestra un empeoramiento progresivo sin activar el failsafe PAP, los recursos
    de crisis son clínicamente relevantes. En sadness estable o mejora, P4 queda
    excluido - el failsafe PAP (src/safety/crisis_detector.py) intercepta los
    mensajes críticos aislados antes de llegar al retriever.

    Args:
        chroma_path: Ruta al directorio ChromaDB persistente
                     (data/vectorstore/chroma_v2/).
        documents: Lista de Document LangChain. Deben incluir metadata["pillar"]
                   y metadata["chunk_id"]. Se almacenan para construir el BM25
                   filtrado por consulta.
        embedding_model_name: Modelo SentenceTransformer para los embeddings.
                               Debe coincidir con el modelo de la ingesta. Ganador Exp. 2:
                               hackathon-pln-es/paraphrase-spanish-distilroberta.

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

        # Corpus almacenado para construir BM25 filtrado en cada retrieve()
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
    def build_pillar_filter(
        self, emotion: str, trend: str
    ) -> list[int] | None:
        """Routing determinista por pilar basado en emoción + tendencia GRU.

        Canaliza la búsqueda hacia el recurso clínico más relevante según
        la emoción detectada por el calsificador emocional y la tendencia temporal del GRU:
          - Miedo --> regulación afectiva (P2) + crisis (P4) + protocolo (P1)
          - Tristeza con mejora --> esperanza (P3) + TCC (P2)
          - Tristeza deterioro --> TCC (P2) + crisis (P4) [deterioro GRU activa P4]
          - Tristeza estable --> TCC (P2) + contranarrativa (P3)
          - Enfado --> protocolo (P1) + TCC (P2)
          - Asco --> TCC (P2) + contranarrativa (P3)
          - Alegría --> esperanza (P3)
          - Sorpresa --> TCC (P2) + protocolo (P1)
          - Otros --> sin filtro (búsqueda global)

        Args:
            emotion: Etiqueta de emoción del clasificador emocional
                     (fear, sadness, anger, disgust, joy, surprise, others).
            trend: Tendencia temporal del GRU
                   (mejora, deterioro, estable).

        Returns:
            Lista ordenada de pilares a incluir, o None para búsqueda global.
        """
        if emotion == "fear":
            return [2, 4, 1]
        if emotion == "sadness":
            if trend == "mejora":
                return [3, 2]
            if trend == "deterioro":
                return [2, 4]   # deterioro progresivo: incluir P4 aunque PAP no haya activado
            return [2, 3]       # estable: P4 ausente — PAP intercepta crisis aisladas
        if emotion == "anger":
            return [1, 2]
        if emotion == "disgust":
            return [2, 3]
        if emotion == "joy":
            return [3]
        if emotion == "surprise":
            return [2, 1]
        return None          # others --> sin filtro

    # Recuperación
    def retrieve(self, query: str, emotion: str = "", trend: str = "", pillar_filter: list[int] | None = None) -> list[Document]:
        """Búsqueda híbrida Config D: BM25 dinámico + ChromaDB pre-filtrado por pilar.

        Para cada consulta construye un EnsembleRetriever sobre el subconjunto filtrado:
          - pillar_filter None --> BM25 sobre corpus completo + ChromaDB global (Config B).
          - pillar_filter lista --> BM25 sobre documentos del pilar + ChromaDB con
            WHERE pillar IN [lista] (Config D). No se aplica post-filtro.

        Args:
            query: Texto del mensaje del usuario.
            emotion: No usado directamente (el filtro se computa antes en retrieve_with_routing).
            trend: No usado directamente.
            pillar_filter: Lista de pilares a incluir, o None para búsqueda global.

        Returns:
            Lista de hasta top_k=3 Document, ordenados por relevancia RRF.
        """
        if pillar_filter is None:
            # Sin filtro: ensemble sobre corpus completo (Config B)
            bm25 = BM25Retriever.from_documents(self._documents, k=TOP_K)
            chroma_r = self._chroma.as_retriever(search_kwargs={"k": TOP_K})
        else:
            # Config D: filtrar corpus antes de instanciar BM25
            filtered_docs = [
                d for d in self._documents
                if d.metadata.get("pillar") in pillar_filter
            ]
            if not filtered_docs:
                log.warning("Filtro vacío. Activando fallback a corpus global para BM25.")
                filtered_docs = self._documents

            bm25 = BM25Retriever.from_documents(filtered_docs, k=TOP_K)
            chroma_r = self._chroma.as_retriever(
                search_kwargs={
                    "k": TOP_K,
                    "filter": {"pillar": {"$in": pillar_filter}},
                }
            )

        # Fusión matemática de rankings
        ensemble = EnsembleRetriever(
            retrievers=[bm25, chroma_r],
            weights=[0.4, 0.6],
        )
        return ensemble.invoke(query)[:TOP_K]

    def retrieve_with_routing(self, query: str, emotion: str, trend: str) -> list[Document]:
        """Calcula el filtro de pilar y delega en retrieve().

        Shortcut para el pipeline V2: recibe emoción y tendencia directamente
        del clasificador emocional + GRU, computa el filtro de pilar (Config D) y
        delega en retrieve().

        Args:
            query: Texto del mensaje del usuario.
            emotion: Etiqueta clasificador emocional (fear, sadness, anger, disgust, joy,
                     surprise, others).
            trend: Tendencia GRU (mejora, deterioro, estable).

        Returns:
            Lista de hasta 3 Document relevantes con routing por pilar.
        """
        pillar_filter = self.build_pillar_filter(emotion, trend)
        log.info(f"Query recibida. Emotion='{emotion}', Trend='{trend}' -> Filtro={pillar_filter}")
        return self.retrieve(query, emotion, trend, pillar_filter)
