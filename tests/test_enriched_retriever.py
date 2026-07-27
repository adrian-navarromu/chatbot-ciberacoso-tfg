"""
Tests del módulo EnrichedRetriever y verificación del orden de llamadas
en el pipeline V2 (cortocircuito PAP).

Grupos:
  A — Routing (build_pillar_filter): unidad pura, sin dependencias externas.
  B — Recuperación semántica (Config C): pre-filtro por pilar, fallback global
      y penalización P4 blanda, con ChromaDB simulado.
  C — Integración PAP: CrisisDetector intercepta ANTES de consultar el RAG.

Los tests del Grupo B usan los documentos reales del corpus (data/rag_corpus/)
y un ChromaDB simulado cuyo similarity_search_with_relevance_scores emula el
pre-filtro WHERE por pilar. Esto verifica la lógica de recuperación de la
Config C ganadora (semántico puro, sin BM25) de forma determinista y sin
GPU ni modelos pesados.

Ejecución:
    pytest tests/test_enriched_retriever.py -v

Nota: No requiere que los experimentos se hayan ejecutado.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _EmptyRetriever(BaseRetriever):
    """Stub de retriever que devuelve lista vacía. Reemplaza Chroma en tests."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return []


# ---------------------------------------------------------------------------
# Helper: carga del corpus para el índice BM25
# ---------------------------------------------------------------------------


def _load_corpus_documents() -> list[Document]:
    """Parsea el corpus RAG y lo convierte a Documents de LangChain para BM25.

    Replica la lógica de load_manual_chunks de los scripts de experimento sin
    importarlos, manteniendo los tests independientes de los experimentos.

    Returns:
        Lista de Document con page_content=contenido y metadata={'chunk_id', 'pillar'}.
    """
    corpus_dir = Path("data/rag_corpus")
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
                    },
                )
            )
    return documents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_documents() -> list[Document]:
    """Documentos del corpus cargados una sola vez para todos los tests del módulo."""
    docs = _load_corpus_documents()
    assert docs, "No se encontraron documentos en data/rag_corpus/"
    return docs


def _make_fake_semantic_search(corpus: list[Document]):
    """Crea un doble de similarity_search_with_relevance_scores para ChromaDB.

    Emula el pre-filtro WHERE `pillar $in [...]` de ChromaDB sobre el corpus real
    y asigna scores de relevancia sintéticos descendentes (deterministas), sin
    embeddings ni vectorstore. Reproduce la Config C: semántico + pre-filtro.

    Args:
        corpus: Documentos del corpus con metadata['pillar'] y ['chunk_id'].

    Returns:
        Función con la firma (query, k, filter) -> list[tuple[Document, float]].
    """

    def _search(query: str, k: int = 10, filter: dict | None = None):  # noqa: A002
        docs = corpus
        if filter is not None:
            allowed = set(filter["pillar"]["$in"])
            docs = [d for d in docs if d.metadata.get("pillar") in allowed]
        return [(d, 1.0 - i * 0.001) for i, d in enumerate(docs[:k])]

    return _search


@pytest.fixture
def retriever(corpus_documents: list[Document]):
    """EnrichedRetriever (Config C) con ChromaDB simulado sobre el corpus real.

    Evita cargar SentenceTransformer (paraphrase-spanish-distilroberta) y ChromaDB
    persistente. El doble de similarity_search_with_relevance_scores emula el
    pre-filtro WHERE por pilar y devuelve scores deterministas, de modo que se
    ejercita la lógica real de retrieve(): filtrado por pilar, fallback global y
    penalización P4. Los tests que necesitan scores concretos sobreescriben
    `retriever._chroma.similarity_search_with_relevance_scores`.

    Yields:
        EnrichedRetriever listo para usar en los tests.
    """
    from src.rag.enriched_retriever import EnrichedRetriever

    tmp_dir = tempfile.mkdtemp()

    fake_chroma = MagicMock()
    fake_chroma.as_retriever.return_value = _EmptyRetriever()
    fake_chroma.similarity_search_with_relevance_scores.side_effect = (
        _make_fake_semantic_search(corpus_documents)
    )

    with (
        patch("src.rag.enriched_retriever._STEmbeddingsAdapter"),
        patch("src.rag.enriched_retriever.Chroma", return_value=fake_chroma),
        patch("src.rag.enriched_retriever.chromadb.PersistentClient"),
    ):
        er = EnrichedRetriever(tmp_dir, corpus_documents)

    yield er

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Grupo A — Routing determinista (unidad pura)
# ---------------------------------------------------------------------------


def test_pillar_routing_fear(retriever: "EnrichedRetriever") -> None:
    """fear con cualquier trend → [2, 4, 1] (regulación + crisis + protocolo)."""
    assert retriever.build_pillar_filter("fear", "estable") == [2, 4, 1]
    assert retriever.build_pillar_filter("fear", "mejora") == [2, 4, 1]
    assert retriever.build_pillar_filter("fear", "deterioro") == [2, 4, 1]


def test_pillar_routing_hope(retriever: "EnrichedRetriever") -> None:
    """sadness + mejora → [3, 2] (esperanza primero, luego TCC)."""
    assert retriever.build_pillar_filter("sadness", "mejora") == [3, 2]


def test_pillar_routing_sadness_deterioro(retriever: "EnrichedRetriever") -> None:
    """sadness + deterioro → [2, 4]; sadness + estable → [2, 3, 4] (P4 elegible con penalización).

    deterioro activa P4 sin penalización: empeoramiento progresivo que requiere recursos
    de crisis. En estable, P4 es elegible pero penalizado (×0.7) para que solo ascienda
    si su relevancia es muy alta; P2/P3 son prioritarios en ranking normal.
    """
    assert retriever.build_pillar_filter("sadness", "deterioro") == [2, 4]
    assert retriever.build_pillar_filter("sadness", "estable") == [2, 3, 4]


def test_pillar_routing_disgust(retriever: "EnrichedRetriever") -> None:
    """disgust → [2, 5, 1]: TCC + digital + protocolo (exposición pública, sextorsión)."""
    assert retriever.build_pillar_filter("disgust", "estable") == [2, 5, 1]
    assert retriever.build_pillar_filter("disgust", "mejora") == [2, 5, 1]


def test_no_pillar_filter_others(retriever: "EnrichedRetriever") -> None:
    """others → None (sin filtro, búsqueda global sobre todos los pilares)."""
    assert retriever.build_pillar_filter("others", "estable") is None


# ---------------------------------------------------------------------------
# Grupo B — Recuperación semántica (Config C): filtro, fallback y penalización
# ---------------------------------------------------------------------------


def test_returns_documents(retriever: "EnrichedRetriever") -> None:
    """retrieve() devuelve una lista no vacía de Document (máx. TOP_K)."""
    results = retriever.retrieve("me están acosando y no sé qué hacer")
    assert isinstance(results, list), "retrieve() debe devolver list"
    assert 0 < len(results) <= 3, "retrieve() debe devolver entre 1 y TOP_K resultados"
    assert all(isinstance(d, Document) for d in results), (
        "Todos los resultados deben ser instancias de Document"
    )


def test_pillar_filter_restricts_results(retriever: "EnrichedRetriever") -> None:
    """Con pillar_filter=[5], el WHERE por pilar limita los resultados a P5.

    El pre-filtro se pasa a ChromaDB como {'pillar': {'$in': [5]}}; todos los
    documentos devueltos deben pertenecer al pilar 5 (recuperación Config C).
    """
    results = retriever.retrieve("bloquear instagram", pillar_filter=[5])
    assert len(results) > 0, "retrieve() con pillar_filter=[5] no devolvió resultados"
    for doc in results:
        assert doc.metadata.get("pillar") == 5, (
            f"Resultado fuera del pilar filtrado (pillar_filter=[5]): "
            f"pillar={doc.metadata.get('pillar')}, chunk_id={doc.metadata.get('chunk_id')}"
        )


def test_empty_filter_falls_back_to_global(retriever: "EnrichedRetriever") -> None:
    """Un pillar_filter sin documentos activa el fallback a búsqueda global.

    Si ningún chunk del corpus pertenece a los pilares pedidos, retrieve() debe
    consultar ChromaDB sin filtro (corpus completo) en lugar de devolver vacío.
    """
    results = retriever.retrieve("me están acosando", pillar_filter=[99])
    assert len(results) > 0, "El fallback global debe devolver resultados"


def test_p4_soft_penalty_demotes_crisis_chunk(retriever: "EnrichedRetriever") -> None:
    """La penalización P4 blanda (×0.7) reordena un chunk de crisis por debajo.

    Con un chunk P4 de score bruto superior (0.90) y uno P2 inferior (0.80),
    aplicar p4_penalty=0.7 deja P4 en 0.63 < 0.80, de modo que P2 pasa a top-1.
    """
    p4 = Document(page_content="conducta suicida 024", metadata={"pillar": 4, "chunk_id": "P4_007"})
    p2 = Document(page_content="reestructuración cognitiva", metadata={"pillar": 2, "chunk_id": "P2_001"})
    retriever._chroma.similarity_search_with_relevance_scores.side_effect = None
    retriever._chroma.similarity_search_with_relevance_scores.return_value = [(p4, 0.90), (p2, 0.80)]

    results = retriever.retrieve("da igual", pillar_filter=[2, 3, 4], p4_penalty=0.7)
    assert results[0].metadata["chunk_id"] == "P2_001", (
        "Con p4_penalty=0.7, el chunk P4 debe caer por debajo del P2"
    )


def test_retrieve_with_routing_applies_penalty(retriever: "EnrichedRetriever") -> None:
    """retrieve_with_routing('sadness','estable') cablea filtro [2,3,4] y penaliza P4.

    Verifica el atajo del pipeline V2: computa build_pillar_filter + _p4_penalty y
    delega en retrieve(). En sadness+estable la penalización P4 (×0.7) debe demover
    el chunk de crisis frente a uno de TCC con score bruto ligeramente menor.
    """
    p4 = Document(page_content="crisis", metadata={"pillar": 4, "chunk_id": "P4_007"})
    p2 = Document(page_content="tcc", metadata={"pillar": 2, "chunk_id": "P2_001"})
    retriever._chroma.similarity_search_with_relevance_scores.side_effect = None
    retriever._chroma.similarity_search_with_relevance_scores.return_value = [(p4, 0.90), (p2, 0.80)]

    results = retriever.retrieve_with_routing("estoy fatal", emotion="sadness", trend="estable")
    assert results[0].metadata["chunk_id"] == "P2_001", (
        "El routing sadness+estable debe aplicar la penalización P4 blanda"
    )


# ---------------------------------------------------------------------------
# Grupo C — Integración PAP: orden de llamadas en el pipeline V2
# ---------------------------------------------------------------------------


def test_crisis_handled_by_failsafe() -> None:
    """CrisisDetector.detect() se llama ANTES del RAG; un mensaje de crisis
    devuelve ChatbotV2Response con emotion_label='crisis' sin haber consultado
    el retriever.

    Verifica que el cortocircuito PAP funciona como primera capa de seguridad:
    el pipeline devuelve inmediatamente la respuesta de primeros auxilios
    psicológicos sin invocar EmotionDetector, EnrichedRetriever ni el SLM.
    """
    from src.pipeline.v2 import ChatbotV2

    crisis_msg = "ya no quiero seguir viviendo nadie me va a echar de menos"

    with (
        patch("src.pipeline.v2.EmotionDetector"),
        patch("src.pipeline.v2.DocumentIngesterV2"),
        patch("src.pipeline.v2.EnrichedRetriever") as mock_rag_cls,
        patch("src.pipeline.v2.EmotionalMemoryTracker") as mock_memory_cls,
        patch("src.pipeline.v2.ChatOllama"),
        patch("src.pipeline.v2.CrisisDetector") as mock_crisis_cls,
    ):
        # Configurar CrisisDetector para activar nivel HIGH
        mock_crisis_instance = mock_crisis_cls.return_value
        mock_crisis_instance.detect.return_value = MagicMock(
            level="HIGH",
            response="[PAP] Llama al 024 ahora mismo.",
            triggered_keywords=["no quiero seguir viviendo"],
        )

        # mock_rag_instance representa la instancia del retriever
        mock_rag_instance = mock_rag_cls.return_value

        # La memoria emocional debe poder ser consultada en el bypass PAP
        mock_memory_cls.return_value.get_prompt_context.return_value = ""

        chatbot = ChatbotV2()
        result = chatbot.run(crisis_msg, [])

    # ── Verificación 1: CrisisDetector.detect() fue llamado con el mensaje ──
    mock_crisis_instance.detect.assert_called_once_with(crisis_msg)

    # ── Verificación 2: el RAG NO fue consultado (cortocircuito PAP) ──────
    mock_rag_instance.retrieve_with_routing.assert_not_called()

    # ── Verificación 3: la respuesta señala 'crisis' correctamente ────────
    assert result.emotion_label == "crisis", (
        f"Se esperaba emotion_label='crisis', "
        f"se obtuvo '{result.emotion_label}'"
    )
    assert result.crisis_level in ("HIGH", "MEDIUM"), (
        f"Se esperaba crisis_level HIGH o MEDIUM, "
        f"se obtuvo '{result.crisis_level}'"
    )
    assert result.rag_chunks == [], (
        "En bypass de crisis rag_chunks debe estar vacío "
        "(el retriever no fue consultado)"
    )
