"""
Tests del módulo EnrichedRetriever y verificación del orden de llamadas
en el pipeline V2 (cortocircuito PAP).

Grupos:
  A — Routing (build_pillar_filter): unidad pura, sin dependencias externas.
  B — Recuperación BM25: índice léxico real sobre el corpus + Chroma mockeado.
  C — Integración PAP: CrisisDetector intercepta ANTES de consultar el RAG.

Los tests del Grupo B usan BM25Retriever con los documentos reales del corpus
(data/rag_corpus/) y un Chroma vacío mockeado. Esto permite verificar el
comportamiento léxico de forma determinista y sin GPU/modelos pesados.

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


@pytest.fixture(scope="module")
def retriever(corpus_documents: list[Document]):
    """EnrichedRetriever con BM25 real sobre el corpus + Chroma vacío mockeado.

    Evita cargar SentenceTransformer (paraphrase-spanish-distilroberta)
    y ChromaDB persistente para mantener los tests rápidos.
    En Config D, retrieve() construye BM25 por consulta sobre self._documents
    (corpus real); Chroma mockeado devuelve [] → los resultados son BM25 puros,
    lo que hace las búsquedas léxicas auténticas y deterministas.

    Yields:
        EnrichedRetriever listo para usar en los tests.
    """
    from src.rag.enriched_retriever import EnrichedRetriever

    tmp_dir = tempfile.mkdtemp()

    # _EmptyRetriever es un BaseRetriever real → EnsembleRetriever lo acepta
    fake_chroma = MagicMock()
    fake_chroma.as_retriever.return_value = _EmptyRetriever()

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
    """sadness + deterioro → [2, 4]; sadness + estable → [2, 3].

    deterioro GRU activa P4: empeoramiento progresivo sin activar PAP puede
    requerir recursos de crisis aunque no se hayan usado keywords críticas.
    En estable, P4 queda excluido — el failsafe PAP intercepta crisis aisladas.
    """
    assert retriever.build_pillar_filter("sadness", "deterioro") == [2, 4]
    assert retriever.build_pillar_filter("sadness", "estable") == [2, 3]


def test_no_pillar_filter_others(retriever: "EnrichedRetriever") -> None:
    """others → None (sin filtro, búsqueda global sobre todos los pilares)."""
    assert retriever.build_pillar_filter("others", "estable") is None


# ---------------------------------------------------------------------------
# Grupo B — Recuperación con BM25 real
# ---------------------------------------------------------------------------


def test_returns_documents(retriever: "EnrichedRetriever") -> None:
    """retrieve() devuelve una lista no vacía de Document de LangChain."""
    results = retriever.retrieve("me están acosando y no sé qué hacer")
    assert isinstance(results, list), "retrieve() debe devolver list"
    assert len(results) > 0, "retrieve() no debe devolver lista vacía"
    assert all(isinstance(d, Document) for d in results), (
        "Todos los resultados deben ser instancias de Document"
    )


def test_bm25_instagram(retriever: "EnrichedRetriever") -> None:
    """Config D con pillar_filter=[5] → top-1 es un chunk de pilar 5.

    En Config D, BM25 se instancia solo sobre los documentos P5 (digital),
    garantizando que 'bloquear' e 'Instagram' compiten únicamente dentro
    del subespacio correcto. Sin routing (corpus global), otros chunks
    con 'bloquear' pueden obtener mayor TF-IDF y desplazar a los P5.

    Con paraphrase-spanish-distilroberta como modelo semántico, el canal
    semántico ya captura 'instagram' razonablemente bien, pero BM25
    garantiza la recuperación exacta independientemente del modelo.
    """
    results = retriever.retrieve("bloquear instagram", pillar_filter=[5])
    assert len(results) > 0, "El retriever no devolvió resultados con pillar_filter=[5]"
    top1_pillar = results[0].metadata.get("pillar")
    top1_id = results[0].metadata.get("chunk_id")
    assert top1_pillar == 5, (
        f"Se esperaba pillar=5 en top-1, se obtuvo pillar={top1_pillar} "
        f"(chunk_id={top1_id})"
    )


def test_config_d_pillar_filter(retriever: "EnrichedRetriever") -> None:
    """Con pillar_filter=[5], retrieve() instancia BM25 solo sobre chunks P5.

    Verifica que Config D limita ambos canales al subconjunto del pilar:
    todos los resultados devueltos deben pertenecer al pilar 5.
    (Chroma mockeado devuelve [] → resultados son BM25 puros sobre P5.)
    """
    results = retriever.retrieve("bloquear instagram", pillar_filter=[5])
    assert len(results) > 0, "retrieve() con pillar_filter=[5] no devolvió resultados"
    for doc in results:
        assert doc.metadata.get("pillar") == 5, (
            f"Resultado fuera del pilar filtrado (pillar_filter=[5]): "
            f"pillar={doc.metadata.get('pillar')}, chunk_id={doc.metadata.get('chunk_id')}"
        )


def test_bm25_phone(retriever: "EnrichedRetriever") -> None:
    """Config D con routing fear → P4_007 (línea 024) en los resultados.

    Con pillar_filter=[2, 4, 1] (routing de miedo), BM25 se instancia solo
    sobre documentos de esos pilares. El token '024' está presente en P4_007
    ('024 — Línea de atención a la conducta suicida') y con el corpus reducido
    obtiene TF-IDF suficiente para aparecer en top-3.

    Con paraphrase-spanish-distilroberta como modelo semántico, el canal
    semántico ya captura '024' razonablemente bien, pero BM25 garantiza
    la recuperación exacta independientemente del modelo.
    """
    pillar_filter = retriever.build_pillar_filter("fear", "estable")  # [2, 4, 1]
    results = retriever.retrieve("024", pillar_filter=pillar_filter)
    chunk_ids = [d.metadata.get("chunk_id") for d in results]
    assert "P4_007" in chunk_ids, (
        f"P4_007 no encontrado con routing fear {pillar_filter}. "
        f"chunk_ids obtenidos: {chunk_ids}"
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
        patch("src.pipeline.v2.EmotionalMemoryGRU") as mock_memory_cls,
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
