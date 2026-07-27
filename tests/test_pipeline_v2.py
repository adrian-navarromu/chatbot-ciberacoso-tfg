"""
Smoke tests del pipeline final ChatbotV2.

Cubre las tres ramas de ChatbotV2.run() con TODAS las dependencias mockeadas
(CrisisDetector, EmotionDetector, EnrichedRetriever, EmotionalMemoryTracker,
DocumentIngesterV2 y ChatOllama), de modo que la suite corre sin GPU, sin
modelos descargados y sin Ollama levantado:

  1. HIGH   → cortocircuito PAP determinista: el SLM NO se invoca (0 ms).
  2. MEDIUM → contención PAP con generación: recupera pilar 4 y llama al SLM.
  3. NONE   → flujo normal: emoción → memoria → RAG enrutado → SLM.

Ejecución:
    pytest tests/test_pipeline_v2.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


@contextmanager
def _mocked_chatbot_v2(crisis_mock, emotion_label="sadness", emotion_confidence=0.9):
    """Instancia ChatbotV2 con todas sus dependencias mockeadas.

    Args:
        crisis_mock: MagicMock que se devolverá desde CrisisDetector.detect().
        emotion_label: Etiqueta que devolverá EmotionDetector.detect().
        emotion_confidence: Confianza que devolverá EmotionDetector.detect().

    Yields:
        (chatbot, mocks) donde mocks expone los dobles relevantes para aserciones.
    """
    from src.pipeline.v2 import ChatbotV2

    with (
        patch("src.pipeline.v2.EmotionDetector") as mock_emotion_cls,
        patch("src.pipeline.v2.DocumentIngesterV2") as mock_ingester_cls,
        patch("src.pipeline.v2.EnrichedRetriever") as mock_rag_cls,
        patch("src.pipeline.v2.EmotionalMemoryTracker") as mock_memory_cls,
        patch("src.pipeline.v2.ChatOllama") as mock_ollama_cls,
        patch("src.pipeline.v2.CrisisDetector") as mock_crisis_cls,
    ):
        mock_crisis_cls.return_value.detect.return_value = crisis_mock

        mock_emotion_cls.return_value.detect.return_value = MagicMock(
            label=emotion_label, confidence=emotion_confidence
        )

        mock_ingester_cls.return_value.load_corpus.return_value = []

        mock_memory_cls.return_value.detect_trend.return_value = "estable"
        mock_memory_cls.return_value.get_prompt_context.return_value = ""

        mock_ollama_cls.return_value.invoke.return_value = MagicMock(
            content="respuesta generada por el SLM"
        )

        chatbot = ChatbotV2()
        mocks = MagicMock(
            crisis=mock_crisis_cls.return_value,
            emotion=mock_emotion_cls.return_value,
            rag=mock_rag_cls.return_value,
            memory=mock_memory_cls.return_value,
            ollama_cls=mock_ollama_cls,
        )
        yield chatbot, mocks


def test_high_crisis_bypasses_slm() -> None:
    """HIGH: respuesta PAP fija, sin invocar SLM, RAG ni clasificador emocional."""
    crisis = MagicMock(level="HIGH", subtype="ideacion", response="[PAP] Llama al 024.")

    with _mocked_chatbot_v2(crisis) as (chatbot, mocks):
        result = chatbot.run("ya no quiero seguir viviendo", [])

    assert result.crisis_level == "HIGH"
    assert result.emotion_label == "crisis"
    assert result.response == "[PAP] Llama al 024."
    assert result.latency_ms == 0.0
    assert result.rag_chunks == []
    # El SLM nunca se instancia/invoca en HIGH
    mocks.ollama_cls.assert_not_called()
    # El clasificador emocional tampoco corre
    mocks.emotion.detect.assert_not_called()
    mocks.rag.retrieve_with_routing.assert_not_called()
    # Historial: turno usuario + respuesta PAP
    assert len(result.history) == 2
    assert result.history[-1]["content"] == "[PAP] Llama al 024."


def test_medium_crisis_generates_with_pillar4() -> None:
    """MEDIUM: recupera pilar 4 (sugerido) y genera contención con el SLM."""
    crisis = MagicMock(
        level="MEDIUM", requires_generation=True, suggested_pillar=4
    )

    with _mocked_chatbot_v2(crisis) as (chatbot, mocks):
        mocks.rag.retrieve.return_value = [
            Document(page_content="Recurso PAP", metadata={"chunk_id": "P4_001", "pillar": 4})
        ]
        mocks.ollama_cls.return_value.invoke.return_value = MagicMock(content="Estoy aquí contigo.")

        result = chatbot.run("no puedo más con todo esto", [])

    assert result.crisis_level == "MEDIUM"
    assert result.emotion_label == "crisis"
    assert result.response == "Estoy aquí contigo."
    # Recuperación forzada al pilar sugerido (4), no routing emocional
    mocks.rag.retrieve.assert_called_once()
    assert mocks.rag.retrieve.call_args.kwargs["pillar_filter"] == [4]
    mocks.rag.retrieve_with_routing.assert_not_called()
    # El SLM sí se invoca en MEDIUM
    mocks.ollama_cls.return_value.invoke.assert_called_once()


def test_normal_flow_routes_and_generates() -> None:
    """NONE: emoción → memoria → RAG enrutado por emoción → SLM."""
    crisis = MagicMock(level="NONE")

    with _mocked_chatbot_v2(crisis, emotion_label="sadness", emotion_confidence=0.88) as (chatbot, mocks):
        mocks.rag.retrieve_with_routing.return_value = [
            Document(page_content="Chunk TCC", metadata={"chunk_id": "P2_003", "pillar": 2})
        ]

        result = chatbot.run("me siento fatal desde el acoso", [])

    assert result.crisis_level is None
    assert result.emotion_label == "sadness"
    assert result.emotion_confidence == 0.88
    assert result.trend == "estable"
    assert result.response == "respuesta generada por el SLM"
    # Routing emocional con la emoción detectada
    mocks.rag.retrieve_with_routing.assert_called_once()
    assert mocks.rag.retrieve_with_routing.call_args.kwargs["emotion"] == "sadness"
    # El SLM genera y la memoria emocional se actualiza
    mocks.ollama_cls.return_value.invoke.assert_called_once()
    mocks.memory.update.assert_called_once()
    assert result.rag_chunks == [{"chunk_id": "P2_003", "pillar": 2}]


def test_change_model_and_reset_session() -> None:
    """change_model() actualiza el modelo en caliente; run() lo respeta."""
    crisis = MagicMock(level="HIGH", subtype="ideacion", response="[PAP]")

    with _mocked_chatbot_v2(crisis) as (chatbot, _mocks):
        chatbot.change_model("mistral:7b")
        result = chatbot.run("quiero desaparecer", [])

    assert chatbot.model_name == "mistral:7b"
    assert result.model_name == "mistral:7b"
