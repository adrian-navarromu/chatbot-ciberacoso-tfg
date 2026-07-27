"""
Tests de los constructores de prompt (V1 y V2).

Cubre: build_prompt, get_system_prompt_preview (builder.py)
       build_prompt_v2, get_system_prompt_preview_v2 (builder_v2.py)

No requieren GPU ni Ollama: funciones puras sin dependencias externas.

Grupos:
  A — build_prompt (V1): estructura, umbral de confianza, inyección de RAG y memoria
  B — get_system_prompt_preview (V1): coherencia con build_prompt
  C — build_prompt_v2 (V2): bloques etiquetados, fallbacks, umbral de confianza
  D — get_system_prompt_preview_v2 (V2): coherencia con build_prompt_v2

Ejecución:
    pytest tests/test_prompt_builder.py -v
"""

from __future__ import annotations

import pytest

from src.prompts.builder_v1 import (
    EMOTION_VARIANTS,
    build_prompt,
    get_system_prompt_preview,
)
from src.prompts.builder_v2 import (
    build_crisis_prompt_v2,
    build_prompt_v2,
    get_system_prompt_preview_v2,
)

VALID_EMOTIONS: list[str] = ["fear", "sadness", "anger", "disgust", "joy", "surprise", "others"]

SAMPLE_HISTORY: list[dict] = [
    {"role": "user", "content": "Me están haciendo la vida imposible."},
    {"role": "assistant", "content": "Entiendo que describes una situación muy difícil."},
]


# ---------------------------------------------------------------------------
# Grupo A — build_prompt (V1)
# ---------------------------------------------------------------------------


class TestBuildPromptV1:

    def test_structure_no_history(self) -> None:
        """Sin historial → lista de 1 mensaje con role='system' y content no vacío."""
        result = build_prompt("fear", "", [])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], str)
        assert result[0]["content"]

    def test_structure_with_history(self) -> None:
        """Con historial → system en posición 0, historial añadido sin modificar."""
        result = build_prompt("sadness", "", SAMPLE_HISTORY)
        assert len(result) == 1 + len(SAMPLE_HISTORY)
        assert result[0]["role"] == "system"
        assert result[1:] == SAMPLE_HISTORY

    @pytest.mark.parametrize("emotion", VALID_EMOTIONS)
    def test_all_emotions_no_error(self, emotion: str) -> None:
        """Las 7 emociones válidas no lanzan KeyError."""
        result = build_prompt(emotion, "", [], confidence=1.0)
        assert result[0]["role"] == "system"

    def test_low_confidence_uses_others(self) -> None:
        """confidence < 0.4 → variante 'others' aunque se pase 'fear'."""
        result = build_prompt("fear", "", [], confidence=0.3)
        system = result[0]["content"]
        assert EMOTION_VARIANTS["others"] in system
        assert EMOTION_VARIANTS["fear"] not in system

    def test_exact_confidence_threshold_keeps_emotion(self) -> None:
        """confidence == 0.4 → se mantiene la emoción original."""
        result = build_prompt("fear", "", [], confidence=0.4)
        assert EMOTION_VARIANTS["fear"] in result[0]["content"]

    def test_rag_context_injected(self) -> None:
        """rag_context no vacío → el texto RAG aparece en el system."""
        rag = "Protocolo de actuación ante ciberacoso: conserva pruebas y denuncia."
        result = build_prompt("sadness", rag, [])
        assert rag in result[0]["content"]

    def test_rag_context_absent_when_empty(self) -> None:
        """rag_context vacío → el bloque RAG no se inyecta."""
        result = build_prompt("fear", "", [])
        system = result[0]["content"]
        assert "CONTEXTO CLÍNICO" not in system

    def test_emotional_context_injected(self) -> None:
        """emotional_context no vacío → se inyecta en el system."""
        ctx = "Emociones registradas: fear, fear, sadness."
        result = build_prompt("fear", "", [], emotional_context=ctx)
        assert ctx in result[0]["content"]

    def test_emotional_context_absent_when_empty(self) -> None:
        """emotional_context vacío → el bloque de memoria no aparece."""
        result = build_prompt("fear", "", [], emotional_context="")
        assert "ESTADO EMOCIONAL DEL USUARIO" not in result[0]["content"]


# ---------------------------------------------------------------------------
# Grupo B — get_system_prompt_preview (V1)
# ---------------------------------------------------------------------------


class TestGetSystemPromptPreviewV1:

    def test_returns_non_empty_string(self) -> None:
        """Devuelve str no vacío."""
        preview = get_system_prompt_preview("fear")
        assert isinstance(preview, str)
        assert preview

    def test_matches_build_prompt_system(self) -> None:
        """El resultado es idéntico al content del system de build_prompt."""
        rag = "Contexto clínico de ejemplo."
        preview = get_system_prompt_preview("sadness", rag_context=rag)
        messages = build_prompt("sadness", rag, [], confidence=1.0)
        assert preview == messages[0]["content"]


# ---------------------------------------------------------------------------
# Grupo C — build_prompt_v2 (V2)
# ---------------------------------------------------------------------------


class TestBuildPromptV2:

    def test_structure_no_history(self) -> None:
        """Sin historial → lista de 1 mensaje con role='system' y content no vacío."""
        result = build_prompt_v2("fear", "", [])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], str)
        assert result[0]["content"]

    def test_structure_with_history(self) -> None:
        """Con historial → system en posición 0, historial añadido sin modificar."""
        result = build_prompt_v2("sadness", "", SAMPLE_HISTORY)
        assert len(result) == 1 + len(SAMPLE_HISTORY)
        assert result[0]["role"] == "system"
        assert result[1:] == SAMPLE_HISTORY

    def test_four_blocks_present(self) -> None:
        """Los 4 bloques etiquetados de la variante D aparecen en el system."""
        result = build_prompt_v2("fear", "Contexto RAG", [])
        system = result[0]["content"]
        for block in (
            "[ROL_Y_LIMITES]",
            "[ESTADO_EMOCIONAL_USUARIO]",
            "[CONTEXTO_CLINICO]",
            "[OBJETIVO_TURNO]",
        ):
            assert block in system, f"Bloque {block!r} no encontrado en el system"

    @pytest.mark.parametrize("emotion", VALID_EMOTIONS)
    def test_all_emotions_no_error(self, emotion: str) -> None:
        """Las 7 emociones válidas no lanzan KeyError."""
        result = build_prompt_v2(emotion, "", [], confidence=1.0)
        assert result[0]["role"] == "system"

    def test_low_confidence_uses_others(self) -> None:
        """confidence < 0.4 → 'others' en el bloque ESTADO_EMOCIONAL_USUARIO."""
        result = build_prompt_v2("fear", "", [], confidence=0.3)
        assert "Emoción detectada: others" in result[0]["content"]

    def test_exact_confidence_threshold_keeps_emotion(self) -> None:
        """confidence == 0.4 → emoción original en el bloque ESTADO_EMOCIONAL_USUARIO."""
        result = build_prompt_v2("fear", "", [], confidence=0.4)
        assert "Emoción detectada: fear" in result[0]["content"]

    def test_confidence_percentage_in_system(self) -> None:
        """La confianza aparece como porcentaje en el bloque ESTADO_EMOCIONAL_USUARIO."""
        result = build_prompt_v2("sadness", "", [], confidence=0.85)
        assert "85%" in result[0]["content"]

    def test_rag_context_injected(self) -> None:
        """rag_context no vacío → el texto RAG aparece en el bloque CONTEXTO_CLINICO."""
        rag = "El 024 atiende 24 horas de forma gratuita."
        result = build_prompt_v2("sadness", rag, [])
        assert rag in result[0]["content"]

    def test_rag_fallback_when_empty(self) -> None:
        """rag_context vacío → mensaje de fallback en el bloque CONTEXTO_CLINICO."""
        result = build_prompt_v2("fear", "", [])
        assert "Sin contexto clínico disponible" in result[0]["content"]

    def test_emotional_context_injected(self) -> None:
        """emotional_context no vacío → aparece en el system."""
        ctx = "Tendencia emocional: el usuario mostró fear como emoción predominante."
        result = build_prompt_v2("fear", "", [], emotional_context=ctx)
        assert ctx in result[0]["content"]

    def test_emotional_context_fallback_when_empty(self) -> None:
        """emotional_context vacío → mensaje de fallback en ESTADO_EMOCIONAL_USUARIO."""
        result = build_prompt_v2("fear", "", [], emotional_context="")
        assert "Sin datos de sesión previa" in result[0]["content"]

    def test_trend_in_system(self) -> None:
        """El trend pasado aparece en el bloque ESTADO_EMOCIONAL_USUARIO."""
        for trend in ("estable", "mejora", "deterioro"):
            result = build_prompt_v2("sadness", "", [], trend=trend)
            assert trend in result[0]["content"], f"Trend '{trend}' no encontrado en el system"


# ---------------------------------------------------------------------------
# Grupo D — get_system_prompt_preview_v2 (V2)
# ---------------------------------------------------------------------------


class TestGetSystemPromptPreviewV2:

    def test_returns_non_empty_string(self) -> None:
        """Devuelve str no vacío."""
        preview = get_system_prompt_preview_v2("fear")
        assert isinstance(preview, str)
        assert preview

    def test_matches_build_prompt_v2_system(self) -> None:
        """El resultado es idéntico al content del system de build_prompt_v2."""
        rag = "Contexto clínico de ejemplo."
        ctx = "Emociones registradas: fear, sadness."
        preview = get_system_prompt_preview_v2(
            "sadness", rag_context=rag, confidence=0.9, emotional_context=ctx, trend="deterioro"
        )
        messages = build_prompt_v2(
            "sadness", rag, [], confidence=0.9, emotional_context=ctx, trend="deterioro"
        )
        assert preview == messages[0]["content"]


# ---------------------------------------------------------------------------
# Grupo E — build_crisis_prompt_v2 (V2): rama de contención PAP MEDIUM
# ---------------------------------------------------------------------------


class TestBuildCrisisPromptV2:

    def test_structure_no_history(self) -> None:
        """Sin historial: devuelve solo el mensaje system."""
        result = build_crisis_prompt_v2("", [])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_structure_with_history(self) -> None:
        """Con historial: system + los mensajes previos, sin añadir el del usuario."""
        result = build_crisis_prompt_v2("", SAMPLE_HISTORY)
        assert result[0]["role"] == "system"
        assert result[1:] == SAMPLE_HISTORY

    def test_contains_containment_blocks(self) -> None:
        """El system incluye los bloques etiquetados y el objetivo de contención PAP."""
        system = build_crisis_prompt_v2("Recurso de apoyo", [])[0]["content"]
        assert "[ROL_Y_LIMITES]" in system
        assert "[CONTEXTO_CLINICO]" in system
        assert "[OBJETIVO_TURNO]" in system
        assert "CONTENCIÓN PAP" in system

    def test_rag_context_injected(self) -> None:
        """El contexto clínico (pilar 4) se inserta en el bloque correspondiente."""
        system = build_crisis_prompt_v2("Línea de ayuda 024 disponible", [])[0]["content"]
        assert "Línea de ayuda 024 disponible" in system

    def test_rag_fallback_when_empty(self) -> None:
        """Sin contexto RAG, usa el texto de fallback."""
        system = build_crisis_prompt_v2("", [])[0]["content"]
        assert "Sin recursos de apoyo disponibles." in system
