"""
Tests del módulo EmotionalMemoryTracker (antes EmotionalMemoryGRU).

Cubre: update, get_window, get_dominant_emotion, detect_trend,
       get_trend_result, get_prompt_context, reset, to_dict.

Ejecución:
    pytest tests/test_emotional_memory.py -v
"""

from __future__ import annotations

import pytest

from src.memory.emotional_memory import (
    EMOTION_LABELS,
    NEGATIVE_HIGH,
    NEGATIVE_LOW,
    POSITIVE,
    EmotionalMemoryTracker,
    TrendResult,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mem() -> EmotionalMemoryTracker:
    """Instancia fresca para cada test."""
    return EmotionalMemoryTracker(window_size=6)


# ---------------------------------------------------------------------------
# Constantes de dominio
# ---------------------------------------------------------------------------


class TestConstants:
    def test_emotion_labels_count(self) -> None:
        assert len(EMOTION_LABELS) == 7

    def test_valence_groups_disjoint(self) -> None:
        all_emotions = NEGATIVE_HIGH + NEGATIVE_LOW + POSITIVE
        assert len(all_emotions) == len(set(all_emotions)), "Grupos solapados"

    def test_valence_groups_cover_labels(self) -> None:
        all_emotions = set(NEGATIVE_HIGH + NEGATIVE_LOW + POSITIVE)
        assert all_emotions == set(EMOTION_LABELS)


# ---------------------------------------------------------------------------
# update + get_window
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_single_update_increments_turn_count(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        assert mem.turn_count == 1

    def test_low_confidence_records_others(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("anger", 0.39)
        assert mem.emotion_history[-1] == "others"

    def test_exact_threshold_keeps_label(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("sadness", 0.4)
        assert mem.emotion_history[-1] == "sadness"

    def test_window_respects_window_size(self, mem: EmotionalMemoryTracker) -> None:
        for _ in range(10):
            mem.update("fear", 0.9)
        assert len(mem.get_window()) == 6

    def test_full_history_grows_beyond_window(self, mem: EmotionalMemoryTracker) -> None:
        for _ in range(10):
            mem.update("joy", 0.9)
        assert len(mem.emotion_history) == 10


# ---------------------------------------------------------------------------
# get_dominant_emotion
# ---------------------------------------------------------------------------


class TestDominantEmotion:
    def test_empty_history_returns_others(self, mem: EmotionalMemoryTracker) -> None:
        assert mem.get_dominant_emotion() == "others"

    def test_clear_majority(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["fear", "fear", "fear", "joy", "joy"]:
            mem.update(e, 0.9)
        assert mem.get_dominant_emotion() == "fear"

    def test_tie_returns_most_recent(self, mem: EmotionalMemoryTracker) -> None:
        # fear y sadness empatados; sadness es el más reciente
        for e in ["fear", "sadness", "fear", "sadness"]:
            mem.update(e, 0.9)
        assert mem.get_dominant_emotion() == "sadness"

    def test_single_turn(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("anger", 0.8)
        assert mem.get_dominant_emotion() == "anger"


# ---------------------------------------------------------------------------
# detect_trend
# ---------------------------------------------------------------------------


class TestDetectTrend:
    def test_empty_is_stable(self, mem: EmotionalMemoryTracker) -> None:
        assert mem.detect_trend() == "estable"

    def test_single_turn_is_stable(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        assert mem.detect_trend() == "estable"

    def test_mejora(self, mem: EmotionalMemoryTracker) -> None:
        # Primera mitad: fear (0), Segunda mitad: joy (2) → diff = +2 > 0.3
        for e in ["fear", "fear", "fear", "joy", "joy", "joy"]:
            mem.update(e, 0.9)
        assert mem.detect_trend() == "mejora"

    def test_deterioro(self, mem: EmotionalMemoryTracker) -> None:
        # Primera mitad: joy (2), Segunda mitad: fear (0) → diff = -2 < -0.3
        for e in ["joy", "joy", "joy", "fear", "fear", "fear"]:
            mem.update(e, 0.9)
        assert mem.detect_trend() == "deterioro"

    def test_estable_same_emotion(self, mem: EmotionalMemoryTracker) -> None:
        for _ in range(6):
            mem.update("sadness", 0.9)
        assert mem.detect_trend() == "estable"

    def test_estable_within_threshold(self, mem: EmotionalMemoryTracker) -> None:
        # Primera mitad: sadness (1), Segunda mitad: sadness (1) → diff = 0
        for e in ["sadness", "sadness", "sadness", "sadness", "sadness", "sadness"]:
            mem.update(e, 0.9)
        assert mem.detect_trend() == "estable"

    def test_trend_only_uses_window(self, mem: EmotionalMemoryTracker) -> None:
        # Historia larga de fear seguida de ventana de joy → debe detectar mejora
        for _ in range(20):
            mem.update("fear", 0.9)
        for _ in range(6):
            mem.update("joy", 0.9)
        assert mem.detect_trend() == "estable"  # toda la ventana es joy → estable


# ---------------------------------------------------------------------------
# get_prompt_context
# ---------------------------------------------------------------------------


class TestGetPromptContext:
    def test_no_turns_empty_string(self, mem: EmotionalMemoryTracker) -> None:
        assert mem.get_prompt_context() == ""

    def test_turn_1(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        ctx = mem.get_prompt_context()
        assert ctx.startswith("Primera interacción.")
        assert "fear" in ctx

    def test_turn_2(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        mem.update("sadness", 0.9)
        ctx = mem.get_prompt_context()
        assert "Emociones registradas:" in ctx
        assert "sadness" in ctx

    def test_turn_3(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["fear", "sadness", "anger"]:
            mem.update(e, 0.9)
        ctx = mem.get_prompt_context()
        assert "Emociones registradas:" in ctx
        assert "anger" in ctx

    def test_turn_4_contains_tendencia(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["fear", "fear", "fear", "sadness"]:
            mem.update(e, 0.9)
        ctx = mem.get_prompt_context()
        assert "Tendencia emocional:" in ctx
        assert "fear" in ctx
        assert "sadness" in ctx

    def test_turn_4_mejora_phrase(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["fear", "fear", "joy", "joy"]:
            mem.update(e, 0.9)
        ctx = mem.get_prompt_context()
        assert "mejora" in ctx

    def test_turn_4_deterioro_phrase(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["joy", "joy", "fear", "fear"]:
            mem.update(e, 0.9)
        ctx = mem.get_prompt_context()
        assert "deterioro" in ctx
        assert "presta especial atención" in ctx

    def test_turn_4_estable_phrase(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["sadness", "sadness", "sadness", "sadness"]:
            mem.update(e, 0.9)
        ctx = mem.get_prompt_context()
        assert "estable" in ctx

    def test_window_size_reflected_in_context(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["fear", "fear", "fear", "fear"]:
            mem.update(e, 0.9)
        ctx = mem.get_prompt_context()
        assert "6" in ctx  # window_size=6 debe aparecer


# ---------------------------------------------------------------------------
# get_trend_result
# ---------------------------------------------------------------------------


class TestGetTrendResult:
    def test_returns_trend_result_instance(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("joy", 0.9)
        result = mem.get_trend_result()
        assert isinstance(result, TrendResult)

    def test_current_emotion_matches_last_update(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        mem.update("joy", 0.8)
        result = mem.get_trend_result()
        assert result.current_emotion == "joy"

    def test_turn_count_correct(self, mem: EmotionalMemoryTracker) -> None:
        for _ in range(5):
            mem.update("sadness", 0.9)
        result = mem.get_trend_result()
        assert result.turn_count == 5

    def test_window_emotions_bounded(self, mem: EmotionalMemoryTracker) -> None:
        for _ in range(10):
            mem.update("anger", 0.9)
        result = mem.get_trend_result()
        assert len(result.window_emotions) <= 6

    def test_prompt_context_non_empty_after_update(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("sadness", 0.9)
        result = mem.get_trend_result()
        assert isinstance(result.prompt_context, str)
        assert len(result.prompt_context) > 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_history(self, mem: EmotionalMemoryTracker) -> None:
        for _ in range(5):
            mem.update("fear", 0.9)
        mem.reset()
        assert mem.emotion_history == []
        assert mem.turn_count == 0

    def test_reset_allows_new_session(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        mem.reset()
        mem.update("joy", 0.9)
        assert mem.turn_count == 1
        assert mem.emotion_history == ["joy"]


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_keys(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("joy", 0.9)
        d = mem.to_dict()
        expected_keys = {"turn_count", "window", "dominant_emotion", "trend", "full_history"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_types(self, mem: EmotionalMemoryTracker) -> None:
        mem.update("fear", 0.9)
        d = mem.to_dict()
        assert isinstance(d["turn_count"], int)
        assert isinstance(d["window"], list)
        assert isinstance(d["dominant_emotion"], str)
        assert isinstance(d["trend"], str)
        assert isinstance(d["full_history"], list)

    def test_to_dict_empty_memory(self, mem: EmotionalMemoryTracker) -> None:
        d = mem.to_dict()
        assert d["turn_count"] == 0
        assert d["window"] == []
        assert d["full_history"] == []

    def test_to_dict_full_history_exceeds_window(self, mem: EmotionalMemoryTracker) -> None:
        for e in ["fear"] * 4 + ["joy"] * 6:
            mem.update(e, 0.9)
        d = mem.to_dict()
        assert len(d["full_history"]) == 10
        assert len(d["window"]) == 6


# ---------------------------------------------------------------------------
# Integración: sesión completa
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_session_deterioro(self, mem: EmotionalMemoryTracker) -> None:
        """Sesión que empieza bien y empeora → debe detectar deterioro."""
        progression = ["joy", "joy", "joy", "fear", "anger", "disgust"]
        for e in progression:
            mem.update(e, 0.85)
        result = mem.get_trend_result()
        assert result.trend == "deterioro"
        assert result.current_emotion == "disgust"

    def test_full_session_mejora(self, mem: EmotionalMemoryTracker) -> None:
        """Sesión que empieza mal y mejora → debe detectar mejora."""
        progression = ["fear", "anger", "disgust", "sadness", "joy", "joy"]
        for e in progression:
            mem.update(e, 0.85)
        result = mem.get_trend_result()
        assert result.trend == "mejora"
        assert result.current_emotion == "joy"

    def test_low_confidence_affects_trend(self, mem: EmotionalMemoryTracker) -> None:
        """Confianza baja degrada a 'others' (valencia 2) y puede cambiar la tendencia."""
        # Con confianza alta: fear → deterioro
        # Con confianza baja: fear se convierte en others (positivo) → no deterioro
        for _ in range(3):
            mem.update("joy", 0.9)
        for _ in range(3):
            mem.update("fear", 0.2)   # confianza baja → registrado como 'others'
        result = mem.get_trend_result()
        # Toda la segunda mitad es 'others' (valencia 2), no hay deterioro
        assert result.trend != "deterioro"

    def test_custom_window_size(self) -> None:
        """window_size personalizado respetado correctamente."""
        m = EmotionalMemoryTracker(window_size=3)
        for e in ["fear", "fear", "fear", "joy", "joy", "joy"]:
            m.update(e, 0.9)
        assert len(m.get_window()) == 3
        assert m.get_window() == ["joy", "joy", "joy"]


