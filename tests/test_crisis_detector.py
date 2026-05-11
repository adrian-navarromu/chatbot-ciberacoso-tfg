"""
Tests del módulo CrisisDetector (capa de seguridad PAP).

Grupos:
  A — 20 casos de crisis (HIGH o MEDIUM, cero falsos negativos tolerados)
  B — 20 casos normales (NONE, cero falsos positivos tolerados)

Ejecución:
    pytest tests/test_crisis_detector.py -v
"""

from __future__ import annotations

import pytest

from src.safety.crisis_detector import (
    CRISIS_RESPONSE_HIGH,
    CrisisDetector,
    CrisisResult,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector() -> CrisisDetector:
    """Instancia compartida del detector para todos los tests del módulo."""
    return CrisisDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_crisis(result: CrisisResult) -> None:
    assert result.level in ("HIGH", "MEDIUM"), (
        f"Se esperaba HIGH o MEDIUM, se obtuvo NONE. "
        f"Texto no detectado como crisis."
    )
    assert isinstance(result.triggered_keywords, list)
    assert len(result.triggered_keywords) > 0
    assert isinstance(result.response, str)


def _assert_normal(result: CrisisResult, text: str) -> None:
    assert result.level == "NONE", (
        f"Falso positivo — texto normal detectado como {result.level!r}: {text!r}. "
        f"Keywords: {result.triggered_keywords}"
    )
    assert isinstance(result.triggered_keywords, list)
    assert result.response is None


# ---------------------------------------------------------------------------
# GRUPO A — Casos de crisis (deben activar HIGH o MEDIUM)
# ---------------------------------------------------------------------------


class TestCrisisHigh:
    """12 casos de ideación autolesiva / amenaza vital → deben ser HIGH."""

    @pytest.mark.parametrize(
        "text",
        [
            "ya no quiero seguir viviendo nadie me va a echar de menos",
            "me quiero morir de verdad no es un decir",
            "estoy pensando en hacerme daño esta noche",
            "me voy a quitar la vida",
            "qué sentido tiene seguir si nadie me quiere",
            "me amenaza con matarme si se lo cuento a alguien",
            "me van a pegar a la salida del insti mañana",
            "quiero quitarme de en medio",
            "todo seria mejor sin mi",
            "me quiero morir en serio",
            "llevo semanas pensando en suicidarme",
            "ya no aguanto mas me voy a hacer algo",
        ],
    )
    def test_high_crisis_detected(self, detector: CrisisDetector, text: str) -> None:
        """Verifica que casos de alta gravedad son detectados como HIGH."""
        result = detector.detect(text)
        assert result.level == "HIGH", (
            f"Se esperaba HIGH, se obtuvo {result.level!r} para: {text!r}"
        )
        _assert_crisis(result)
        assert result.requires_immediate_response is True


class TestCrisisMedium:
    """8 casos de malestar extremo sin ideación explícita → HIGH o MEDIUM."""

    @pytest.mark.parametrize(
        "text",
        [
            "no puedo más con todo esto ya no aguanto",
            "estoy completamente sola no tengo a nadie en quien confiar",
            "no me queda nadie nadie me escucha",
            "me van a pegar si no hago lo que dicen",
            "no puedo más y nadie me ayuda",
            "estoy desesperada y no sé qué hacer",
            "me van a agredir si hablo",
            "no aguanto más y no tengo a nadie",
        ],
    )
    def test_medium_crisis_detected(self, detector: CrisisDetector, text: str) -> None:
        """Verifica que casos de malestar extremo activan al menos MEDIUM."""
        result = detector.detect(text)
        _assert_crisis(result)


# ---------------------------------------------------------------------------
# GRUPO B — Casos normales (NO deben activar crisis)
# ---------------------------------------------------------------------------


NORMAL_CASES: list[str] = [
    "estoy muy triste porque me ignoran en clase",
    "me han insultado por instagram",
    "tengo miedo de ir al instituto",
    "me siento sola desde que cambié de colegio",
    "cómo bloqueo a alguien en whatsapp",
    "quiero denunciar lo que me hacen",
    "me han publicado fotos sin permiso",
    "estoy enfadada con mis amigas",
    "me da vergüenza lo que pasó",
    "no sé cómo contárselo a mis padres",
    "me han hackeado la cuenta",
    "me acosan por ser diferente",
    "hola qué puedes hacer por mí",
    "necesito consejo sobre cómo actuar",
    "un chico mayor me escribe cosas raras",
    "me siento muy mal pero creo que va a mejorar",
    "llevo días llorando pero hoy estoy mejor",
    "todo el mundo me odia en clase",
    "ya no quiero usar las redes sociales",
    # Expresión coloquial metafórica — NO debe activar crisis
    "quiero morirme de vergüenza cuando lo recuerdo",
]


class TestNormalCases:
    """20 casos de malestar real o preguntas neutras que no son crisis."""

    @pytest.mark.parametrize("text", NORMAL_CASES)
    def test_no_crisis_detected(self, detector: CrisisDetector, text: str) -> None:
        """Verifica que casos normales no activan el detector de crisis."""
        result = detector.detect(text)
        _assert_normal(result, text)


# ---------------------------------------------------------------------------
# Test de contenido de respuesta HIGH
# ---------------------------------------------------------------------------


def test_response_contains_numbers() -> None:
    """CRISIS_RESPONSE_HIGH debe contener los tres teléfonos de emergencia."""
    assert "024" in CRISIS_RESPONSE_HIGH, "Falta el teléfono 024"
    assert "900 202 010" in CRISIS_RESPONSE_HIGH, "Falta el teléfono ANAR 900 202 010"
    assert "112" in CRISIS_RESPONSE_HIGH, "Falta el teléfono de emergencias 112"


# ---------------------------------------------------------------------------
# Tests de estructura del resultado
# ---------------------------------------------------------------------------


class TestCrisisResultStructure:
    """Verifica la integridad del dataclass CrisisResult."""

    def test_none_result_structure(self, detector: CrisisDetector) -> None:
        """Resultado NONE tiene response=None y lista vacía."""
        result = detector.detect("hola, ¿cómo estás?")
        assert result.level == "NONE"
        assert result.triggered_keywords == []
        assert result.requires_immediate_response is False
        assert result.response is None

    def test_high_result_structure(self, detector: CrisisDetector) -> None:
        """Resultado HIGH tiene requires_immediate_response=True."""
        result = detector.detect("me quiero morir")
        assert result.level == "HIGH"
        assert result.requires_immediate_response is True
        assert isinstance(result.response, str)
        assert len(result.response) > 0

    def test_medium_result_structure(self, detector: CrisisDetector) -> None:
        """Resultado MEDIUM tiene requires_immediate_response=False."""
        result = detector.detect("no puedo más con todo esto")
        assert result.level in ("MEDIUM", "HIGH")
        assert isinstance(result.triggered_keywords, list)
        assert isinstance(result.response, str)
