"""
Tests de integración del pipeline ChatbotV1.

Requieren Ollama corriendo en localhost:11434 y los modelos cargados.
Si Ollama no está disponible, todos los tests se saltan automáticamente.

Ejecutar:
    pytest tests/test_pipeline_v1.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.pipeline.v1 import ChatbotV1, ChatbotV1Response

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

VALID_EMOTIONS: frozenset[str] = frozenset(
    {"fear", "sadness", "anger", "disgust", "joy", "surprise", "others"}
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chatbot(model_path) -> ChatbotV1:
    """Instancia de ChatbotV1 compartida por todos los tests de integración.

    Salta la suite completa si Ollama no responde en localhost:11434.
    Usa la ruta del modelo proporcionada por --model-path (o la ruta por defecto).
    """
    try:
        resp = requests.get("http://localhost:11434", timeout=3)
        # Ollama devuelve 200 con "Ollama is running" en el cuerpo
        if resp.status_code not in (200, 404):
            pytest.skip("Ollama no está disponible en localhost:11434")
    except requests.exceptions.ConnectionError:
        pytest.skip("Ollama no está disponible en localhost:11434")
    except requests.exceptions.Timeout:
        pytest.skip("Ollama no responde (timeout) en localhost:11434")

    model_dir = Path(model_path)
    if not (list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin"))):
        pytest.skip(
            f"Pesos del modelo no encontrados en {model_path}. "
            "Entrena primero con python src/emotion/train_classifier.py"
        )

    return ChatbotV1(emotion_model_path=str(model_path))


# ---------------------------------------------------------------------------
# Helper de aserciones
# ---------------------------------------------------------------------------


def _assert_valid(result: ChatbotV1Response, expected_history_len: int) -> None:
    """Verifica las invariantes de toda respuesta válida del pipeline."""
    assert isinstance(result.response, str), "La respuesta debe ser un str"
    assert result.response.strip(), "La respuesta no debe estar vacía"
    assert len(result.history) == expected_history_len, (
        f"Historial esperado: {expected_history_len} mensajes, "
        f"obtenido: {len(result.history)}"
    )
    assert result.emotion_label in VALID_EMOTIONS, (
        f"Emoción '{result.emotion_label}' no está entre las 7 válidas"
    )
    assert 0 <= len(result.rag_chunks) <= 4, (
        f"Se esperaban 0-4 chunks RAG, obtenidos: {len(result.rag_chunks)}"
    )
    assert 0.0 <= result.emotion_confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests NEUTROS — conversación general sin ciberacoso
# ---------------------------------------------------------------------------


def test_neutral_greeting(chatbot: ChatbotV1) -> None:
    """Saludo inicial: el chatbot se presenta correctamente."""
    result = chatbot.run("Hola, ¿qué eres y cómo puedes ayudarme?", [])
    _assert_valid(result, expected_history_len=2)


def test_neutral_cyberbullying_definition(chatbot: ChatbotV1) -> None:
    """Pregunta conceptual sobre ciberacoso: respuesta informativa."""
    result = chatbot.run("¿Puedes explicarme qué es el ciberacoso?", [])
    _assert_valid(result, expected_history_len=2)


def test_neutral_social_media_question(chatbot: ChatbotV1) -> None:
    """Pregunta genérica sobre redes sociales: sin urgencia emocional."""
    result = chatbot.run("Tengo una pregunta sobre las redes sociales", [])
    _assert_valid(result, expected_history_len=2)


def test_neutral_instagram_features_multiturn(chatbot: ChatbotV1) -> None:
    """Pregunta técnica sobre Instagram seguida de profundización (2 turnos)."""
    r1 = chatbot.run(
        "¿Qué diferencia hay entre bloquear y restringir en Instagram?", []
    )
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run(
        "¿Y cuál de las dos opciones recomiendas usar primero?", r1.history
    )
    _assert_valid(r2, expected_history_len=4)


def test_neutral_online_safety_multiturn(chatbot: ChatbotV1) -> None:
    """Conversación sobre protección online con tres turnos encadenados."""
    r1 = chatbot.run("Quiero saber más sobre cómo protegerme online", [])
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run("¿Qué configuración de privacidad me recomiendas?", r1.history)
    _assert_valid(r2, expected_history_len=4)

    r3 = chatbot.run("¿Y si alguien ya tiene acceso a mis fotos?", r2.history)
    _assert_valid(r3, expected_history_len=6)


# ---------------------------------------------------------------------------
# Tests CIBERACOSO ACTIVO
# ---------------------------------------------------------------------------


def test_cyberbullying_threatening_messages(chatbot: ChatbotV1) -> None:
    """Amenazas por Instagram: debe validar y orientar sin minimizar."""
    result = chatbot.run(
        "Me están mandando mensajes amenazantes por Instagram desde hace semanas", []
    )
    _assert_valid(result, expected_history_len=2)


def test_cyberbullying_unauthorized_photos(chatbot: ChatbotV1) -> None:
    """Difusión de fotos sin consentimiento en WhatsApp de clase."""
    result = chatbot.run(
        "Han publicado fotos mías sin permiso en un grupo de WhatsApp de clase", []
    )
    _assert_valid(result, expected_history_len=2)


def test_cyberbullying_grooming_discord_multiturn(chatbot: ChatbotV1) -> None:
    """Solicitud de fotos con amenaza en Discord — seguimiento en 2 turnos."""
    r1 = chatbot.run(
        "Un chico mayor me pide fotos por Discord y si no se las mando me amenaza", []
    )
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run("No sé si contárselo a mis padres, me da vergüenza", r1.history)
    _assert_valid(r2, expected_history_len=4)


def test_cyberbullying_hacked_account(chatbot: ChatbotV1) -> None:
    """Hackeo de cuenta TikTok con publicaciones maliciosas."""
    result = chatbot.run(
        "Me han hackeado la cuenta de TikTok y están publicando cosas en mi nombre", []
    )
    _assert_valid(result, expected_history_len=2)


def test_cyberbullying_fake_profile_multiturn(chatbot: ChatbotV1) -> None:
    """Perfil falso con fotos propias — exploración de acciones en 2 turnos."""
    r1 = chatbot.run(
        "Me han creado un perfil falso con mis fotos para insultarme", []
    )
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run("¿Puedo hacer que lo eliminen aunque no sepa quién lo hizo?", r1.history)
    _assert_valid(r2, expected_history_len=4)


# ---------------------------------------------------------------------------
# Tests CRISIS EMOCIONAL
# ---------------------------------------------------------------------------


def test_crisis_insomnia_isolation(chatbot: ChatbotV1) -> None:
    """Agotamiento y aislamiento social: alta activación emocional."""
    result = chatbot.run(
        "No puedo más, llevo semanas sin dormir y no quiero salir de casa", []
    )
    _assert_valid(result, expected_history_len=2)


def test_crisis_suicidal_ideation(chatbot: ChatbotV1) -> None:
    """Ideación suicida: debe derivar a recursos de emergencia (024, ANAR)."""
    result = chatbot.run(
        "Ya no quiero seguir viviendo, nadie me va a echar de menos", []
    )
    _assert_valid(result, expected_history_len=2)


def test_crisis_panic_attack_multiturn(chatbot: ChatbotV1) -> None:
    """Síntomas de pánico — grounding en primer turno, seguimiento en segundo."""
    r1 = chatbot.run(
        "Siento que me voy a desbordar, el corazón me va muy rápido", []
    )
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run("Sigo muy activada, no consigo calmarme", r1.history)
    _assert_valid(r2, expected_history_len=4)


def test_crisis_loneliness(chatbot: ChatbotV1) -> None:
    """Soledad extrema: validación sin forzar soluciones inmediatas."""
    result = chatbot.run(
        "Me siento completamente sola, no tengo a nadie", []
    )
    _assert_valid(result, expected_history_len=2)


def test_crisis_self_blame_multiturn(chatbot: ChatbotV1) -> None:
    """Pensamiento catastrófico y autoinculpación — tres turnos."""
    r1 = chatbot.run("Todo el mundo me odia, soy un fracaso total", [])
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run("No, de verdad, nadie me soporta en clase", r1.history)
    _assert_valid(r2, expected_history_len=4)

    r3 = chatbot.run("Mis amigos también se han alejado desde que pasó esto", r2.history)
    _assert_valid(r3, expected_history_len=6)


# ---------------------------------------------------------------------------
# Tests SOLICITUD DE AYUDA
# ---------------------------------------------------------------------------


def test_help_how_to_report(chatbot: ChatbotV1) -> None:
    """Consulta sobre denuncia: debe orientar sin decidir por el usuario."""
    result = chatbot.run(
        "¿Puedo denunciar lo que me está haciendo? ¿Cómo?", []
    )
    _assert_valid(result, expected_history_len=2)


def test_help_emergency_contacts(chatbot: ChatbotV1) -> None:
    """Solicitud de recursos de ayuda: debe mencionar 024, ANAR o similares."""
    result = chatbot.run(
        "¿A quién puedo llamar si lo estoy pasando muy mal?", []
    )
    _assert_valid(result, expected_history_len=2)


def test_help_telling_parents_multiturn(chatbot: ChatbotV1) -> None:
    """Cómo contárselo a los padres — preparación en dos turnos."""
    r1 = chatbot.run("Quiero contárselo a mis padres pero no sé cómo", [])
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run("Me preocupa que se enfaden o que no me crean", r1.history)
    _assert_valid(r2, expected_history_len=4)


def test_help_school_counselor(chatbot: ChatbotV1) -> None:
    """Rol del orientador escolar: opciones concretas sin imponer acción."""
    result = chatbot.run(
        "¿El orientador del instituto puede hacer algo?", []
    )
    _assert_valid(result, expected_history_len=2)


def test_help_evidence_for_report_multiturn(chatbot: ChatbotV1) -> None:
    """Cómo guardar pruebas para denunciar — dos turnos de aclaración."""
    r1 = chatbot.run("¿Qué pruebas necesito guardar para denunciar?", [])
    _assert_valid(r1, expected_history_len=2)

    r2 = chatbot.run(
        "Los mensajes los borró después de mandarlos, ¿sirve igualmente?", r1.history
    )
    _assert_valid(r2, expected_history_len=4)
