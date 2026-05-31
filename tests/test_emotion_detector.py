"""
Tests de EmotionDetector.

Requieren que exista un modelo fine-tuned en models/emotion_classifier/
(directorio con config.json, pytorch_model.bin / model.safetensors y tokenizer).

Ejecución:
    pytest tests/test_emotion_detector.py -v
    pytest tests/test_emotion_detector.py -v --model-path models/emotion_classifier/robertuito
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.emotion.emotion_detector import EmotionDetector, EmotionResult


# ---------------------------------------------------------------------------
# Fixtures  (model_path y pytest_addoption viven en conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def detector(model_path: Path) -> EmotionDetector:
    """Instancia única de EmotionDetector reutilizada en toda la sesión."""
    if not model_path.exists():
        pytest.skip(
            f"Modelo no encontrado en {model_path}. "
            "Entrena primero con python src/emotion/train_classifier.py"
        )
    return EmotionDetector(model_path=model_path)


@pytest.fixture(scope="session")
def detector_cpu(model_path: Path) -> EmotionDetector:
    """Instancia de EmotionDetector forzada a CPU."""
    if not model_path.exists():
        pytest.skip(f"Modelo no encontrado en {model_path}.")
    return EmotionDetector(model_path=model_path, device="cpu")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def assert_valid_result(result: EmotionResult) -> None:
    """Verifica que EmotionResult tiene todos los campos correctamente formados."""
    assert isinstance(result, EmotionResult), "El resultado debe ser EmotionResult"
    assert isinstance(result.label, str) and result.label != "", "label debe ser str no vacío"
    assert 0.0 <= result.confidence <= 1.0, "confidence debe estar en [0, 1]"
    assert isinstance(result.all_scores, dict), "all_scores debe ser dict"
    assert len(result.all_scores) == 7, "all_scores debe tener 7 clases"
    assert isinstance(result.is_low_confidence, bool), "is_low_confidence debe ser bool"
    total = sum(result.all_scores.values())
    assert abs(total - 1.0) < 1e-4, f"Probabilidades no suman 1: {total}"
    valid_labels = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "others"}
    assert set(result.all_scores.keys()) == valid_labels, (
        f"Etiquetas incorrectas: {set(result.all_scores.keys())}"
    )
    if result.is_low_confidence:
        assert result.label == "others", "Si is_low_confidence=True, label debe ser 'others'"


# ---------------------------------------------------------------------------
# Tests por emoción (uno por clase)
# Las frases han sido validadas contra el modelo robertuito fine-tuneado.
# ---------------------------------------------------------------------------


class TestEmotionDetection:

    def test_anger(self, detector: EmotionDetector) -> None:
        """Detecta anger en un mensaje de rabia explícita."""
        result = detector.detect(
            "Estoy harto, me tienes hasta las narices. ¡No te soporto más!"
        )
        assert_valid_result(result)
        assert result.label == "anger", f"Esperaba anger, obtuvo {result.label}"

    def test_disgust(self, detector: EmotionDetector) -> None:
        """Detecta disgust en un mensaje de náuseas y desagrado."""
        result = detector.detect(
            "Me da náuseas, es lo más desagradable que he visto."
        )
        assert_valid_result(result)
        assert result.label == "disgust", f"Esperaba disgust, obtuvo {result.label}"

    def test_fear(self, detector: EmotionDetector) -> None:
        """Detecta fear en un mensaje de miedo relacionado con ciberacoso."""
        result = detector.detect(
            "Tengo miedo de ir al insti mañana, me están amenazando por WhatsApp."
        )
        assert_valid_result(result)
        assert result.label == "fear", f"Esperaba fear, obtuvo {result.label}"

    def test_joy(self, detector: EmotionDetector) -> None:
        """Detecta joy en un mensaje de alegría."""
        result = detector.detect(
            "¡Qué alegría tan grande! Ha sido el mejor día de mi vida, estoy feliz."
        )
        assert_valid_result(result)
        assert result.label == "joy", f"Esperaba joy, obtuvo {result.label}"

    def test_sadness(self, detector: EmotionDetector) -> None:
        """Detecta sadness en un mensaje de tristeza."""
        result = detector.detect(
            "Estoy muy triste, me siento solo y nadie me hace caso en clase."
        )
        assert_valid_result(result)
        assert result.label == "sadness", f"Esperaba sadness, obtuvo {result.label}"

    def test_surprise(self, detector: EmotionDetector) -> None:
        """Detecta surprise en un mensaje de sorpresa."""
        result = detector.detect(
            "¡No me lo puedo creer! Jamás habría pensado que algo así podía pasar."
        )
        assert_valid_result(result)
        assert result.label == "surprise", f"Esperaba surprise, obtuvo {result.label}"

    def test_others(self, detector: EmotionDetector) -> None:
        """Detecta others en un mensaje informativo sin carga emocional marcada."""
        result = detector.detect(
            "El partido de fútbol empieza a las ocho de la noche en el estadio."
        )
        assert_valid_result(result)
        assert result.label == "others", f"Esperaba others, obtuvo {result.label}"


# ---------------------------------------------------------------------------
# Test de baja confianza
# ---------------------------------------------------------------------------


class TestLowConfidence:

    def test_ambiguous_text_low_confidence(self, detector: EmotionDetector) -> None:
        """Una frase de emociones contradictorias debe activar is_low_confidence.

        El modelo no logra decantarse por una emoción dominante y distribuye
        la probabilidad entre varias clases, dejando la confianza < 0.4.
        """
        result = detector.detect("¿por qué no sé qué siento?")
        assert_valid_result(result)
        assert result.is_low_confidence is True, (
            f"Se esperaba baja confianza (< 0.4), obtenida={result.confidence:.3f}"
        )
        assert result.label == "others", "Con baja confianza el label debe ser 'others'"


# ---------------------------------------------------------------------------
# Tests de batch
# ---------------------------------------------------------------------------


class TestBatchDetection:

    def test_batch_three_texts(self, detector: EmotionDetector) -> None:
        """Procesa 3 frases en lote y verifica que devuelve 3 resultados válidos."""
        texts = [
            "¡Qué feliz estoy, ha salido todo perfecto!",
            "Me están haciendo la vida imposible, tengo miedo.",
            "El tren llega a las doce.",
        ]
        results = detector.detect_batch(texts)
        assert len(results) == len(texts), (
            f"Se esperaban {len(texts)} resultados, se obtuvieron {len(results)}"
        )
        for i, result in enumerate(results):
            assert_valid_result(result)

    def test_batch_single_item(self, detector: EmotionDetector) -> None:
        """Un batch de un solo elemento debe devolver una lista de longitud 1."""
        results = detector.detect_batch(["Estoy enfadado."])
        assert len(results) == 1
        assert_valid_result(results[0])


# ---------------------------------------------------------------------------
# Test de CPU
# ---------------------------------------------------------------------------


class TestCPUDevice:

    def test_detect_on_cpu(self, detector_cpu: EmotionDetector) -> None:
        """El detector funciona correctamente forzando device=cpu."""
        assert detector_cpu.device == torch.device("cpu"), "El detector debería usar CPU"
        result = detector_cpu.detect("Estoy muy triste y me siento abandonado.")
        assert_valid_result(result)