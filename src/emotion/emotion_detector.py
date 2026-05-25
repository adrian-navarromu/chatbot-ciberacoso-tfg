"""
Módulo de inferencia del clasificador emocional para el pipeline del chatbot.

Uso:
    detector = EmotionDetector("models/emotion_classifier/maria")
    result = detector.detect("Me están amenazando y tengo mucho miedo")
    print(result.label, result.confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Dataclass de resultado
@dataclass
class EmotionResult:
    """Resultado de la clasificación emocional para un texto.

    Atributos:
        label: Emoción predicha. Se fuerza a 'others' si is_low_confidence.
        confidence: Probabilidad de la clase más probable (0-1).
        all_scores: Probabilidad de cada una de las 7 emociones.
        is_low_confidence: True cuando confidence < LOW_CONFIDENCE_THRESHOLD.
    """

    label: str
    confidence: float
    all_scores: dict[str, float]
    is_low_confidence: bool


# Umbral de baja confianza
LOW_CONFIDENCE_THRESHOLD = 0.4


# EmotionDetector
class EmotionDetector:
    """Clasificador emocional basado en un transformer fine-tuneado sobre EmoEvent.

    Solo detecta emociones (anger, disgust, fear, joy, sadness, surprise,
    others). La detección de ofensividad se implementará en V2 como
    clasificador separado.

    Args:
        model_path: Ruta al directorio con el modelo fine-tuned y el tokenizer.
        device: 'cuda', 'cpu' o None (autodetección).
        max_length: Longitud máxima de tokenización (debe coincidir con la del entrenamiento).
    """

    def __init__(self, model_path: str | Path, device: str | None = None, max_length: int = 128) -> None:
        self.model_path = Path(model_path)
        self.max_length = max_length

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

    def _load_model(self) -> None:
        """Carga el tokenizer y el modelo desde disco y los mueve al device."""
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path)).to(self.device)
        self.model.eval()

        # Construir mapas id <-> etiqueta desde la configuración del modelo
        self.id2label: dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self.label2id: dict[str, int] = {
            v: int(k) for k, v in self.model.config.id2label.items()
        }

    # Inferencia sobre un único texto
    def detect(self, text: str) -> EmotionResult:
        """Clasifica la emoción principal de un texto en español.

        Args:
            text: Texto a clasificar (tweet, mensaje, etc.).

        Returns:
            EmotionResult con la emoción predicha y las probabilidades.
        """
        return self.detect_batch([text])[0]

    # Inferencia en lote
    def detect_batch(self, texts: list[str]) -> list[EmotionResult]:
        """Clasifica la emoción de una lista de textos en una sola pasada.

        Args:
            texts: Lista de textos a clasificar.

        Returns:
            Lista de EmotionResult en el mismo orden que la entrada.
        """
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu()

        results: list[EmotionResult] = []
        for prob_row in probs:
            all_scores = {
                self.id2label[i]: float(prob_row[i]) for i in range(len(prob_row))
            }
            confidence = float(prob_row.max())
            predicted_id = int(prob_row.argmax())
            predicted_label = self.id2label[predicted_id]

            is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
            label = "others" if is_low_confidence else predicted_label

            results.append(
                EmotionResult(
                    label=label,
                    confidence=confidence,
                    all_scores=all_scores,
                    is_low_confidence=is_low_confidence,
                )
            )
        return results
