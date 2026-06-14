"""
Módulo de memoria emocional (EmotionalMemoryTracker).

Rastrea la evolución afectiva del usuario turno a turno y genera
metadatos para inyectar en el prompt dinámico.

Inspirado en el patrón arquitectónico del DMCGES (Pan et al., 2026, UMU),
versión simplificada para texto sin modalidad audio/visual.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

# Configuración del logger
log = logging.getLogger(__name__)


# Constantes de dominio emocional
EMOTION_LABELS: list[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "others",
]

# Convierte clases discretas en un espacio continuo simple (0 a 2) para poder calcular derivadas (tendencias) matemáticas en la ventana de tiempo.
NEGATIVE_HIGH: list[str] = ["fear", "anger", "disgust"]    # 0 - malestar intenso
NEGATIVE_LOW: list[str] = ["sadness", "surprise"]          # 1 - malestar moderado
POSITIVE: list[str] = ["joy", "others"]                    # 2 - neutro o positivo

_VALENCE_MAP: dict[str, float] = (
    {e: 0.0 for e in NEGATIVE_HIGH}
    | {e: 1.0 for e in NEGATIVE_LOW}
    | {e: 2.0 for e in POSITIVE}
)


def _valence(emotion: str) -> float:
    """Devuelve el valor numérico de la emoción. Default 1.0 (neutro/moderado) por seguridad."""
    return _VALENCE_MAP.get(emotion, 1.0)


# Dataclass resultado
@dataclass
class TrendResult:
    """Resultado del análisis de tendencia emocional para un turno dado."""
    dominant_emotion: str
    current_emotion: str
    trend: str
    window_emotions: list[str] = field(default_factory=list)
    turn_count: int = 0
    prompt_context: str = ""


# Clase principal
class EmotionalMemoryTracker:
    """
    Memoria emocional de sesión con detección de tendencias por ventana deslizante.

    Implementa una versión heurística (sin GRU real) del rastreo afectivo del DMCGES,
    operando únicamente sobre la modalidad texto para mantener el estado
    latente de la conversación a corto/medio plazo.
    """

    def __init__(self, window_size: int = 6) -> None:
        """
        Inicializa la memoria.
        Args:
            window_size: Tamaño de la memoria a corto plazo (turnos a evaluar).
                         6 turnos = aprox. los últimos 3 intercambios completos.
        """
        self.window_size: int = window_size
        self.emotion_history: list[str] = []
        self.turn_count: int = 0


    # Registro
    def update(self, emotion: str, confidence: float) -> None:
        """
        Registra la emoción del turno actual.

        Filtro de incertidumbre: Si confidence < 0.4, registra 
        'others' para evitar contaminar la memoria con falsos positivos del clasificador.
        """
        effective_emotion = emotion if confidence >= 0.4 else "others"
        self.emotion_history.append(effective_emotion)
        self.turn_count += 1


    # Consultas sobre la ventana
    def get_window(self) -> list[str]:
        """Devuelve los últimos window_size turnos."""
        return self.emotion_history[-self.window_size :]

    def get_dominant_emotion(self) -> str:
        """
        Calcula la emoción más frecuente en la ventana actual.
        Sistema de desempate temporal: En caso de empate de frecuencias, 
        prioriza la emoción que haya ocurrido más recientemente.
        """
        window = self.get_window()
        if not window:
            return "others"
        
        # Desempate por posición más reciente: índice mayor en window = más reciente
        counts = Counter(window)
        max_count = max(counts.values())
        candidates = [e for e, c in counts.items() if c == max_count]

        if len(candidates) == 1:
            return candidates[0]
        
        # Desempate iterando desde el más reciente (final de la lista) hacia atrás
        for emotion in reversed(window):
            if emotion in candidates:
                return emotion
        
        return window[-1]  # Fallback de seguridad (unreachable)


    # Tendencia
    def detect_trend(self) -> str:
        """
        Detecta la tendencia calculando el gradiente de valencia entre la primera 
        y la segunda mitad de la ventana de memoria.

        Mapeo: NEGATIVE_HIGH=0, NEGATIVE_LOW=1, POSITIVE=2.
        Umbral de cambio significativo: ±0.3.
        """
        window = self.get_window()
        if len(window) < 2:
            return "estable"

        mid = len(window) // 2
        first_half = window[:mid]
        second_half = window[mid:]

        first_mean = sum(_valence(e) for e in first_half) / len(first_half)
        second_mean = sum(_valence(e) for e in second_half) / len(second_half)

        diff = second_mean - first_mean
        if diff > 0.3:
            return "mejora"
        if diff < -0.3:
            return "deterioro"
        return "estable"

    # Inyección de Contexto para el LLM (Prompt Engineering)
    def get_trend_result(self) -> TrendResult:
        """Genera el TrendResult completo con todos los campos."""
        current = self.emotion_history[-1] if self.emotion_history else "others"
        return TrendResult(
            dominant_emotion=self.get_dominant_emotion(),
            current_emotion=current,
            trend=self.detect_trend(),
            window_emotions=self.get_window(),
            turn_count=self.turn_count,
            prompt_context=self.get_prompt_context(),
        )

    def get_prompt_context(self) -> str:
        """
        Genera la cadena descriptiva para inyectar en el system prompt.

        - Turno 1: primera interacción.
        - Turnos 2-3: lista simple de emociones registradas.
        - Turno 4+: análisis completo con dominante, tendencia y recuento.
        """
        if self.turn_count == 0:
            return ""

        current = self.emotion_history[-1]

        if self.turn_count == 1:
            return f"Primera interacción. Emoción actual: {current}."

        if self.turn_count <= 3:
            lista = ", ".join(self.emotion_history)
            return f"Emociones registradas: {lista}. Emoción actual: {current}."

        # turn_count >= 4
        dominant = self.get_dominant_emotion()
        window = self.get_window()
        n_dominant = window.count(dominant)
        trend_raw = self.detect_trend()

        trend_phrases = {
            "mejora": "el estado emocional muestra signos de mejora",
            "deterioro": "el estado emocional muestra signos de deterioro, presta especial atención",
            "estable": "el estado emocional se mantiene estable",
        }
        trend_phrase = trend_phrases[trend_raw]

        return (
            f"Tendencia emocional: el usuario mostró {dominant} como emoción "
            f"predominante ({n_dominant} de los últimos {self.window_size} turnos) "
            f"y actualmente expresa {current}. "
            f"Tendencia: {trend_phrase}."
        )


    # Gestión de sesión
    def reset(self) -> None:
        """Reinicia la memoria al comenzar una nueva sesión."""
        self.emotion_history = []
        self.turn_count = 0

    def to_dict(self) -> dict:
        """Serializa el estado para el panel de debug de Gradio."""
        return {
            "turn_count": self.turn_count,
            "window": self.get_window(),
            "dominant_emotion": self.get_dominant_emotion(),
            "trend": self.detect_trend(),
            "full_history": self.emotion_history,
        }
