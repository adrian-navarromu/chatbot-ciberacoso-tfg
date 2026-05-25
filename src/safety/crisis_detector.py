"""
Módulo de detección de crisis (capa de seguridad PAP).

Se ejecuta ANTES del clasificador emocional y de cualquier otro componente
del pipeline. Implementa el protocolo de Primeros Auxilios Psicológicos (PAP)
siguiendo las directrices PNUD/Integra 2022.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

# Configuración del logger
log = logging.getLogger(__name__)

# Keywords de crisis
KEYWORDS_CRISIS_HIGH: list[str] = [
    r"no quiero seguir viviendo",
    r"no quiero seguir aqu[ií]",
    r"quiero morirme",
    r"quiero morir",
    r"me quiero morir",
    r"quitarme la vida",
    r"quitar(?:me|se)?\s+la\s+vida",
    r"quitarme de en medio",
    r"hacerme da[ñn]o",
    r"hacerme algo",
    r"suicidarme",
    r"suicidio",
    r"me van a matar",
    r"me van a pegar",
    r"me va a hacer da[ñn]o",
    r"me amenaza con hacerme da[ñn]o",
    r"me amenaza con matarme",
    r"no tiene sentido seguir",
    r"qu[eé] sentido tiene seguir",
    r"para qu[eé] seguir",
    r"todo ser[ií]a mejor sin m[ií]",
    r"nadie me echar[ií]a de menos",
    r"\bcortarme\b",
    r"\bpastillas\b.*\btomarme\b",
    r"\btomarme\b.*\bpastillas\b",
    r"voy a hacer(?:me)? algo",
    r"acabar con(?:migo| mi vida| todo)",
]

KEYWORDS_CRISIS_MEDIUM: list[str] = [
    r"no puedo m[aá]s",
    r"ya no aguanto m[aá]s",
    r"no aguanto m[aá]s",
    r"estoy desesperada?o?",
    r"me siento completamente sola?",
    r"no tengo a nadie",
    r"nadie me ayuda",
    r"nadie me escucha",
    r"me van a agredir",
]


# Respuestas PAP
CRISIS_RESPONSE_HIGH: str = (
    "Lo que describes es muy serio y entiendo que estás pasando por un momento "
    "muy difícil. No estás solo/a en esto.\n\n"
    "Necesito pedirte que contactes ahora mismo con alguien que puede ayudarte "
    "de verdad: llama al 024 (Línea de atención a la conducta suicida, gratuita "
    "y disponible las 24 horas), al 900 202 010 (ANAR, para menores, también "
    "gratuita, confidencial y disponible 24 horas), o al 112 si sientes que "
    "estás en peligro físico inmediato. Estas personas están preparadas para "
    "escucharte sin juzgarte.\n\n"
    "¿Hay alguien contigo ahora mismo?"
)

CRISIS_RESPONSE_MEDIUM: str = (
    "Escucho que estás pasando por algo muy duro y tiene sentido que te sientas "
    "así. Quiero ayudarte a entender mejor lo que está pasando.\n\n"
    "¿Puedes contarme un poco más sobre lo que está ocurriendo? Si en algún "
    "momento sientes que la situación se vuelve insoportable, el 024 está "
    "disponible de forma gratuita las 24 horas."
)

# Dataclass resultado
@dataclass
class CrisisResult:
    """Resultado de la detección de crisis."""

    level: str
    triggered_keywords: list[str] = field(default_factory=list)
    requires_immediate_response: bool = False
    response: str | None = None


# Detector
class CrisisDetector:
    """
    Primera capa de seguridad del chatbot.

    Detecta ideación autolesiva o malestar extremo mediante coincidencia
    de patrones regex y devuelve una respuesta PAP estructurada.
    """

    # Expresiones que indican uso coloquial/metafórico y NO deben activar HIGH
    _FALSE_POSITIVE_GUARDS: list[re.Pattern[str]] = []

    def __init__(self) -> None:
        """Precompila todos los patrones de crisis."""
        self._high_patterns: list[tuple[str, re.Pattern[str]]] = [
            (kw, re.compile(kw, flags=re.IGNORECASE | re.UNICODE))
            for kw in KEYWORDS_CRISIS_HIGH
        ]
        self._medium_patterns: list[tuple[str, re.Pattern[str]]] = [
            (kw, re.compile(kw, flags=re.IGNORECASE | re.UNICODE))
            for kw in KEYWORDS_CRISIS_MEDIUM
        ]
        # Guarda contextuales que neutralizan falsos positivos en HIGH
        self._false_positive_patterns: list[re.Pattern[str]] = [
            re.compile(p, flags=re.IGNORECASE | re.UNICODE)
            for p in [
                r"de verg[üu]enza",
                r"de risa",
                r"de asco",
                r"de pena",
                r"de susto",
                r"de miedo",
                r"de amor",
                r"de alegr[ií]a",
            ]
        ]

    def _is_false_positive(self, text: str) -> bool:
        """
        Devuelve True si el texto contiene una expresión coloquial que indica
        uso metafórico (p.ej. 'quiero morirme de vergüenza').
        """
        return any(p.search(text) for p in self._false_positive_patterns)

    def detect(self, text: str) -> CrisisResult:
        """
        Analiza el texto y determina el nivel de crisis.

        Reglas de activación:
        - Cualquier HIGH keyword --> nivel HIGH (salvo falso positivo contextual)
        - 2 o más MEDIUM keywords --> nivel HIGH
        - 1 MEDIUM keyword --> nivel MEDIUM
        - Ninguna --> nivel NONE
        """
        triggered_high: list[str] = [
            kw for kw, pat in self._high_patterns if pat.search(text)
        ]

        # Filtrar falsos positivos contextuales (expresiones coloquiales)
        if triggered_high and self._is_false_positive(text):
            log.info("Falso positivo de crisis evitado por contexto pragmático.")
            triggered_high = []

        triggered_medium: list[str] = [
            kw for kw, pat in self._medium_patterns if pat.search(text)
        ]

        if triggered_high or len(triggered_medium) >= 2:
            level = "HIGH"
            triggered = triggered_high + triggered_medium
            log.warning(f"¡CRISIS HIGH DETECTADA! Triggers: {triggered}")
        elif len(triggered_medium) == 1:
            level = "MEDIUM"
            triggered = triggered_medium
            log.info(f"Crisis MEDIUM detectada. Triggers: {triggered}")
        else:
            return CrisisResult(
                level="NONE",
                triggered_keywords=[],
                requires_immediate_response=False,
                response=None,
            )

        return CrisisResult(
            level=level,
            triggered_keywords=triggered,
            requires_immediate_response=(level == "HIGH"),
            response=self.get_response(level),
        )

    def get_response(self, level: str) -> str:
        """Devuelve la respuesta PAP correspondiente al nivel de crisis."""
        if level == "HIGH":
            return CRISIS_RESPONSE_HIGH
        if level == "MEDIUM":
            return CRISIS_RESPONSE_MEDIUM
        return ""
