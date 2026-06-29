"""
Módulo de detección de crisis (capa de seguridad PAP).

Se ejecuta ANTES del clasificador emocional y de cualquier otro componente
del pipeline. Implementa el protocolo de Primeros Auxilios Psicológicos (PAP)
siguiendo las directrices PNUD/Integra 2022.

Niveles y comportamiento:
    HIGH → respuesta determinista fija según SUBTYPE (100% determinista).
           requires_generation=False, response=<plantilla>.
    MEDIUM → marca para generación SLM con prompt de contención PAP.
             requires_generation=True, suggested_pillar=4, response=None.
    NONE → no activa failsafe.

HIGH subtypes (prioridad: IDEACION_PROPIA > AMENAZA_FISICA > TERCERO_RIESGO):
    IDEACION_PROPIA  — ideación suicida/autolesión del propio usuario.
    AMENAZA_FISICA   — amenaza física de un tercero al usuario.
    TERCERO_RIESGO   — el usuario habla de OTRA persona en riesgo inmediato.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

# Configuración del logger
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords HIGH — agrupadas por subtipo
# ---------------------------------------------------------------------------

KEYWORDS_CRISIS_HIGH_IDEACION: list[str] = [
    # Ideación suicida directa
    r"no quiero seguir viviendo",
    r"no quiero seguir aqu[ií]",
    r"quiero morirme",
    r"quiero morir",
    r"me quiero morir",
    r"quitarme la vida",
    r"quitar(?:me|se)?\s+la\s+vida",
    r"quitarme de en medio",
    # Autolesión
    r"hacerme da[ñn]o",
    r"hacerme algo",
    r"\bcortarme\b",
    r"me\s+(?:voy\s+a\s+)?cortar(?:me)?",   # Q081: "me voy a cortar"
    r"\bpastillas\b.*\btomar(?:me(?:las)?)?\b",  # Q075: cubre tomarme y tomármelas
    r"\bpastillas\b.*tom[aá]rmelas",             # Q075: forma acentuada
    r"\btomar(?:me(?:las)?)?\b.*\bpastillas\b",
    r"voy a hacer(?:me)? algo",
    # Palabras clave de suicidio
    r"suicidarme",
    r"suicidio",
    # Desesperanza vital activa
    r"no tiene sentido seguir",
    r"qu[eé] sentido tiene seguir",
    r"para qu[eé] seguir",
    r"todo ser[ií]a mejor sin m[ií]",
    r"nadie me echar[ií]a de menos",
    r"acabar con(?:migo| mi vida| todo)",
]

KEYWORDS_CRISIS_HIGH_AMENAZA: list[str] = [
    # Amenaza física de un tercero AL usuario
    r"me van a matar",
    r"me van a pegar",
    r"me va a hacer da[ñn]o",
    r"me amenaza con hacerme da[ñn]o",
    r"me amenaza con matarme",
    r"hacer(?:me)?\s+da[ñn]o\s+f[ií]sico",              # Q078: daño físico explícito
    r"me\s+(?:dijo|dijeron)\s+que\s+.{0,50}me\s+(?:iba(?:n)?\s+a\s+)?matar(?:me)?",  # Q089: pasado cond.
]

KEYWORDS_CRISIS_HIGH_TERCERO: list[str] = [
    # Otra persona en crisis — Q088: modificador opcional antes del sustantivo; vivir/viviendo
    r"(?:mi|un[ao]?)\s+(?:\w+\s+)?(?:amig[ao]|compañer[ao]|herman[ao]|primo?|pareja|chic[ao]|novi[ao])\s+[^.]{0,60}(?:no\s+quiere?\s+(?:seguir\s+)?viv(?:ir|iendo)|quiere?\s+morir(?:se)?|se\s+quiere?\s+matar|quiere?\s+quitarse\s+la\s+vida)",
    # "me dijo que" + ideación de tercero
    r"me\s+dijo\s+que\s+(?:no\s+quiere?\s+(?:seguir\s+)?vivir|quiere?\s+morir(?:se)?|se\s+va\s+a\s+(?:matar|hacer\s+algo|cortar))",
    # Autolesión activa en tercera persona
    r"se\s+(?:está\s+)?haciendo\s+da[ñn]o\s+(?:a\s+sí\s+mism[ao])?",
    r"se\s+(?:está\s+)?cortando",
    # Plan concreto con pastillas (tercero) — Q093: cubre "las va a usar"
    r"tiene\s+pastillas\s+(?:y|guardadas|para)\s+(?:usarlas|tomárselas|hacerse\s+algo)",
    r"tiene\s+pastillas.{0,40}va\s+a\s+(?:usar(?:las)?|tomar(?:las|melas)?)",
    # Ideación activa de tercero con plan/fecha
    r"(?:está|estaba|dice\s+que)\s+(?:pensando\s+en\s+)?suicidar(?:se|la|lo)",
]

# Keywords MEDIUM — malestar extremo sin ideación explícita
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

# ---------------------------------------------------------------------------
# Respuestas HIGH — deterministas por subtipo
# ---------------------------------------------------------------------------

CRISIS_RESPONSE_IDEACION: str = (
    "Lo que describes es muy serio y entiendo que estás pasando por un momento "
    "muy difícil. No estás solo en esto.\n\n"
    "Necesito pedirte que contactes ahora mismo con alguien que puede ayudarte "
    "de verdad: llama al 024 (Línea de atención a la conducta suicida, gratuita "
    "y disponible las 24 horas), al 900 202 010 (ANAR, para menores, también "
    "gratuita, confidencial y disponible 24 horas), o al 112 si sientes que "
    "estás en peligro físico inmediato. Estas personas están preparadas para "
    "escucharte sin juzgarte.\n\n"
    "¿Hay alguien contigo ahora mismo?"
)

CRISIS_RESPONSE_AMENAZA: str = (
    "Recibir una amenaza como esa es muy serio y entiendo que sientes miedo. "
    "Tu seguridad es lo primero.\n\n"
    "Si crees que el peligro es inmediato, llama al 112 ahora mismo: están "
    "disponibles las 24 horas. También puedes buscar a un adulto de confianza "
    "— un familiar, tutor o profesor — y contarle lo que está pasando para que "
    "puedan protegerte. Tienes derecho a estar a salvo.\n\n"
    "¿Puedes hablar con alguien de confianza ahora mismo?"
)

CRISIS_RESPONSE_TERCERO: str = (
    "Gracias por contarme esto. Lo que describes sobre tu amigo es serio y "
    "es importante que alguien adulto lo sepa hoy mismo.\n\n"
    "La mejor forma de ayudar a alguien que está en ese momento es no guardarlo "
    "en silencio: cuéntaselo a un familiar, tutor o al centro escolar. También "
    "puedes llamar al 024 o al 900 202 010 (ANAR) para pedir consejo sobre cómo "
    "actuar — son gratuitos, confidenciales y están para esto.\n\n"
    "¿Hay un adulto de confianza al que puedas contárselo hoy?"
)

# Alias para compatibilidad con tests e imports existentes
CRISIS_RESPONSE_HIGH: str = CRISIS_RESPONSE_IDEACION

# Respuesta MEDIUM — YA NO SE USA (MEDIUM genera con SLM). Conservada para referencia.
CRISIS_RESPONSE_MEDIUM: str = (
    "Escucho que estás pasando por algo muy duro y tiene sentido que te sientas "
    "así. Quiero ayudarte a entender mejor lo que está pasando.\n\n"
    "¿Puedes contarme un poco más sobre lo que está ocurriendo? Si en algún "
    "momento sientes que la situación se vuelve insoportable, el 024 está "
    "disponible de forma gratuita las 24 horas."
)


# ---------------------------------------------------------------------------
# Dataclass resultado
# ---------------------------------------------------------------------------

@dataclass
class CrisisResult:
    """Resultado de la detección de crisis.

    Atributos:
        level: Nivel de crisis ('HIGH', 'MEDIUM', 'NONE').
        subtype: Subtipo de HIGH ('IDEACION_PROPIA', 'AMENAZA_FISICA',
                 'TERCERO_RIESGO', 'NONE').
        triggered_keywords: Keywords que activaron la detección.
        requires_immediate_response: True si HIGH (respuesta determinista inmediata).
        requires_generation: True si MEDIUM (el pipeline debe generar con SLM + PAP prompt).
        suggested_pillar: Pilar RAG sugerido para la generación MEDIUM (4 = PAP/recursos).
        response: Texto de respuesta PAP fija (solo HIGH). None en MEDIUM y NONE.
    """
    level: str
    subtype: str = "NONE"
    triggered_keywords: list[str] = field(default_factory=list)
    requires_immediate_response: bool = False
    requires_generation: bool = False
    suggested_pillar: int | None = None
    response: str | None = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CrisisDetector:
    """
    Primera capa de seguridad del chatbot.

    Detecta ideación autolesiva, amenaza física o tercero en riesgo mediante
    coincidencia de patrones regex y devuelve un CrisisResult estructurado.

    HIGH → respuesta determinista fija según subtipo (100% determinista).
    MEDIUM → marca para generación SLM con suggested_pillar=4.
    """

    def __init__(self) -> None:
        """Precompila todos los patrones de crisis."""
        self._ideacion_patterns: list[tuple[str, re.Pattern[str]]] = [
            (kw, re.compile(kw, flags=re.IGNORECASE | re.UNICODE))
            for kw in KEYWORDS_CRISIS_HIGH_IDEACION
        ]
        self._amenaza_patterns: list[tuple[str, re.Pattern[str]]] = [
            (kw, re.compile(kw, flags=re.IGNORECASE | re.UNICODE))
            for kw in KEYWORDS_CRISIS_HIGH_AMENAZA
        ]
        self._tercero_patterns: list[tuple[str, re.Pattern[str]]] = [
            (kw, re.compile(kw, flags=re.IGNORECASE | re.UNICODE))
            for kw in KEYWORDS_CRISIS_HIGH_TERCERO
        ]
        self._medium_patterns: list[tuple[str, re.Pattern[str]]] = [
            (kw, re.compile(kw, flags=re.IGNORECASE | re.UNICODE))
            for kw in KEYWORDS_CRISIS_MEDIUM
        ]
        # Guards contextuales que neutralizan falsos positivos en IDEACION y MEDIUM
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
                # Guard académico/escolar (Q087): cancela MEDIUM cuando el malestar
                # es claramente por presión de estudios, no por ciberacoso
                r"(?:el|este|ese|mi)\s+(?:examen|trabajo\s+de\s+\w+|trabajo\s+final|estudio|profesor|clase)",
            ]
        ]

    def _is_false_positive(self, text: str) -> bool:
        """True si el texto contiene expresión coloquial metafórica."""
        return any(p.search(text) for p in self._false_positive_patterns)

    def detect(self, text: str) -> CrisisResult:
        """Analiza el texto y determina el nivel de crisis.

        Reglas de activación:
        - Cualquier IDEACION keyword (sin falso positivo) → HIGH (IDEACION_PROPIA)
        - Cualquier AMENAZA keyword → HIGH (AMENAZA_FISICA)
        - Cualquier TERCERO keyword → HIGH (TERCERO_RIESGO)
        - Prioridad: IDEACION > AMENAZA > TERCERO
        - 2 o más MEDIUM keywords → HIGH (IDEACION_PROPIA por defecto)
        - 1 MEDIUM keyword → MEDIUM (requires_generation=True, response=None)
        - Ninguna → NONE

        Args:
            text: Mensaje del usuario.

        Returns:
            CrisisResult con level, subtype, requires_generation y response.
        """
        triggered_ideacion = [kw for kw, pat in self._ideacion_patterns if pat.search(text)]

        # Guard contextual solo para IDEACION (expresiones coloquiales)
        if triggered_ideacion and self._is_false_positive(text):
            log.info("Falso positivo de crisis evitado por contexto pragmático.")
            triggered_ideacion = []

        triggered_amenaza = [kw for kw, pat in self._amenaza_patterns if pat.search(text)]
        triggered_tercero = [kw for kw, pat in self._tercero_patterns if pat.search(text)]
        triggered_medium = [kw for kw, pat in self._medium_patterns if pat.search(text)]
        # Guard contextual: cancela MEDIUM en contexto académico/escolar (Q087 falso positivo)
        if triggered_medium and self._is_false_positive(text):
            log.info("Guard contextual cancela MEDIUM: contexto coloquial/académico.")
            triggered_medium = []

        # Determinar subtype HIGH (prioridad: IDEACION > AMENAZA > TERCERO)
        if triggered_ideacion:
            subtype = "IDEACION_PROPIA"
            triggered_high = triggered_ideacion
        elif triggered_amenaza:
            subtype = "AMENAZA_FISICA"
            triggered_high = triggered_amenaza
        elif triggered_tercero:
            subtype = "TERCERO_RIESGO"
            triggered_high = triggered_tercero
        else:
            subtype = "NONE"
            triggered_high = []

        if triggered_high or len(triggered_medium) >= 2:
            level = "HIGH"
            # 2+ MEDIUM sin HIGH explícito → subtipo genérico IDEACION_PROPIA
            effective_subtype = subtype if triggered_high else "IDEACION_PROPIA"
            all_triggered = triggered_high + triggered_medium
            log.warning(f"¡CRISIS HIGH! Subtype={effective_subtype}, triggers={all_triggered}")
            return CrisisResult(
                level="HIGH",
                subtype=effective_subtype,
                triggered_keywords=all_triggered,
                requires_immediate_response=True,
                requires_generation=False,
                suggested_pillar=None,
                response=self.get_response("HIGH", effective_subtype),
            )

        if len(triggered_medium) == 1:
            log.info(f"Crisis MEDIUM detectada. Trigger={triggered_medium[0]}. → generación SLM con P4.")
            return CrisisResult(
                level="MEDIUM",
                subtype="NONE",
                triggered_keywords=triggered_medium,
                requires_immediate_response=False,
                requires_generation=True,
                suggested_pillar=4,
                response=None,
            )

        return CrisisResult(
            level="NONE",
            subtype="NONE",
            triggered_keywords=[],
            requires_immediate_response=False,
            requires_generation=False,
            suggested_pillar=None,
            response=None,
        )

    def get_response(self, level: str, subtype: str = "IDEACION_PROPIA") -> str | None:
        """Devuelve la respuesta PAP para un nivel y subtipo dados.

        Args:
            level: 'HIGH', 'MEDIUM' o 'NONE'.
            subtype: 'IDEACION_PROPIA', 'AMENAZA_FISICA' o 'TERCERO_RIESGO'.

        Returns:
            Cadena de respuesta PAP, o None para MEDIUM/NONE.
        """
        if level == "HIGH":
            if subtype == "AMENAZA_FISICA":
                return CRISIS_RESPONSE_AMENAZA
            if subtype == "TERCERO_RIESGO":
                return CRISIS_RESPONSE_TERCERO
            return CRISIS_RESPONSE_IDEACION  # default (IDEACION_PROPIA o 2+ MEDIUM)
        return None
