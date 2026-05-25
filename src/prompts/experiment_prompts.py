"""
4 variantes de prompt para el experimento de Prompt Engineering (Exp. 5).

Variante A — Baseline V1: delegación directa a build_prompt sin cambios.
Variante B — Few-shot clínico: añade 14 ejemplos (2x7 emociones) antes del RAG.
Variante C — Chain-of-Thought moderado: inserta pasos de razonamiento explícitos.
Variante D — Prompt estructurado: bloques etiquetados [ROL], [ESTADO], [CLÍNICO], [OBJETIVO].

Uso:
    from src.prompts.experiment_prompts import build_prompt_variant

    messages = build_prompt_variant(
        variant="B",
        emotion="fear",
        rag_context=ctx,
        history=[{"role": "user", "content": "..."}],
        confidence=0.85,
        emotional_context="",
    )
"""

from src.prompts.builder import BASE_SYSTEM_PROMPT, EMOTION_VARIANTS, TEMPERATURE_BY_EMOTION, build_prompt

# Variante B — sección de ejemplos few-shot (14 pares, 2 por emoción)
_FEW_SHOT_SECTION: str = """
EJEMPLOS DE RESPUESTA CORRECTA:

[Emoción: fear]
Usuario: "me da miedo abrir el móvil porque siempre hay algo malo"
Respuesta correcta: "Ese miedo que describes tiene sentido. Antes de nada,
¿puedes respirar despacio conmigo? Inspira contando hasta 4 y suelta
contando hasta 6. Cuando estés lista, cuéntame un poco más."

[Emoción: fear]
Usuario: "me han enviado un mensaje amenazándome y no puedo dormir"
Respuesta correcta: "Recibir una amenaza así activa el cuerpo entero, y
entiendo que describes un nivel de alerta muy alto. ¿Estás en un sitio
donde te sientes segura ahora mismo? Si en algún momento sientes peligro
real, el 112 está disponible las 24 horas."

[Emoción: sadness]
Usuario: "todo el mundo me ignora en clase desde hace semanas"
Respuesta correcta: "Semanas sintiéndote invisible es muy duro. Eso tiene
nombre y no es exageración. ¿Hay algún momento del día que se haga más
llevadero, o es igual todo el tiempo?"

[Emoción: sadness]
Usuario: "ya no tengo ganas de hacer nada, todo me da igual desde que empezó esto"
Respuesta correcta: "Cuando todo se vuelve gris así, el cansancio emocional
que describes es real y no es debilidad. ¿Desde cuándo más o menos
te sientes así?"

[Emoción: anger]
Usuario: "me dan ganas de responderles igual de mal que ellos, estoy harto"
Respuesta correcta: "Tiene sentido que estés agotado. Alguien que te trata
así provoca exactamente esa rabia. Una cosa: ¿qué crees que pasaría
si les respondes ahora mismo?"

[Emoción: anger]
Usuario: "estoy muy enfadado porque nadie hace nada y siguen igual"
Respuesta correcta: "Esa frustración que describes tiene todo el sentido
cuando ves que la situación no cambia. Separemos dos cosas: lo que han
hecho ellos y lo que está en tu mano. ¿Qué es lo que más te gustaría
que cambiara primero?"

[Emoción: disgust]
Usuario: "me da mucha vergüenza que hayan visto esas fotos, no puedo salir de casa"
Respuesta correcta: "Lo que describes es vergüenza, y es una reacción muy
común cuando alguien expone algo tuyo sin permiso. Eso que sientes no dice
nada malo de ti. ¿Llevas mucho tiempo cargando con esto sola/solo?"

[Emoción: disgust]
Usuario: "me siento sucia después de los mensajes que me enviaron"
Respuesta correcta: "Lo que describes tiene sentido. Esa sensación no es
tuya, es la consecuencia de lo que ellos hicieron. No tienes que explicarlo
todo ahora. ¿Hay alguien de confianza cerca de ti en este momento?"

[Emoción: joy]
Usuario: "hoy he conseguido bloquearlo y me siento un poco más tranquila"
Respuesta correcta: "Bloquearlo es un paso concreto que tú has dado, y eso
importa. ¿Hay algo más que quieras hacer ahora, o prefieres tomarte un tiempo?"

[Emoción: joy]
Usuario: "mi amiga me ha dicho que me cree y me apoya, me ha dado fuerzas"
Respuesta correcta: "Tener a alguien que te cree puede cambiar mucho cómo
te sientes. Es un apoyo real. ¿Qué te ha ayudado de habérselo contado a ella?"

[Emoción: surprise]
Usuario: "me acaban de avisar de que van a publicar fotos mías y no sé qué pensar"
Respuesta correcta: "Es normal que no sepas qué pensar ahora mismo, la
noticia llega de golpe. Antes de hacer nada, ¿puedes contarme un poco
más de qué fotos son y quién te ha avisado?"

[Emoción: surprise]
Usuario: "alguien de clase que no esperaba me ha defendido delante de los demás"
Respuesta correcta: "A veces pasan cosas que no esperamos en ninguna
dirección. ¿Cómo te has sentido tú con eso?"

[Emoción: others]
Usuario: "no sé muy bien por qué he entrado aquí, supongo que quería hablar con alguien"
Respuesta correcta: "Está bien que hayas entrado. No hace falta saber
exactamente qué buscar. ¿Hay algo que te esté rondando estos días,
aunque sea difícil de explicar?"

[Emoción: others]
Usuario: "últimamente me siento raro/rara pero no sé cómo llamar a lo que me pasa"
Respuesta correcta: "No tienes que ponerle nombre todavía. Puedes contarme
lo que quieras a tu ritmo. ¿Qué es lo que está pasando, aunque te cueste
explicarlo?"
"""

# Variante C — sección de razonamiento Chain-of-Thought
_COT_SECTION: str = """
PROCESO DE RAZONAMIENTO (seguir estos pasos antes de responder):
Paso 1 — VALIDACIÓN: Identifica y nombra cognitivamente la emoción
  expresada. Empieza con validación antes de cualquier otra acción.
Paso 2 — SELECCIÓN: Elige UNA técnica o recurso del contexto clínico
  que sea aplicable a este momento concreto de la conversación.
Paso 3 — FORMULACIÓN: Construye la respuesta con el tono correcto
  según la emoción detectada. Verifica que no supera 300 palabras y
  que termina con una pregunta abierta o una propuesta concreta.
"""

# Constructores internos por variante
def _build_variant_a(emotion: str, rag_context: str, history: list[dict], confidence: float, emotional_context: str) -> list[dict]:
    return build_prompt(
        emotion=emotion,
        rag_context=rag_context,
        history=history,
        confidence=confidence,
        emotional_context=emotional_context,
    )


def _build_variant_b(emotion: str, rag_context: str, history: list[dict], confidence: float, emotional_context: str) -> list[dict]:
    """Few-shot: misma estructura que V1 + ejemplos clínicos antes del RAG."""
    effective_emotion = emotion if confidence >= 0.4 else "others"
    system = BASE_SYSTEM_PROMPT + "\n\n" + EMOTION_VARIANTS[effective_emotion]

    if emotional_context:
        system += (
            "\n\nESTADO EMOCIONAL DEL USUARIO (evolución de la sesión):\n"
            + emotional_context
        )

    system += _FEW_SHOT_SECTION

    if rag_context:
        system += (
            "\n\nCONTEXTO CLÍNICO (usa esta información para orientar tu "
            "respuesta si es relevante, sin reproducirla literalmente):\n"
            + rag_context
        )

    return [{"role": "system", "content": system}] + history


def _build_variant_c(emotion: str, rag_context: str, history: list[dict], confidence: float, emotional_context: str) -> list[dict]:
    """CoT: pasos de razonamiento explícitos justo después de la variante emocional."""
    effective_emotion = emotion if confidence >= 0.4 else "others"
    system = BASE_SYSTEM_PROMPT + "\n\n" + EMOTION_VARIANTS[effective_emotion]

    system += _COT_SECTION

    if emotional_context:
        system += (
            "\n\nESTADO EMOCIONAL DEL USUARIO (evolución de la sesión):\n"
            + emotional_context
        )

    if rag_context:
        system += (
            "\n\nCONTEXTO CLÍNICO (usa esta información para orientar tu "
            "respuesta si es relevante, sin reproducirla literalmente):\n"
            + rag_context
        )

    return [{"role": "system", "content": system}] + history


def _build_variant_d(emotion: str, rag_context: str, history: list[dict], confidence: float, emotional_context: str) -> list[dict]:
    """Prompt estructurado con bloques etiquetados explícitamente."""
    effective_emotion = emotion if confidence >= 0.4 else "others"
    temperature = TEMPERATURE_BY_EMOTION[effective_emotion]
    emotion_variant = EMOTION_VARIANTS[effective_emotion]
    rag_block = rag_context if rag_context else "Sin contexto clínico disponible para esta consulta."
    emotional_block = emotional_context if emotional_context else "Sin datos de sesión previa."

    system = (
        "[ROL_Y_LIMITES]\n"
        + BASE_SYSTEM_PROMPT.strip()
        + "\n[/ROL_Y_LIMITES]\n\n"
        + "[ESTADO_EMOCIONAL_USUARIO]\n"
        + f"Emoción detectada: {effective_emotion} (confianza: {confidence:.0%})\n"
        + f"Tendencia de sesión: {emotional_block}\n"
        + "[/ESTADO_EMOCIONAL_USUARIO]\n\n"
        + "[CONTEXTO_CLINICO]\n"
        + rag_block
        + "\n[/CONTEXTO_CLINICO]\n\n"
        + "[OBJETIVO_TURNO]\n"
        + emotion_variant
        + f"\nTemperatura ajustada: {temperature}\n"
        + "[/OBJETIVO_TURNO]"
    )

    return [{"role": "system", "content": system}] + history


# Interfaz pública
_VARIANT_BUILDERS: dict = {
    "A": _build_variant_a,
    "B": _build_variant_b,
    "C": _build_variant_c,
    "D": _build_variant_d,
}


def build_prompt_variant(variant: str, emotion: str, rag_context: str, history: list[dict], confidence: float = 1.0, emotional_context: str = "") -> list[dict]:
    """
    Construye el prompt para una de las 4 variantes del experimento de PE.

    Args:
        variant: Identificador de la variante ('A', 'B', 'C' o 'D').
        emotion: Emoción detectada ('fear', 'sadness', 'anger', etc.)
        rag_context: Texto de los chunks RAG recuperados (puede ser vacío).
        history: Historial en formato OpenAI/Ollama (incluye mensaje del usuario).
        confidence: Confianza del clasificador (< 0.4 → usa 'others').
        emotional_context: Cadena de memoria emocional de sesión (V2).

    Returns:
        Lista de mensajes en formato OpenAI/Ollama chat.

    Raises:
        ValueError: Si variant no está en ('A', 'B', 'C', 'D').
    """
    if variant not in _VARIANT_BUILDERS:
        raise ValueError(
            f"Variante desconocida: {variant!r}. Usa 'A', 'B', 'C' o 'D'."
        )
    return _VARIANT_BUILDERS[variant](
        emotion=emotion,
        rag_context=rag_context,
        history=history,
        confidence=confidence,
        emotional_context=emotional_context,
    )
