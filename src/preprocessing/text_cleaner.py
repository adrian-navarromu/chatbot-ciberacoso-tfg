"""
Limpieza de texto para tweets de EmoEvent.

Diseñado para preservar información emocional (puntuación, tildes, mayúsculas)
mientras elimina ruido específico de Twitter y los placeholders del dataset.
"""

import re
import emoji


# ---------------------------------------------------------------------------
# Patrones de compilación única (coste de compilación pagado una sola vez)
# ---------------------------------------------------------------------------

# Placeholders que EmoEvent usa para anonimizar menciones, hashtags y URLs.
# Orden: variantes con prefijo (@USER, #HASHTAG) antes que la palabra sola
# para evitar que el grupo más corto consuma solo parte del token.
_PLACEHOLDER_RE = re.compile(
    r"@USER\b"          # mención anonimizada con arroba
    r"|#HASHTAG\b"      # hashtag anonimizado con almohadilla
    r"|\bUSER\b"        # mención sin arroba (variante en algunos splits)
    r"|\bHASHTAG\b"     # hashtag sin almohadilla
    r"|\bHTTPURL\b"     # URL anonimizada (variante HTTPURL)
    r"|\bURL\b"         # URL anonimizada (variante corta)
    r"|https?://\S+",   # URL real que haya podido escapar al anonimizador
    re.IGNORECASE,
)

# Comillas escapadas que aparecen como artefactos de serialización CSV/JSON
_ESCAPED_QUOTES_RE = re.compile(r'"{2,}')

# Secuencias de espacios (incluye tabuladores y retornos de carro)
_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")


def clean_tweet(text: str) -> str:
    """Limpia un tweet de EmoEvent preservando la información emocional.

    Pasos aplicados en orden:
    1. Elimina placeholders de anonimización (USER, HASHTAG, URL y variantes).
    2. Convierte emojis a su descripción textual en español (biblioteca emoji).
    3. Reemplaza comillas escapadas (\"\"…\"\") por comilla simple.
    4. Colapsa espacios múltiples a uno solo y elimina espacios marginales.

    No modifica:
    - Signos de puntuación (informativos para detección de emoción).
    - Tildes y caracteres especiales del español.
    - Mayúsculas/minúsculas (BETO y MarIA son case-sensitive).

    Args:
        text: Texto original del tweet.

    Returns:
        Texto limpio listo para tokenización o aumentación de datos.
    """
    if not isinstance(text, str):
        return ""

    # 1. Eliminar placeholders y URLs reales
    text = _PLACEHOLDER_RE.sub("", text)

    # 2. Emojis → descripción en español
    #    delimiters=(' ', ' ') rodea cada descripción con espacios para evitar
    #    que se pegue a palabras adyacentes; los guiones bajos se convierten
    #    en espacios para facilitar la tokenización posterior.
    text = emoji.demojize(text, language="es", delimiters=(" ", " "))
    text = text.replace("_", " ")

    # 3. Comillas escapadas → comilla simple
    text = _ESCAPED_QUOTES_RE.sub("'", text)

    # 4. Colapsar espacios y limpiar márgenes
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
