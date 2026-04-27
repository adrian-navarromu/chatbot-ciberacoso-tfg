"""
Pipeline V1: EmotionDetector → RAGRetriever → SLM (Ollama).

Uso:
    chatbot = ChatbotV1()
    result = chatbot.run("Me están amenazando por Instagram", [])
    print(result.response, result.emotion_label)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.emotion.emotion_detector import EmotionDetector
from src.prompts.builder import TEMPERATURE_BY_EMOTION, build_prompt, get_system_prompt_preview
from src.rag.document_ingestion import RAGRetriever


# ---------------------------------------------------------------------------
# Dataclass de respuesta
# ---------------------------------------------------------------------------


@dataclass
class ChatbotV1Response:
    """Respuesta completa del pipeline V1 para un turno de conversación.

    Atributos:
        response: Texto de respuesta generado por el SLM.
        history: Historial actualizado incluyendo el turno actual.
        emotion_label: Emoción detectada por el clasificador BETO.
        emotion_confidence: Probabilidad de la clase predicha (0-1).
        rag_chunks: Metadatos de cada chunk recuperado del índice FAISS.
        system_prompt_preview: System prompt completo enviado al SLM.
        model_name: Nombre del modelo Ollama utilizado en este turno.
    """

    response: str
    history: list[dict]
    emotion_label: str
    emotion_confidence: float
    rag_chunks: list[dict]
    system_prompt_preview: str
    model_name: str


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


class ChatbotV1:
    """Pipeline completo V1: EmotionDetector → RAGRetriever → SLM (Ollama).

    Carga los tres componentes una sola vez en __init__ y los reutiliza en
    cada turno de conversación. El modelo Ollama se selecciona en cada llamada
    para permitir el cambio en caliente con change_model().

    Args:
        model_name: Nombre del modelo Ollama a utilizar (ej. 'mistral:7b').
        emotion_model_path: Ruta al directorio del clasificador BETO fine-tuned.
        vectorstore_path: Ruta al directorio con el índice FAISS.
    """

    def __init__(
        self,
        model_name: str = "mistral:7b",
        emotion_model_path: str = "models/emotion_classifier/beto",
        vectorstore_path: str = "data/vectorstore/faiss_index_v1",
    ) -> None:
        self.model_name = model_name
        self.detector = EmotionDetector(Path(emotion_model_path))
        self.retriever = RAGRetriever(vectorstore_dir=Path(vectorstore_path))

    def run(
        self,
        user_message: str,
        history: list[dict],
        temperature: float | None = None,
    ) -> ChatbotV1Response:
        """Ejecuta el pipeline completo para un turno de conversación.

        Args:
            user_message: Mensaje del usuario en el turno actual.
            history: Historial anterior en formato OpenAI/Ollama
                     [{"role": "user/assistant", "content": "..."}].
                     No debe incluir el system prompt.
            temperature: Temperatura para el SLM. Si es None, se usa la
                         temperatura automática según la emoción detectada
                         (TEMPERATURE_BY_EMOTION).

        Returns:
            ChatbotV1Response con la respuesta del SLM y todos los metadatos.
        """
        # 1. Detectar emoción
        emotion_result = self.detector.detect(user_message)

        # 2. Recuperar chunks RAG con query enriquecida con el último turno.
        # En el segundo turno en adelante, combinar el último mensaje del usuario
        # con el actual para que el retriever tenga más contexto semántico.
        if history:
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                "",
            )
            rag_query = f"{last_user} {user_message}".strip()
        else:
            rag_query = user_message

        retrieved = self.retriever.retrieve(
            query=rag_query,
            emotion=emotion_result.label,
            top_k=3,
        )
        rag_context = "\n---\n".join(r.chunk.content for r in retrieved)

        # 3. Construir prompt (system + historial)
        messages = build_prompt(
            emotion=emotion_result.label,
            rag_context=rag_context,
            history=history,
            confidence=emotion_result.confidence,
        )
        messages.append({"role": "user", "content": user_message})

        # 4. Llamar al SLM: temperatura manual si se especifica, automática si no
        if temperature is None:
            temperature = TEMPERATURE_BY_EMOTION.get(emotion_result.label, 0.7)
        llm = ChatOllama(
            model=self.model_name,
            temperature=temperature,
            num_predict=300,
        )
        response_msg = llm.invoke(_to_lc_messages(messages))
        response_text: str = response_msg.content

        # 5. Actualizar historial y construir respuesta
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_text},
        ]

        rag_chunks = [
            {
                "id": r.chunk.id,
                "title": r.chunk.title,
                "pillar": r.chunk.pillar,
                "score": round(r.score, 4),
                "forced": r.forced,
            }
            for r in retrieved
        ]

        system_prompt_preview = get_system_prompt_preview(
            emotion=emotion_result.label,
            rag_context=rag_context,
        )

        return ChatbotV1Response(
            response=response_text,
            history=updated_history,
            emotion_label=emotion_result.label,
            emotion_confidence=emotion_result.confidence,
            rag_chunks=rag_chunks,
            system_prompt_preview=system_prompt_preview,
            model_name=self.model_name,
        )

    def change_model(self, model_name: str) -> None:
        """Cambia el modelo SLM en caliente sin recargar EmotionDetector ni RAG.

        Args:
            model_name: Nombre del nuevo modelo Ollama (ej. 'gemma:7b').
        """
        self.model_name = model_name


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------


def _to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convierte dicts de rol/contenido a objetos de mensaje de LangChain."""
    _cls = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    return [_cls[m["role"]](content=m["content"]) for m in messages]
