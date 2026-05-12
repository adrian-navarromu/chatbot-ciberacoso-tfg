"""
Pipeline V2: CrisisDetector → EmotionDetector → EmotionalMemoryGRU → RAG → SLM.

Flujo:
  1. CrisisDetector  → cortocircuito PAP si HIGH/MEDIUM
  2. EmotionDetector → emoción + confianza
  3. EmotionalMemoryGRU.update() → registra emoción del turno
  4. get_prompt_context() → cadena descriptiva para inyectar en prompt
  5. RAGRetriever → chunks clínicos
  6. build_prompt() con emotional_context
  7. ChatOllama → genera respuesta
  8. Actualiza historial y devuelve ChatbotV2Response

Uso:
    chatbot = ChatbotV2()
    result = chatbot.run("Me están amenazando por Instagram", [])
    print(result.response, result.emotion_label, result.crisis_level)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.emotion.emotion_detector import EmotionDetector
from src.memory.emotional_memory import EmotionalMemoryGRU
from src.prompts.builder import TEMPERATURE_BY_EMOTION, build_prompt
from src.rag.document_ingestion import RAGRetriever
from src.safety.crisis_detector import CrisisDetector


# ---------------------------------------------------------------------------
# Dataclass de respuesta
# ---------------------------------------------------------------------------


@dataclass
class ChatbotV2Response:
    """Respuesta completa del pipeline V2 para un turno de conversación.

    Atributos:
        response: Texto de respuesta generado (PAP o SLM).
        history: Historial actualizado incluyendo el turno actual.
        emotion_label: Emoción detectada por BETO, o 'crisis' si hubo bypass PAP.
        emotion_confidence: Probabilidad del clasificador (1.0 en bypass de crisis).
        rag_chunks: Metadatos de chunks recuperados (vacío en bypass de crisis).
        system_prompt_preview: System prompt enviado al SLM (o mensaje PAP si crisis).
        model_name: Nombre del modelo Ollama utilizado.
        crisis_level: Nivel del detector de crisis: 'NONE', 'MEDIUM' o 'HIGH'.
        emotional_context: Cadena de memoria emocional inyectada en el prompt.
    """

    response: str
    history: list[dict]
    emotion_label: str
    emotion_confidence: float
    rag_chunks: list[dict]
    system_prompt_preview: str
    model_name: str
    crisis_level: str
    emotional_context: str


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


class ChatbotV2:
    """Pipeline V2 con detección de crisis, memoria emocional y RAG clínico.

    Añade sobre V1:
    - CrisisDetector como primera capa de seguridad (bypass PAP inmediato)
    - EmotionalMemoryGRU para rastrear la evolución afectiva de la sesión
    - emotional_context inyectado en el system prompt entre la variante
      emocional y el contexto RAG

    Args:
        model_name: Nombre del modelo Ollama (ej. 'gemma:7b').
        emotion_model_path: Ruta al directorio del clasificador BETO fine-tuned.
        vectorstore_path: Ruta al directorio con el índice FAISS.
    """

    def __init__(
        self,
        model_name: str = "gemma:7b",
        emotion_model_path: str = "models/emotion_classifier/beto",
        vectorstore_path: str = "data/vectorstore/faiss_index_v1",
    ) -> None:
        self.model_name = model_name
        self.detector = EmotionDetector(Path(emotion_model_path))
        self.retriever = RAGRetriever(vectorstore_dir=Path(vectorstore_path))
        self.crisis_detector = CrisisDetector()
        self.memory = EmotionalMemoryGRU()

    def run(
        self,
        user_message: str,
        history: list[dict],
        temperature: float | None = None,
    ) -> ChatbotV2Response:
        """Ejecuta el pipeline V2 completo para un turno de conversación.

        Si el CrisisDetector activa HIGH o MEDIUM, la función devuelve
        inmediatamente la respuesta PAP sin invocar el SLM ni actualizar
        la memoria emocional.

        Args:
            user_message: Mensaje del usuario en el turno actual.
            history: Historial anterior en formato OpenAI/Ollama.
            temperature: Temperatura manual; None usa la automática por emoción.

        Returns:
            ChatbotV2Response con la respuesta y todos los metadatos de debug.
        """
        # 1. Crisis check — cortocircuito PAP si se detecta HIGH o MEDIUM
        crisis_result = self.crisis_detector.detect(user_message)
        if crisis_result.level != "NONE":
            updated_history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": crisis_result.response},
            ]
            return ChatbotV2Response(
                response=crisis_result.response,
                history=updated_history,
                emotion_label="crisis",
                emotion_confidence=1.0,
                rag_chunks=[],
                system_prompt_preview=(
                    f"[CRISIS {crisis_result.level}]\n\n{crisis_result.response}"
                ),
                model_name=self.model_name,
                crisis_level=crisis_result.level,
                emotional_context=self.memory.get_prompt_context(),
            )

        # 2. Detectar emoción
        emotion_result = self.detector.detect(user_message)

        # 3. Actualizar memoria emocional con el turno actual
        self.memory.update(emotion_result.label, emotion_result.confidence)

        # 4. Obtener cadena descriptiva para inyectar en el prompt
        emotional_context = self.memory.get_prompt_context()

        # 5. Recuperar chunks RAG con query enriquecida
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

        # 6. Construir prompt con contexto emocional
        messages = build_prompt(
            emotion=emotion_result.label,
            rag_context=rag_context,
            history=history,
            confidence=emotion_result.confidence,
            emotional_context=emotional_context,
        )
        messages.append({"role": "user", "content": user_message})

        # 7. Llamar al SLM
        if temperature is None:
            temperature = TEMPERATURE_BY_EMOTION.get(emotion_result.label, 0.7)
        llm = ChatOllama(
            model=self.model_name,
            temperature=temperature,
            num_predict=300,
        )
        response_text: str = llm.invoke(_to_lc_messages(messages)).content

        # 8. Actualizar historial y construir respuesta
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

        system_prompt_preview = build_prompt(
            emotion=emotion_result.label,
            rag_context=rag_context,
            history=[],
            confidence=emotion_result.confidence,
            emotional_context=emotional_context,
        )[0]["content"]

        return ChatbotV2Response(
            response=response_text,
            history=updated_history,
            emotion_label=emotion_result.label,
            emotion_confidence=emotion_result.confidence,
            rag_chunks=rag_chunks,
            system_prompt_preview=system_prompt_preview,
            model_name=self.model_name,
            crisis_level="NONE",
            emotional_context=emotional_context,
        )

    def change_model(self, model_name: str) -> None:
        """Cambia el modelo SLM en caliente sin recargar EmotionDetector ni RAG."""
        self.model_name = model_name

    def reset_memory(self) -> None:
        """Reinicia la memoria emocional para comenzar una nueva sesión."""
        self.memory.reset()


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
