"""
Pipeline V2: CrisisDetector → EmotionDetector → EmotionalMemoryTracker → RAG → SLM.

Flujo normal:
  1. CrisisDetector → HIGH: cortocircuito PAP determinista
                   → MEDIUM: rama generación PAP con pilar 4
  2. EmotionDetector → emoción + confianza
  3. EmotionalMemoryTracker.update() → registra emoción del turno
  4. detect_trend() + get_prompt_context() → tendencia y contexto de sesión
  5. EnrichedRetriever.retrieve_with_routing() → chunks clínicos con routing por pilar
  6. build_prompt_v2() → prompt estructurado variante D (83.4 %)
  7. ChatOllama → genera respuesta
  8. Actualiza historial y devuelve ChatbotV2Response

Rama crisis MEDIUM:
  Cuando CrisisDetector.requires_generation=True, el pipeline recupera pilar 4,
  construye build_crisis_prompt_v2() y genera con el SLM.

Uso:
    chatbot = ChatbotV2()
    result = chatbot.run("Me están amenazando por Instagram", [])
    print(result.response, result.emotion_label, result.crisis_level)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.emotion.emotion_detector import EmotionDetector
from src.memory.emotional_memory import EmotionalMemoryTracker
from src.prompts.builder_v2 import TEMPERATURE_BY_EMOTION, build_prompt_v2, build_crisis_prompt_v2
from src.rag.document_ingestion_v2 import DocumentIngesterV2
from src.rag.enriched_retriever import EnrichedRetriever
from src.safety.crisis_detector import CrisisDetector


# Dataclass de respuesta
@dataclass
class ChatbotV2Response:
    """Respuesta completa del pipeline V2 para un turno de conversación.

    Atributos:
        response: Texto de respuesta generado (PAP o SLM).
        history: Historial actualizado incluyendo el turno actual.
        emotion_label: Emoción detectada por el classificador emocional, o 'crisis' si hubo bypass PAP.
        emotion_confidence: Probabilidad del clasificador (1.0 en bypass de crisis).
        trend: Tendencia emocional de la sesión ('estable', 'mejora', 'deterioro').
        rag_chunks: Metadatos de chunks recuperados (vacío en bypass de crisis).
        system_prompt_preview: System prompt completo enviado al SLM (o etiqueta PAP).
        model_name: Nombre del modelo Ollama utilizado.
        latency_ms: Latencia de inferencia en milisegundos (0.0 en bypass de crisis).
        crisis_level: Nivel del detector de crisis: 'HIGH', 'MEDIUM' o None.
    """

    response: str
    history: list[dict]
    emotion_label: str
    emotion_confidence: float
    trend: str
    rag_chunks: list[dict]
    system_prompt_preview: str
    model_name: str
    latency_ms: float
    crisis_level: str | None


# Pipeline principal
class ChatbotV2:
    """Pipeline V2 con detección de crisis, clasificación emocional, memoria GRU y RAG híbrido.

    Integra todos los módulos desarrollados en V2:
    - CrisisDetector: primera capa de seguridad, bypass PAP inmediato si HIGH/MEDIUM.
    - EmotionDetector: clasificación emocional fine-tuned.
    - EmotionalMemoryTracker: rastreo de tendencia afectiva por ventana deslizante.
    - EnrichedRetriever: recuperación semántica con routing por pilar.
    - build_prompt_v2: prompt estructurado variante D.

    Args:
        model_name: Nombre del modelo Ollama (ej. 'gemma:7b').
        emotion_model_path: Ruta al directorio del clasificador fine-tuned.
        corpus_dir: Directorio con los archivos .md del corpus RAG.
        chroma_path: Ruta al directorio ChromaDB persistente (chroma_v2).
        embedding_model: Identificador HuggingFace del modelo de embeddings.
        window_size: Tamaño de ventana deslizante para EmotionalMemoryTracker.
    """

    def __init__(self, model_name: str = "gemma:7b",
        emotion_model_path: str = "models/emotion_classifier/robertuito",
        corpus_dir: str = "data/rag_corpus",
        chroma_path: str = "data/vectorstore/chroma_v2",
        embedding_model: str = "hackathon-pln-es/paraphrase-spanish-distilroberta",
        window_size: int = 6,
    ) -> None:
        self.model_name = model_name

        self.crisis_detector = CrisisDetector()
        self.emotion_detector = EmotionDetector(Path(emotion_model_path))
        self.memory = EmotionalMemoryTracker(window_size=window_size)

        # Parsear corpus y construir documentos LangChain para EnrichedRetriever
        ingester = DocumentIngesterV2(
            corpus_dir=corpus_dir,
            embedding_model=embedding_model,
        )
        chunks = ingester.load_corpus()
        documents: list[Document] = [
            Document(
                page_content=c.content,
                metadata={
                    "chunk_id": c.id,
                    "pillar": c.pillar,
                    "title": c.title,
                },
            )
            for c in chunks
        ]
        self.retriever = EnrichedRetriever(chroma_path, documents, embedding_model)

    def run(self, user_message: str, history: list[dict]) -> ChatbotV2Response:
        """Ejecuta el pipeline V2 completo para un turno de conversación.

        Si el CrisisDetector activa HIGH o MEDIUM, devuelve inmediatamente
        la respuesta PAP sin invocar el SLM ni actualizar la memoria emocional.

        Args:
            user_message: Mensaje del usuario en el turno actual.
            history: Historial anterior en formato OpenAI/Ollama
                     [{"role": "user/assistant", "content": "..."}].
                     No debe incluir el system prompt.

        Returns:
            ChatbotV2Response con la respuesta y todos los metadatos de debug.
        """
        # PASO 1 - FAILSAFE PAP (primera capa, antes de cualquier ML)
        crisis_result = self.crisis_detector.detect(user_message)

        if crisis_result.level == "HIGH":
            # Respuesta determinista fija — no se invoca el SLM
            history_updated = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": crisis_result.response},
            ]
            return ChatbotV2Response(
                response=crisis_result.response,
                history=history_updated,
                emotion_label="crisis",
                emotion_confidence=1.0,
                trend="deterioro",
                rag_chunks=[],
                system_prompt_preview=(
                    f"[FAILSAFE PAP HIGH — subtype={crisis_result.subtype}]"
                ),
                model_name=self.model_name,
                latency_ms=0.0,
                crisis_level="HIGH",
            )

        if crisis_result.level == "MEDIUM" and crisis_result.requires_generation:
            # Contención PAP con generación SLM: pilar 4 + prompt de crisis
            docs = self.retriever.retrieve(
                user_message,
                pillar_filter=[crisis_result.suggested_pillar],
            )
            rag_context = "\n---\n".join(d.page_content for d in docs)
            messages = build_crisis_prompt_v2(rag_context=rag_context, history=history)
            messages.append({"role": "user", "content": user_message})

            llm = ChatOllama(
                model=self.model_name,
                temperature=0.5,
                num_predict=200,
            )
            t0 = time.time()
            llm_response = llm.invoke(messages)
            latency_ms = (time.time() - t0) * 1000

            history_updated = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": llm_response.content},
            ]
            return ChatbotV2Response(
                response=llm_response.content,
                history=history_updated,
                emotion_label="crisis",
                emotion_confidence=1.0,
                trend="deterioro",
                rag_chunks=[d.metadata for d in docs],
                system_prompt_preview=messages[0]["content"],
                model_name=self.model_name,
                latency_ms=round(latency_ms, 1),
                crisis_level="MEDIUM",
            )

        # PASO 2 - CLASIFICADOR EMOCIONAL
        emotion_result = self.emotion_detector.detect(user_message)

        # PASO 3 - ACTUALIZAR MEMORIA GRU
        self.memory.update(emotion_result.label, emotion_result.confidence)
        trend = self.memory.detect_trend()
        emotional_context = self.memory.get_prompt_context()

        # PASO 4 - RECUPERACIÓN RAG ENRIQUECIDA
        if history:
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                "",
            )
            rag_query = f"{last_user} {user_message}".strip()
        else:
            rag_query = user_message

        docs = self.retriever.retrieve_with_routing(
            rag_query,
            emotion=emotion_result.label,
            trend=trend,
        )
        rag_context = "\n---\n".join([d.page_content for d in docs])

        # PASO 5 - CONSTRUIR PROMPT V2 (variante D)
        messages = build_prompt_v2(
            emotion=emotion_result.label,
            rag_context=rag_context,
            history=history,
            confidence=emotion_result.confidence,
            emotional_context=emotional_context,
            trend=trend,
        )
        messages.append({"role": "user", "content": user_message})

        # PASO 6 - INFERENCIA SLM
        temperature = TEMPERATURE_BY_EMOTION.get(emotion_result.label, 0.7)
        llm = ChatOllama(
            model=self.model_name,
            temperature=temperature,
            num_predict=300,
        )
        t0 = time.time()
        llm_response = llm.invoke(messages)
        latency_ms = (time.time() - t0) * 1000

        # PASO 7 - ACTUALIZAR HISTORIAL Y DEVOLVER
        history_updated = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": llm_response.content},
        ]
        return ChatbotV2Response(
            response=llm_response.content,
            history=history_updated,
            emotion_label=emotion_result.label,
            emotion_confidence=emotion_result.confidence,
            trend=trend,
            rag_chunks=[d.metadata for d in docs],
            system_prompt_preview=messages[0]["content"],
            model_name=self.model_name,
            latency_ms=round(latency_ms, 1),
            crisis_level=None,
        )

    def change_model(self, model_name: str) -> None:
        """Cambia el modelo SLM en caliente sin recargar EmotionDetector ni RAG."""
        self.model_name = model_name

    def reset_session(self) -> None:
        """Reinicia la memoria emocional para una nueva sesión."""
        self.memory.reset()
