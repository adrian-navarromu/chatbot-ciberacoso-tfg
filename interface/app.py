"""
Interfaz Gradio V1 — Chatbot de apoyo emocional para ciberacoso adolescente.

Lanzar:
    python interface/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Garantiza que el directorio raíz del proyecto esté en el path cuando el
# script se lanza directamente (python interface/app.py), ya que Python añade
# interface/ al path, no la raíz.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from src.pipeline.v1 import ChatbotV1

AVAILABLE_MODELS = ["mistral:7b", "gemma:2b", "gemma:7b", "phi3:mini", "tinyllama"]
DEFAULT_MODEL = "gemma:7b"

# ---------------------------------------------------------------------------
# Instancia compartida (sesión única — demo TFG)
# Se inicializa de forma lazy en el primer mensaje para no bloquear el arranque.
# ---------------------------------------------------------------------------

_chatbot: ChatbotV1 | None = None


def _get_chatbot() -> ChatbotV1:
    global _chatbot
    if _chatbot is None:
        _chatbot = ChatbotV1(model_name=DEFAULT_MODEL)
    return _chatbot


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _send_message(
    user_message: str,
    history_state: list[dict],
    auto_temp: bool,
    manual_temp: float,
) -> tuple:
    """Procesa un mensaje del usuario y devuelve la respuesta con metadatos debug."""
    if not user_message.strip():
        return history_state, history_state, "", {}, [], "", 0

    chatbot = _get_chatbot()
    t0 = time.time()
    result = chatbot.run(
        user_message=user_message,
        history=history_state,
        temperature=None if auto_temp else manual_temp,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    # El historial result.history ya tiene el formato {"role", "content"}
    # que gr.Chatbot de Gradio 6 entiende directamente (único formato soportado).
    emotion_display = {result.emotion_label: result.emotion_confidence}

    rag_rows = [
        [c["id"], c["title"], c["pillar"], c["score"]]
        for c in result.rag_chunks
    ]

    return (
        result.history,       # chatbot_ui — actualiza el display
        result.history,       # history_state — persiste el historial interno
        "",                   # limpia el input
        emotion_display,      # gr.Label emoción
        rag_rows,             # gr.Dataframe chunks RAG
        result.system_prompt_preview,  # gr.Textbox system prompt (completo)
        elapsed_ms,           # gr.Number tiempo de respuesta
    )


def _change_model(model_name: str) -> None:
    """Cambia el modelo SLM en la instancia compartida."""
    _get_chatbot().change_model(model_name)


def _clear_conversation() -> tuple:
    """Reinicia la conversación eliminando el historial."""
    return [], [], ""


def _toggle_temp_slider(auto: bool) -> dict:
    """Muestra u oculta el slider de temperatura manual."""
    return gr.update(visible=not auto)


# ---------------------------------------------------------------------------
# Layout Gradio
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    """Construye y devuelve el bloque Gradio de la interfaz V1."""
    with gr.Blocks(title="Chatbot Ciberacoso V1") as demo:
        gr.Markdown(
            "## Chatbot de apoyo emocional — Ciberacoso adolescente\n"
            "_Proyecto TFG · Ciencia de Datos · UMU_"
        )

        with gr.Row():
            # ----------------------------------------------------------------
            # Barra lateral izquierda
            # ----------------------------------------------------------------
            with gr.Column(scale=1, min_width=230):
                gr.Markdown("### Configuración")
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                    label="Modelo SLM (Ollama)",
                )
                auto_temp_checkbox = gr.Checkbox(
                    value=True,
                    label="Temperatura automática por emoción",
                )
                temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.7,
                    label="Temperatura manual",
                    visible=False,
                )

            # ----------------------------------------------------------------
            # Área principal de chat
            # ----------------------------------------------------------------
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(
                    height=500,
                    label="Conversación",
                    show_label=False,
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Cuéntame qué está pasando...",
                        label="",
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)
                clear_btn = gr.Button("Limpiar conversación", variant="secondary")

        # --------------------------------------------------------------------
        # Panel de debug (colapsado por defecto)
        # --------------------------------------------------------------------
        with gr.Accordion("Debug", open=True):
            with gr.Row():
                emotion_label_ui = gr.Label(
                    label="Emoción detectada (label: confianza)",
                    num_top_classes=1,
                )
            rag_table = gr.Dataframe(
                headers=["id", "title", "pillar", "score"],
                label="Chunks RAG recuperados",
                interactive=False,
                wrap=True,
            )
            system_prompt_ui = gr.Textbox(
                label="System prompt",
                lines=20,
                autoscroll=False,
                interactive=False,
            )
            response_time_ui = gr.Number(
                label="Tiempo de respuesta (ms)",
                interactive=False,
            )

        # --------------------------------------------------------------------
        # Estado por sesión
        # --------------------------------------------------------------------
        history_state = gr.State([])

        # --------------------------------------------------------------------
        # Eventos
        # --------------------------------------------------------------------
        _outputs = [
            chatbot_ui,
            history_state,
            user_input,
            emotion_label_ui,
            rag_table,
            system_prompt_ui,
            response_time_ui,
        ]
        _inputs = [user_input, history_state, auto_temp_checkbox, temp_slider]

        send_btn.click(fn=_send_message, inputs=_inputs, outputs=_outputs)
        user_input.submit(fn=_send_message, inputs=_inputs, outputs=_outputs)

        clear_btn.click(
            fn=_clear_conversation,
            outputs=[chatbot_ui, history_state, user_input],
        )
        model_dropdown.change(fn=_change_model, inputs=[model_dropdown])
        auto_temp_checkbox.change(
            fn=_toggle_temp_slider,
            inputs=[auto_temp_checkbox],
            outputs=[temp_slider],
        )

    return demo


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
