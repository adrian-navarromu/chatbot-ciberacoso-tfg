"""
Interfaz Gradio — Chatbot de apoyo emocional para ciberacoso adolescente.
Soporta pipeline V1 (sin memoria) y V2 (CrisisDetector + EmotionalMemoryGRU).

Lanzar:
    python interface/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Configuración
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from src.pipeline.v1 import ChatbotV1
from src.pipeline.v2 import ChatbotV2

# Catálogo de modelos locales
AVAILABLE_MODELS = ["mistral:7b", "gemma:2b", "gemma:7b", "phi3:mini", "tinyllama"]
DEFAULT_MODEL = "gemma:7b"


# Instancias que se crean en el primer mensaje de cada versión.
# Se inicializan como None para evitar cargar los pesados modelos de HuggingFace/FAISS en VRAM hasta que el usuario explícitamente envíe el primer mensaje.
_chatbot_v1: ChatbotV1 | None = None
_chatbot_v2: ChatbotV2 | None = None
_current_model: str = DEFAULT_MODEL


def _get_chatbot_v1() -> ChatbotV1:
    global _chatbot_v1
    if _chatbot_v1 is None:
        _chatbot_v1 = ChatbotV1(model_name=_current_model)
    return _chatbot_v1


def _get_chatbot_v2() -> ChatbotV2:
    global _chatbot_v2
    if _chatbot_v2 is None:
        _chatbot_v2 = ChatbotV2(model_name=_current_model)
    return _chatbot_v2


# Lógica de la interfaz
def _send_message(
    user_message: str,
    history_state: list[dict],
    auto_temp: bool,
    manual_temp: float,
    version: str,
) -> tuple:
    """Procesa un mensaje del usuario con el pipeline seleccionado."""
    if not user_message.strip():
        # Retorno de seguridad para evitar inferencias vacías
        return history_state, history_state, "", {}, [], "", 0, "", {}

    t0 = time.time()
    temp = None if auto_temp else manual_temp  # Control de temperatura: dinámica (auto) o forzada por UI (manual)

    # Incorpora detección de crisis aguda y actualización de la memoria recurrente (GRU) antes de la recuperación RAG y el SLM.
    if version == "V2":
        chatbot = _get_chatbot_v2()
        result = chatbot.run(user_message=user_message, history=history_state)
        
        # Telemetría extendida específica de V2
        crisis_level = result.crisis_level
        memory_state = chatbot.memory.to_dict()
        emotion_display = {result.emotion_label: result.emotion_confidence}
        rag_rows = [
            [c.get("chunk_id", ""), c.get("title", ""),
             c.get("pillar", ""), "—"]
            for c in result.rag_chunks
        ]
        system_prompt = result.system_prompt_preview

    #Pipeline base: Inferencia emocional directa + RAG + SLM.
    else:
        chatbot = _get_chatbot_v1()
        result = chatbot.run(user_message=user_message, history=history_state)
        
        # Valores nulos/vacíos para telemetría ausente en V1
        crisis_level = ""
        memory_state = {}
        emotion_display = {result.emotion_label: result.emotion_confidence}
        rag_rows = [
            [c.get("chunk_id", ""), c.get("title", ""),
             c.get("pillar", ""), "—"]
            for c in result.rag_chunks
        ]
        system_prompt = result.system_prompt_preview

    elapsed_ms = int((time.time() - t0) * 1000)

    # Retorno coincidente con los 'outputs' definidos en la interfaz
    return (
        result.history,       # Actualiza el UI del Chat
        result.history,       # Actualiza el gr.State (memoria de sesión)
        "",                   # Limpia la caja de texto
        emotion_display,      # Actualiza UI: Emoción
        rag_rows,             # Actualiza UI: Tabla RAG
        system_prompt,        # Actualiza UI: Prompt
        elapsed_ms,           # Actualiza UI: Tiempo
        crisis_level,         # Actualiza UI: Nivel Crisis (V2)
        memory_state,         # Actualiza UI: Estado Memoria (V2)
    )


def _change_model(model_name: str) -> None:
    """Cambia el modelo SLM en las instancias ya inicializadas."""
    global _current_model
    _current_model = model_name
    if _chatbot_v1 is not None:
        _chatbot_v1.change_model(model_name)
    if _chatbot_v2 is not None:
        _chatbot_v2.change_model(model_name)


def _clear_conversation(version: str) -> tuple:
    """Reinicia la conversación. En V2 también resetea la memoria emocional."""
    if version == "V2" and _chatbot_v2 is not None:
        _chatbot_v2.reset_session()
    return [], [], ""


def _toggle_temp_slider(auto: bool) -> dict:
    """Oculta/Muestra el control de temperatura manual en el UI."""
    return gr.update(visible=not auto)


def _toggle_v2_debug(version: str) -> tuple:
    """Muestra u oculta los paneles exclusivos de V2 en el panel Debug."""
    is_v2 = version == "V2"
    return gr.update(visible=is_v2), gr.update(visible=is_v2)


# Layout Gradio
def build_interface() -> gr.Blocks:
    """Construye y devuelve el bloque Gradio con soporte V1/V2."""
    with gr.Blocks(title="Chatbot Ciberacoso") as demo:
        gr.Markdown(
            "## Chatbot de apoyo emocional — Ciberacoso adolescente\n"
            "_Proyecto TFG · Ciencia de Datos · UMU_"
        )

        with gr.Row():

            # Barra lateral izquierda
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Configuración")

                # Selector de arquitectura
                version_radio = gr.Radio(
                    choices=["V1", "V2"],
                    value="V1",
                    label="Versión del pipeline",
                    info="V1: emoción+RAG | V2: crisis+memoria+emoción+RAG",
                )
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

            # Área principal de chat
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

        # Panel Inferior de debug
        with gr.Accordion("Debug", open=True):
            with gr.Row():
                emotion_label_ui = gr.Label(
                    label="Emoción detectada (label: confianza)",
                    num_top_classes=1,
                )
                crisis_level_ui = gr.Textbox(
                    label="Nivel de crisis (V2)",
                    interactive=False,
                    visible=False,
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
            memory_state_ui = gr.JSON(
                label="Memoria emocional (V2)",
                visible=False,
            )
            response_time_ui = gr.Number(
                label="Tiempo de respuesta (ms)",
                interactive=False,
            )

        # Estado por sesión
        history_state = gr.State([])

        # Mapeo de Eventos
        _outputs = [
            chatbot_ui,
            history_state,
            user_input,
            emotion_label_ui,
            rag_table,
            system_prompt_ui,
            response_time_ui,
            crisis_level_ui,
            memory_state_ui,
        ]
        _inputs = [user_input, history_state, auto_temp_checkbox, temp_slider, version_radio]

        send_btn.click(fn=_send_message, inputs=_inputs, outputs=_outputs)
        user_input.submit(fn=_send_message, inputs=_inputs, outputs=_outputs)

        clear_btn.click(
            fn=_clear_conversation,
            inputs=[version_radio],
            outputs=[chatbot_ui, history_state, user_input],
        )
        model_dropdown.change(fn=_change_model, inputs=[model_dropdown])
        auto_temp_checkbox.change(
            fn=_toggle_temp_slider,
            inputs=[auto_temp_checkbox],
            outputs=[temp_slider],
        )
        version_radio.change(
            fn=_toggle_v2_debug,
            inputs=[version_radio],
            outputs=[crisis_level_ui, memory_state_ui],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
