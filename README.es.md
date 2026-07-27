# 🛡️ Chatbot Emocionalmente Adaptativo de Apoyo frente al Ciberacoso

**Un agente de apoyo emocional *privacy-first* para víctimas adolescentes de ciberacoso, ejecutado íntegramente sobre Small Language Models locales.**

Trabajo de Fin de Grado (TFG) — Grado en Ciencia e Ingeniería de Datos, Universidad de Murcia · Nota: 9,4/10

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Local First](https://img.shields.io/badge/inference-100%25%20local-orange)

![Interfaz del chatbot](docs/figures/arquitectura/interfaz_gradio.png)

---

## Por qué este proyecto

En España, **el 22,5 % de los adolescentes sufre ciberacoso** (UNICEF), y las víctimas presentan **4× más ideación suicida**. El acoso es continuo, anónimo y sigue a la víctima hasta casa a través del móvil, pero la barrera para pedir ayuda a un adulto es enorme.

Este proyecto explora si un agente conversacional bien diseñado puede ofrecer primeros auxilios emocionales seguros y clínicamente fundamentados **sin que ningún dato salga nunca del dispositivo**. Esa restricción es innegociable cuando los usuarios son menores, y descarta por completo las APIs de LLM en la nube: todo se ejecuta sobre SLMs locales mediante [Ollama](https://ollama.com).

**El hueco que cubre:** ningún sistema publicado combina (1) apoyo directo a la víctima, (2) inferencia 100 % local, (3) memoria emocional turno a turno y (4) detección de crisis determinista. El precedente más cercano, PrevenIA (U. de La Rioja), depende de modelos en la nube y se dirige a terceros, no a la víctima.

## Resultados clave

| Componente | Métrica | Resultado |
|---|---|---|
| Enrutamiento RAG emocional | Precisión de recuperación P@1 | **0,67 → 0,90** frente a recuperación semántica global |
| Enrutamiento RAG emocional | Precisión de recuperación P@3 | **0,59 → 0,84** |
| Detector de crisis (determinista, 3 niveles) | Recall y precisión en crisis de riesgo ALTO | **0,905** (19/21), validado en casos adversariales |
| Clasificador emocional (RoBERTuito fine-tuned, pérdida ponderada por clase) | F1 en *tristeza* (clase crítica) | **> 0,81** |
| Benchmark SLM (5 modelos) | LLM-as-a-judge vs. evaluadores humanos | **ρ = 0,71** (Spearman), r = 0,68 |

Los 8 objetivos específicos del proyecto se cumplieron con evidencia empírica. Los informes completos de los experimentos son navegables en [`reports/`](reports/).

## Arquitectura

El sistema evolucionó en dos versiones, y el rediseño V1 → V2 es el núcleo de la historia de ingeniería del proyecto.

![Arquitectura V2](docs/figures/arquitectura/arquitectura_v2.png)

**V1 — línea base secuencial:** clasificador emocional → recuperación FAISS global → ensamblado de prompt → SLM. Las pruebas revelaron tres fallos: la emoción se leía por turno sin memoria de su evolución, el riesgo vital se delegaba al modelo generativo y la recuperación no tenía enrutamiento.

**V2 — seguridad primero:** un **`CrisisDetector` determinista se ejecuta antes que nada** y gradúa cada mensaje en tres niveles:

- **ALTO** → el SLM nunca se invoca; se devuelve en 0 ms una respuesta fija de primeros auxilios psicológicos con derivación (línea 024).
- **MEDIO** → la recuperación se fuerza al pilar de primeros auxilios psicológicos con un prompt de contención.
- **NINGUNO** → flujo normal: clasificación emocional → `EmotionalMemoryTracker` (tendencia de sesión: estable / mejora / deterioro) → **recuperación enrutada por emoción** sobre ChromaDB → prompt estructurado → SLM local.

Principio de diseño: *el SLM se guía, nunca se le confía el riesgo*. El comportamiento crítico de seguridad es determinista y auditable, no probabilístico.

**Base de conocimiento:** un corpus clínico curado a mano y organizado en 5 pilares (psicoeducación y protocolos, TCC, esperanza y contranarrativas, primeros auxilios psicológicos, autodefensa digital), fragmentado en unidades clínicas de ~150 palabras con metadatos por emoción ([`data/rag_corpus/`](data/rag_corpus/)).

## Qué se evaluó (y cómo)

Cada componente se seleccionó mediante experimentos controlados — véanse los notebooks numerados en [`notebooks/`](notebooks/) y sus exportaciones HTML en [`reports/`](reports/):

1. **Clasificador emocional:** RoBERTuito vs. BETO vs. MarIA, con/sin pérdida ponderada por clase, sobre el corpus EmoEvent (sin datos sintéticos — la ampliación por traducción ganó a los sintéticos generados por LLM, ver [`decisions.md`](decisions.md)).
2. **RAG:** 4 experimentos factoriales — estrategias de chunking, embeddings en español vs. multilingües, FAISS vs. ChromaDB, y BM25 híbrido × enrutamiento emocional (conclusión: el canal léxico no aporta; lo que importa es el enrutamiento).
3. **Generación:** 5 SLMs (Gemma 7B, Mistral 7B, Phi-3 Mini, Gemma 2B, TinyLlama) × variantes de prompt, puntuados por un **LLM-as-a-judge validado contra evaluadores humanos** (ρ = 0,71) en 5 dimensiones clínicas. La elección de modelo domina sobre la de prompt (rango de score 0,42 vs. 0,03). Gemma 7B se sitúa en la frontera calidad/latencia.
4. **Seguridad:** evaluación por matriz de confusión del CrisisDetector sobre un banco de 100 consultas de prueba que incluye prompts adversariales.

## Inicio rápido

```bash
# 1. Clonar e instalar
git clone https://github.com/adrian-navarromu/chatbot-ciberacoso-tfg.git
cd chatbot-ciberacoso-tfg
conda env create -f environment.yml && conda activate chatbot-tfg
# o: pip install -r requirements.txt

# 2. Descargar el SLM local (requiere Ollama instalado)
ollama pull gemma:7b
# La app también permite elegir mistral:7b, gemma:2b, phi3:mini o tinyllama —
# descarga el que quieras probar, p. ej.  ollama pull mistral:7b

# 3. Lanzar la interfaz Gradio (V1 y V2 seleccionables en la app)
python interface/app.py
```

> **Nota:** la inferencia en GPU del clasificador emocional usa la build CUDA de PyTorch.
> `requirements.txt` fija la build CPU portable; para GPU instala la wheel correspondiente:
> `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.

## Tests

```bash
pytest tests/                    # suite completa (necesita el modelo emocional local)
pytest -m "not integration"      # subconjunto para CI — sin Ollama ni modelos descargados
```

Los tests de integración (marcados con `@pytest.mark.integration`) requieren el clasificador
emocional fine-tuned en disco y Ollama en ejecución; todo lo demás está mockeado.

## Estructura del repositorio

```
├── src/
│   ├── safety/          # CrisisDetector determinista (primera capa, 3 niveles)
│   ├── emotion/         # Clasificador emocional transformer fine-tuned + entrenamiento
│   ├── memory/          # EmotionalMemoryTracker (tendencia afectiva de sesión)
│   ├── rag/             # Ingesta, recuperador enrutado por emoción + 4 experimentos
│   ├── prompts/         # Constructores de prompt según emoción/estado (V1, V2)
│   ├── pipeline/        # Orquestadores ChatbotV1 y ChatbotV2
│   └── evaluation/      # LLM-as-a-judge, acuerdo, evaluación de crisis
├── interface/           # App Gradio con panel de debug (emoción, nivel de crisis, chunks)
├── data/                # Corpus RAG (5 pilares) y datasets procesados
├── docs/                # PDF de arquitectura, figuras de resultados, documentos de diseño
├── eval/                # Resultados brutos de cada experimento (JSON/CSV + gráficas)
├── notebooks/           # 01–09: EDA → selección de modelo → RAG → benchmark
├── reports/             # Exportaciones HTML de todos los notebooks
├── tests/               # Tests unitarios por módulo
└── decisions.md         # Registro de decisiones de diseño (qué, alternativas, por qué)
```

## Decisiones de diseño

Tres destacadas del registro completo en [`decisions.md`](decisions.md):

- **Detección de crisis determinista, no un guardrail probabilístico.** Cuando el modo de fallo es no detectar un mensaje de ideación suicida, la auditabilidad importa más que la elegancia del modelo.
- **SLM local en lugar de API en la nube.** Los datos de los menores nunca salen de la máquina; el proyecto demuestra que un modelo 7B bien diseñado puede sostener todo el sistema.
- **Datos reales traducidos frente a generación sintética.** Los sintéticos generados por LLM introducían duplicados y emociones mal etiquetadas; la traducción EN→ES de EmoEvent + pérdida ponderada por clase gestionó mejor el desbalance miedo/tristeza.

## ⚠️ Aviso importante

Esto es un **prototipo de investigación**, no una herramienta clínica. No sustituye la ayuda profesional. Si tú o alguien que conoces está sufriendo acoso, hostigamiento o malestar emocional en España, hay ayuda gratuita y confidencial 24/7 en la **línea 024** (prevención del suicidio) y en **ANAR: 900 20 20 10** (menores).

## Autor

**Adrián Navarro Muñoz** — Grado en Ciencia e Ingeniería de Datos, Universidad de Murcia (primera promoción)
Dirigido por **Rafael Valencia García** y **Ronghao Pan** (grupo TECNOMOD, UMU)

[LinkedIn](https://www.linkedin.com/in/adrian-navarro-munoz) · adrian.navarromu@gmail.com

---

*English version: [README.md](README.md)*
