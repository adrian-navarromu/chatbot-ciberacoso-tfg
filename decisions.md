# Registro de Decisiones de Diseño

Log cronológico de las decisiones de diseño del proyecto, con las alternativas
descartadas y el motivo. Las rectificaciones se mantienen visibles a propósito:
el valor de un registro está en documentar también los cambios de criterio.

| Fecha | Módulo | Decisión | Alternativas descartadas | Motivo |
|---|---|---|---|---|
| Abr 2026 | Data augmentation | Traducción EN→ES + class weights | Generación sintética con Ollama | Sintéticos con duplicados, prefijos y emociones incorrectas; traducción aporta datos reales con etiquetas validadas |
| Abr 2026 | Clasificador emocional | BETO (bert-base-spanish-wwm-cased) con class weights | RoBERTuito (mayor F1 macro), MarIA (tokenizer inestable) | BETO tiene el mejor F1 en fear (0.57), emoción crítica en ciberacoso; diferencia en F1 macro es solo 0.007. **⚠️ Rectificada en May 2026 — ver más abajo** |
| Abr 2026 | SLM desarrollo | Cambio de Mistral 7B a Gemma 7B como modelo de trabajo | Mistral 7B | Gemma 7B respeta mejor las instrucciones del system prompt (no saluda en cada turno), produce respuestas semánticamente más coherentes y completa las frases. Mistral 7B ignora la instrucción de no saludar y corta las respuestas. Validado tras pruebas con el chatbot V1 |
| May 2026 | Detección de acoso | Descartar MTLHate como señal auxiliar | Modelo multitarea MTLHate para reforzar la detección | Validación empírica (nb05): no aporta señal útil sobre el clasificador emocional; se simplifica el pipeline |
| May 2026 | Seguridad | `CrisisDetector` **determinista** de 3 niveles (NONE / MEDIUM / HIGH) como primera capa, antes del SLM | Guardrail probabilístico; delegar el riesgo vital al modelo generativo | El modo de fallo crítico es no detectar ideación suicida. La conducta de seguridad debe ser auditable y reproducible (respuesta fija con derivación al 024 en 0 ms en HIGH), no depender de un modelo probabilístico |
| May 2026 | Memoria | `EmotionalMemoryTracker`: tendencia afectiva de sesión (estable / mejora / empeora) inyectada en el prompt | Lectura de emoción por turno, sin memoria | Permite adaptar el tono a la evolución emocional del usuario a lo largo de la conversación, no solo al mensaje actual |
| May 2026 | RAG | Recuperación **enrutada por emoción** sobre ChromaDB (pilares filtrados según la emoción detectada) | Recuperación semántica global sobre todo el corpus | Sube la precisión de recuperación P@1 de 0.67→0.90 y P@3 de 0.59→0.84 frente a la recuperación global |
| May 2026 | RAG | Descartar el canal léxico **BM25** de la búsqueda híbrida; quedarse solo con el enrutamiento emocional | Búsqueda híbrida BM25 × routing emocional | Los 4 experimentos factoriales muestran que el canal léxico no aporta mejora; el enrutamiento emocional es lo que mueve la métrica (nb06) |
| May 2026 | Arquitectura | Rediseño **V1 → V2** («safety first»): crisis antes que nada, memoria emocional y RAG enrutado | Mantener el pipeline V1 secuencial (emoción → RAG global → SLM) | V1 leía la emoción por turno sin memoria de su evolución, delegaba el riesgo vital al SLM y no enrutaba la recuperación. V2 corrige los tres fallos |
| May 2026 | Clasificador emocional | **RECTIFICACIÓN: RoBERTuito con class weights** como clasificador final de producción | BETO (elección previa de Abr 2026) | Con el split y los pesos de clase definitivos, RoBERTuito ofrece mejor F1 macro y F1 en *sadness* (>0.81), clase crítica. Es el modelo que carga `pipeline/v2.py`. **Supersede la decisión de Abr 2026** |
| Jun 2026 | Generación (SLM) | **Gemma 7B** como SLM de producción | Mistral 7B, Phi-3 Mini, Gemma 2B, TinyLlama | Benchmark de 5 SLMs puntuado por LLM-as-a-judge (validado contra evaluadores humanos, ρ = 0.71): la elección de modelo domina sobre la de prompt; Gemma 7B se sitúa en la frontera calidad/latencia (nb08, nb09) |
| Jun 2026 | Evaluación | **LLM-as-a-judge** validado contra evaluadores humanos para puntuar generaciones | Solo evaluación humana (no escala); solo métricas automáticas (no capturan calidad clínica) | Correlación juez↔humano ρ = 0.71 (Spearman), r = 0.68 sobre 5 dimensiones clínicas; permite escalar la evaluación de generaciones manteniendo validez |

## Rectificaciones

### Clasificador emocional: BETO → RoBERTuito (May 2026, commit `7003e14`)

La entrada de **Abr 2026** fijó **BETO** como clasificador final, con el argumento de
que maximizaba el F1 en *fear* y que la diferencia en F1 macro frente a RoBERTuito era
marginal (~0.007).

Tras rehacer el split de datos y consolidar los *class weights* definitivos, la
comparativa se invirtió: **RoBERTuito con pérdida ponderada por clase** pasó a dar mejor
F1 macro y, en particular, mejor F1 en *sadness* (>0.81) — la clase crítica para el caso
de uso. Es el modelo que carga el pipeline de producción (`src/pipeline/v2.py`).

Se conserva deliberadamente la entrada original de BETO: el registro documenta el criterio
y su rectificación, no solo el resultado final.
