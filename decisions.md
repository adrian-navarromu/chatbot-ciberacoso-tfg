# Registro de Decisiones de Diseño

| Fecha | Módulo | Decisión | Alternativas descartadas | Motivo |
|---|---|---|---|---|
| Abr 2026 | Data augmentation | Traducción EN→ES + class weights | Generación sintética con Ollama | Sintéticos con duplicados, prefijos y emociones incorrectas; traducción aporta datos reales con etiquetas validadas |
| Abr 2026 | Clasificador emocional | BETO (bert-base-spanish-wwm-cased) con class weights | RoBERTuito (mayor F1 macro), MarIA-copy (tokenizer inestable) | BETO tiene el mejor F1 en fear (0.57), emocion critica en ciberacoso; diferencia en F1 macro es solo 0.007 |
| Abr 2026 | SLM desarrollo | Cambio de Mistral 7B a Gemma 7B como modelo de trabajo | Mistral 7B | Gemma 7B respeta mejor las instrucciones del system prompt (no saluda en cada turno), produce respuestas semánticamente más coherentes y completa las frases. Mistral 7B ignora la instrucción de no saludar y corta las respuestas. Validado tras pruebas con el chatbot V1