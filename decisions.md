# Registro de Decisiones de Diseño

| Fecha | Módulo | Decisión | Alternativas descartadas | Motivo |
|---|---|---|---|---|
| Abr 2026 | Data augmentation | Traducción EN→ES + class weights | Generación sintética con Ollama | Sintéticos con duplicados, prefijos y emociones incorrectas; traducción aporta datos reales con etiquetas validadas |
| Abr 2026 | Clasificador emocional | BETO (bert-base-spanish-wwm-cased) con class weights | RoBERTuito (mayor F1 macro), MarIA-copy (tokenizer inestable) | BETO tiene el mejor F1 en fear (0.57), emocion critica en ciberacoso; diferencia en F1 macro es solo 0.007 |
