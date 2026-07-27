# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-06-29

Safety-first redesign of the pipeline (V2). The core engineering change is moving
life-critical risk handling out of the generative model into a deterministic layer.

### Added
- **Deterministic `CrisisDetector`** (3 levels: NONE / MEDIUM / HIGH) as the first
  layer, before any model runs. On HIGH the SLM is never invoked: a fixed
  psychological-first-aid response with referral to the 024 hotline is returned.
- **`EmotionalMemoryTracker`**: session-level affect trend (stable / improving /
  deteriorating) injected into the prompt.
- **Emotion-routed RAG** over ChromaDB: retrieval is filtered by pillar according to
  the detected emotion (P@1 0.67→0.90, P@3 0.59→0.84 vs. global retrieval).
- **Structured V2 prompt builder** (`prompts/builder_v2.py`) with crisis-containment
  variant and emotion/state-aware assembly.
- **LLM-as-a-judge evaluation** validated against human raters (ρ = 0.71) across 5
  clinical dimensions; SLM benchmark over 5 models and prompt variants.
- 100-query test bank (including adversarial cases) and confusion-matrix evaluation
  of the CrisisDetector.
- Hand-curated clinical corpus reorganised into 5 pillars with per-emotion metadata.

### Changed
- **Emotion classifier: RoBERTuito with class-weighted loss** replaces BETO as the
  production model (better macro-F1 and F1 on *sadness*). See `decisions.md`.
- RAG stack moved from FAISS (global) to ChromaDB with emotional routing.
- Pipeline orchestration split into `ChatbotV1` and `ChatbotV2`.

### Removed
- Lexical **BM25** channel from hybrid search: the 4 factorial RAG experiments showed
  it adds nothing over emotional routing.
- MTLHate as an auxiliary harassment signal (no empirical improvement).

## [1.0.0] — 2026-04-27

First end-to-end prototype (V1): sequential emotion → retrieval → generation pipeline
with a web interface.

### Added
- Fine-tuned transformer **emotion classifier** (7 emotions) trained on EmoEvent with
  class-weighted loss and EN→ES translation-based augmentation.
- **RAG** over a 5-pillar therapeutic corpus with FAISS indexing and global semantic
  retrieval.
- **Emotion-aware prompt builder** with per-emotion temperature.
- Local **SLM generation** via Ollama (Gemma 7B as working model).
- **Gradio interface** with a debug panel (detected emotion, retrieved chunks).
- Experiment notebooks (EDA, preprocessing, model selection, RAG test) and test suite.

[2.0.0]: https://github.com/adrian-navarromu/chatbot-ciberacoso-tfg/releases/tag/v2.0.0
[1.0.0]: https://github.com/adrian-navarromu/chatbot-ciberacoso-tfg/releases/tag/v1.0.0
