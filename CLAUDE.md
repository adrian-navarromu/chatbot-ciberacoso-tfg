# CLAUDE.md

## Project Overview
Chatbot de apoyo emocional para ciberacoso adolescente (TFG Ciencia de Datos, UMU).
Detección de emociones + RAG clínico + memoria emocional + SLMs locales.

## Tech Stack
-Python 3.11, PyTorch 2.11+cu128, conda environment: chatbots
-LangChain (LCEL), Ollama (SLMs locales), FAISS/ChromaDB
-Transformers (MarIA/BETO para clasificación emocional)
-Gradio (interfaz web)
-GPU: NVIDIA RTX 5080 (16 GB VRAM)

## Conventions
-Type hints obligatorios en todas las funciones
-Docstrings en español para funciones públicas (memoria del TFG)
-PEP 8 + black formatter
-Tests en tests/ usando pytest
-Commits: feat(módulo): descripción

## Commands
-Activar entorno: conda activate chatbots
-Tests: pytest tests/
-Jupyter: jupyter lab
-Interfaz: python interface/app.py
-Benchmark SLMs: python eval/benchmark_slm.py