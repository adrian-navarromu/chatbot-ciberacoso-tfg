"""
Experimento 1: Comparación de estrategias de chunking para RAG de ciberacoso.

Hipótesis:
    El chunking manual por unidades terapéuticas debe superar al automático
    porque preserva la integridad semántica de cada técnica clínica (un
    ejercicio completo en un chunk, no partido a la mitad), lo que permite
    que el vector capture una idea coherente sin ruido adicional.

Uso:
    python src/rag/experiments/chunking_comparison.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Garantiza que la raíz del proyecto está en sys.path al ejecutar el script directamente.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.experiments.utils import (  # noqa: E402
    CORPUS_DIR,
    RESULTS_DIR,
    TEST_CASES,
    compute_hit_at_any,
    compute_metrics_for_run,
    compute_precision_at_k,
    load_chunks_from_markdown,
    print_results_table,
    save_results,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 3


# ---------------------------------------------------------------------------
# Dataclass auxiliar (solo Estrategia B — auto-chunking sin front matter)
# ---------------------------------------------------------------------------


@dataclass
class EvalChunk:
    """Chunk mínimo con pilar inferido del nombre de archivo (solo Estrategia B)."""

    pillar: int
    content: str


# ---------------------------------------------------------------------------
# Carga de corpus — Estrategia B (auto-chunking, específico de este experimento)
# ---------------------------------------------------------------------------


def _pillar_from_filename(path: Path) -> int:
    """Extrae el número de pilar del nombre de archivo (pillar3_hope.md → 3)."""
    match = re.search(r"pillar(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_auto_chunks(corpus_dir: Path) -> list[EvalChunk]:
    """Genera chunks automáticos ignorando el front matter YAML.

    Estrategia B: extrae el texto plano de los mismos archivos .md y aplica
    RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50). Puede
    partir técnicas terapéuticas a la mitad, degradando la coherencia semántica
    de cada vector. El pilar se infiere del nombre del archivo fuente.

    Args:
        corpus_dir: Directorio con los archivos .md del corpus RAG.

    Returns:
        Lista de EvalChunk con pillar (inferido del fichero) y fragmento de texto.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[EvalChunk] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        pillar = _pillar_from_filename(md_file)
        text = md_file.read_text(encoding="utf-8")
        # Split por '---' y conservar solo las partes de contenido (índices pares ≥ 2)
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        content_parts = [
            parts[i].strip() for i in range(2, len(parts), 2) if i < len(parts)
        ]
        plain_text = "\n\n".join(p for p in content_parts if p)
        if not plain_text:
            continue
        for fragment in splitter.split_text(plain_text):
            if fragment.strip():
                chunks.append(EvalChunk(pillar=pillar, content=fragment.strip()))
    return chunks


# ---------------------------------------------------------------------------
# Indexación FAISS (in-memory)
# ---------------------------------------------------------------------------


def build_faiss_index(
    docs: list[Document], model: SentenceTransformer
) -> faiss.IndexFlatIP:
    """Genera embeddings e indexa en FAISS en memoria. No persiste en disco.

    Usa IndexFlatIP con vectores normalizados, equivalente a similitud coseno exacta.

    Args:
        docs: Lista de Document de LangChain a indexar.
        model: Modelo SentenceTransformer para generar embeddings.

    Returns:
        Índice FAISS IndexFlatIP cargado en memoria.
    """
    texts = [doc.page_content for doc in docs]
    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def evaluate_strategy(
    docs: list[Document],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """Evalúa precision@k y hit@any para las 10 consultas estándar.

    Args:
        docs: Documents indexados (mismo orden que el índice FAISS).
        index: Índice FAISS IndexFlatIP en memoria.
        model: Modelo de embeddings para codificar las queries.
        top_k: Número de resultados a recuperar por consulta.

    Returns:
        Lista de dicts con métricas por consulta.
    """
    results: list[dict] = []
    for tc in TEST_CASES:
        query_vec = model.encode(
            [tc["query"]], normalize_embeddings=True
        ).astype(np.float32)
        _, indices = index.search(query_vec, top_k)

        retrieved_pillars = [
            docs[idx].metadata["pillar"]
            for idx in indices[0]
            if 0 <= idx < len(docs)
        ]
        expected = tc["expected_pillars"]

        results.append(
            {
                "label": tc["label"],
                "query": tc["query"],
                "expected_pillars": sorted(expected),
                "retrieved_pillars": retrieved_pillars,
                "precision_at_3": compute_precision_at_k(retrieved_pillars, expected, top_k),
                "hit_at_any": compute_hit_at_any(retrieved_pillars, expected),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------------


def _print_summary(
    manual_results: list[dict],
    auto_results: list[dict],
    n_manual: int,
    n_auto: int,
) -> None:
    """Imprime resumen comparativo entre estrategias."""
    m_a = compute_metrics_for_run(manual_results)
    m_b = compute_metrics_for_run(auto_results)
    prec_a, hit_a = m_a["precision_at_3_media"], m_a["hit_at_any_rate"]
    prec_b, hit_b = m_b["precision_at_3_media"], m_b["hit_at_any_rate"]

    sep = "=" * 74
    print(f"\n{sep}")
    print("  RESUMEN COMPARATIVO")
    print(sep)
    print(
        f"  {'Estrategia':<46} {'chunks':>6} {'P@3':>8} {'hit@any':>8}"
    )
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*8}")
    print(
        f"  {'A — Manual terapéutico (V1)':<46} {n_manual:>6}"
        f" {prec_a:>8.4f} {hit_a:>8.4f}"
    )
    print(
        f"  {'B — Automático (chunk_size=300, overlap=50)':<46} {n_auto:>6}"
        f" {prec_b:>8.4f} {hit_b:>8.4f}"
    )
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Ejecuta el experimento comparativo de chunking y guarda resultados en JSON."""
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")
    print("Cargando modelo...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Estrategia A: chunking manual terapéutico ──────────────────────────
    print("\n[A] Cargando chunks manuales (unidades terapéuticas)...")
    manual_docs = load_chunks_from_markdown(CORPUS_DIR)
    print(f"    {len(manual_docs)} chunks cargados.")
    print("[A] Indexando en FAISS (in-memory)...")
    manual_index = build_faiss_index(manual_docs, model)
    print("[A] Evaluando con 10 consultas estándar...")
    manual_results = evaluate_strategy(manual_docs, manual_index, model)

    # ── Estrategia B: chunking automático ─────────────────────────────────
    print("\n[B] Generando chunks automáticos (RecursiveCharacterTextSplitter)...")
    auto_chunks = load_auto_chunks(CORPUS_DIR)
    print(f"    {len(auto_chunks)} chunks generados.")
    # Convertir EvalChunk a Document para usar la misma interfaz de evaluación
    auto_docs = [
        Document(page_content=c.content, metadata={"pillar": c.pillar, "chunk_id": ""})
        for c in auto_chunks
    ]
    print("[B] Indexando en FAISS (in-memory)...")
    auto_index = build_faiss_index(auto_docs, model)
    print("[B] Evaluando con 10 consultas estándar...")
    auto_results = evaluate_strategy(auto_docs, auto_index, model)

    # ── Tablas individuales ────────────────────────────────────────────────
    print_results_table(
        f"Estrategia A — Manual terapéutico (V1)  [{len(manual_docs)} chunks]",
        manual_results,
    )
    print_results_table(
        f"Estrategia B — Automático RecursiveCharacterTextSplitter  [{len(auto_docs)} chunks]",
        auto_results,
    )

    # ── Resumen comparativo ────────────────────────────────────────────────
    _print_summary(manual_results, auto_results, len(manual_docs), len(auto_docs))

    # ── Guardar JSON ───────────────────────────────────────────────────────
    m_a = compute_metrics_for_run(manual_results)
    m_b = compute_metrics_for_run(auto_results)
    prec_a, hit_a = m_a["precision_at_3_media"], m_a["hit_at_any_rate"]
    prec_b, hit_b = m_b["precision_at_3_media"], m_b["hit_at_any_rate"]

    output: dict = {
        "experimento": "chunking_comparison",
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "estrategia_a": {
            "nombre": "Manual terapéutico (V1)",
            "descripcion": "Chunks delimitados por front matter YAML, un chunk por unidad terapéutica",
            "n_chunks": len(manual_docs),
            "precision_at_3_media": round(prec_a, 4),
            "hit_at_any_rate": round(hit_a, 4),
            "resultados_por_consulta": manual_results,
        },
        "estrategia_b": {
            "nombre": "Automático RecursiveCharacterTextSplitter",
            "descripcion": "Texto plano sin front matter, chunk_size=300, chunk_overlap=50",
            "chunk_size": 300,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", ". ", " "],
            "n_chunks": len(auto_docs),
            "precision_at_3_media": round(prec_b, 4),
            "hit_at_any_rate": round(hit_b, 4),
            "resultados_por_consulta": auto_results,
        },
    }

    save_results("chunking_results.json", output)


if __name__ == "__main__":
    main()
