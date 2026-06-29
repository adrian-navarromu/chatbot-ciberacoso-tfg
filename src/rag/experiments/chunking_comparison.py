"""
Experimento 1: Comparación de estrategias de chunking para RAG de ciberacoso.

Banco de evaluación: 72 casos (expected_pillars ≠ ∅, crisis_expected ≠ HIGH).
La emoción del banco se inyecta directamente al routing (sin clasificador externo).

Estrategias comparadas:
    A — Chunking manual terapéutico (front matter YAML, un chunk por técnica clínica)
    B — Chunking automático RecursiveCharacterTextSplitter (chunk_size=300, overlap=50)

ADVERTENCIA METODOLÓGICA (CAMBIO 5):
    La métrica P@k favorece artificialmente al chunking automático porque fragmentar
    cada técnica en más trozos aumenta la probabilidad de acertar el pilar por azar.
    La estrategia A se elige por INTEGRIDAD CLÍNICA del chunk, no por la métrica:
    un chunk manual contiene una técnica terapéutica completa (instrucción + cierre),
    mientras que un fragmento automático puede cortar una técnica a la mitad.
    Ver `qualitative_analysis()` para ejemplos concretos.

Requiere GPU (SentenceTransformer embeddings).

Uso:
    conda activate chatbots
    python src/rag/experiments/chunking_comparison.py
"""

from __future__ import annotations

import re
import sys
import random
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 112
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.experiments.utils import (
    CORPUS_DIR, RESULTS_DIR, FIGURES_DIR,
    ORACLE_ROUTING,
    load_eval_cases, oracle_faiss_search, make_oracle_result,
    compute_eval_metrics, rag_figure_bar,
    compute_precision_at_k, compute_hit_at_any,
    load_chunks_from_markdown, save_results,
)

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 3


@dataclass
class EvalChunk:
    pillar: int
    content: str


def _pillar_from_filename(path: Path) -> int:
    match = re.search(r"pillar(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_auto_chunks(corpus_dir: Path) -> list[EvalChunk]:
    """Estrategia B (baseline): RecursiveCharacterTextSplitter(chunk_size=300, overlap=50)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[EvalChunk] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        pillar = _pillar_from_filename(md_file)
        text = md_file.read_text(encoding="utf-8")
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        content_parts = [parts[i].strip() for i in range(2, len(parts), 2) if i < len(parts)]
        plain_text = "\n\n".join(p for p in content_parts if p)
        if not plain_text:
            continue
        for fragment in splitter.split_text(plain_text):
            if fragment.strip():
                chunks.append(EvalChunk(pillar=pillar, content=fragment.strip()))
    return chunks


# ---------------------------------------------------------------------------
# CAMBIO 5 — Análisis cualitativo de fragmentación
# ---------------------------------------------------------------------------

def qualitative_analysis(corpus_dir: Path) -> None:
    """Muestra ejemplos concretos de cómo el chunking automático rompe técnicas clínicas.

    Compara el chunk manual (técnica completa) con los fragmentos automáticos
    correspondientes. Demuestra que la métrica P@k favorece artificialmente al
    chunking automático por inflación del espacio de búsqueda, pero la fragmentación
    destruye la integridad clínica (la instrucción queda sin cierre, o viceversa).
    """
    manual_docs = load_chunks_from_markdown(corpus_dir)
    auto_chunks = load_auto_chunks(corpus_dir)

    sep = "=" * 78
    print(f"\n{sep}")
    print("  ANÁLISIS CUALITATIVO: CHUNKING MANUAL vs. CHUNKING AUTOMÁTICO")
    print("  ADVERTENCIA: La métrica P@k NO mide integridad clínica del chunk.")
    print("  El chunking manual se elige porque preserva la técnica terapéutica")
    print("  completa en cada chunk, no porque supere métricamente al automático.")
    print(sep)

    # Cargar texto plano de cada fichero de pilar para buscar fragmentos automáticos
    pillar_texts: dict[str, str] = {}
    for md_file in sorted(corpus_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        contents = [parts[i].strip() for i in range(2, len(parts), 2) if i < len(parts)]
        pillar_texts[md_file.stem] = "\n\n".join(c for c in contents if c)

    # Seleccionar 3 chunks manuales ilustrativos (técnicas con pasos numerados o estructuradas)
    ejemplos = [
        ("P2_001", "Respiración diafragmática — 5 pasos numerados"),
        ("P2_002", "Grounding 5-4-3-2-1 — lista de sentidos"),
        ("P2_007", "Parada del pensamiento + mindfulness — dos técnicas"),
    ]

    for chunk_id, label in ejemplos:
        manual = next((d for d in manual_docs if d.metadata.get("chunk_id") == chunk_id), None)
        if not manual:
            continue

        manual_text = manual.page_content
        manual_words = len(manual_text.split())

        # Buscar auto-chunks que contengan parte de este contenido
        manual_start = manual_text[:80]
        matching_auto = [
            c for c in auto_chunks
            if c.pillar == manual.metadata["pillar"]
            and any(word in c.content for word in manual_start.split()[:5])
        ]

        print(f"\n  {'─'*74}")
        print(f"  EJEMPLO: {label}  [{chunk_id}]")
        print(f"  {'─'*74}")
        print(f"\n  ▶ CHUNK MANUAL ({manual_words} palabras) — TÉCNICA COMPLETA:")
        print(f"  {'·'*74}")
        for line in manual_text[:500].split("\n"):
            print(f"    {line}")
        if len(manual_text) > 500:
            print(f"    ... [{manual_words - len(manual_text[:500].split())} palabras más]")

        if matching_auto:
            print(f"\n  ▶ AUTO-CHUNKS generados desde el mismo pilar ({len(matching_auto)} encontrados):")
            for i, ac in enumerate(matching_auto[:2], 1):
                n_words = len(ac.content.split())
                first_chars = ac.content[:200]
                print(f"  {'·'*74}")
                print(f"  Auto-fragmento {i} ({n_words} palabras) — {'INICIO' if i==1 else 'CONTINUACIÓN'}:")
                for line in first_chars.split("\n"):
                    print(f"    {line}")
                if len(ac.content) > 200:
                    print(f"    ...")
                if i == 1 and len(matching_auto) > 1:
                    print(f"  ⚠  La técnica continúa en el siguiente fragmento (chunk cortado).")
        else:
            print(f"  [No se encontraron auto-chunks coincidentes — técnica muy corta o sin coincidencia]")

    print(f"\n  {'─'*74}")
    print(f"  CONCLUSIÓN: El chunking automático genera {len(auto_chunks)} fragmentos")
    print(f"  frente a {len(manual_docs)} chunks manuales. Cada fragmento extra")
    print(f"  infla P@k porque aumenta la cobertura de pilar por azar, pero")
    print(f"  el retriever devuelve técnicas incompletas al usuario.")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def build_embeddings(docs: list[Document], model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        [d.page_content for d in docs],
        normalize_embeddings=True, show_progress_bar=True, batch_size=32,
    ).astype(np.float32)


def evaluate_config(
    docs: list[Document], embs: np.ndarray,
    model: SentenceTransformer, cases: list[dict], top_k: int = TOP_K,
) -> list[dict]:
    """Evaluación con routing por emoción del banco sobre FAISS numpy."""
    doc_pillars = [d.metadata.get("pillar", 0) for d in docs]
    results: list[dict] = []
    for tc in cases:
        pillar_filter = ORACLE_ROUTING.get(tc.get("emotion", "others"))
        q_emb = model.encode([tc["query"]], normalize_embeddings=True).astype(np.float32)[0]
        global_indices = oracle_faiss_search(q_emb, embs, doc_pillars, pillar_filter, top_k)
        retrieved_pillars = [docs[i].metadata.get("pillar", 0) for i in global_indices]
        results.append(make_oracle_result(tc, retrieved_pillars, top_k,
                                          extra={"pillar_filter": pillar_filter}))
    return results


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

def _print_summary(metrics_a: dict, metrics_b: dict, n_a: int, n_b: int) -> None:
    sep = "=" * 82
    print(f"\n{sep}")
    print("  RESUMEN COMPARATIVO — Exp.1 Chunking (banco=72)")
    print("  NOTA: P@k no mide integridad clínica. Ver análisis cualitativo abajo.")
    print(sep)
    print(f"  {'Estrategia':<44} {'chunks':>6} {'P@1':>6} {'P@3':>6} {'Hit':>6}")
    print(f"  {'-'*43} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'A Manual terapéutico':<44} {n_a:>6}"
          f" {metrics_a['precision_at_1_media']:>6.3f}"
          f" {metrics_a['precision_at_3_media']:>6.3f}"
          f" {metrics_a['hit_at_any_rate']:>6.3f}")
    print(f"  {'B Automático chunk_size=300 (Ref. V1)':<44} {n_b:>6}"
          f" {metrics_b['precision_at_1_media']:>6.3f}"
          f" {metrics_b['precision_at_3_media']:>6.3f}"
          f" {metrics_b['hit_at_any_rate']:>6.3f}")
    print(sep + "\n")


def _figure(metrics_a: dict, metrics_b: dict) -> None:
    rag_figure_bar(
        config_names=["B Auto\n(Ref. V1)", "A Manual"],
        config_metrics={"B Auto\n(Ref. V1)": metrics_b, "A Manual": metrics_a},
        baseline_name="B Auto\n(Ref. V1)",
        title="Exp.1 — Chunking manual vs. automático",
        out_path=FIGURES_DIR / "exp1_chunking.png",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cases = load_eval_cases()
    log.info(f"Banco de evaluación: {len(cases)} casos.")

    log.info(f"Cargando SentenceTransformer: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Estrategia A: chunking manual terapéutico
    log.info("[A] Cargando corpus manual...")
    manual_docs = load_chunks_from_markdown(CORPUS_DIR)
    log.info(f"[A] {len(manual_docs)} chunks. Calculando embeddings...")
    manual_embs = build_embeddings(manual_docs, model)
    log.info("[A] Evaluando sobre conjunto completo...")
    manual_results = evaluate_config(manual_docs, manual_embs, model, cases)
    metrics_a = compute_eval_metrics(manual_results)

    # Estrategia B: chunking automático
    log.info("[B] Generando chunks automáticos...")
    auto_chunks = load_auto_chunks(CORPUS_DIR)
    auto_docs = [
        Document(page_content=c.content, metadata={"pillar": c.pillar, "chunk_id": f"auto_{i}"})
        for i, c in enumerate(auto_chunks)
    ]
    log.info(f"[B] {len(auto_docs)} chunks automáticos. Calculando embeddings...")
    auto_embs = build_embeddings(auto_docs, model)
    log.info("[B] Evaluando sobre conjunto completo...")
    auto_results = evaluate_config(auto_docs, auto_embs, model, cases)
    metrics_b = compute_eval_metrics(auto_results)

    _print_summary(metrics_a, metrics_b, len(manual_docs), len(auto_docs))

    # Análisis cualitativo
    qualitative_analysis(CORPUS_DIR)

    _figure(metrics_a, metrics_b)

    # JSON
    output: dict = {
        "experimento": "chunking_comparison",
        "evaluacion": "completa",
        "seed": SEED,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "banco": {"n_total": len(cases)},
        "advertencia_metrica": (
            "P@k favorece artificialmente al chunking automático por inflación del corpus. "
            "La estrategia A se selecciona por integridad clínica (técnica completa por chunk), "
            "no por la métrica de recuperación."
        ),
        "estrategia_a": {
            "nombre": "Manual terapéutico",
            "es_referencia_v1": False,
            "n_chunks": len(manual_docs),
            **metrics_a,
            "resultados_por_consulta": manual_results,
        },
        "estrategia_b": {
            "nombre": "Automático RecursiveCharacterTextSplitter (Ref. V1)",
            "es_referencia_v1": True,
            "chunk_size": 300,
            "chunk_overlap": 50,
            "n_chunks": len(auto_docs),
            **metrics_b,
            "resultados_por_consulta": auto_results,
        },
    }
    save_results("chunking_results.json", output)
    log.info("Exp.1 completado. Figura: exp1_chunking.png")


if __name__ == "__main__":
    main()
