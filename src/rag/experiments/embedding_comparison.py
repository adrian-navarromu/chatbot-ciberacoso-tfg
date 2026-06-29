"""
Experimento 2: Comparación de modelos de embeddings para RAG de ciberacoso.

Banco de evaluación: 72 casos (expected_pillars ≠ ∅, crisis_expected ≠ HIGH).
La emoción del banco se inyecta directamente al routing (sin clasificador externo).
Usa chunking manual terapéutico y FAISS numpy para aislar el impacto del embedding.

Modelos evaluados:
    paraphrase-multilingual-mpnet   (Ref. V1 — genérico multilingüe)
    multilingual-e5-large           (estado del arte, prefijos query:/passage:)
    BAAI/bge-m3                     (estado del arte 2024)
    paraphrase-spanish-distilroberta (español nativo, Knowledge Distillation)

Requiere GPU (~1–2 GB VRAM por modelo × 4 modelos).

Uso:
    conda activate chatbots
    python src/rag/experiments/embedding_comparison.py
"""

from __future__ import annotations

import sys
import time
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

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
    CORPUS_DIR, FIGURES_DIR,
    ORACLE_ROUTING,
    load_eval_cases, oracle_faiss_search, make_oracle_result,
    compute_eval_metrics, rag_figure_bar,
    load_chunks_from_markdown, save_results,
)

TOP_K = 3
BASELINE_MODEL = "paraphrase-multilingual-mpnet"


@dataclass
class ModelConfig:
    short_name: str
    model_id: str
    dim: int
    requires_e5_prefix: bool = False
    description: str = ""
    is_reference_v1: bool = False

    def __str__(self) -> str:
        return self.short_name


MODELS: list[ModelConfig] = [
    ModelConfig(
        short_name="paraphrase-multilingual-mpnet",
        model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        dim=768,
        is_reference_v1=True,
        description="Ref. V1 · genérico multilingüe · dim=768 · ~278M params",
    ),
    ModelConfig(
        short_name="multilingual-e5-large",
        model_id="intfloat/multilingual-e5-large",
        dim=1024,
        requires_e5_prefix=True,
        description="dim=1024 · ~560M params · prefijos query:/passage: (Wang et al. 2024)",
    ),
    ModelConfig(
        short_name="BAAI/bge-m3",
        model_id="BAAI/bge-m3",
        dim=1024,
        description="dim=1024 · ~568M params · Apache 2.0 · estado del arte multilingual 2024",
    ),
    ModelConfig(
        short_name="paraphrase-spanish-distilroberta",
        model_id="hackathon-pln-es/paraphrase-spanish-distilroberta",
        dim=768,
        description="español nativo · KD (Reimers & Gurevych EMNLP 2020) · dim=768",
    ),
]


class E5LargeEmbeddings:
    def __init__(self) -> None:
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            [f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=32
        ).astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0].astype(np.float32).tolist()

    def embed_queries_timed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        latencies, embeddings = [], []
        for t in texts:
            t0 = time.perf_counter()
            emb = self.model.encode([f"query: {t}"], normalize_embeddings=True)[0]
            latencies.append((time.perf_counter() - t0) * 1000)
            embeddings.append(emb.astype(np.float32).tolist())
        return embeddings, float(np.mean(latencies))


def load_embeddings(config: ModelConfig) -> Any:
    if config.requires_e5_prefix:
        log.info("  Usando E5LargeEmbeddings (prefijos query:/passage:)")
        return E5LargeEmbeddings()
    return HuggingFaceEmbeddings(
        model_name=config.model_id,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def measure_query_latency(emb_obj: Any, queries: list[str]) -> tuple[list[list[float]], float]:
    if isinstance(emb_obj, E5LargeEmbeddings):
        return emb_obj.embed_queries_timed(queries)
    latencies, query_embs = [], []
    for q in queries:
        t0 = time.perf_counter()
        emb = emb_obj.embed_query(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        query_embs.append(emb)
    return query_embs, float(np.mean(latencies))


def evaluate_model(
    config: ModelConfig,
    docs: list,
    cases: list[dict],
    top_k: int = TOP_K,
) -> dict:
    """Evalúa un modelo de embeddings sobre el conjunto de evaluación completo."""
    tag = " (Ref. V1)" if config.is_reference_v1 else ""
    log.info(f"Evaluando: {config.short_name}{tag}")
    emb_obj = load_embeddings(config)

    t0 = time.perf_counter()
    doc_embs = np.array(
        emb_obj.embed_documents([d.page_content for d in docs]), dtype=np.float32
    )
    t_corpus = (time.perf_counter() - t0) * 1000
    log.info(f"  Corpus ({len(docs)} chunks) codificado en {t_corpus:.0f} ms.")

    doc_pillars = [d.metadata.get("pillar", 0) for d in docs]
    all_queries = [tc["query"] for tc in cases]
    query_embs_list, latencia_ms = measure_query_latency(emb_obj, all_queries)
    query_embs = np.array(query_embs_list, dtype=np.float32)
    log.info(f"  Latencia media query: {latencia_ms:.1f} ms")

    results: list[dict] = []
    for tc, q_emb in zip(cases, query_embs):
        pillar_filter = ORACLE_ROUTING.get(tc.get("emotion", "others"))
        global_indices = oracle_faiss_search(q_emb, doc_embs, doc_pillars, pillar_filter, top_k)
        retrieved_pillars = [docs[i].metadata.get("pillar", 0) for i in global_indices]
        results.append(make_oracle_result(tc, retrieved_pillars, top_k,
                                          extra={"pillar_filter": pillar_filter}))

    eval_m = compute_eval_metrics(results)
    log.info(f"  P@1={eval_m['precision_at_1_media']:.3f} "
             f"P@3={eval_m['precision_at_3_media']:.3f} "
             f"Hit={eval_m['hit_at_any_rate']:.3f}")

    return {
        "short_name": config.short_name,
        "model_id": config.model_id,
        "dim": config.dim,
        "description": config.description,
        "es_referencia_v1": config.is_reference_v1,
        "n_chunks": len(docs),
        "latencia_media_ms": round(latencia_ms, 2),
        **eval_m,
        "resultados_por_consulta": results,
    }


def _print_results_table(all_results: list[dict]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  TABLA COMPARATIVA — MODELOS DE EMBEDDINGS (banco=72)")
    print(sep)
    print(f"  {'Modelo':<37} {'dim':>5} {'P@1':>6} {'P@3':>6} {'Hit':>6} {'lat_ms':>8}")
    print(f"  {'-'*36} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for r in all_results:
        tag = " [Ref.V1]" if r["es_referencia_v1"] else ""
        print(f"  {r['short_name'] + tag:<37} {r['dim']:>5}"
              f" {r['precision_at_1_media']:>6.3f} {r['precision_at_3_media']:>6.3f}"
              f" {r['hit_at_any_rate']:>6.3f} {r['latencia_media_ms']:>8.1f}")
    print(sep + "\n")


def _figure(all_results: list[dict]) -> None:
    config_names = [r["short_name"] for r in all_results]
    config_metrics = {r["short_name"]: r for r in all_results}
    rag_figure_bar(
        config_names=config_names,
        config_metrics=config_metrics,
        baseline_name=BASELINE_MODEL,
        title="Exp.2 — Modelos de embeddings",
        out_path=FIGURES_DIR / "exp2_embeddings.png",
    )


def main() -> None:
    cases = load_eval_cases()
    log.info(f"Banco de evaluación: {len(cases)} casos.")

    docs = load_chunks_from_markdown(CORPUS_DIR)
    log.info(f"Corpus: {len(docs)} chunks (chunking manual terapéutico).")

    all_results: list[dict] = []
    for config in MODELS:
        result = evaluate_model(config, docs, cases, TOP_K)
        all_results.append(result)

    _print_results_table(all_results)
    _figure(all_results)

    prec_vals = [r["precision_at_3_media"] for r in all_results]
    output: dict = {
        "experimento": "embedding_comparison",
        "evaluacion": "completa",
        "seed": SEED,
        "chunking": "manual_terapeutico",
        "top_k": TOP_K,
        "banco": {"n_total": len(cases)},
        "max_diff_precision_at_3": round(max(prec_vals) - min(prec_vals), 4),
        "modelos_evaluados": all_results,
    }
    save_results("embedding_results.json", output)
    log.info("Exp.2 completado. Figura: exp2_embeddings.png")


if __name__ == "__main__":
    main()
