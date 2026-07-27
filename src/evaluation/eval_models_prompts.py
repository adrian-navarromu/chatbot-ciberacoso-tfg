"""
Generación factorial SLM × variantes de prompt.

Ejecuta el pipeline V2 completo (CrisisDetector → RAG Config C → SLM Ollama)
para las 2000 combinaciones de 5 modelos × 4 variantes × 100 casos del banco
y vuelca todas las generaciones en results/generations/slm_prompt_generations.csv

Este script es exclusivamente de GENERACIÓN.
La evaluación con rúbrica es la Tarea 7 (juez LLM sobre este CSV).

Flujo por (modelo, variante, caso):
  - HIGH detectado  → respuesta determinista del CrisisDetector. Sin llamada SLM.
  - MEDIUM detectado → build_crisis_prompt_variant(variante) + generación SLM.
  - NONE            → build_prompt_variant(variante, emocion_gold) + generación SLM.

La crisis detection y el RAG son deterministas e independientes del modelo SLM:
se pre-computan una vez para los 100 casos y se cachean.

Ejecución:
    conda activate chatbots
    python src/evaluation/eval_models_prompts.py

Salida: results/generations/slm_prompt_generations.csv  (~2000 filas)
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Añadir raíz del proyecto al path para imports src.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.evaluation.test_bank import SEED, TEST_BANK
from src.prompts.experiment_prompts import build_crisis_prompt_variant, build_prompt_variant, TEMPERATURE_BY_EMOTION
from src.rag.document_ingestion_v2 import DocumentIngesterV2
from src.rag.enriched_retriever import EnrichedRetriever
from src.safety.crisis_detector import CrisisDetector

# Silenciar logs de nivel INFO de los módulos internos durante la ejecución larga
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración fija del experimento (Tarea 5)
# ---------------------------------------------------------------------------
MODELS: list[str] = ["mistral:7b", "gemma:2b", "gemma:7b", "phi3:mini", "tinyllama"]
VARIANTS: list[str] = ["A", "B", "C", "D"]
TREND: str = "estable"
CONFIDENCE: float = 1.0
OUTPUT_PATH: Path = _PROJECT_ROOT / "eval" / "generations" / "slm_prompt_generations.csv"

FIELDNAMES: list[str] = [
    "modelo",
    "variante",
    "id_caso",
    "intent_type",
    "emocion_gold",
    "crisis_expected",
    "crisis_detected",
    "crisis_acierto",
    "crisis_subtype",
    "rama",
    "expected_pillars",
    "pillars_recuperados",
    "trend",
    "confidence",
    "query",
    "respuesta",
    "latencia_ms",
    "n_palabras",
    "seed",
    "genera_slm",
    "es_adversarial",
    "tiene_pilar_esperado",
]


# ---------------------------------------------------------------------------
# Inicialización del recuperador RAG (Config C)
# ---------------------------------------------------------------------------

def _build_retriever() -> EnrichedRetriever:
    """Carga el corpus RAG y devuelve el recuperador semántico Config C."""
    corpus_dir = str(_PROJECT_ROOT / "data" / "rag_corpus")
    chroma_path = str(_PROJECT_ROOT / "data" / "vectorstore" / "chroma_v2")
    embedding_model = "hackathon-pln-es/paraphrase-spanish-distilroberta"

    ingester = DocumentIngesterV2(corpus_dir=corpus_dir, embedding_model=embedding_model)
    chunks = ingester.load_corpus()
    documents: list[Document] = [
        Document(
            page_content=c.content,
            metadata={"chunk_id": c.id, "pillar": c.pillar, "title": c.title},
        )
        for c in chunks
    ]
    return EnrichedRetriever(chroma_path, documents, embedding_model)


# ---------------------------------------------------------------------------
# Pre-cómputo por caso (independiente de modelo y variante)
# ---------------------------------------------------------------------------

def _precompute_cache(
    cases: list[dict],
    detector: CrisisDetector,
    retriever: EnrichedRetriever,
) -> dict[str, dict]:
    """Calcula crisis detection y recuperación RAG para cada caso del banco.

    Ambos son deterministas e independientes del modelo SLM y la variante de
    prompt. Pre-computarlos una sola vez evita 19 recálculos por caso
    (5 modelos × 4 variantes − 1).
    """
    cache: dict[str, dict] = {}
    for i, case in enumerate(cases, 1):
        query = case["query"]
        crisis_result = detector.detect(query)

        if crisis_result.level == "HIGH":
            docs: list[Document] = []
        elif crisis_result.level == "MEDIUM" and crisis_result.requires_generation:
            docs = retriever.retrieve(
                query,
                pillar_filter=[crisis_result.suggested_pillar],
            )
        else:
            docs = retriever.retrieve_with_routing(
                query,
                emotion=case["emotion"],
                trend=TREND,
            )

        cache[case["id"]] = {
            "crisis_result": crisis_result,
            "rag_context": "\n---\n".join(d.page_content for d in docs),
            "pillars_recuperados": [d.metadata.get("pillar") for d in docs],
        }

        if i % 25 == 0:
            print(f"    {i}/{len(cases)} casos pre-computados...")

    return cache


# ---------------------------------------------------------------------------
# Llamada al SLM con manejo de errores
# ---------------------------------------------------------------------------

def _call_slm(
    model_name: str,
    messages: list[dict],
    temperature: float,
    num_predict: int,
) -> tuple[str, float, str | None]:
    """Invoca ChatOllama y devuelve (respuesta, latencia_ms, error_msg|None)."""
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
        seed=SEED   
    )
    t0 = time.time()
    try:
        result = llm.invoke(messages)
        latency_ms = round((time.time() - t0) * 1000, 1)
        return result.content, latency_ms, None
    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return "", latency_ms, str(exc)


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TAREA 5 — Generación factorial SLM × variantes de prompt")
    print("=" * 60)

    print("\n[1/3] Inicializando RAG (ChromaDB + embeddings)...")
    retriever = _build_retriever()

    detector = CrisisDetector()

    print(f"\n[2/3] Pre-computando crisis detection + RAG ({len(TEST_BANK)} casos)...")
    case_cache = _precompute_cache(TEST_BANK, detector, retriever)

    total_expected = len(MODELS) * len(VARIANTS) * len(TEST_BANK)
    print(
        f"\n[3/3] Generando: {len(MODELS)} modelos × {len(VARIANTS)} variantes "
        f"× {len(TEST_BANK)} casos = {total_expected} filas\n"
    )

    errors: list[str] = []
    # Contadores para el resumen (se llenan al vuelo)
    rama_counts: Counter = Counter()
    latencias_por_modelo: dict[str, list[float]] = defaultdict(list)
    crisis_hits_por_combo: dict[tuple, int] = {}

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for model in MODELS:
            print(f"╔══ Modelo: {model}")

            for variant in VARIANTS:
                print(f"  ▶ Variante {variant} ... ", end="", flush=True)
                t_variant = time.time()
                crisis_hits = 0

                for case in TEST_BANK:
                    cid = case["id"]
                    cached = case_cache[cid]
                    crisis_result = cached["crisis_result"]
                    rag_context = cached["rag_context"]
                    pillars_recuperados = cached["pillars_recuperados"]

                    crisis_detected = crisis_result.level
                    crisis_subtype = crisis_result.subtype if crisis_detected == "HIGH" else ""
                    crisis_acierto = crisis_detected == case["crisis_expected"]
                    if crisis_acierto:
                        crisis_hits += 1

                    # --- Rama del pipeline ---
                    if crisis_detected == "HIGH":
                        rama = "HIGH_determinista"
                        respuesta = crisis_result.response or ""
                        latencia_ms = 0.0

                    elif crisis_detected == "MEDIUM" and crisis_result.requires_generation:
                        rama = "MEDIUM_generado"
                        messages = build_crisis_prompt_variant(
                            variant=variant,
                            rag_context=rag_context,
                            history=[],
                        )
                        messages.append({"role": "user", "content": case["query"]})
                        respuesta, latencia_ms, err = _call_slm(
                            model_name=model,
                            messages=messages,
                            temperature=0.5,
                            num_predict=200,
                        )
                        if err:
                            errors.append(f"{model}/{variant}/{cid}: {err}")
                        latencias_por_modelo[model].append(latencia_ms)

                    else:
                        rama = "normal_generado"
                        messages = build_prompt_variant(
                            variant=variant,
                            emotion=case["emotion"],
                            rag_context=rag_context,
                            history=[],
                            confidence=CONFIDENCE,
                            emotional_context="",
                            trend=TREND,
                        )
                        messages.append({"role": "user", "content": case["query"]})
                        temperature = TEMPERATURE_BY_EMOTION.get(case["emotion"], 0.7)
                        respuesta, latencia_ms, err = _call_slm(
                            model_name=model,
                            messages=messages,
                            temperature=temperature,
                            num_predict=300,
                        )
                        if err:
                            errors.append(f"{model}/{variant}/{cid}: {err}")
                        latencias_por_modelo[model].append(latencia_ms)

                    rama_counts[rama] += 1
                    genera_slm = rama != "HIGH_determinista"

                    writer.writerow({
                        "modelo": model,
                        "variante": variant,
                        "id_caso": cid,
                        "intent_type": case["intent_type"],
                        "emocion_gold": case["emotion"],
                        "crisis_expected": case["crisis_expected"],
                        "crisis_detected": crisis_detected,
                        "crisis_acierto": crisis_acierto,
                        "crisis_subtype": crisis_subtype,
                        "rama": rama,
                        "expected_pillars": sorted(case["expected_pillars"]),
                        "pillars_recuperados": pillars_recuperados,
                        "trend": TREND,
                        "confidence": CONFIDENCE,
                        "query": case["query"],
                        "respuesta": respuesta,
                        "latencia_ms": latencia_ms,
                        "n_palabras": len(respuesta.split()),
                        "seed": SEED,
                        "genera_slm": genera_slm,
                        "es_adversarial": case["intent_type"] == "adversarial",
                        "tiene_pilar_esperado": len(case["expected_pillars"]) > 0,
                    })
                    csvfile.flush()

                elapsed = time.time() - t_variant
                crisis_hits_por_combo[(model, variant)] = crisis_hits
                print(f"{len(TEST_BANK)} casos OK  ({elapsed:.0f}s)")

            print()

    _print_summary(rama_counts, crisis_hits_por_combo, latencias_por_modelo, errors)


# ---------------------------------------------------------------------------
# Resumen de control (Cambio 3)
# ---------------------------------------------------------------------------

def _print_summary(
    rama_counts: Counter,
    crisis_hits_por_combo: dict[tuple, int],
    latencias_por_modelo: dict[str, list[float]],
    errors: list[str],
) -> None:
    total_rows = sum(rama_counts.values())
    total_expected = len(MODELS) * len(VARIANTS) * len(TEST_BANK)

    print("=" * 60)
    print("RESUMEN DE CONTROL — TAREA 5")
    print("=" * 60)
    print(f"Total filas generadas : {total_rows}  (esperado: {total_expected})")

    # Recuento por rama
    print("\nPor rama:")
    for rama in ["HIGH_determinista", "MEDIUM_generado", "normal_generado"]:
        print(f"  {rama:<25}  {rama_counts.get(rama, 0):>5}")

    # Consistencia del CrisisDetector
    # El detector es determinista → mismo resultado en todos los combos modelo×variante
    unique_hits = set(crisis_hits_por_combo.values())
    print(f"\nConsistencia CrisisDetector ({len(crisis_hits_por_combo)} combos modelo×variante):")
    if len(unique_hits) == 1:
        hits = unique_hits.pop()
        print(f"  OK — {hits}/{len(TEST_BANK)} casos correctos en todos los combos (homogéneo)")
    else:
        print(f"  AVISO — variación inesperada entre combos: valores únicos = {unique_hits}")
        for combo, hits in sorted(crisis_hits_por_combo.items()):
            if hits != max(unique_hits):
                print(f"    {combo[0]}/{combo[1]}: {hits}/{len(TEST_BANK)}")

    # Latencia media por modelo
    print("\nLatencia media de generación SLM por modelo:")
    for model in MODELS:
        lats = latencias_por_modelo.get(model, [])
        if lats:
            media = sum(lats) / len(lats)
            p50 = sorted(lats)[len(lats) // 2]
            print(f"  {model:<15}  media={media:>7.0f}ms  mediana={p50:>7.0f}ms  (n={len(lats)})")

    # Respuestas vacías
    # (solo informativo; el CSV ya tiene todas las filas)
    slm_calls = sum(len(v) for v in latencias_por_modelo.values())
    print(f"\nTotal llamadas SLM realizadas: {slm_calls}")

    if errors:
        print(f"\nAVISO — {len(errors)} errores de llamada a Ollama:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} más")
    else:
        print("\nSin errores de llamada a Ollama.")

    print(f"\nCSV guardado en: {OUTPUT_PATH}")
    print("=" * 60)
    print("\nSiguiente paso — Tarea 7: evaluación con rúbrica (juez LLM) sobre este CSV.")


if __name__ == "__main__":
    main()
