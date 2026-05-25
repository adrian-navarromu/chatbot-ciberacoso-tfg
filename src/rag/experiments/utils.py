"""
Utilidades compartidas para los experimentos de comparación RAG (Exp. 1-4).

Proporciona la fuente canónica de casos de test, funciones de métricas,
carga del corpus, construcción de índice FAISS en memoria y guardado de JSON.

Importar con:
    from src.rag.experiments.utils import (
        CORPUS_DIR, RESULTS_DIR,
        TEST_CASES, BM25_TEST_CASES,
        compute_precision_at_k, compute_precision_at_1, compute_hit_at_any,
        compute_metrics_for_run, print_results_table,
        load_chunks_from_markdown,
        build_faiss_index_from_embeddings,
        save_results,
    )
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Union

import faiss
import numpy as np
import yaml
from langchain_core.documents import Document

# Configuración del logger para consistencia con el proyecto
log = logging.getLogger(__name__)

# Rutas canónicas - fuente única para todos los experimentos
CORPUS_DIR: Path = Path("data/rag_corpus")
RESULTS_DIR: Path = Path("eval/rag_experiments")

# Casos de test canónicos - fuente única para todos los experimentos
TEST_CASES: list[dict] = [
    {
        "id": "Q01",
        "label": "Q1",
        "query": "tengo miedo de ir al instituto porque me amenazan por internet",
        "emotion": "fear",
        "expected_pillars": {1, 2},
        "set": "standard",
    },
    {
        "id": "Q02",
        "label": "Q2",
        "query": "me han publicado fotos mías sin permiso y no sé qué hacer",
        "emotion": "fear",
        "expected_pillars": {1, 5},
        "set": "standard",
    },
    {
        "id": "Q03",
        "label": "Q3",
        "query": "no puedo parar de llorar me siento muy mal y no sé por qué",
        "emotion": "sadness",
        "expected_pillars": {2},
        "set": "standard",
    },
    {
        "id": "Q04",
        "label": "Q4",
        "query": "todo el mundo me odia en clase y en los grupos de whatsapp me ignoran",
        "emotion": "sadness",
        "expected_pillars": {2},
        "set": "standard",
    },
    {
        "id": "Q05",
        "label": "Q5",
        "query": "quiero denunciar al que me acosa pero no sé cómo hacerlo",
        "emotion": "anger",
        "expected_pillars": {1},
        "set": "standard",
    },
    {
        "id": "Q06",
        "label": "Q6",
        "query": "cómo bloqueo a alguien en instagram que me está insultando",
        "emotion": "others",
        "expected_pillars": {5},
        "set": "standard",
    },
    {
        "id": "Q07",
        "label": "Q7",
        "query": "todo lo que hago sale mal y no sirvo para nada",
        "emotion": "sadness",
        "expected_pillars": {3},
        "set": "standard",
    },
    {
        "id": "Q08",
        "label": "Q8",
        "query": "me van a pegar si no hago lo que dicen me da muchísimo miedo",
        "emotion": "fear",
        "expected_pillars": {4},
        "set": "standard",
    },
    {
        "id": "Q09",
        "label": "Q9",
        "query": "ya no quiero seguir viviendo nadie me va a echar de menos",
        "emotion": "sadness",
        "expected_pillars": {4},
        "set": "standard",
    },
    {
        "id": "Q10",
        "label": "Q10",
        "query": "me da mucha vergüenza lo que pasó y no puedo contárselo a nadie",
        "emotion": "disgust",
        "expected_pillars": {2},
        "set": "standard",
    },
]

# Set de pruebas Lexical-Search (BM25). Evalúa la capacidad del sistema para recuperar entidades exactas, contrarrestando la debilidad de los modelos densos (embeddings) con acrónimos.
BM25_TEST_CASES: list[dict] = [
    {
        "id": "BM25_01",
        "label": "BM25_1",
        "query": "cómo bloqueo en instagram",
        "emotion": "others",
        "expected_pillars": {5},
        "set": "bm25",
        "note": "Nombre de plataforma exacto --> pillar5_digital",
    },
    {
        "id": "BM25_02",
        "label": "BM25_2",
        "query": "teléfono 024",
        "emotion": "fear",
        "expected_pillars": {4},
        "set": "bm25",
        "note": "Número de línea de crisis --> pillar4_pap",
    },
    {
        "id": "BM25_03",
        "label": "BM25_3",
        "query": "ANAR 900 202 010",
        "emotion": "fear",
        "expected_pillars": {3, 4},
        "set": "bm25",
        "note": "Número exacto ANAR --> pillar4_pap y pillar3_hope",
    },
    {
        "id": "BM25_04",
        "label": "BM25_4",
        "query": "discord bloquear usuario",
        "emotion": "others",
        "expected_pillars": {5},
        "set": "bm25",
        "note": "Plataforma Discord --> pillar5_digital",
    },
    {
        "id": "BM25_05",
        "label": "BM25_5",
        "query": "AEPD canal joven",
        "emotion": "others",
        "expected_pillars": {1, 4, 5},
        "set": "bm25",
        "note": "Sigla organismo --> pillar1, pillar4 y pillar5",
    },
    {
        "id": "BM25_06",
        "label": "BM25_6",
        "query": "llamar al 017 incibe",
        "emotion": "others",
        "expected_pillars": {1, 4, 5},
        "set": "bm25",
        "note": "Número y sigla clave en ciberseguridad (INCIBE)",
    },
    {
        "id": "BM25_07",
        "label": "BM25_7",
        "query": "me están haciendo sexting",
        "emotion": "fear",
        "expected_pillars": {1},
        "set": "bm25",
        "note": "Término técnico de ciberacoso",
    },
    {
        "id": "BM25_08",
        "label": "BM25_8",
        "query": "hacer capturas de pantalla notario",
        "emotion": "others",
        "expected_pillars": {1},
        "set": "bm25",
        "note": "Recolección de pruebas exactas --> pillar1_action",
    }
]


# Funciones de métricas
def compute_precision_at_k(retrieved: list[int], expected: set[int], k: int) -> float:
    """Fracción de los primeros k resultados que pertenecen a un pilar esperado.

    Args:
        retrieved: Lista de pilares recuperados (en orden de ranking).
        expected: Conjunto de pilares correctos para la consulta.
        k: Número de resultados a considerar.

    Returns:
        Valor entre 0.0 y 1.0, redondeado a 4 decimales.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for p in retrieved[:k] if p in expected)
    return round(hits / k, 4)


def compute_precision_at_1(retrieved: list[int], expected: set[int]) -> float:
    """1.0 si el primer resultado pertenece a un pilar esperado, 0.0 si no.

    Args:
        retrieved: Lista de pilares recuperados.
        expected: Conjunto de pilares correctos para la consulta.

    Returns:
        1.0 o 0.0.
    """
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] in expected else 0.0


def compute_hit_at_any(retrieved: list[int], expected: set[int]) -> bool:
    """True si al menos un resultado recuperado pertenece a un pilar esperado.

    Args:
        retrieved: Lista de pilares recuperados.
        expected: Conjunto de pilares correctos para la consulta.

    Returns:
        bool.
    """
    return any(p in expected for p in retrieved)


def compute_metrics_for_run(results: list[dict]) -> dict:
    """Calcula métricas agregadas para una lista de resultados por consulta.

    Args:
        results: Lista de dicts con al menos las claves 'precision_at_3' y
                 'hit_at_any'. Si los dicts incluyen 'precision_at_1',
                 también se calcula su media.

    Returns:
        Dict con 'precision_at_3_media', 'hit_at_any_rate' y, si está disponible
        en los resultados, 'precision_at_1_media'.
    """
    n = len(results)
    if n == 0:
        return {"precision_at_3_media": 0.0, "hit_at_any_rate": 0.0}

    p3 = round(sum(r["precision_at_3"] for r in results) / n, 4)
    hit = round(sum(1 for r in results if r["hit_at_any"]) / n, 4)
    out: dict = {"precision_at_3_media": p3, "hit_at_any_rate": hit}

    if "precision_at_1" in results[0]:
        p1 = round(sum(r["precision_at_1"] for r in results) / n, 4)
        out["precision_at_1_media"] = p1

    return out


# Presentación de resultados
def print_results_table(config_name: str, results: list[dict], show_emotion: bool = False) -> None:
    """Imprime tabla de resultados por consulta en formato de consola.

    Detecta automáticamente si los resultados incluyen 'precision_at_1'
    y añade esa columna a la tabla en ese caso.

    Args:
        config_name: Nombre de la configuración (aparece en la cabecera).
        results: Lista de dicts con métricas por consulta.
        show_emotion: Si True, incluye la columna de emoción detectada.
    """
    if not results:
        return

    has_p1 = "precision_at_1" in results[0]
    n = len(results)
    p3 = round(sum(r["precision_at_3"] for r in results) / n, 4)
    hit = round(sum(1 for r in results if r["hit_at_any"]) / n, 4)

    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {config_name}")
    print(sep)

    if show_emotion and has_p1:
        print(
            f"  {'Consulta':<8} {'Emoción':<10} {'Esperado':<12}"
            f" {'Recuperado':<16} {'P@1':>5} {'P@3':>6} {'Hit':>5}"
        )
        print(
            f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*5} {'-'*6} {'-'*5}"
        )
        for r in results:
            exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
            ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
            hit_sym = "V" if r["hit_at_any"] else "X"
            print(
                f"  {r['label']:<8} {r.get('emotion',''):<10} {exp_str:<12}"
                f" {ret_str:<16} {r.get('precision_at_1',0):>5.0f}"
                f" {r['precision_at_3']:>6.2f} {hit_sym:>5}"
            )
        print(
            f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*5} {'-'*6} {'-'*5}"
        )
        p1 = round(sum(r.get("precision_at_1", 0) for r in results) / n, 4)
        print(
            f"  {'MEDIA':<8} {'':<10} {'':<12} {'':<16}"
            f" {p1:>5.2f} {p3:>6.2f} {hit:>5.2f}"
        )

    elif show_emotion:
        print(
            f"  {'Consulta':<8} {'Emoción':<10} {'Esperado':<12}"
            f" {'Recuperado':<16} {'P@3':>6} {'Hit':>5}"
        )
        print(f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*6} {'-'*5}")
        for r in results:
            exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
            ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
            hit_sym = "V" if r["hit_at_any"] else "X"
            print(
                f"  {r['label']:<8} {r.get('emotion',''):<10} {exp_str:<12}"
                f" {ret_str:<16} {r['precision_at_3']:>6.2f} {hit_sym:>5}"
            )
        print(f"  {'-'*7} {'-'*9} {'-'*11} {'-'*15} {'-'*6} {'-'*5}")
        print(
            f"  {'MEDIA':<8} {'':<10} {'':<12} {'':<16}"
            f" {p3:>6.2f} {hit:>5.2f}"
        )

    else:
        print(
            f"  {'Consulta':<8} {'Esperado':<12} {'Recuperado':<18}"
            f" {'P@3':>6} {'Hit':>5}"
        )
        print(f"  {'-'*7} {'-'*11} {'-'*17} {'-'*6} {'-'*5}")
        for r in results:
            exp_str = ",".join(f"P{p}" for p in r["expected_pillars"])
            ret_str = ",".join(f"P{p}" for p in r["retrieved_pillars"])
            hit_sym = "V" if r["hit_at_any"] else "X"
            print(
                f"  {r['label']:<8} {exp_str:<12} {ret_str:<18}"
                f" {r['precision_at_3']:>6.2f} {hit_sym:>5}"
            )
        print(f"  {'-'*7} {'-'*11} {'-'*17} {'-'*6} {'-'*5}")
        print(
            f"  {'MEDIA':<8} {'':<12} {'':<18} {p3:>6.2f} {hit:>5.2f}"
        )

    print(sep)


# Índice FAISS en memoria
def build_faiss_index_from_embeddings(embeddings: Union[list[list[float]], np.ndarray]) -> faiss.IndexFlatIP:
    """Construye un índice FAISS IndexFlatIP en memoria a partir de vectores precalculados.

    IndexFlatIP con vectores normalizados equivale a similitud coseno exacta.
    No persiste en disco; reproducible en cada ejecución.

    Args:
        embeddings: Lista de vectores normalizados o array numpy (n × dim).

    Returns:
        Índice FAISS listo para búsqueda.
    """
    arr = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


# Guardado de resultados
def save_results(filename: str, output: dict) -> Path:
    """Guarda resultados en JSON dentro de RESULTS_DIR y confirma en consola.

    Args:
        filename: Nombre del archivo (ej. 'chunking_results.json').
        output: Diccionario serializable a JSON.

    Returns:
        Path del archivo guardado.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Resultados de evaluación guardados en {out_path}")
    return out_path


# Carga del corpus
def load_chunks_from_markdown(corpus_dir: Path) -> list[Document]:
    """Parsea los archivos .md del corpus RAG y devuelve Documents de LangChain.

    Fuente canónica de carga del corpus para todos los experimentos.
    Cada chunk está delimitado por bloques '---' con front matter YAML.
    Solo se cargan chunks con los campos 'id' y 'pillar' presentes en el YAML.

    Los metadatos se serializan compatibles con ChromaDB:
    - emotions (list) --> JSON string
    - therapeutic_technique (None) --> ""

    Args:
        corpus_dir: Directorio con los archivos .md del corpus RAG.

    Returns:
        Lista de Document de LangChain con page_content y metadatos completos.
    """
    documents: list[Document] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        parts = [p for p in parts if p.strip()]
        for i in range(0, len(parts) - 1, 2):
            try:
                meta: dict = yaml.safe_load(parts[i]) or {}
            except yaml.YAMLError:
                continue
            if "id" not in meta or "pillar" not in meta:
                continue
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": str(meta["id"]),
                        "pillar": int(meta["pillar"]),
                        "title": str(meta.get("title", "")),
                        "source": str(meta.get("source", "")),
                        "emotions": json.dumps(
                            list(meta.get("emotions") or []), ensure_ascii=False
                        ),
                        "trigger": str(meta.get("trigger", "conditional")),
                        "therapeutic_technique": meta.get("therapeutic_technique") or "",
                        "audience": str(meta.get("audience", "adolescente")),
                    },
                )
            )
    return documents
