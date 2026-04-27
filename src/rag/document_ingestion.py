"""
Pipeline de ingesta de documentos RAG y recuperador semántico.

Uso como script (construye el índice desde cero):
    python -m src.rag.document_ingestion

Uso como módulo (recuperar en el pipeline):
    from src.rag.document_ingestion import RAGRetriever
    retriever = RAGRetriever()
    results = retriever.retrieve("me están amenazando", emotion="fear")
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CORPUS_DIR = Path("data/rag_corpus")
VECTORSTORE_DIR = Path("data/vectorstore/faiss_index_v1")
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

# Emociones para las que se fuerza la inclusión de chunks always_include.
# Son las etiquetas de alta urgencia del clasificador BETO.
ALWAYS_INCLUDE_EMOTIONS: frozenset[str] = frozenset({"fear", "sadness"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """Representa un único chunk del corpus RAG con su front matter y contenido.

    Atributos:
        id: Identificador único del chunk (ej. P1_001).
        pillar: Número de pilar temático (1-5).
        title: Título descriptivo del chunk.
        source: Cita bibliográfica de la fuente.
        emotions: Lista de emociones BETO que activan este chunk.
        trigger: 'always_include' o 'conditional'.
        therapeutic_technique: Técnica terapéutica asociada (puede ser None).
        audience: Audiencia objetivo ('adolescente').
        content: Texto completo del chunk sin el front matter.
    """

    id: str
    pillar: int
    title: str
    source: str
    emotions: list[str] = field(default_factory=list)
    trigger: str = "conditional"
    therapeutic_technique: str | None = None
    audience: str = "adolescente"
    content: str = ""


@dataclass
class RetrievedChunk:
    """Resultado de una consulta al índice RAG.

    Atributos:
        chunk: Chunk recuperado con todos sus metadatos.
        score: Similitud coseno con la consulta (0-1). Valor 1.0 cuando forced=True.
        forced: True cuando el chunk fue incluido por la regla always_include,
                no por similitud semántica.
    """

    chunk: Chunk
    score: float
    forced: bool = False


# ---------------------------------------------------------------------------
# DocumentIngester
# ---------------------------------------------------------------------------


class DocumentIngester:
    """Parsea el corpus RAG, genera embeddings e indexa en FAISS.

    Genera un índice IndexFlatIP (producto interior) sobre vectores normalizados,
    equivalente a similitud coseno. Guarda el índice binario y un archivo JSON
    con los metadatos paralelos en vectorstore_dir.

    Args:
        corpus_dir: Directorio con los archivos .md del corpus.
        vectorstore_dir: Directorio destino para el índice y los metadatos.
        model_name: Nombre del modelo sentence-transformers para los embeddings.
    """

    def __init__(
        self,
        corpus_dir: Path = CORPUS_DIR,
        vectorstore_dir: Path = VECTORSTORE_DIR,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_md_file(self, path: Path) -> list[Chunk]:
        """Parsea un archivo .md con múltiples chunks delimitados por front matter YAML.

        Cada chunk tiene la estructura:
            ---
            campo: valor
            ---
            Contenido del chunk en texto libre.

        Args:
            path: Ruta al archivo .md.

        Returns:
            Lista de Chunk extraídos del archivo.
        """
        text = path.read_text(encoding="utf-8")
        # Divide en los delimitadores '---' que ocupan una línea completa
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        # Resultado: ["", yaml1, content1, yaml2, content2, ...]
        # El primer elemento es cadena vacía (o whitespace antes del primer ---)
        parts = [p for p in parts if p.strip()]

        chunks: list[Chunk] = []
        for i in range(0, len(parts) - 1, 2):
            yaml_str = parts[i]
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            try:
                meta: dict = yaml.safe_load(yaml_str) or {}
            except yaml.YAMLError:
                continue

            if "id" not in meta:
                continue

            chunks.append(
                Chunk(
                    id=str(meta["id"]),
                    pillar=int(meta["pillar"]),
                    title=str(meta["title"]),
                    source=str(meta["source"]),
                    emotions=list(meta.get("emotions") or []),
                    trigger=str(meta.get("trigger", "conditional")),
                    therapeutic_technique=meta.get("therapeutic_technique"),
                    audience=str(meta.get("audience", "adolescente")),
                    content=content,
                )
            )
        return chunks

    def load_corpus(self) -> list[Chunk]:
        """Carga todos los chunks de los archivos .md del corpus.

        Returns:
            Lista de todos los Chunk, ordenados por nombre de archivo.
        """
        all_chunks: list[Chunk] = []
        md_files = sorted(self.corpus_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(
                f"No se encontraron archivos .md en {self.corpus_dir}"
            )
        for md_file in md_files:
            parsed = self.parse_md_file(md_file)
            all_chunks.extend(parsed)
        return all_chunks

    # ------------------------------------------------------------------
    # Construcción del índice
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Lee el corpus, genera embeddings e indexa en FAISS. Guarda en disco.

        Sobrescribe cualquier índice previo en vectorstore_dir.
        """
        chunks = self.load_corpus()

        texts = [c.content for c in chunks]
        embeddings: np.ndarray = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).astype(np.float32)

        # IndexFlatIP con vectores normalizados = similitud coseno exacta
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.vectorstore_dir / "index.faiss"))

        metadata = [
            {
                "id": c.id,
                "pillar": c.pillar,
                "title": c.title,
                "source": c.source,
                "emotions": c.emotions,
                "trigger": c.trigger,
                "therapeutic_technique": c.therapeutic_technique,
                "audience": c.audience,
                "content": c.content,
            }
            for c in chunks
        ]
        (self.vectorstore_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _print_summary(chunks, self.vectorstore_dir)


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """Recuperador semántico sobre el índice FAISS del corpus RAG.

    Implementa dos modalidades de recuperación:
    - Semántica: top-k chunks por similitud coseno con la consulta.
    - Forzada: chunks con trigger='always_include' se añaden siempre cuando
      la emoción detectada pertenece a ALWAYS_INCLUDE_EMOTIONS.

    Args:
        vectorstore_dir: Directorio con index.faiss y metadata.json.
        model_name: Nombre del modelo sentence-transformers para los embeddings.
    """

    def __init__(
        self,
        vectorstore_dir: Path = VECTORSTORE_DIR,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.vectorstore_dir = Path(vectorstore_dir)
        self.model = SentenceTransformer(model_name)
        self._load()

    def _load(self) -> None:
        """Carga el índice FAISS y los metadatos desde disco."""
        index_path = self.vectorstore_dir / "index.faiss"
        meta_path = self.vectorstore_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Índice no encontrado en {self.vectorstore_dir}. "
                "Ejecuta DocumentIngester.build_index() primero."
            )

        self.index: faiss.IndexFlatIP = faiss.read_index(str(index_path))
        self.metadata: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))

        # Precalcular índices de chunks always_include para búsqueda O(1)
        self._always_include_indices: list[int] = [
            i
            for i, m in enumerate(self.metadata)
            if m.get("trigger") == "always_include"
        ]

    def retrieve(
        self,
        query: str,
        emotion: str | None = None,
        top_k: int = 3,
        pillar: int | None = None,
    ) -> list[RetrievedChunk]:
        """Recupera los chunks más relevantes para una consulta.

        Args:
            query: Texto del usuario sobre el que buscar en el corpus.
            emotion: Emoción detectada por el clasificador (anger, fear, sadness,
                     disgust, joy, surprise, others). Si es fear o sadness,
                     se añaden los chunks always_include al resultado.
            top_k: Número máximo de chunks semánticos a devolver.
            pillar: Si se especifica (1-5), filtra los resultados semánticos
                    al pilar indicado. Los chunks always_include se incluyen
                    siempre, independientemente de este filtro.

        Returns:
            Lista de RetrievedChunk. Los semánticos primero (ordenados por score
            descendente), seguidos de los forced al final.
        """
        query_vec = (
            self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        )

        # Over-fetch para poder aplicar el filtro por pilar sin quedarnos cortos
        fetch_k = min(top_k * 6 if pillar is not None else top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)

        seen_ids: set[str] = set()
        results: list[RetrievedChunk] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or len(results) == top_k:
                break
            meta = self.metadata[idx]
            if pillar is not None and meta["pillar"] != pillar:
                continue
            chunk_id: str = meta["id"]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            results.append(RetrievedChunk(chunk=Chunk(**meta), score=float(score)))

        # Regla always_include: añadir chunks de crisis cuando la emoción es de alerta
        if emotion in ALWAYS_INCLUDE_EMOTIONS:
            for idx in self._always_include_indices:
                meta = self.metadata[idx]
                if meta["id"] not in seen_ids:
                    seen_ids.add(meta["id"])
                    results.append(
                        RetrievedChunk(chunk=Chunk(**meta), score=1.0, forced=True)
                    )

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_summary(chunks: list[Chunk], vectorstore_dir: Path) -> None:
    """Imprime el resumen de la ingesta al stdout."""
    by_pillar = Counter(c.pillar for c in chunks)
    n_always = sum(1 for c in chunks if c.trigger == "always_include")

    print(f"\n{'=' * 52}")
    print(f"  Ingesta completada — {len(chunks)} chunks indexados")
    print(f"{'=' * 52}")
    print("  Distribución por pilar:")
    for p in sorted(by_pillar):
        print(f"    Pilar {p}: {by_pillar[p]:>2} chunks")
    print(f"  Chunks always_include : {n_always}")
    print(f"  Dimensión embeddings  : {EMBEDDING_DIM}")
    print(f"  Índice guardado en    : {vectorstore_dir}")
    print(f"{'=' * 52}\n")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingesta del corpus RAG y test de recuperación."
    )
    parser.add_argument(
        "--query",
        default="me están amenazando y tengo mucho miedo",
        help="Consulta de prueba para validar el retriever tras la ingesta.",
    )
    parser.add_argument(
        "--emotion",
        default="fear",
        choices=["anger", "disgust", "fear", "joy", "sadness", "surprise", "others"],
        help="Emoción detectada para la consulta de prueba.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Número de chunks semánticos a recuperar.",
    )
    parser.add_argument(
        "--pillar",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Filtrar resultados por pilar temático.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Saltar la ingesta y usar el índice existente directamente.",
    )
    return parser.parse_args()


def main() -> None:
    """Construye el índice FAISS y valida el retriever con una consulta de prueba."""
    args = _parse_args()

    if not args.skip_ingest:
        print("Iniciando ingesta del corpus RAG...")
        ingester = DocumentIngester()
        ingester.build_index()

    print(f"\nTest de recuperación")
    print(f"  Query  : {args.query}")
    print(f"  Emoción: {args.emotion}")
    print(f"  Top-k  : {args.top_k}")
    if args.pillar:
        print(f"  Pilar  : {args.pillar}")
    print()

    retriever = RAGRetriever()
    results = retriever.retrieve(
        query=args.query,
        emotion=args.emotion,
        top_k=args.top_k,
        pillar=args.pillar,
    )

    for i, r in enumerate(results, 1):
        forced_tag = " [FORZADO]" if r.forced else ""
        print(
            f"  {i}. [{r.chunk.id}] P{r.chunk.pillar} — {r.chunk.title}"
            f"\n     score={r.score:.4f}{forced_tag}"
            f"\n     trigger={r.chunk.trigger} | emotions={r.chunk.emotions}"
        )


if __name__ == "__main__":
    main()
