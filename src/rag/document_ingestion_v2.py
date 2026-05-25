"""
Ingesta definitiva del corpus RAG para V2.

Configuración ganadora de los experimentos comparativos:
    - Chunking   : manual terapéutico — un chunk por unidad clínica completa
                   (Exp. 1 — precision@3 0.467 vs 0.433)
    - Embeddings : hackathon-pln-es/paraphrase-spanish-distilroberta
                   (Exp. 2 — p@3=0.600, p@1=0.800, latencia 3.7ms;
                   ganador frente a e5-large, bge-m3 y mpnet)
    - Vectorstore: ChromaDB SQLite (Exp. 3 — habilita routing por metadatos)
    - Recuperación: Híbrida BM25+semántica con routing emocional (Exp. 4)

Persiste el índice en data/vectorstore/chroma_v2/.
NO modifica data/vectorstore/faiss_index_v1/.

Uso como script (construye el índice desde cero):
    python src/rag/document_ingestion_v2.py

Uso como módulo (acceder a los chunks parseados):
    from src.rag.document_ingestion_v2 import DocumentIngesterV2
    ingester = DocumentIngesterV2()
    chunks = ingester.load_corpus()
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Constantes

# Ganador Experimento 2: p@3=0.600, p@1=0.800, latencia=3.7ms
# Knowledge Distillation desde paraphrase-mpnet (teacher) +
# BERTIN (student). Supera a BAAI/bge-m3 en corpus pequeño en español.
# Referencia: Reimers & Gurevych, EMNLP 2020.
EMBEDDING_MODEL_V2 = "hackathon-pln-es/paraphrase-spanish-distilroberta"

CORPUS_DIR = Path("data/rag_corpus")
CHROMA_V2_DIR = Path("data/vectorstore/chroma_v2")

COLLECTION_NAME = "rag_ciberacoso_v2"


# Dataclass
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


# DocumentIngesterV2
class DocumentIngesterV2:
    """Parsea el corpus RAG, genera embeddings con paraphrase-spanish-distilroberta e indexa en ChromaDB.

    Persiste la colección en CHROMA_V2_DIR usando chromadb.PersistentClient.
    Los metadatos de cada chunk se almacenan como campos filtrables en SQLite,
    permitiendo el routing por pilar que implementa EnrichedRetriever.

    Args:
        corpus_dir: Directorio con los archivos .md del corpus RAG.
        chroma_dir: Directorio destino para la colección ChromaDB persistente.
        embedding_model: Nombre HuggingFace del modelo de embeddings.
    """

    def __init__(self, corpus_dir: Path = CORPUS_DIR, chroma_dir: Path = CHROMA_V2_DIR, embedding_model: str = EMBEDDING_MODEL_V2) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.chroma_dir = Path(chroma_dir)
        self.embedding_model = embedding_model
        self._model: SentenceTransformer | None = None

    # Parsing
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
        parts = re.split(r"^---$", text, flags=re.MULTILINE)
        parts = [p for p in parts if p.strip()]

        chunks: list[Chunk] = []
        for i in range(0, len(parts) - 1, 2):
            yaml_str = parts[i]
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if not content:
                continue
            try:
                meta: dict = yaml.safe_load(yaml_str) or {}
            except yaml.YAMLError:
                continue
            if "id" not in meta or "pillar" not in meta:
                continue
            chunks.append(
                Chunk(
                    id=str(meta["id"]),
                    pillar=int(meta["pillar"]),
                    title=str(meta.get("title", "")),
                    source=str(meta.get("source", "")),
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

        Raises:
            FileNotFoundError: Si no hay archivos .md en corpus_dir.
        """
        md_files = sorted(self.corpus_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(
                f"No se encontraron archivos .md en {self.corpus_dir}"
            )
        all_chunks: list[Chunk] = []
        for md_file in md_files:
            all_chunks.extend(self.parse_md_file(md_file))
        return all_chunks

    # Modelo de embeddings
    @property
    def model(self) -> SentenceTransformer:
        """Carga el modelo de embeddings de forma diferida (lazy)."""
        if self._model is None:
            print(f"  Cargando modelo: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    # Serialización de metadatos para ChromaDB
    @staticmethod
    def _serialize_metadata(chunk: Chunk) -> dict:
        """Convierte un Chunk a un dict de metadatos compatible con ChromaDB.

        ChromaDB solo acepta valores escalares (str, int, float, bool) en los
        metadatos. Las listas y None se convierten a str antes de almacenar.

        Args:
            chunk: Chunk con metadatos originales.

        Returns:
            Dict con todos los metadatos serializados para ChromaDB.
        """
        return {
            "chunk_id": chunk.id,
            "pillar": chunk.pillar,
            "title": chunk.title,
            "source": chunk.source,
            "emotions": json.dumps(chunk.emotions, ensure_ascii=False),
            "trigger": chunk.trigger,
            "therapeutic_technique": chunk.therapeutic_technique or "",
            "audience": chunk.audience,
        }

    # Construcción del índice ChromaDB
    def build_index(self) -> None:
        """Parsea el corpus, genera embeddings e indexa en ChromaDB persistente.

        Sobrescribe cualquier colección previa en chroma_dir. NO toca
        data/vectorstore/faiss_index_v1/.

        Los embeddings se proporcionan explícitamente (embedding_function=None)
        para garantizar que se usa EMBEDDING_MODEL_V2 y no el modelo por defecto
        de ChromaDB.
        """
        chunks = self.load_corpus()
        log.info(f"Corpus leído. {len(chunks)} chunks en memoria.")

        # Generar embeddings
        log.info(f"Generando embeddings topológicos con {self.embedding_model}...")
        embeddings = self.model.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).tolist()

        # Limpiar colección previa y crear nueva
        if self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)
            log.info(f"Índice previo eliminado en {self.chroma_dir}.")
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

        # Indexar en lotes para evitar límites de ChromaDB
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings[start : start + batch_size],
                documents=[c.content for c in batch],
                metadatas=[self._serialize_metadata(c) for c in batch],
            )

        _print_summary(chunks, self.chroma_dir, self.embedding_model)


# Helpers
def _print_summary(
    chunks: list[Chunk], chroma_dir: Path, embedding_model: str
) -> None:
    """Imprime el resumen de la ingesta en stdout."""
    by_pillar = Counter(c.pillar for c in chunks)
    n_always = sum(1 for c in chunks if c.trigger == "always_include")
    techniques = Counter(c.therapeutic_technique for c in chunks if c.therapeutic_technique)

    print(f"\n{'=' * 56}")
    print(f"  Ingesta V2 completada — {len(chunks)} chunks indexados")
    print(f"{'=' * 56}")
    print(f"  Modelo embeddings : {embedding_model}")
    print(f"  Persistido en     : {chroma_dir}")
    print(f"  Colección         : {COLLECTION_NAME}")
    print(f"  Distribución por pilar:")
    for p in sorted(by_pillar):
        print(f"    Pilar {p}: {by_pillar[p]:>2} chunks")
    print(f"  Chunks always_include : {n_always}")
    print(f"  Técnicas terapéuticas : {len(techniques)} tipos")
    print(f"{'=' * 56}\n")


# Main
def main() -> None:
    """Construye el índice ChromaDB V2 con la configuración ganadora de los experimentos."""
    log.info("INICIANDO PIPELINE DE INGESTA V2")

    # Inicializar inyector usando la configuración
    ingester = DocumentIngesterV2(embedding_model=EMBEDDING_MODEL_V2)

    # Lanzar procesamiento ETL
    ingester.build_index()


if __name__ == "__main__":
    main()
