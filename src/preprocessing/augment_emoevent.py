"""
Pipeline de preprocesamiento y aumentación de datos para EmoEvent (español).

Genera archivos intermedios y splits separados para permitir entrenar y comparar
modelos con y sin datos sintéticos de forma reproducible.

Pasos:
  1. Construir emoevent_preprocessed.csv
     (ES original limpio + EN traducido fear/disgust/surprise + undersampling others).
  2. Splits SIN sintéticos → data/processed/no_synthetic/
  3. Generación sintética con Ollama → emoevent_preprocessed_synthetic.csv
     (solo si NO se pasa --skip-synthetic).
  4. Splits CON sintéticos → data/processed/with_synthetic/
     Las muestras sintéticas van SOLO al split de train.

Uso:
    python src/preprocessing/augment_emoevent.py
    python src/preprocessing/augment_emoevent.py --skip-synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import ollama
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel, MarianTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.text_cleaner import clean_tweet  # noqa: E402

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "emoevent"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ES_CSV = RAW_DIR / "emoevent_es.csv"
EN_CSV = RAW_DIR / "emoevent_en.csv"

PREPROCESSED_CSV = PROCESSED_DIR / "emoevent_preprocessed.csv"
PREPROCESSED_SYNTHETIC_CSV = PROCESSED_DIR / "emoevent_preprocessed_synthetic.csv"

NO_SYNTH_DIR = PROCESSED_DIR / "no_synthetic"
WITH_SYNTH_DIR = PROCESSED_DIR / "with_synthetic"

TRANSLATE_CLASSES: list[str] = ["fear", "disgust", "surprise"]
SYNTHETIC_CONFIG: dict[str, int] = {"fear": 100, "disgust": 80}
OTHERS_MAX = 1800
SEED = 42
MT_MODEL = "Helsinki-NLP/opus-mt-en-es"
OLLAMA_MODEL = "mistral:7b"

CYBERBULLYING_CONTEXTS = [
    "amenazas online",
    "difusión de imágenes sin consentimiento",
    "chantaje digital",
    "exclusión de grupos de WhatsApp o redes sociales",
    "miedo a ir al instituto por acoso",
]


# ---------------------------------------------------------------------------
# Carga de datos crudos
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los CSVs español e inglés de EmoEvent (separador tabulador).

    Returns:
        Tupla (df_es, df_en) con los datos crudos de cada idioma.

    Raises:
        FileNotFoundError: Si alguno de los CSVs no existe en RAW_DIR.
    """
    for path in (ES_CSV, EN_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"CSV no encontrado: {path}\n"
                f"Descarga EmoEvent desde https://sinai.ujaen.es/investigacion/datos/emoevent "
                f"y coloca los archivos en {RAW_DIR}/"
            )

    df_es = pd.read_csv(ES_CSV, sep="\t", encoding="utf-8")
    df_en = pd.read_csv(EN_CSV, sep="\t", encoding="utf-8")

    for df in (df_es, df_en):
        df.columns = df.columns.str.lower().str.strip()
        for col in df.columns:
            if col in ("tweet", "sentence", "content") and "text" not in df.columns:
                df.rename(columns={col: "text"}, inplace=True)
                break

    log.info("CSV español cargado: %d filas", len(df_es))
    log.info("CSV inglés cargado:  %d filas", len(df_en))
    return df_es, df_en


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica clean_tweet() a la columna 'text' de un DataFrame.

    Args:
        df: DataFrame con columna 'text'.

    Returns:
        Copia del DataFrame con 'text' limpio y filas vacías eliminadas.
    """
    df = df.copy()
    df["text"] = df["text"].apply(clean_tweet)
    mask_empty = df["text"].str.strip() == ""
    n_removed = int(mask_empty.sum())
    if n_removed:
        log.warning("Eliminadas %d filas vacías tras limpieza", n_removed)
        df = df[~mask_empty].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Traducción EN → ES
# ---------------------------------------------------------------------------


def translate_english_classes(df_en: pd.DataFrame) -> pd.DataFrame:
    """Filtra y traduce al español las clases fear, disgust y surprise del CSV inglés.

    Utiliza MarianMT (Helsinki-NLP/opus-mt-en-es) directamente sin el pipeline
    de HuggingFace (compatible con Transformers 5.x). La traducción se realiza
    en lotes de 16 muestras para aprovechar la GPU.

    Args:
        df_en: DataFrame inglés ya limpio con columnas 'text' y 'emotion'.

    Returns:
        DataFrame con los mismos campos que el español y columna 'source'
        con valor 'translated_en'.
    """
    subset = df_en[df_en["emotion"].isin(TRANSLATE_CLASSES)].copy()
    log.info(
        "Textos a traducir: %d (%s)",
        len(subset),
        subset["emotion"].value_counts().to_dict(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info(
            "VRAM libre antes de traducción: %.1f GB",
            torch.cuda.mem_get_info()[0] / 1024 ** 3,
        )

    log.info("Cargando modelo de traducción: %s (device=%s)", MT_MODEL, device)
    tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(device)
    model.eval()

    texts = subset["text"].tolist()
    log.info("Traduciendo %d textos...", len(texts))

    translated_texts: list[str] = []
    for batch_size in (16, 8):
        translated_texts = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=150,
                ).to(device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs)
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                translated_texts.extend(decoded)
                log.info(
                    "  Traducidos %d/%d (batch=%d)",
                    min(i + batch_size, len(texts)), len(texts), batch_size,
                )
            break  # batch_size funcionó, salir del bucle de reintentos
        except torch.cuda.OutOfMemoryError:
            log.warning(
                "CUDA OOM con batch_size=%d. Vaciando caché y reintentando con %d...",
                batch_size, batch_size // 2,
            )
            torch.cuda.empty_cache()

    subset = subset.copy()
    subset["text"] = [clean_tweet(t) for t in translated_texts]
    subset["source"] = "translated_en"

    # Liberar VRAM explícitamente para que Ollama encuentre la GPU libre
    # cuando cargue mistral:7b en el paso de generación sintética.
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info(
            "VRAM liberada tras traducción: %.1f GB libres",
            torch.cuda.mem_get_info()[0] / 1024 ** 3,
        )

    log.info("Traducción completada: %d filas", len(subset))
    return subset.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Undersampling
# ---------------------------------------------------------------------------


def undersample_others(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce la clase 'others' a OTHERS_MAX muestras aleatorias.

    Args:
        df: Dataset completo.

    Returns:
        Dataset con 'others' recortada a como máximo OTHERS_MAX filas,
        mezclado aleatoriamente.
    """
    others_mask = df["emotion"] == "others"
    n_others = int(others_mask.sum())

    if n_others <= OTHERS_MAX:
        log.info(
            "Undersampling omitido: 'others' tiene %d muestras (≤ %d).",
            n_others,
            OTHERS_MAX,
        )
        return df

    keep = df[others_mask].sample(n=OTHERS_MAX, random_state=SEED)
    df_out = pd.concat([df[~others_mask], keep], ignore_index=True)
    log.info("Undersampling 'others': %d → %d muestras.", n_others, OTHERS_MAX)
    return df_out.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deduplicación
# ---------------------------------------------------------------------------


def deduplicate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con texto exactamente idéntico del dataset completo.

    La deduplicación debe realizarse ANTES del split para evitar que el mismo
    texto aparezca en más de un split (data leakage).

    Args:
        df: Dataset preprocesado con columna 'text'.

    Returns:
        Dataset sin duplicados de texto, con índice reseteado.
    """
    n_before = len(df)
    df_dedup = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_removed = n_before - len(df_dedup)
    if n_removed:
        log.warning(
            "Deduplicación: eliminados %d textos duplicados (%d → %d filas).",
            n_removed, n_before, len(df_dedup),
        )
    else:
        log.info("Deduplicación: sin duplicados encontrados (%d filas).", n_before)
    return df_dedup


# ---------------------------------------------------------------------------
# PASO 1 — Construir y guardar emoevent_preprocessed.csv
# ---------------------------------------------------------------------------


def build_preprocessed(df_es: pd.DataFrame, df_en: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset preprocesado base (sin sintéticos).

    Aplica limpieza a ambos idiomas, traduce las clases seleccionadas del
    inglés, concatena con el español original y aplica undersampling a 'others'.

    Args:
        df_es: DataFrame crudo en español.
        df_en: DataFrame crudo en inglés.

    Returns:
        DataFrame preprocesado con columnas 'text', 'emotion', 'source'
        y cualquier otra columna presente en ambos CSVs.
    """
    log.info("── PASO 1: construyendo dataset preprocesado ──")

    log.info("Limpiando textos en español...")
    df_es = apply_cleaning(df_es)
    df_es["source"] = "original"

    log.info("Limpiando textos en inglés...")
    df_en = apply_cleaning(df_en)

    df_translated = translate_english_classes(df_en)

    # Conservar solo columnas presentes en ambos DataFrames
    common_cols = sorted(set(df_es.columns) & set(df_translated.columns))
    df_combined = pd.concat(
        [df_es[common_cols], df_translated[common_cols]],
        ignore_index=True,
    )
    log.info("Combinado (original + traducido): %d filas", len(df_combined))

    df_preprocessed = undersample_others(df_combined)
    log.info("Dataset preprocesado final: %d filas", len(df_preprocessed))
    return df_preprocessed


def save_preprocessed(df: pd.DataFrame, path: Path = PREPROCESSED_CSV) -> None:
    """Guarda el dataset preprocesado en disco.

    Args:
        df: DataFrame a guardar.
        path: Ruta de destino (por defecto PREPROCESSED_CSV).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Guardado: %s (%d filas)", path, len(df))


# ---------------------------------------------------------------------------
# PASO 2 — Split estratificado 70/15/15
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide un dataset de forma estratificada por emoción (70 / 15 / 15).

    Args:
        df: Dataset a dividir. Debe contener la columna 'emotion'.

    Returns:
        Tupla (train, val, test) con índices reseteados.
    """
    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["emotion"],
        random_state=SEED,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,        # 50 % de 30 % = 15 % del total
        stratify=temp["emotion"],
        random_state=SEED,
    )
    log.info(
        "Split → train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def verify_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verifica que ningún texto aparece en más de un split (data leakage).

    Compara los tres pares posibles (train↔val, train↔test, val↔test) usando
    el texto limpio como clave. Registra un WARNING si se detecta solapamiento
    y un error explícito si train↔val o train↔test contienen duplicados.

    Args:
        train: Split de entrenamiento.
        val: Split de validación.
        test: Split de test.
    """
    sets = {
        "train": set(train["text"].str.strip()),
        "val":   set(val["text"].str.strip()),
        "test":  set(test["text"].str.strip()),
    }
    leakage_found = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            log.error(
                "DATA LEAKAGE detectado: %d textos comunes en %s ∩ %s.",
                len(overlap), a, b,
            )
            leakage_found = True
        else:
            log.info("Leakage check %s ∩ %s: ✓ OK (0 solapamientos).", a, b)
    if not leakage_found:
        log.info("Verificación de leakage completada: sin solapamientos.")


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    subdir: str,
) -> None:
    """Guarda los tres splits en un subdirectorio de data/processed/.

    Args:
        train: Split de entrenamiento.
        val: Split de validación.
        test: Split de test.
        subdir: Nombre del subdirectorio ('no_synthetic' o 'with_synthetic').
    """
    out_dir = PROCESSED_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"emoevent_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        log.info("Guardado: %s (%d filas)", path, len(df))


# ---------------------------------------------------------------------------
# PASO 3 — Generación sintética con Ollama
# ---------------------------------------------------------------------------


def generate_synthetic_tweets(
    emotion: str,
    context: list[str],
    n: int,
) -> list[str]:
    """Genera tweets sintéticos en español adolescente usando Ollama.

    Args:
        emotion: Nombre de la emoción objetivo (p. ej. 'fear', 'disgust').
        context: Lista de situaciones de ciberacoso para el prompt.
        n: Número de tweets a generar.

    Returns:
        Lista de tweets sintéticos limpios (puede ser menor que n si el modelo
        devuelve líneas vacías o con menos de 5 tokens).
    """
    contexto_str = ", ".join(context)
    prompt = textwrap.dedent(f"""
        Genera {n} tweets en español coloquial adolescente (14-17 años) que expresen
        {emotion} en contexto de ciberacoso.
        Situaciones: {contexto_str}.
        Un tweet por línea, 10-40 palabras, sin hashtags, máxima variedad.
        No numeres las líneas. No añadas explicaciones. Solo los tweets.
    """).strip()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.9, "num_predict": n * 60},
    )

    raw_text: str = response.message.content
    tweets = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip() and len(line.split()) >= 5
    ]
    log.info(
        "Generados %d/%d tweets sintéticos para '%s'",
        len(tweets), n, emotion,
    )
    return tweets[:n]


def collect_synthetic_rows() -> pd.DataFrame:
    """Genera todos los tweets sintéticos configurados y los devuelve como DataFrame.

    Returns:
        DataFrame con columnas 'text', 'emotion' y 'source' = 'synthetic'.
        Puede estar vacío si Ollama no devuelve resultados válidos.
    """
    frames: list[pd.DataFrame] = []

    for emotion, n_target in SYNTHETIC_CONFIG.items():
        log.info("Generando %d tweets sintéticos para '%s'...", n_target, emotion)
        tweets = generate_synthetic_tweets(emotion, CYBERBULLYING_CONTEXTS, n_target)
        if not tweets:
            log.warning("No se generaron tweets para '%s'", emotion)
            continue

        synth_df = pd.DataFrame({
            "text": [clean_tweet(t) for t in tweets],
            "emotion": emotion,
            "source": "synthetic",
        })
        synth_df = synth_df[synth_df["text"].str.strip() != ""].reset_index(drop=True)
        log.info("  → %d tweets válidos para '%s'", len(synth_df), emotion)
        frames.append(synth_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# PASO 4 — Splits con sintéticos (sintéticos solo en train)
# ---------------------------------------------------------------------------


def build_splits_with_synthetic(
    train_base: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye los splits con datos sintéticos añadidos exclusivamente al train.

    Val y test son idénticos a los splits sin sintéticos para garantizar
    que la evaluación se realiza siempre sobre datos reales.

    Args:
        train_base: Split de entrenamiento sin sintéticos.
        val: Split de validación (sin modificar).
        test: Split de test (sin modificar).
        synthetic: DataFrame con las muestras sintéticas generadas.

    Returns:
        Tupla (train_augmented, val, test).
    """
    train_augmented = pd.concat(
        [train_base, synthetic], ignore_index=True
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    log.info(
        "Train aumentado: %d base + %d sintéticos = %d total",
        len(train_base), len(synthetic), len(train_augmented),
    )
    return train_augmented, val, test


# ---------------------------------------------------------------------------
# Resumen comparativo
# ---------------------------------------------------------------------------


def _emotion_table(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Construye tabla de distribución de emociones por split."""
    tbl = pd.DataFrame(
        {
            "train": train["emotion"].value_counts(),
            "val": val["emotion"].value_counts(),
            "test": test["emotion"].value_counts(),
        }
    ).fillna(0).astype(int)
    tbl["TOTAL"] = tbl.sum(axis=1)
    tbl.loc["TOTAL"] = tbl.sum()
    return tbl


def _source_table(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame | None:
    """Construye tabla de distribución por fuente de datos."""
    if "source" not in train.columns:
        return None
    tbl = pd.DataFrame(
        {
            "train": train["source"].value_counts(),
            "val": val["source"].value_counts(),
            "test": test["source"].value_counts(),
        }
    ).fillna(0).astype(int)
    tbl["TOTAL"] = tbl.sum(axis=1)
    tbl.loc["TOTAL"] = tbl.sum()
    return tbl


def print_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    title: str = "",
) -> None:
    """Imprime distribución de clases y fuentes para un conjunto de splits.

    Args:
        train: Split de entrenamiento.
        val: Split de validación.
        test: Split de test.
        title: Encabezado opcional de la tabla.
    """
    sep = "=" * 65
    header = f"  {title}" if title else "  DISTRIBUCIÓN DE CLASES POR SPLIT"
    print(f"\n{sep}\n{header}\n{sep}")
    print(_emotion_table(train, val, test).to_string())

    src = _source_table(train, val, test)
    if src is not None:
        print(f"\n{sep}\n  DISTRIBUCIÓN POR FUENTE DE DATOS\n{sep}")
        print(src.to_string())
    print(sep)


def print_comparison(
    train_ns: pd.DataFrame,
    val_ns: pd.DataFrame,
    test_ns: pd.DataFrame,
    train_ws: pd.DataFrame,
    val_ws: pd.DataFrame,
    test_ws: pd.DataFrame,
) -> None:
    """Imprime tabla comparativa entre splits sin y con sintéticos.

    Muestra la variación en número de muestras por emoción en el train
    y confirma que val/test son iguales en ambas versiones.

    Args:
        train_ns: Train SIN sintéticos.
        val_ns: Val SIN sintéticos.
        test_ns: Test SIN sintéticos.
        train_ws: Train CON sintéticos.
        val_ws: Val CON sintéticos (debe ser idéntico a val_ns).
        test_ws: Test CON sintéticos (debe ser idéntico a test_ns).
    """
    sep = "=" * 75

    # --- Comparación de train ---
    emo_ns = train_ns["emotion"].value_counts().rename("train_no_synth")
    emo_ws = train_ws["emotion"].value_counts().rename("train_with_synth")
    comp = pd.concat([emo_ns, emo_ws], axis=1).fillna(0).astype(int)
    comp["diff"] = comp["train_with_synth"] - comp["train_no_synth"]
    comp.loc["TOTAL"] = comp.sum()

    print(f"\n{sep}")
    print("  COMPARATIVA TRAIN: sin sintéticos vs con sintéticos")
    print(sep)
    print(comp.to_string())

    # --- Confirmación val/test iguales ---
    val_equal = val_ns.shape == val_ws.shape
    test_equal = test_ns.shape == test_ws.shape
    print(f"\n  Val  idéntico en ambas versiones: {'✓' if val_equal  else '✗'} ({len(val_ns)} filas)")
    print(f"  Test idéntico en ambas versiones: {'✓' if test_equal else '✗'} ({len(test_ns)} filas)")
    print(sep)

    # --- Tablas individuales ---
    print_summary(train_ns, val_ns, test_ns, title="SIN SINTÉTICOS — distribución por split")
    print_summary(train_ws, val_ws, test_ws, title="CON SINTÉTICOS — distribución por split")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento y aumentación de EmoEvent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Ejemplos:
              python src/preprocessing/augment_emoevent.py
              python src/preprocessing/augment_emoevent.py --skip-synthetic
              python src/preprocessing/augment_emoevent.py --skip-build --skip-synthetic
        """),
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        default=False,
        help="Omitir la generación de tweets sintéticos con Ollama (pasos 3 y 4).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        default=False,
        help=(
            "Omitir el paso 1 (traducción + limpieza) y cargar directamente "
            f"{PREPROCESSED_CSV.name} ya existente. Útil para regenerar splits "
            "sin repetir la traducción."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta el pipeline completo de preprocesamiento y aumentación."""
    args = parse_args()
    log.info("=== Pipeline EmoEvent — inicio ===")

    # ── PASO 1: construir o cargar dataset preprocesado base ─────────────────
    if args.skip_build:
        if not PREPROCESSED_CSV.exists():
            log.error(
                "--skip-build activo pero no existe %s. Ejecuta sin --skip-build primero.",
                PREPROCESSED_CSV,
            )
            sys.exit(1)
        log.info("── PASO 1 omitido: cargando %s ──", PREPROCESSED_CSV)
        df_preprocessed = pd.read_csv(PREPROCESSED_CSV)
        log.info("Cargadas %d filas desde disco.", len(df_preprocessed))
    else:
        df_es, df_en = load_data()
        df_preprocessed = build_preprocessed(df_es, df_en)

    # Deduplicar ANTES de guardar el CSV intermedio
    df_preprocessed = deduplicate_dataset(df_preprocessed)
    save_preprocessed(df_preprocessed, PREPROCESSED_CSV)

    # ── PASO 2: splits sin sintéticos ────────────────────────────────────────
    log.info("── PASO 2: splits sin sintéticos ──")
    train_base, val, test = split_dataset(df_preprocessed)
    verify_no_leakage(train_base, val, test)
    save_splits(train_base, val, test, subdir="no_synthetic")

    if args.skip_synthetic:
        log.info("--skip-synthetic activo: se omiten los pasos 3 y 4.")
        print_summary(train_base, val, test, title="SIN SINTÉTICOS — distribución por split")
        log.info("=== Pipeline EmoEvent — completado (sin sintéticos) ===")
        return

    # ── PASO 3: generación sintética ─────────────────────────────────────────
    log.info("── PASO 3: generación sintética con Ollama ──")
    synthetic = collect_synthetic_rows()

    if synthetic.empty:
        log.error("No se generaron muestras sintéticas. Abortando pasos 3 y 4.")
        print_summary(train_base, val, test, title="SIN SINTÉTICOS — distribución por split")
        log.info("=== Pipeline EmoEvent — completado (sin sintéticos) ===")
        return

    # Combinar, deduplicar y guardar CSV sintético limpio ANTES del split
    df_preprocessed_synthetic = pd.concat(
        [df_preprocessed, synthetic], ignore_index=True
    )
    df_preprocessed_synthetic = deduplicate_dataset(df_preprocessed_synthetic)
    save_preprocessed(df_preprocessed_synthetic, PREPROCESSED_SYNTHETIC_CSV)

    # Extraer los sintéticos que sobrevivieron la deduplicación
    synthetic_clean = df_preprocessed_synthetic[
        df_preprocessed_synthetic["source"] == "synthetic"
    ].reset_index(drop=True)
    log.info(
        "Sintéticos tras deduplicación: %d (de %d originales).",
        len(synthetic_clean), len(synthetic),
    )

    # ── PASO 4: splits con sintéticos (sintéticos solo en train) ─────────────
    log.info("── PASO 4: splits con sintéticos ──")
    train_augmented, val_ws, test_ws = build_splits_with_synthetic(
        train_base, val, test, synthetic_clean
    )
    verify_no_leakage(train_augmented, val_ws, test_ws)
    save_splits(train_augmented, val_ws, test_ws, subdir="with_synthetic")

    # ── Resumen comparativo ───────────────────────────────────────────────────
    print_comparison(
        train_ns=train_base, val_ns=val,       test_ns=test,
        train_ws=train_augmented, val_ws=val_ws, test_ws=test_ws,
    )
    log.info("=== Pipeline EmoEvent — completado ===")


if __name__ == "__main__":
    main()
