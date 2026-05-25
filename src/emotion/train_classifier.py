"""
Pipeline de fine-tuning del clasificador emocional sobre EmoEvent.

Entrena un modelo transformer (robertuito/maria-copy/BETO) para clasificar 7 emociones:
anger, disgust, fear, joy, sadness, surprise, others.

Aplica class weights para compensar el desequilibrio de clases y
early stopping basado en f1_macro en validación.

Uso:
    python src/emotion/train_classifier.py
    python src/emotion/train_classifier.py --model_name dccuchile/bert-base-spanish-wwm-cased
    python src/emotion/train_classifier.py --model_name dccuchile/bert-base-spanish-wwm-cased --no-class-weights
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

# Sklearn métricas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuración
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "no_synthetic"
MODELS_DIR = PROJECT_ROOT / "models" / "emotion_classifier"
EVAL_DIR = PROJECT_ROOT / "eval" / "emotion_results"

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "others"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}

MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
SEED = 112

MODEL_SHORT_NAMES = {
    "pysentimiento/robertuito-base-uncased": "robertuito",
    "PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy": "maria-copy",
    "dccuchile/bert-base-spanish-wwm-cased": "beto",
}

# Dataset
class EmoEventDataset(torch.utils.data.Dataset):
    """Dataset PyTorch para los splits de EmoEvent tokenizados."""

    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# CustomTrainer con class weights
class WeightedTrainer(Trainer):
    """Trainer que aplica CrossEntropyLoss ponderada por frecuencia de clase.

    Los pesos se pasan al constructor y se mueven al mismo device que el
    modelo en cada llamada a compute_loss.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)  # Asegura que los pesos estén en el mismo dispositivo que los logits en cada batch
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Métricas
def build_compute_metrics(id2label: dict[int, str]):
    """Devuelve la función compute_metrics con cierre sobre id2label."""

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0))
        }
        f1_per = f1_score(labels, preds, average=None, zero_division=0)
        for idx, f1 in enumerate(f1_per):
            metrics[f"f1_{id2label[idx]}"] = float(f1)

        return metrics

    return compute_metrics


# Carga de datos
def load_split(csv_path: Path) -> pd.DataFrame:
    """Carga un split CSV y valida que contiene las columnas necesarias."""
    df = pd.read_csv(csv_path)
    missing = {"text", "emotion"} - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en {csv_path}: {missing}")
    df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
    return df


def tokenize_split(df: pd.DataFrame, tokenizer, label2id: dict[str, int]) -> EmoEventDataset:
    """Tokeniza textos y convierte etiquetas a índices numéricos."""
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    labels = [label2id[e] for e in df["emotion"]]
    return EmoEventDataset(encodings, labels)


# Cálculo de class weights
def compute_weights(df_train: pd.DataFrame, label2id: dict[str, int]) -> torch.Tensor:
    """Calcula pesos balanceados inversamente proporcionales a la frecuencia."""
    classes = np.array(sorted(label2id.values()))
    y = np.array([label2id[e] for e in df_train["emotion"]])
    weights = compute_class_weight("balanced", classes=classes, y=y)

    log.info("Pesos de clase calculados:")
    for label, w in zip(EMOTIONS, weights):
        log.info("  %s: %.4f", label.ljust(10), w)
    return torch.tensor(weights, dtype=torch.float32)


# Figuras de evaluación
def plot_confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[str], out_path: Path) -> None:
    """Guarda heatmap de la matriz de confusión normalizada."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
            
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (normalizada por fila)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Matriz de confusión guardada en: %s", out_path.name)


def plot_f1_per_class(f1_scores: dict[str, float], out_path: Path) -> None:
    """Barplot de F1 por clase."""
    labels = list(f1_scores.keys())
    values = list(f1_scores.values())
    
    # Procesar los colores elemento por elemento
    colors = [plt.cm.RdYlGn(v) for v in values] 
    
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
                
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1 por clase — conjunto de test")
    
    mean_val = np.mean(values)
    ax.axhline(mean_val, color="steelblue", linestyle="--", linewidth=1.2, label=f"Media: {mean_val:.3f}")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico de F1 por clase guardado en: %s", out_path.name)


def plot_training_history(log_history: list[dict], out_path: Path) -> None:
    """Curvas de loss y F1-macro por época de train y validación."""
    train_loss, val_loss, val_f1 = [], [], []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_loss.append((entry["epoch"], entry["loss"]))
        if "eval_loss" in entry:
            val_loss.append((entry["epoch"], entry["eval_loss"]))
            val_f1.append((entry["epoch"], entry.get("eval_f1_macro", 0.0)))

    if not val_loss:
        log.warning("Sin historial de validación suficiente para generar la figura de entrenamiento.")
        return

    ep_val = [x[0] for x in val_loss]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if train_loss:
        ax1.plot([x[0] for x in train_loss], [x[1] for x in train_loss], "o-", label="Train loss", color="#4878CF")
    ax1.plot(ep_val, [x[1] for x in val_loss], "s-", label="Val loss", color="#D65F5F")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Curva de pérdida")
    ax1.legend()

    ax2.plot(ep_val, [x[1] for x in val_f1], "s-", color="#6ACC65")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("F1-macro")
    ax2.set_title("F1-macro en validación")
    ax2.set_ylim(0, 1)

    plt.suptitle("Historial de entrenamiento", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Historial de entrenamiento guardado en: %s", out_path.name)


# Evaluación en test
def evaluate_on_test(trainer: WeightedTrainer, test_dataset: EmoEventDataset, id2label: dict[int, str], out_dir: Path) -> dict:
    """Evalúa en test, guarda classification_report y todas las figuras."""
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    label_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)

    log.info("Reporte de Clasificación:\n%s", report)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {id2label[i]: float(f) for i, f in enumerate(f1_per)}

    plot_confusion_matrix(y_true.tolist(), y_pred.tolist(), label_names, out_dir / "confusion_matrix.png")
    plot_f1_per_class(f1_dict, out_dir / "f1_per_class.png")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_dict,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    
    log.info("Métricas Finales - Accuracy: %.4f, F1-Macro: %.4f, Precisión: %.4f, Recall: %.4f", metrics['accuracy'], metrics['f1_macro'], metrics['precision_macro'], metrics['recall_macro'])
    
    return metrics

# Argparse
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning del clasificador emocional sobre EmoEvent."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pysentimiento/robertuito-base-uncased",
        help=(
            "Modelo HuggingFace a fine-tunear. "
            "Opciones: pysentimiento/robertuito-base-uncased | "
            "PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy | "
            "dccuchile/bert-base-spanish-wwm-cased"
        ),
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        default=False,
        help=(
            "Desactiva los pesos de clase y usa CrossEntropyLoss estándar. "
            "Los resultados se guardan en {model_short}_no_weights/."
        ),
    )
    return parser.parse_args()


# Main
def main() -> None:
    args = parse_args()
    set_seed(SEED)

    model_name = args.model_name
    use_class_weights = not args.no_class_weights
    model_short = MODEL_SHORT_NAMES.get(model_name, model_name.split("/")[-1])
    
    if not use_class_weights:
        model_short = f"{model_short}_no_weights"

    model_out_dir = MODELS_DIR / model_short
    eval_out_dir = EVAL_DIR / model_short

    log.info("CLASIFICADOR EMOCIONAL: Fine-tuning")
    log.info("Modelo: %s", model_name)
    log.info("Class weights: %s", "SÍ" if use_class_weights else "[DESACTIVADOS]")

    # Carga de datos
    log.info("[1/6] Cargando particiones de datos...")
    df_train = load_split(DATA_DIR / "emoevent_train.csv")
    df_val = load_split(DATA_DIR / "emoevent_val.csv")
    df_test = load_split(DATA_DIR / "emoevent_test.csv")

    # Tokenización
    log.info("[2/6] Tokenizando corpus con %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    train_ds = tokenize_split(df_train, tokenizer, LABEL2ID)
    val_ds = tokenize_split(df_val, tokenizer, LABEL2ID)
    test_ds = tokenize_split(df_test, tokenizer, LABEL2ID)

    # Modelo
    log.info("[3/6] Inicializando modelo base y cabeza de clasificación...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTIONS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Class weights
    if use_class_weights:
        log.info("[4/6] Calculando ponderación de clases (Class Weights)...")
        class_weights = compute_weights(df_train, LABEL2ID)
    else:
        log.info("[4/6] [OMITIDO] Usando CrossEntropyLoss estándar sin ponderación.")
        class_weights = None

    # Training arguments
    log.info("[5/6] Configurando hiperparámetros de entrenamiento...")

    training_args = TrainingArguments(
        output_dir=str(model_out_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(ID2LABEL),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    if use_class_weights:
        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    # Entrenamiento
    log.info("[6/6] Iniciando bucle de entrenamiento...")
    log.info("Acelerador detectado: %s", 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU')

    trainer.train()

    trainer.save_model(str(model_out_dir))
    tokenizer.save_pretrained(str(model_out_dir))
    log.info("Entrenamiento finalizado. Modelo guardado en: %s", model_out_dir)

    # Figuras de historial 
    plot_training_history(trainer.state.log_history, eval_out_dir / "training_history.png")

    # Evaluación en test
    evaluate_on_test(trainer, test_ds, ID2LABEL, eval_out_dir)

    log.info("Pipeline Completado")


if __name__ == "__main__":
    main()
