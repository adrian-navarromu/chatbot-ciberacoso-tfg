"""Configuración de pytest compartida por todos los tests del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emotion_classifier" / "beto"


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta al directorio del modelo fine-tuned de EmotionDetector.",
    )
