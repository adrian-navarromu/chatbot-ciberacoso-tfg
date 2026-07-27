"""Configuración de pytest compartida por todos los tests del proyecto."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emotion_classifier" / "robertuito"

# Añade la raíz del proyecto al path para que `from src.xxx import ...` funcione
# en todos los tests sin que cada archivo lo tenga que hacer individualmente.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requiere Ollama levantado y modelos descargados "
        "(excluir con `pytest -m 'not integration'`).",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta al directorio del modelo fine-tuned de EmotionDetector.",
    )


@pytest.fixture(scope="session")
def model_path(request) -> Path:
    """Ruta al clasificador emocional, configurable con --model-path."""
    return Path(request.config.getoption("--model-path"))
