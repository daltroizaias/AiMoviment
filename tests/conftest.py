import numpy as np
import pytest

from aimoviment.image_processor import ImageProcessor, PoseConfig


@pytest.fixture
def sample_frame():
    """Retorna um frame de exemplo (imagem preta 640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def processor():
    """Retorna uma instância de ImageProcessor para testes."""
    return ImageProcessor(PoseConfig())
