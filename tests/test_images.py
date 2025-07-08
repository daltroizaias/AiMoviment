import cv2
import pytest


def test_prepare_frame(processor, sample_frame):
    """Testa a preparação do frame (rotação)."""
    # Teste sem rotação
    frame = processor._prepare_frame(sample_frame, rotate=False)
    assert frame.shape == sample_frame.shape

    # Teste com rotação
    frame_rotated = processor._prepare_frame(sample_frame, rotate=True)
    assert frame_rotated.shape == (640, 480, 3)  # Dimensões invertidas


def test_resize_frame(processor, sample_frame):
    """Testa o redimensionamento do frame."""
    frame_resized, dims = processor._resize_frame(sample_frame, 0.5)
    assert dims == (320, 240)  # 50% de 640x480
    assert frame_resized.shape == (240, 320, 3)


def test_detect_pose(processor, sample_frame):
    """Testa a detecção de pose (deve retornar None em frame preto)."""
    frame_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
    landmarks = processor._detect_pose(frame_rgb)
    assert landmarks is None  # Nenhuma pose detectada em frame preto


def test_process_frame(processor, sample_frame):
    """Testa o processamento completo do frame."""
    processed = processor._process_frame(sample_frame)
    assert processed.shape == (240, 320, 3)  # Redimensionado para 50%

    # Teste com rotação
    processed_rotated = processor._process_frame(sample_frame, rotate=True)
    assert processed_rotated.shape == (320, 240, 3)


def test_video_source(processor):
    """Testa a obtenção da fonte de vídeo."""
    # Teste com arquivo (deve falhar sem file_path)
    with pytest.raises(
        ValueError,
        match='Você deve fornecer um file_path ou definir web_cam=True.',
    ):
        processor._get_video_source(web_cam=False)

    # Teste com webcam (pode falhar se não houver webcam)
    try:
        cap = processor._get_video_source(web_cam=True)
        cap.release()
    except IOError:
        pytest.skip('Webcam não disponível para teste')
