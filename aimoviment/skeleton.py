# %%
from image_processor import ImageProcessor, PoseConfig

file_path = './movies/d533328e-faf9-4625-a301-5acbac64c3f9.mp4'


# --- Exemplo de Uso ---

# Configuração da pose
pose_config = PoseConfig(
    model_complexity=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Para usar a webcam:
processor_webcam = ImageProcessor(pose_config=pose_config, file_path=file_path)
print('Iniciando detecção de pose pela webcam...')
# %%
processor = ImageProcessor(PoseConfig(), file_path=file_path)
# processor.save("output_with_landmarks.mp4")
processor.run(scale_percent=0.7, rotate=True)
# %%
