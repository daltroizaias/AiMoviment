# %%

import cv2
import mediapipe as mp
from pydantic import BaseModel

# %%

file_path = "./movies/20250701_123834000_iOS.MOV"


# %%

class PoseConfig(BaseModel):
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class ImageProcessor:
    def __init__(self, pose_config: PoseConfig, file_path: str = None):
        self.media_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.media_pose.Pose(
            model_complexity=pose_config.model_complexity,
            min_detection_confidence=pose_config.min_detection_confidence,
            min_tracking_confidence=pose_config.min_tracking_confidence
        )
        self.file_path = file_path

    def _get_video_source(self, web_cam: bool = False) -> cv2.VideoCapture:
        """
        Retorna o objeto VideoCapture para webcam ou arquivo.
        """
        if web_cam:
            cap = cv2.VideoCapture(0)  # Usar a webcam
        elif self.file_path:
            cap = cv2.VideoCapture(self.file_path)
        else:
            raise ValueError(
                "Você deve fornecer um file_path ou definir web_cam=True."
            )

        if not cap.isOpened():
            raise IOError("Erro ao abrir a câmera/vídeo.")
        return cap

    def _process_frame(
            self,
            frame: cv2.Mat,
            scale_percent: float = 0.5,
            rotate: bool = False
        ) -> cv2.Mat:
        """
        Processa um único frame: rotaciona (opcional), redimensiona e converte para RGB.
        Em seguida, detecta e desenha os landmarks.

        Args:
            frame (cv2.Mat): O frame original lido do VideoCapture.
            scale_percent (float): Percentual de redimensionamento (ex: 0.5 para 50%).
            rotate (bool): Se True, rotaciona o frame em 90 graus no sentido horário.

        Returns:
            cv2.Mat: O frame processado com os landmarks desenhados.
        """  # noqa: E501
        # 1. Rotacionar o frame (se rotate for True)
        if rotate:
            frame_to_process = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            frame_to_process = frame.copy()  # Garante que trabalhamos em uma cópia  # noqa: E501

        # 2. Calcular as novas dimensões para redimensionar (mantendo a proporção)  # noqa: E501
        h_temp, w_temp = frame_to_process.shape[:2]
        width_resized = int(w_temp * scale_percent)
        height_resized = int(h_temp * scale_percent)
        dimensions = (width_resized, height_resized)

        # 3. Redimensionar o frame para o tamanho desejado
        frame_resized = cv2.resize(
            frame_to_process,
            dimensions,
            interpolation=cv2.INTER_AREA
        )

        # 4. Converter o frame redimensionado para RGB (MediaPipe espera RGB)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 5. Processar o frame redimensionado com MediaPipe Pose
        results = self.pose.process(frame_rgb)

        # 6. Desenhar os landmarks no frame redimensionado (frame_resized)
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame_resized,
                results.pose_landmarks,
                self.media_pose.POSE_CONNECTIONS,  # Use self.media_pose aqui
                self.mp_draw.DrawingSpec(
                    color=(245, 117, 66),
                    thickness=2,
                    circle_radius=2
                ),
                self.mp_draw.DrawingSpec(
                    color=(245, 66, 230),
                    thickness=2,
                    circle_radius=2
                )
            )

        return frame_resized    # Retorna o frame pronto para exibição

    def run(
            self,
            web_cam: bool = False,
            scale_percent: float = 0.5,
            rotate: bool = False
        ):
        """
        Executa o loop principal de processamento de vídeo.
        """
        cap = self._get_video_source(web_cam)

        # Obter e imprimir dimensões originais (apenas uma vez)
        width_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Dimensões originais do vídeo/câmera: {width_original}x{height_original}"  # noqa: E501
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Processar o frame individualmente
                processed_frame = self._process_frame(
                    frame,
                    scale_percent,
                    rotate
                )

                # Exibir o frame processado
                cv2.imshow("Pose Detection", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:  # Garante que os recursos são liberados mesmo se houver erro
            cap.release()
            self.pose.close()   # Fechar o objeto pose do MediaPipe
            cv2.destroyAllWindows()


# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Configuração da pose
    pose_config = PoseConfig(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Para usar a webcam:
    processor_webcam = ImageProcessor(
        pose_config=pose_config,
        file_path=file_path
    )
    print("Iniciando detecção de pose pela webcam...")
    processor_webcam.run(
        scale_percent=0.6,
        rotate=True
    )
# %%
