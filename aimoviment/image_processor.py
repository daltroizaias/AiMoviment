from typing import List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from pydantic import BaseModel


class PoseConfig(BaseModel):
    """Configurações para o modelo de detecção de pose."""

    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class ImageProcessor:
    """Processador de imagem para detecção de pose usando MediaPipe.

    Attributes:
        media_pose: Objeto MediaPipe para detecção de pose.
        mp_draw: Utilitário de desenho do MediaPipe.
        pose: Instância do detector de pose.
        file_path: Caminho para o arquivo de vídeo (opcional).
        drawing_specs: Especificações de desenho para landmarks.
    """

    # Constantes de desenho (evita repetição nos métodos)
    LANDMARK_COLOR = (245, 117, 66)
    CONNECTION_COLOR = (245, 66, 230)
    THICKNESS = 2
    CIRCLE_RADIUS = 2

    def __init__(self, pose_config: PoseConfig, file_path: str = None):
        """Inicializa o processador de imagem.

        Args:
            pose_config: Configurações para o detector de pose.
            file_path: Caminho opcional para arquivo de vídeo.
        """
        self.media_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.media_pose.Pose(
            model_complexity=pose_config.model_complexity,
            min_detection_confidence=pose_config.min_detection_confidence,
            min_tracking_confidence=pose_config.min_tracking_confidence,
        )
        self.file_path = file_path

        # Configurações de desenho reutilizáveis
        self.drawing_specs = (
            self.mp_draw.DrawingSpec(
                color=self.LANDMARK_COLOR,
                thickness=self.THICKNESS,
                circle_radius=self.CIRCLE_RADIUS,
            ),
            self.mp_draw.DrawingSpec(
                color=self.CONNECTION_COLOR,
                thickness=self.THICKNESS,
                circle_radius=self.CIRCLE_RADIUS,
            ),
        )

    def _get_video_source(self, web_cam: bool = False) -> cv2.VideoCapture:
        """Obtém a fonte de vídeo (webcam ou arquivo).

        Args:
            web_cam: Se True, usa a webcam. Se False, usa o arquivo.
        Returns:
            Objeto VideoCapture configurado.

        Raises:
            ValueError: Se não houver file_path e web_cam for False.
            IOError: Se não conseguir abrir a fonte de vídeo.
        """
        if web_cam:
            cap = cv2.VideoCapture(0)
        elif self.file_path:
            cap = cv2.VideoCapture(self.file_path)
        else:
            raise ValueError(
                'Você deve fornecer um file_path ou definir web_cam=True.'
            )

        if not cap.isOpened():
            raise IOError('Erro ao abrir a câmera/vídeo.')
        return cap

    @staticmethod
    def _prepare_frame(frame: cv2.Mat, rotate: bool) -> cv2.Mat:
        """Prepara o frame rotacionando se necessário.

        Args:
            frame: Frame original.
            rotate: Se True, rotaciona 90 graus no sentido horário.

        Returns:
            Frame processado (rotacionado ou cópia).
        """
        return (
            cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if rotate
            else frame.copy()
        )

    @staticmethod
    def _resize_frame(
        frame: cv2.Mat, scale_percent: float
    ) -> Tuple[cv2.Mat, Tuple[int, int]]:
        """Redimensiona o frame conforme o percentual especificado.

        Args:
            frame: Frame a ser redimensionado.
            scale_percent: Percentual de redimensionamento (0-1).

        Returns:
            Tupla contendo o frame redimensionado e suas dimensões.
        """
        height, width = frame.shape[:2]
        new_dim = (int(width * scale_percent), int(height * scale_percent))
        resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
        return resized, new_dim

    def _detect_pose(
        self, frame_rgb: cv2.Mat
    ) -> Optional[mp.solutions.pose.PoseLandmark]:
        """Detecta landmarks de pose no frame.

        Args:
            frame_rgb: Frame no formato RGB.

        Returns:
            Resultados da detecção de pose ou None.
        """
        return self.pose.process(frame_rgb).pose_landmarks

    def _process_frame(
        self, frame: cv2.Mat, scale_percent: float = 0.5, rotate: bool = False
    ) -> cv2.Mat:
        """Processa um frame completo: rotaciona, redimensiona e detecta pose.

        Args:
            frame: Frame original.
            scale_percent: Percentual de redimensionamento (padrão: 0.5).
            rotate: Se deve rotacionar o frame (padrão: False).

        Returns:
            Frame processado com landmarks desenhados (se detectados).
        """
        # 1. Preparar frame (rotacionar se necessário)
        frame_processed = self._prepare_frame(frame, rotate)

        # 2. Redimensionar frame
        frame_resized, _ = self._resize_frame(frame_processed, scale_percent)

        # 3. Converter para RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 4. Detectar pose
        landmarks = self._detect_pose(frame_rgb)

        # 5. Desenhar landmarks se detectados
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame_resized,
                landmarks,
                self.media_pose.POSE_CONNECTIONS,
                *self.drawing_specs,
            )

        return frame_resized

    def run(
        self,
        web_cam: bool = False,
        scale_percent: float = 0.5,
        rotate: bool = False,
    ):
        """Executa o loop principal de processamento de vídeo.

        Args:
            web_cam: Usar webcam (True) ou arquivo (False).
            scale_percent: Percentual de redimensionamento.
            rotate: Rotacionar frames.
        """
        cap = self._get_video_source(web_cam)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._process_frame(
                    frame, scale_percent, rotate
                )
                cv2.imshow('Pose Detection', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            self.pose.close()
            cv2.destroyAllWindows()

    def save(
        self,
        output_path: Optional[str] = None,
        return_frames: bool = False,
        web_cam: bool = False,
        scale_percent: float = 0.5,
        rotate: bool = False,
    ) -> Union[None, List[np.ndarray]]:
        """Processa e salva o vídeo ou retorna frames processados em memória.

        Args:
            output_path: Caminho para salvar o vídeo. Se None, não salva em disco.
            return_frames: Se True, retorna os frames processados em memória.
            web_cam: Se True, captura da webcam. Se False, usa file_path.
            scale_percent: Percentual de redimensionamento dos frames.
            rotate: Se True, rotaciona os frames 90° no sentido horário.

        Returns:
            Se return_frames=True, retorna lista de frames processados.
            Caso contrário, retorna None.

        Raises:
            ValueError: Se ambos output_path e return_frames forem False.
            IOError: Se não conseguir criar o arquivo de vídeo.
        """
        if not output_path and not return_frames:
            raise ValueError(
                'Deve especificar output_path ou setar return_frames=True'
            )

        cap = self._get_video_source(web_cam)
        frames = []

        # Configuração do VideoWriter se for salvar em disco
        if output_path:
            # Obter dimensões do primeiro frame processado
            ret, frame = cap.read()
            if not ret:
                raise IOError('Não foi possível ler o primeiro frame')

            sample_frame = self._process_frame(frame, scale_percent, rotate)
            height, width = sample_frame.shape[:2]

            # Criar VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Voltar ao início

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._process_frame(
                    frame, scale_percent, rotate
                )

                if output_path:
                    out.write(processed_frame)

                if return_frames:
                    frames.append(processed_frame)

        finally:
            cap.release()
            if output_path:
                out.release()
            self.pose.close()

        return frames if return_frames else None

    def run_and_save(
        self,
        output_path: Optional[str] = None,
        show_preview: bool = True,
        web_cam: bool = False,
        scale_percent: float = 0.5,
        rotate: bool = False,
    ) -> Union[None, List[np.ndarray]]:
        """Versão combinada de run() e save() que mostra preview enquanto processa.

        Args:
            output_path: Caminho para salvar o vídeo. Se None, não salva em disco.
            show_preview: Se True, mostra uma janela com o preview.
            web_cam: Se True, captura da webcam. Se False, usa file_path.
            scale_percent: Percentual de redimensionamento dos frames.
            rotate: Se True, rotaciona os frames 90° no sentido horário.

        Returns:
            Lista de frames processados se output_path=None, senão None.
        """
        cap = self._get_video_source(web_cam)
        frames = []

        # Configuração do VideoWriter se for salvar em disco
        if output_path:
            ret, frame = cap.read()
            if not ret:
                raise IOError('Não foi possível ler o primeiro frame')

            sample_frame = self._process_frame(frame, scale_percent, rotate)
            height, width = sample_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._process_frame(
                    frame, scale_percent, rotate
                )

                if output_path:
                    out.write(processed_frame)

                frames.append(processed_frame)

                if show_preview:
                    cv2.imshow('Pose Detection - Preview', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            if output_path:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            self.pose.close()

        return frames if not output_path else None
