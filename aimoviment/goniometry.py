from typing import List

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from mediapipe.python.solutions.pose import PoseLandmark as PL


class PoseAnalyzer:
    def __init__(self, landmarks: List[NormalizedLandmark]):
        """
        Inicializa com a lista de landmarks.
        """
        self.landmarks = landmarks

    def get_point(self, idx: int) -> np.ndarray:
        lm = self.landmarks.landmark[idx]
        return np.array([lm.x, lm.y, lm.z])

    def calcular_angulo(self, a_idx: int, b_idx: int, c_idx: int) -> float:
        """
        Calcula o ângulo entre três landmarks: A-B-C
        """
        a = self.get_point(a_idx)
        b = self.get_point(b_idx)
        c = self.get_point(c_idx)

        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angulo = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angulo

    def detectar_plano_movimento(self) -> str:
        """
        Heurística para detectar plano principal do movimento baseado na orientação dos ombros e quadris.

        Return:
            'sagital' - visão de perfil (movimento predominante para frente/trás)
            'frontal' - visão de frente (movimento predominante lateral)
            'transversal' - visão de cima (rotação)
        """

        # Obter coordenadas dos ombros
        left_shoulder = self.get_point(PL.LEFT_SHOULDER.value)
        right_shoulder = self.get_point(PL.RIGHT_SHOULDER.value)

        # Obter coordenadas dos quadris
        left_hip = self.get_point(PL.LEFT_HIP.value)
        right_hip = self.get_point(PL.RIGHT_HIP.value)

        # Calcular diferenças nas coordenadas X e Z entre ombros
        shoulder_diff_x = abs(left_shoulder[0] - right_shoulder[0])  # diferença horizontal
        shoulder_diff_z = abs(left_shoulder[1] - right_shoulder[1])  # diferença vertical (na imagem 2D)

        # Calcular diferenças nos quadris
        hip_diff_x = abs(left_hip[0] - right_hip[0])
        hip_diff_z = abs(left_hip[1] - right_hip[1])

        # Média das diferenças
        avg_diff_x = (shoulder_diff_x + hip_diff_x) / 2
        avg_diff_z = (shoulder_diff_z + hip_diff_z) / 2

        # Determinar o plano predominante
        if avg_diff_z > avg_diff_x * 1.5:  # diferença vertical significativamente maior
            return 'sagital'  # visão de perfil
        elif avg_diff_x > avg_diff_z * 1.5:  # diferença horizontal significativamente maior
            return 'frontal'  # visão de frente
        else:
            return 'transversal'  # caso intermediário (possível rotação)
