from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
from base64 import b64encode

from aimoviment.goniometry import PoseAnalyzer
from aimoviment.image_processor import ImageProcessor, PoseConfig

main = Blueprint('main', __name__)

pose_config = PoseConfig(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=0,
)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/video_feed', methods=['POST'])
def video_feed():
    if 'image' not in request.files:
        return jsonify({'error': 'Arquivo de imagem não enviado'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_processor = ImageProcessor(pose_config=pose_config)
    frame_processed, landmarks = image_processor.process_frame(frame)

    # Compactar imagem com qualidade reduzida
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    _, buffer = cv2.imencode('.jpg', frame_processed, encode_params)
    frame_base64 = b64encode(buffer).decode('utf-8')

    if landmarks:
        analyser = PoseAnalyzer(landmarks)
        plano = analyser.detectar_plano_movimento()
        return jsonify({'status': 'ok', 'plano': plano, 'frame_base64': frame_base64})
    else:
        return jsonify({'status': 'ok', 'plano': None, 'frame_base64': frame_base64})
