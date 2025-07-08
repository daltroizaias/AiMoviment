import os

from flask import Blueprint, redirect, render_template, url_for

from aimoviment.settings import app_config
from aimoviment.skeleton import ImageProcessor

main = Blueprint('main', __name__)


@main.route('/')
def index():
    video_file = 'C:\\Users\\daltr\\OneDrive\\Repositorio GIT\\AiMoviment\\aimoviment\\static\\videos\\20250701_124007000_iOS.MOV'  # noqa: E501
    processed_file = 'processed_' + video_file
    processed_path = os.path.join(app_config.VIDEO_FOLDER, processed_file)

    # Processa o vídeo se ainda não estiver feito
    if not os.path.exists(processed_path):
        processor = ImageProcessor(video_file)
        processor.process_and_save()

    video_url = url_for(
        'static', filename=os.path.join(app_config.VIDEO_URL, processed_file)
    )  # noqa: E501
    return render_template('index.html', video_url=video_url)


@main.route('/process/<filename>')
def process_video(filename):
    processor = ImageProcessor(filename)
    processor.process_and_save()

    return redirect(url_for('main.index'))
