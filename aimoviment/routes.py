import os

from flask import Blueprint, redirect, render_template, url_for

from aimoviment.process_video import ProcessVideo
from aimoviment.settings import config

main = Blueprint('main', __name__)


@main.route('/')
def index():
    video_file = "C:\\Users\\daltr\\OneDrive\\Repositorio GIT\\AiMoviment\\aimoviment\\static\\videos\\20250701_124007000_iOS.MOV"
    processed_file = "processed_" + video_file
    processed_path = os.path.join(config.VIDEO_FOLDER, processed_file)

    # Processa o vídeo se ainda não estiver feito
    if not os.path.exists(processed_path):
        processor = ProcessVideo(video_file)
        processor.process_and_save()

    video_url = url_for('static', filename=os.path.join(config.VIDEO_URL, processed_file))
    return render_template('index.html', video_url=video_url)


@main.route('/process/<filename>')
def process_video(filename):
    processor = ProcessVideo(filename)
    processor.process_and_save()

    return redirect(url_for('main.index'))
