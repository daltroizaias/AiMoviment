from flask import Flask

from aimoviment.routes import main
from aimoviment.settings import Config


def create_app(config: Config = Config) -> Flask:
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(config)

    app.register_blueprint(main, url_prefix='/')

    return app
