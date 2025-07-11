from aimoviment import create_app
from aimoviment.settings import app_config

app = create_app(config=app_config)
