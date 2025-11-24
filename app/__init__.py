from flask import Flask
from .extensions import redis_client
from .config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    redis_client.init_app(app)

    from .main import main_bp

    app.register_blueprint(main_bp)

    return app
