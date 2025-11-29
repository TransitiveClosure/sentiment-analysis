from flask import Flask
from flask_socketio import SocketIO
from flask_executor import Executor

socketio = SocketIO(cors_allowed_origins="*")
executor = Executor()

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "secret!"

    # инициализация здесь
    socketio.init_app(app)
    executor.init_app(app)

    # blueprint импортируем ПОСЛЕ создания app
    from .routes import main
    app.register_blueprint(main)

    return app