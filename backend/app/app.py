from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_mapping(SECRET_KEY="dev")  # для разработки

    @app.route("/")
    def index():
        return "Hello, Flask!"

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)