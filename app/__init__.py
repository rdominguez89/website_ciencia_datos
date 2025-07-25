from flask import Flask, request, abort
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    from app.routes import bp
    app.register_blueprint(bp)

    @app.before_request
    def restrict_host():
        """
        Restrict access to only allow requests coming from http://127.0.0.1:5000/ 
        and https://rastro.pythonanywhere.com/.
        """
        allowed_hosts = {"127.0.0.1:5000", "rastro.pythonanywhere.com"}
        host = request.headers.get("Host", "")
        if host not in allowed_hosts:
            abort(403)

    return app