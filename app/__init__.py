from flask import Flask, request, abort, session
from flask_babel import Babel
from config import Config

def get_locale():
    """Determine the best locale for the user."""
    # 1. Try to get language from session
    if 'language' in session:
        return session['language']
    
    # 2. Try to get from cookie
    language = request.cookies.get('language')
    if language in ['en', 'es']:
        return language
    
    # 3. Fall back to browser's accepted languages
    return request.accept_languages.best_match(['en', 'es']) or 'en'

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize Babel
    babel = Babel(app, locale_selector=get_locale)

    from app.routes import bp
    app.register_blueprint(bp)

    @app.before_request
    def restrict_host():
        """
        Restrict access to only allow requests coming from http://127.0.0.1:5000/ 
        and https://rastro.pythonanywhere.com/.
        """
        allowed_hosts = {"rastro.pythonanywhere.com"}
        host = request.headers.get("Host", "")
        if host not in allowed_hosts:
            abort(403)

    return app