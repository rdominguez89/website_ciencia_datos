from app import create_app
from flask_cors import CORS

app = create_app()

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://rastro.pythonanywhere.com"],  # No trailing slash
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

if __name__ == '__main__':
    app.run()