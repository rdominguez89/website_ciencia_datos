from flask import Blueprint, render_template, request, jsonify, send_file, current_app, abort
import pandas as pd
import os
import re
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from flask_talisman import Talisman
from werkzeug.exceptions import HTTPException
from .utils import allowed_file, load_dataframe, get_stats_summary, clean_dataframe, create_visualizations_function, remove_outliers_function, create_correlation_plots_function, save_dataframe
from .analyze_supervised import analyze_data
from flask_cors import CORS
from traceback import format_exc

bp = Blueprint('main', __name__)

# Initialize security extensions
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
csrf = CSRFProtect()

CORS(bp, resources={
    r"/api/*": {
        "origins": ["https://rastro.pythonanywhere.com"],  # No trailing slash
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# New before-request hook for API origin validation
@bp.before_request
def restrict_api_origin():
    """
    Reject requests to API endpoints that do not have an allowed Origin header.
    This ensures that API endpoints are only accessible from our webpages.
    """
    if request.path.startswith('/api/'):
        allowed_origins = {"http://127.0.0.1:5000", "https://rastro.pythonanywhere.com"}
        origin = request.headers.get("Origin")
        if origin not in allowed_origins:
            current_app.logger.error(f"Blocked API access from Origin: {origin}")
            abort(403)

# Error handling
@bp.errorhandler(Exception)
def handle_exception(e):
    # Log the full traceback for debugging
    current_app.logger.error(f"Error: {str(e)}\n{format_exc()}")

    # Extract error information
    error_message = str(e)
    error_type = type(e).__name__

    # Return error information to client
    if isinstance(e, HTTPException):
        return jsonify({'error': e.description, 'error_type': error_type}, e.code)
    return jsonify({'error': error_message, 'error_type': error_type}), 500

# Security headers (configure in your app factory)
talisman = Talisman(
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'style-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
    }
)

def validate_filename(filename):
    """Validate and sanitize filenames"""
    if not re.match(r'^[\w,\s-]+\.[A-Za-z]{3}$', filename):
        raise ValueError("Invalid filename")
    return secure_filename(filename)

@bp.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@bp.route('/cleaning')
def cleaning():
    """Render the data cleaning page."""
    return render_template('cleaning.html')

@bp.route('/supervised')
def supervised():
    """Render the supervised learning page."""
    return render_template('supervised.html')

@bp.route('/nosupervised')
def nosupervised():
    """Render the unsupervised learning page."""
    return render_template('nosupervised.html')

@bp.route('/inference')
def inference():
    """Render the inference page."""
    return render_template('inference.html')

@bp.route('/api/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    """Handle file upload and validate CSV input."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Validate filename
            filename = validate_filename(file.filename)
            
            # Verify file is actually CSV
            if not file.content_type == 'text/csv':
                return jsonify({'error': 'Invalid file type'}), 400
                
            # Process file directly from memory
            df = load_dataframe(file)
            
            return jsonify({
                'success': True,
                'df': df.where(pd.notnull(df), '-').to_dict(orient='records')
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': 'File processing failed'}), 400
    return jsonify({'error': 'Invalid file type'}), 400

@bp.route('/api/statsummary', methods=['POST'])
@limiter.limit("20 per minute")
def statsummary():
    """Generate statistical summary of the uploaded dataset."""
    try:
        data = request.json
        df = pd.DataFrame.from_dict(data['df']).replace('-', None)
        response = get_stats_summary(df)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': 'Failed to generate statistics'}), 500

@bp.route('/api/clean_data', methods=['POST'])
@limiter.limit("15 per minute")
def clean_data():
    """Clean the dataset by removing nulls and duplicates as specified."""
    try:
        data = request.json
        df = pd.DataFrame.from_dict(data['df']).replace('-', None)
        df = clean_dataframe(df, data.get('remove_nulls', False), data.get('remove_duplicates', False))
        return jsonify({
            'success': True,
            'df': df.where(pd.notnull(df), '-').to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': 'Data cleaning failed'}), 500

@bp.route('/api/create_visualizations', methods=['POST'])
@limiter.limit("10 per minute")
def create_visualizations():
    """Generate boxplots and distribution plots for the dataset."""
    try:
        data = request.json
        df = pd.DataFrame.from_dict(data['df']).replace('-', None)
        visualizations = create_visualizations_function(df)
        return jsonify(visualizations)
    except Exception as e:
        return jsonify({'error': 'Visualization generation failed'}), 500

@bp.route('/api/remove_outliers', methods=['POST'])
@limiter.limit("15 per minute")
def remove_outliers():
    """Remove specified outliers from the dataset."""
    try:
        data = request.json
        df = pd.DataFrame.from_dict(data['df']).replace('-', None)
        result = remove_outliers_function(df, data.get('outliers', []))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': 'Outlier removal failed', 'success': False}), 400

@bp.route('/api/create_correlation_plots', methods=['POST'])
@limiter.limit("10 per minute")
def create_correlation_plots():
    """Generate pairplot and correlation heatmap for numerical features."""
    try:
        data = request.json
        df = pd.DataFrame.from_dict(data['df'])
        plots = create_correlation_plots_function(df)
        return jsonify(plots)
    except Exception as e:
        return jsonify({'error': 'Correlation plot generation failed', 'success': False}), 500

@bp.route('/api/save_data', methods=['POST'])
@limiter.limit("5 per minute")
def save_data():
    """Save the processed dataset as a CSV file."""
    try:
        data = request.json
        df = pd.DataFrame(data['df'])
        return save_dataframe(df, data.get('columns', df.columns.tolist()), data.get('filename', 'processed_data'))
    except Exception as e:
        return jsonify({'error': 'Data saving failed', 'success': False}), 500

@bp.route('/api/analyze', methods=['POST'])
@limiter.limit("5 per minute")
def analyze():
    """Perform machine learning analysis based on user-specified parameters."""
    try:
        data = request.json
        df = pd.DataFrame.from_dict(data['df'])
        result = analyze_data(df, data)
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Analysis failed: {str(e)}\n{format_exc()}")
        return jsonify({'error': str(e), 'error_type': type(e).__name__, 'success': False}), 400