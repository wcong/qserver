"""
QServer Flask Application
A web server for qlib data training and visualization
"""

import logging
import sys
from flask import Flask


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object('app.config.Config')
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if app.config.get('DEBUG') else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
        ]
    )
    
    # Set Flask's logger level
    app.logger.setLevel(logging.DEBUG if app.config.get('DEBUG') else logging.INFO)
    
    # Also log werkzeug (Flask's internal server)
    logging.getLogger('werkzeug').setLevel(logging.DEBUG)
    
    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    app.logger.info("QServer Flask application initialized")
    
    return app
