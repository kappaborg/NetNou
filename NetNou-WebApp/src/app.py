"""Flask application initialization."""

from flask import Flask
from .routes.api_routes import api
from .routes.web_routes import web
from .config import Config

def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Load configuration
    app.config.from_object(config_class)
    
    # Register blueprints
    app.register_blueprint(api)
    app.register_blueprint(web)
    
    return app 