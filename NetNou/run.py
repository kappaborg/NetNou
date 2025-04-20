"""Main entry point for the NetNou application."""

import os
from src.app import create_app
from src.config import DevelopmentConfig, ProductionConfig

# Determine environment
env = os.environ.get('FLASK_ENV', 'development')

# Create app with appropriate config
if env == 'production':
    app = create_app(ProductionConfig)
else:
    app = create_app(DevelopmentConfig)

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = env != 'production'
    
    print(f"Starting NetNou in {env} mode on {host}:{port}")
    app.run(host=host, port=port, debug=debug) 