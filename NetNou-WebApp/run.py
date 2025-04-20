"""Main entry point for the NetNou application."""

import os
import argparse
from src.app import create_app
from src.config import DevelopmentConfig, ProductionConfig

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the NetNou application')
parser.add_argument('--port', type=int, help='Port to run the server on')
parser.add_argument('--host', type=str, help='Host to run the server on')
args = parser.parse_args()

# Determine environment
env = os.environ.get('FLASK_ENV', 'development')

# Create app with appropriate config
if env == 'production':
    app = create_app(ProductionConfig)
else:
    app = create_app(DevelopmentConfig)

if __name__ == '__main__':
    host = args.host or os.environ.get('HOST', '0.0.0.0')
    port = args.port or int(os.environ.get('PORT', 5001))
    debug = env != 'production'
    
    print(f"Starting NetNou in {env} mode on {host}:{port}")
    app.run(host=host, port=port, debug=debug) 