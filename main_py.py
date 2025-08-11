#!/usr/bin/env python3
"""
Main entry point for the AI Resume Builder application
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'backend'))

# Import and run the Flask app from backend
from app import app

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"""
    🚀 AI Resume Builder Starting...
    ================================
    URL: http://localhost:{port}
    Debug: {debug}
    
    Open your browser to:
    → http://localhost:{port}/resume-builder.html
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)