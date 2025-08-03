"""
Entry point for Streamlit Cloud deployment.
This file simply imports and runs the main dashboard application.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main dashboard
from dashboard.app import main

if __name__ == "__main__":
    main() 