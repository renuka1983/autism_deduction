# This is the main entry point for Streamlit Cloud
# It simply imports and runs the main app

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
if __name__ == "__main__":
    # This will be executed when running on Streamlit Cloud
    exec(open('app.py').read())
