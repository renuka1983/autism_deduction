#!/usr/bin/env python3
"""
Setup script for Autism Detection App
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data/face/autistic', 'data/face/non_autistic', 
                   'data/handwriting/autistic', 'data/handwriting/non_autistic',
                   'data/voice/autistic', 'data/voice/non_autistic']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def main():
    print("Setting up Autism Detection App...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        print("\nSetup completed successfully!")
        print("\nTo run the app:")
        print("  streamlit run app.py")
        print("\nTo run the training dashboard:")
        print("  streamlit run train_dashboard.py")
    else:
        print("\nSetup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
