import streamlit as st
import os
import tempfile
from datetime import datetime

st.set_page_config(
    page_title="Autism Screening — Live Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Autism Screening — Live Prediction")
st.markdown("**Multi-modal analysis using facial landmarks, handwriting patterns, and voice characteristics**")

# Add disclaimer
with st.expander("⚠️ Important Disclaimer", expanded=False):
    st.warning("""
    **Medical Disclaimer**: This application is for research and educational purposes only. 
    It should not be used as a substitute for professional medical diagnosis or screening. 
    Always consult with qualified healthcare professionals for medical decisions.
    """)

# Check if we can import the heavy dependencies
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    # Try to import utils
    from utils.model_loader import load_models
    from utils.preprocess import extract_landmarks_from_image, preprocess_handwriting_image, extract_mfcc_from_audio
    from utils.pairs_config import PAIRS
    
    DEPENDENCIES_AVAILABLE = True
    st.success("✅ All dependencies loaded successfully!")
    
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    st.error(f"❌ Missing dependency: {e}")
    st.info("This is a simplified version of the app. Some features may not be available.")
    
    # Create a fallback version
    st.markdown("### Simplified Interface")
    st.info("Please install the full requirements to access all features.")
    
    # Show what the app would do
    st.markdown("#### This app would normally:")
    st.markdown("- Analyze facial landmarks using MediaPipe")
    st.markdown("- Process handwriting patterns with CNN")
    st.markdown("- Extract voice characteristics using MFCC")
    st.markdown("- Provide ensemble predictions")
    st.markdown("- Generate PDF reports")
    
    st.stop()

# If we get here, all dependencies are available
st.markdown("### Full App Interface")

# Layout: left (inputs) | center (preview) | right (legend + distances + result)
left_col, center_col, right_col = st.columns([1, 1.3, 0.9])

# ---------------- Sidebar / Left inputs ----------------
with left_col:
    st.header("Inputs")
    st.markdown("Upload the child's data. **Face image is required** for screening output.")
    face_file = st.file_uploader("Face image (required)", type=["jpg", "jpeg", "png"])
    handwriting_file = st.file_uploader("Handwriting image (optional)", type=["jpg", "jpeg", "png"])
    audio_file = st.file_uploader("Voice recording (WAV, optional)", type=["wav"])
    child_id = st.text_input("Child name / ID (optional)")
    predict_btn = st.button("Predict")

# ---------------- Right legend and results ----------------
with right_col:
    st.header("Landmark Legend & Results")
    st.markdown("**Landmark pairs used (MediaPipe indices):**")
    legend_lines = [f"{i+1}. {a}–{b}" for i, (a, b) in enumerate(PAIRS)]
    st.code("\n".join(legend_lines), language="text")
    st.markdown("---")
    results_box = st.empty()  # placeholder for results summary / probabilities
    distances_box = st.empty()  # placeholder for distances
    pdf_button_box = st.empty()  # placeholder for PDF export button

# Load models
with st.spinner("Loading models..."):
    try:
        face_model, handwriting_model, audio_model = load_models()
        
        # Show model status
        col1, col2, col3 = st.columns(3)
        with col1:
            if face_model:
                st.success("✅ Face Model Loaded")
            else:
                st.warning("⚠️ Face Model Not Found")
        with col2:
            if handwriting_model:
                st.success("✅ Handwriting Model Loaded")
            else:
                st.warning("⚠️ Handwriting Model Not Found")
        with col3:
            if audio_model:
                st.success("✅ Audio Model Loaded")
            else:
                st.warning("⚠️ Audio Model Not Found")
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        face_model, handwriting_model, audio_model = None, None, None

# Rest of the app logic would go here...
st.info("This is a simplified version. The full app logic would be implemented here once all dependencies are properly installed.")
