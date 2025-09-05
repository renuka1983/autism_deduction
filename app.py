# app.py
import os
import io
import tempfile
from datetime import datetime

# Import with error handling for Streamlit Cloud compatibility
try:
    import cv2
    import numpy as np
    import streamlit as st
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError as e:
    st.error(f"Missing required dependency: {e}")
    st.error("Please check your requirements.txt file and ensure all dependencies are installed.")
    st.stop()

# utils - ensure these modules exist in utils/
try:
    from utils.model_loader import load_models
    from utils.preprocess import extract_landmarks_from_image, preprocess_handwriting_image, extract_mfcc_from_audio
    from utils.pairs_config import PAIRS  # pairs of mediapipe indices used for distances
except ImportError as e:
    st.error(f"Error importing utility modules: {e}")
    st.error("Please ensure all files are in the correct directory structure.")
    st.stop()

st.set_page_config(
    page_title="Autism Screening — Live Prediction", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/autism-detector',
        'Report a bug': "https://github.com/your-repo/autism-detector/issues",
        'About': "Multi-modal autism screening application using facial landmarks, handwriting, and voice analysis."
    }
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

# Load models (face_model, handwriting_model, audio_model) - may be None if not trained
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

# helper to show resized image nicely in center
def safe_image_show(img_rgb, caption=None, max_width=520):
    if img_rgb is None:
        return
    h, w = img_rgb.shape[:2]
    scale = min(max_width / max(w, 1), 1.0)
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    st.image(img_rgb, caption=caption, use_column_width=False)

# ---------------- Prediction flow ----------------
if predict_btn:
    # reset placeholders
    results_box.empty()
    distances_box.empty()
    pdf_button_box.empty()

    modality_probs = {}
    distances_texts = []
    annotated_image = None
    handwriting_preview_rgb = None
    waveform_fig = None

    # ---- FACE processing (required for ensemble) ----
    if face_file is None:
        st.error("Face image is required. Please upload a face image.")
    else:
        with st.spinner("Processing face image..."):
            # save temp face file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                tf.write(face_file.read())
                face_temp_path = tf.name

            # extract landmarks, annotated overlay, and used points using utils.preprocess
            feats, annotated_overlay, used_points = extract_landmarks_from_image(face_temp_path, return_overlay=True)

            if annotated_overlay is None or feats is None:
                st.warning("Could not detect face landmarks in the provided face image.")
            else:
                annotated_image = annotated_overlay.copy()
                # display annotated image in center
                with center_col:
                    st.subheader("Face preview")
                    safe_image_show(annotated_image, caption="Facial landmarks (selected points highlighted)")

                # show distances in right column
                with right_col:
                    st.subheader("Computed Distances (normalized)")
                    distances_lines = []
                    for i, (p1, p2) in enumerate(PAIRS):
                        # feats is expected to be normalized distances corresponding to PAIRS ordering
                        try:
                            val = feats[i]
                            line = f"{i+1}. {p1}-{p2}: {val:.4f}"
                        except Exception:
                            line = f"{i+1}. {p1}-{p2}: N/A"
                        distances_lines.append(line)
                    distances_box.code("\n".join(distances_lines), language="text")

                # face model prediction (if loaded) - ensure shape matches training (feats is 1D vector)
                if face_model is not None:
                    try:
                        face_prob = float(face_model.predict(np.expand_dims(feats, axis=0), verbose=0)[0][0])
                        modality_probs["face"] = face_prob
                        with right_col:
                            st.info(f"Face model probability (ASD): {face_prob:.3f}")
                    except Exception as e:
                        with right_col:
                            st.error(f"Face model prediction error: {e}")
                else:
                    with right_col:
                        st.info("Face model not loaded (models/face_model.h5).")

    # ---- HANDWRITING processing (optional) ----
    handwriting_prob = None
    if handwriting_file is not None:
        with st.spinner("Processing handwriting image..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                tf.write(handwriting_file.read())
                hw_temp_path = tf.name

        # preview handwriting in center
        img_bgr = cv2.imread(hw_temp_path)
        if img_bgr is not None:
            handwriting_preview_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            with center_col:
                st.subheader("Handwriting preview")
                safe_image_show(handwriting_preview_rgb, caption="Handwriting sample")

        img_proc = preprocess_handwriting_image(hw_temp_path)
        if img_proc is None:
            with right_col:
                st.warning("Could not preprocess handwriting image.")
        else:
            if handwriting_model is not None:
                try:
                    handwriting_prob = float(handwriting_model.predict(np.expand_dims(img_proc, axis=0), verbose=0)[0][0])
                    modality_probs["handwriting"] = handwriting_prob
                    with right_col:
                        st.info(f"Handwriting model probability (ASD): {handwriting_prob:.3f}")
                except Exception as e:
                    with right_col:
                        st.error(f"Handwriting model prediction error: {e}")
            else:
                with right_col:
                    st.info("Handwriting model not loaded (models/handwriting_model.h5).")

    # ---- AUDIO processing (optional) ----
    audio_prob = None
    if audio_file is not None:
        with st.spinner("Processing audio file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                tf.write(audio_file.read())
                audio_temp_path = tf.name

        # waveform preview in center
        try:
            y, sr = librosa.load(audio_temp_path, sr=None)
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Audio waveform")
            with center_col:
                st.subheader("Audio preview")
                st.pyplot(fig, clear_figure=True)
            waveform_fig = fig  # to save into PDF later
        except Exception:
            with center_col:
                st.warning("Could not load audio for waveform preview.")

        mfcc = extract_mfcc_from_audio(audio_temp_path)
        if mfcc is None:
            with right_col:
                st.warning("Could not extract MFCC features from audio.")
        else:
            if audio_model is not None:
                try:
                    # mfcc expected shape e.g., (40,100,1) -> model training shapes may vary
                    x = np.transpose(np.expand_dims(mfcc, 0), (0, 2, 1, 3))  # (1, time, mfcc, ch) maybe needed
                    audio_prob = float(audio_model.predict(x, verbose=0)[0][0])
                    modality_probs["voice"] = audio_prob
                    with right_col:
                        st.info(f"Voice model probability (ASD): {audio_prob:.3f}")
                except Exception as e:
                    with right_col:
                        st.error(f"Audio model prediction error: {e}")
            else:
                with right_col:
                    st.info("Audio model not loaded (models/audio_model.h5).")

    # ---- Ensemble aggregation & final result ----
    if modality_probs:
        avg_prob = float(np.mean(list(modality_probs.values())))
        final_label = "Positive (screening)" if avg_prob >= 0.5 else "Negative (screening)"
        with right_col:
            st.subheader("Final Ensemble Prediction")
            st.success(f"ASD likelihood: {avg_prob:.3f} → {final_label}")
            if child_id:
                st.caption(f"Child ID: {child_id}")
    else:
        with right_col:
            st.warning("No model predictions available. Ensure at least one model is loaded and inputs are valid.")

    # ---------------- PDF Export (ReportLab) ----------------
    # create a downloadable PDF with results and embedded previews
    if modality_probs:
        def create_pdf_bytes():
            # create temp files for images
            tmp_dir = tempfile.gettempdir()
            pdf_name = f"autism_report_{(child_id or 'child')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf_path = os.path.join(tmp_dir, pdf_name)

            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            margin = 50
            y = height - margin

            # Header
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, y, "Autism Screening Report")
            y -= 25
            c.setFont("Helvetica", 10)
            c.drawString(margin, y, f"Child ID: {child_id or 'N/A'}")
            c.drawRightString(width - margin, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 20
            c.line(margin, y, width - margin, y)
            y -= 20

            # Face distances & probabilities
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Face Analysis")
            y -= 16
            c.setFont("Helvetica", 10)
            if 'face' in modality_probs:
                c.drawString(margin, y, f"Face model probability (ASD): {modality_probs['face']:.3f}")
            else:
                c.drawString(margin, y, "Face model probability: N/A")
            y -= 14

            # list distances (truncate if too many)
            if annotated_image is not None and feats is not None:
                c.drawString(margin, y, "Distances (normalized):")
                y -= 14
                for i, (p1, p2) in enumerate(PAIRS):
                    if i >= len(feats):
                        break
                    line = f"{i+1}. {p1}-{p2}: {feats[i]:.4f}"
                    c.drawString(margin + 10, y, line)
                    y -= 12
                    if y < 120:
                        c.showPage()
                        y = height - margin

                # embed annotated face image (save temp)
                try:
                    face_img_temp = os.path.join(tmp_dir, f"annot_face_{datetime.now().timestamp()}.png")
                    # annotated_image may be RGB
                    cv2.imwrite(face_img_temp, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    # place image and adjust y
                    img_w = 200
                    img_h = 200 * annotated_image.shape[0] / annotated_image.shape[1]
                    if y - img_h < margin:
                        c.showPage()
                        y = height - margin
                    c.drawImage(face_img_temp, margin, y - img_h, width=img_w, height=img_h)
                    y -= img_h + 10
                except Exception:
                    pass

            # Handwriting block
            if 'handwriting' in modality_probs or handwriting_preview_rgb is not None:
                if y < 160:
                    c.showPage()
                    y = height - margin
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, y, "Handwriting Analysis")
                y -= 16
                c.setFont("Helvetica", 10)
                if 'handwriting' in modality_probs:
                    c.drawString(margin, y, f"Handwriting model probability (ASD): {modality_probs['handwriting']:.3f}")
                else:
                    c.drawString(margin, y, "Handwriting model probability: N/A")
                y -= 12

                if handwriting_preview_rgb is not None:
                    try:
                        hw_temp = os.path.join(tmp_dir, f"hw_{datetime.now().timestamp()}.png")
                        cv2.imwrite(hw_temp, cv2.cvtColor(handwriting_preview_rgb, cv2.COLOR_RGB2BGR))
                        img_w = 200
                        img_h = 200 * handwriting_preview_rgb.shape[0] / handwriting_preview_rgb.shape[1]
                        if y - img_h < margin:
                            c.showPage()
                            y = height - margin
                        c.drawImage(hw_temp, margin, y - img_h, width=img_w, height=img_h)
                        y -= img_h + 10
                    except Exception:
                        pass

            # Audio block
            if 'voice' in modality_probs or waveform_fig is not None:
                if y < 160:
                    c.showPage()
                    y = height - margin
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, y, "Voice Analysis")
                y -= 16
                c.setFont("Helvetica", 10)
                if 'voice' in modality_probs:
                    c.drawString(margin, y, f"Voice model probability (ASD): {modality_probs['voice']:.3f}")
                else:
                    c.drawString(margin, y, "Voice model probability: N/A")
                y -= 12

                if waveform_fig is not None:
                    try:
                        wf_temp = os.path.join(tmp_dir, f"wave_{datetime.now().timestamp()}.png")
                        waveform_fig.savefig(wf_temp, bbox_inches="tight")
                        img_w = 400
                        img_h = 200
                        if y - img_h < margin:
                            c.showPage()
                            y = height - margin
                        c.drawImage(wf_temp, margin, y - img_h, width=img_w, height=img_h)
                        y -= img_h + 10
                    except Exception:
                        pass

            # Final ensemble
            if y < 120:
                c.showPage()
                y = height - margin
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Final Ensemble Result")
            y -= 16
            c.setFont("Helvetica", 11)
            c.drawString(margin, y, f"Aggregate ASD likelihood: {avg_prob:.3f} → {final_label}")
            y -= 20

            c.save()

            # read bytes
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            # cleanup temp images (optionally)
            try:
                if 'face_img_temp' in locals() and os.path.exists(face_img_temp):
                    os.remove(face_img_temp)
                if 'hw_temp' in locals() and os.path.exists(hw_temp):
                    os.remove(hw_temp)
                if 'wf_temp' in locals() and os.path.exists(wf_temp):
                    os.remove(wf_temp)
            except Exception:
                pass

            return pdf_bytes, os.path.basename(pdf_path)

        # show download button
        pdf_bytes, pdf_filename = create_pdf_bytes()
        pdf_button_box.download_button("📄 Download Report (PDF)", data=pdf_bytes, file_name=pdf_filename, mime="application/pdf")
# Full Streamlit prediction app code with face landmarks, handwriting, voice, and PDF export.
