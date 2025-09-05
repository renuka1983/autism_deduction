import os
from tensorflow.keras.models import load_model

MODEL_PATH = "models"

def load_models():
    face_model, handwriting_model, audio_model = None, None, None
    if os.path.exists(os.path.join(MODEL_PATH, "face_model.h5")):
        face_model = load_model(os.path.join(MODEL_PATH, "face_model.h5"))
    if os.path.exists(os.path.join(MODEL_PATH, "handwriting_model.h5")):
        handwriting_model = load_model(os.path.join(MODEL_PATH, "handwriting_model.h5"))
    if os.path.exists(os.path.join(MODEL_PATH, "audio_model.h5")):
        audio_model = load_model(os.path.join(MODEL_PATH, "audio_model.h5"))
    return face_model, handwriting_model, audio_model
