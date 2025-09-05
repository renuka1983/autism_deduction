import cv2
import numpy as np
import librosa

# Check for MediaPipe availability
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from utils.pairs_config import PAIRS
except ImportError:
    # Fallback if import fails
    PAIRS = [(1,2),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17)]

def extract_landmarks_from_image(path, return_overlay=False):
    if not MEDIAPIPE_AVAILABLE:
        # Return dummy data when MediaPipe is not available
        dummy_distances = np.random.random(len(PAIRS)) * 0.1 + 0.05  # Random normalized distances
        img = cv2.imread(path)
        if img is None:
            return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return dummy_distances, img_rgb, None
    
    img = cv2.imread(path)
    if img is None:
        return None, None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, None, None

    h, w, _ = img.shape
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.append((int(lm.x * w), int(lm.y * h)))
    landmarks = np.array(landmarks)

    distances = []
    for (p1, p2) in PAIRS:
        d = np.linalg.norm(landmarks[p1] - landmarks[p2])
        distances.append(d)

    distances = np.array(distances)
    annotated_img = None
    if return_overlay:
        annotated_img = img_rgb.copy()
        for (p1, p2) in PAIRS:
            cv2.circle(annotated_img, tuple(landmarks[p1]), 3, (0,255,0), -1)
            cv2.circle(annotated_img, tuple(landmarks[p2]), 3, (0,0,255), -1)
            cv2.line(annotated_img, tuple(landmarks[p1]), tuple(landmarks[p2]), (255,0,0), 1)

    return distances, annotated_img, landmarks

def preprocess_handwriting_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def extract_mfcc_from_audio(path, max_pad_len=100):
    try:
        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc[..., np.newaxis]
    except Exception:
        return None
