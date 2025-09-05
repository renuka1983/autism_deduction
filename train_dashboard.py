import streamlit as st
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, TimeDistributed
from utils.preprocess import extract_landmarks_from_image, preprocess_handwriting_image, extract_mfcc_from_audio

st.set_page_config(page_title="Training Dashboard", layout="wide")
st.title("Unified Autism Detection Trainer")

DATA_PATH, SAVE_PATH = "data", "models"
os.makedirs(SAVE_PATH, exist_ok=True)

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.legend()
    st.pyplot(fig)

if st.button("Train All"):
    # FACE
    X, y = [], []
    for label in os.listdir(os.path.join(DATA_PATH,"face")):
        for file in os.listdir(os.path.join(DATA_PATH,"face",label)):
            feats,_,_ = extract_landmarks_from_image(os.path.join(DATA_PATH,"face",label,file))
            if feats is not None:
                X.append(feats)
                y.append(1 if label=="autistic" else 0)
    if X:
        X,y = np.array(X), np.array(y)
        Xt,Xv,yt,yv = train_test_split(X,y,test_size=0.2)
        model = Sequential([Dense(128,activation='relu',input_shape=(X.shape[1],)),Dropout(0.3),
                            Dense(64,activation='relu'),Dense(1,activation='sigmoid')])
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        hist = model.fit(Xt,yt,validation_data=(Xv,yv),epochs=5,verbose=0)
        model.save(os.path.join(SAVE_PATH,"face_model.h5"))
        st.line_chart(hist.history["accuracy"])
        yp = model.predict(Xv).flatten()
        plot_cm(yv,yp,"Face CM"); plot_roc(yv,yp,"Face ROC")

    # HANDWRITING
    X,y = [],[]
    for label in os.listdir(os.path.join(DATA_PATH,"handwriting")):
        for file in os.listdir(os.path.join(DATA_PATH,"handwriting",label)):
            img = preprocess_handwriting_image(os.path.join(DATA_PATH,"handwriting",label,file))
            if img is not None:
                X.append(img); y.append(1 if label=="autistic" else 0)
    if X:
        X,y = np.array(X), np.array(y)
        Xt,Xv,yt,yv = train_test_split(X,y,test_size=0.2)
        model = Sequential([Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)),
                            MaxPooling2D(),Conv2D(64,(3,3),activation='relu'),
                            MaxPooling2D(),Flatten(),Dense(64,activation='relu'),
                            Dense(1,activation='sigmoid')])
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        hist = model.fit(Xt,yt,validation_data=(Xv,yv),epochs=5,verbose=0)
        model.save(os.path.join(SAVE_PATH,"handwriting_model.h5"))
        st.line_chart(hist.history["accuracy"])
        yp = model.predict(Xv).flatten()
        plot_cm(yv,yp,"HW CM"); plot_roc(yv,yp,"HW ROC")

    # VOICE
    X,y = [],[]
    for label in os.listdir(os.path.join(DATA_PATH,"voice")):
        for file in os.listdir(os.path.join(DATA_PATH,"voice",label)):
            mfcc = extract_mfcc_from_audio(os.path.join(DATA_PATH,"voice",label,file))
            if mfcc is not None:
                X.append(mfcc); y.append(1 if label=="autistic" else 0)
    if X:
        X,y = np.array(X), np.array(y)
        Xt,Xv,yt,yv = train_test_split(X,y,test_size=0.2)
        model = Sequential([TimeDistributed(Conv2D(32,(3,3),activation='relu'),input_shape=(40,100,1)),
                            TimeDistributed(MaxPooling2D()),TimeDistributed(Flatten()),
                            LSTM(64),Dense(1,activation='sigmoid')])
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        hist = model.fit(Xt,yt,validation_data=(Xv,yv),epochs=5,verbose=0)
        model.save(os.path.join(SAVE_PATH,"audio_model.h5"))
        st.line_chart(hist.history["accuracy"])
        yp = model.predict(Xv).flatten()
        plot_cm(yv,yp,"Audio CM"); plot_roc(yv,yp,"Audio ROC")
