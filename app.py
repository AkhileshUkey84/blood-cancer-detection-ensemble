import streamlit as st
import numpy as np
from PIL import Image
try:
    import cv2
    st.success("OpenCV is successfully imported!")
except ImportError as e:
    st.error(f"Error importing OpenCV: {e}")
import tensorflow as tf
from tensorflow.keras.applications import VGG16, Xception, InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xcept_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as incep_preprocess
import joblib

label_map = {'Benign': 0, '[Malignant] Pre-B': 1, '[Malignant] Pro-B': 2, '[Malignant] early Pre-B': 3}

import gdown

# Download models from Google Drive
def download_models():
    import os

    def download_file(url, filename, expected_min_size_kb):
        if not os.path.exists(filename) or os.path.getsize(filename) < expected_min_size_kb * 1024:
            gdown.download(url, filename, quiet=False)
            size_kb = os.path.getsize(filename) // 1024
            if size_kb < expected_min_size_kb:
                raise ValueError(f"{filename} might be corrupted. Size too small: {size_kb} KB")

    download_file("https://drive.google.com/uc?id=178mgl9upoITsR1YlWJ4AJLnzX3SJoN0N", "cnn_blood_cancer_model.h5", 25000)
    download_file("https://drive.google.com/uc?id=1K4MmvR5-FlV98CfADY8t7MQB_4SkZ3jp", "inceptionv3_blood_cancer_model.h5", 90000)
    download_file("https://drive.google.com/uc?id=1P0AVe1fA5n1_nILogO7oMqoarQky4Y-V", "logistic_regression_model.pkl", 300)
    download_file("https://drive.google.com/uc?id=1v7-jcuOIE4xfZcWG3PuY5__AAZTRGKLY", "svm_model.pkl", 300)
    download_file("https://drive.google.com/uc?id=14A73FT3MzdGu4h1m9DEFvwm7XoW3erfF", "xception_blood_cancer_model.h5", 90000)

    # Repeat similarly for the other models

with st.spinner("Downloading models (first time only)..."):
    download_models()

# Load models
svm_model = joblib.load("svm_model.pkl")
log_model = joblib.load("logistic_regression_model.pkl")
cnn_model = load_model("cnn_blood_cancer_model.h5")
xcept_model = load_model("xception_blood_cancer_model.h5")
incep_model = load_model("inceptionv3_blood_cancer_model.h5")


# Load feature extractor
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

def extract_vgg_features(img_array):
    img_array = vgg_preprocess(img_array)
    features = vgg_model.predict(np.expand_dims(img_array, axis=0))
    return features.flatten()

def preprocess_image(img, size):
    img = img.resize(size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return img_array

def predict_ensemble(img):
    # Reverse label map
    index_to_class = {v: k for k, v in label_map.items()}

    # Preprocess for VGG16
    vgg_img = preprocess_image(img, (224, 224))
    vgg_features = extract_vgg_features(vgg_img)
    svm_pred = svm_model.predict([vgg_features])[0]
    log_pred = log_model.predict([vgg_features])[0]

    # Preprocess for CNN
    cnn_img = preprocess_image(img, (224, 224)) / 255.0
    cnn_class_idx = np.argmax(cnn_model.predict(np.expand_dims(cnn_img, axis=0)), axis=1)[0]

    # Preprocess for Xception
    xcept_img = preprocess_image(img, (224, 224))
    xcept_img = xcept_preprocess(xcept_img)
    xcept_class_idx = np.argmax(xcept_model.predict(np.expand_dims(xcept_img, axis=0)), axis=1)[0]

    # Preprocess for InceptionV3
    incep_img = preprocess_image(img, (224, 224))
    incep_img = incep_preprocess(incep_img)
    incep_class_idx = np.argmax(incep_model.predict(np.expand_dims(incep_img, axis=0)), axis=1)[0]

    # Voting
    predictions_idx = [svm_pred, log_pred, cnn_class_idx, xcept_class_idx, incep_class_idx]
    final_pred_idx = max(set(predictions_idx), key=predictions_idx.count)

    # Convert to class names
    predictions = [index_to_class[i] for i in predictions_idx]
    final_pred = index_to_class[final_pred_idx]

    return predictions, final_pred



# Streamlit UI
st.title("🧬 Blood Cancer Image Classification")
st.write("Upload an image from the blood cancer dataset to classify using ensemble learning.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        preds, final = predict_ensemble(image)

    st.subheader("🔍 Predictions:")
    st.write(f"SVM: {preds[0]}")
    st.write(f"Logistic Regression: {preds[1]}")
    st.write(f"CNN: {preds[2]}")
    st.write(f"Xception: {preds[3]}")
    st.write(f"InceptionV3: {preds[4]}")
    
    st.success(f"✅ Final Prediction (via voting): **{final}**")
