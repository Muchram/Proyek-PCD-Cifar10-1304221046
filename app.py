import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def gaussian_denoise(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def median_denoise(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def clahe_on_l_channel(img, clipLimit=2.0, tileGridSize=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def contrast_stretch(img):
    img_f = img.astype(np.float32) / 255.0
    p2, p98 = np.percentile(img_f, (2, 98))
    out = (img_f - p2) / (p98 - p2 + 1e-8)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

def preprocess(img, cfg):
    out = img.copy()
    if cfg["use_gaussian"]:
        out = gaussian_denoise(out, 3)
    if cfg["use_median"]:
        out = median_denoise(out, 3)
    if cfg["use_clahe"]:
        out = clahe_on_l_channel(out)
    if cfg["use_contrast"]:
        out = contrast_stretch(out)
    return out

@st.cache_resource
def load_models():
    clf = joblib.load("svm_cifar10_mobilenet.joblib")
    cfg = joblib.load("preprocess_config.joblib")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224,224,3)
    )
    base_model.trainable = False
    return clf, cfg, base_model

def extract_feature_single(img_rgb_uint8, cfg, base_model):
    img_p = preprocess(img_rgb_uint8, cfg)

    x = tf.image.resize(img_p, (224,224)).numpy().astype(np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    feat = base_model.predict(x, verbose=0)
    return img_p, feat

st.title("CIFAR-10 Classifier (Preprocess + MobileNetV2 + SVM) by Y Muchram Anarqi")

uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img).astype(np.uint8)

    clf, cfg, base_model = load_models()
    img_after, feat = extract_feature_single(img_np, cfg, base_model)

    pred = clf.predict(feat)[0]
    scores = clf.decision_function(feat).flatten()
    top3 = np.argsort(scores)[::-1][:3]

    st.subheader("Preview")
    st.image(img_np, caption="Original", use_container_width=True)
    st.image(img_after, caption="After Preprocessing", use_container_width=True)

    st.subheader("Prediction")
    st.write("Predicted class:", class_names[pred])

    st.subheader("Top-3 (SVM scores)")
    for i in top3:
        st.write(f"- {class_names[i]} (score={scores[i]:.3f})")