import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# ===============================
# PAGE CONFIG (HARUS PALING ATAS)
# ===============================
st.set_page_config(
    page_title="Sistem Klasifikasi Alat Musik",
    layout="wide"
)

# ===============================
# FIXED THEME (BROWN DARK + SOFT BROWN)
# ===============================
bg = "#5C4632"          # background utama (dark brown)
sidebar_bg = "#2A241E"  # sidebar dark soft
card_bg = "#3A3228"     # card
accent = "#C8A46A"      # gold brown
text = "#EDE6D8"        # cream text

# ===============================
# APPLY CSS
# ===============================
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}

section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg};
}}

section[data-testid="stSidebar"] * {{
    color: {text};
}}

h1, h2, h3 {{
    color: {accent};
    font-weight: 700;
}}

div.stAlert,
div.stSuccess,
div.stInfo {{
    background-color: {card_bg} !important;
    color: {text} !important;
    border-radius: 14px;
}}

div[data-testid="stFileUploader"] {{
    background-color: #FFFFFF10;
    border-radius: 14px;
    padding: 12px;
}}

button {{
    background-color: {accent} !important;
    color: #000 !important;
    border-radius: 10px;
}}

button:hover {{
    background-color: #B8935A !important;
}}

canvas {{
    background-color: {card_bg} !important;
    border-radius: 12px;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR (TANPA PILIHAN WARNA)
# ===============================
with st.sidebar:
    st.header("⚙️ Tampilan")
    view_mode = st.radio(
        "Mode Tampilan",
        ["Single Model", "Bandingkan Semua Model"]
    )

    st.divider()

    st.header("💡 Tentang Sistem")
    st.write("""
    Sistem klasifikasi citra alat musik berbasis **Deep Learning**.

    Model yang digunakan:
    - CNN (Non-Pretrained)
    - MobileNetV2
    - EfficientNetB0
    """)

# ===============================
# HEADER
# ===============================
st.markdown(
    "<h1 style='text-align:center;'>🎵 Sistem Klasifikasi Alat Musik</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Kelas: gitar, piano, drum, biola, saxophone, cello</p>",
    unsafe_allow_html=True
)
st.divider()

# ===============================
# LOAD CLASS NAMES
# ===============================
TRAIN_DIR = "dataset/train"
class_names = sorted(os.listdir(TRAIN_DIR))

# ===============================
# MODEL PATHS
# ===============================
MODEL_PATHS = {
    "CNN Base": "models/cnn_scratch.h5",
    "MobileNetV2": "models/mobilenetv2.h5",
    "EfficientNetB0": "models/efficientnetb0.h5"
}

# ===============================
# CACHE MODEL
# ===============================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ===============================
# LOAD MODEL(S)
# ===============================
if view_mode == "Single Model":
    selected_model = st.selectbox(
        "Pilih Model",
        list(MODEL_PATHS.keys())
    )
    models = {
        selected_model: load_model(MODEL_PATHS[selected_model])
    }
else:
    models = {
        name: load_model(path)
        for name, path in MODEL_PATHS.items()
    }

# ===============================
# IMAGE UPLOAD
# ===============================
st.subheader(" 🗂️Upload Gambar Alat Musik")
uploaded_file = st.file_uploader(
    "Format JPG / PNG",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col_img, col_result = st.columns([1, 2])
    with col_img:
        st.image(img, caption="Gambar Input", width=350)


    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if view_mode == "Single Model":
        model_name, model = list(models.items())[0]
        preds = model.predict(img_array)[0]

        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx] * 100

        with col_result:
            st.subheader(f"📊 Hasil Prediksi - {model_name}")
            st.success(f"🎯 {class_names[pred_idx]}")
            st.info(f"Confidence: {confidence:.2f}%")

            prob_df = pd.DataFrame({
                "Kelas": class_names,
                "Probabilitas (%)": preds * 100
            })
            st.bar_chart(prob_df.set_index("Kelas"))

    else:
        st.subheader("📊 Perbandingan Semua Model")
        cols = st.columns(3)

        for col, (model_name, model) in zip(cols, models.items()):
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx] * 100

            with col:
                st.markdown(f"### 🔹 {model_name}")
                st.success(class_names[pred_idx])
                st.caption(f"Confidence: {confidence:.2f}%")

                prob_df = pd.DataFrame({
                    "Kelas": class_names,
                    "Probabilitas (%)": preds * 100
                })
                st.bar_chart(prob_df.set_index("Kelas"), height=200)
