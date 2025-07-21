import streamlit as st
import tempfile
import cv2
from utils.predict import predict_disease

st.title("🌿 AgroVision: Plant Disease Detector")
st.write("Upload a plant leaf image to classify and localize the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
class_names = ['Healthy', 'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold']

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.image(temp_path, caption='Uploaded Image', use_column_width=True)

    st.write("🔍 Classifying...")
    result_image, label = predict_disease(temp_path, "model/model.h5", class_names)

    st.image(result_image, caption=f"Prediction: {label}", use_column_width=True)