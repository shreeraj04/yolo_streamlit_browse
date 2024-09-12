import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

def load_model(version, size):
    model_name = f"yolo{version}{size}"
    return YOLO(model_name)

def detect_objects(model, image):
    results = model(image)
    return results[0].plot()

def main():
    st.title("YOLO Object Detection App")

    col1, col2 = st.columns(2)
    with col1:
        version = st.selectbox("Select YOLO version", ["v8", "v9"])
    with col2:
        size = st.selectbox("Select model size", ["n", "s", "m", "l", "x"])

    model = load_model(version, size)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            result_image = detect_objects(model, image)
            st.image(result_image, caption="Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()