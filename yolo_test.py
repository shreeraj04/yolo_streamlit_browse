import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import requests
import torch
import cv2
import numpy as np
import os

def load_model(version, size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if version in ["v8", "v9"]:
        model_name = f"yolo{version}{size}.pt"
        model_path = os.path.join(current_dir, model_name)
        return YOLO(model_path)
    elif version == "v7":
        current_dir = os.path.dirname(__file__)  # Gets the current directory of the .py file
        model_path = os.path.join(current_dir, "yolov7.pt")

        # Load from the cloned local yolov7 repo
        model = torch.hub.load('./yolov7', 'custom', model_path, source='local')
        model.eval()
        return model

def detect_objects(_model, image_bytes, max_size=640, version="v8"):
    image = Image.open(io.BytesIO(image_bytes))
    
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    
    if version in ["v8", "v9"]:
        results = _model(image)
        return results[0]
    elif version == "v7":
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = _model(image_np)
        return results

def load_image_from_url(url, max_size=640):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size))
            return image
        else:
            st.error("Failed to load image from URL.")
            return None
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

def is_valid_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def plot_yolov7_results(image, results):
    image_np = np.array(image)
    for *xyxy, conf, cls in results.xyxy[0]:
        label = f'{results.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(image_np, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(image_np, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

def main():
    st.title("YOLO Object Detection App")

    col1, col2 = st.columns(2)
    with col1:
        version = st.selectbox("Select YOLO version", ["v9", "v8", "v7"])
    with col2:
        if version == "v8":
            size = st.selectbox("Select model size", ["n", "s", "m", "l"])
        elif version == "v9":
            size = st.selectbox("Select model size", ["t", "s", "m", "c"])
        elif version == "v7":
            size = st.selectbox("Select model size", ["base"])

    option = st.radio("Choose an option to provide the image", ["Upload Image", "Image URL"])

    image = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    elif option == "Image URL":
        image_url = st.text_input("Enter image URL")
        if image_url and is_valid_image_file(image_url):
            image = load_image_from_url(image_url)
            if image:
                st.image(image, caption="Image from URL", use_column_width=True)
        elif image_url:
            st.warning("Please provide a valid image URL with jpg, jpeg, or png extension.")

    if image is not None and st.button("Detect Objects"):
        with st.spinner("Loading model... This may take a moment."):
            st.text(f"Using YOLO {version}-{size}")
            model = load_model(version, size)

        with st.spinner("Detecting objects..."):
            image_bytes = image_to_bytes(image)
            results = detect_objects(model, image_bytes, version=version)

        if version in ["v8", "v9"]:
            st.image(results.plot(), caption="Detection Result", use_column_width=True)
            detections = results.boxes.data.cpu().numpy()
            classes = results.names
        elif version == "v7":
            result_image = plot_yolov7_results(image, results)
            st.image(result_image, caption="Detection Result", use_column_width=True)
            detections = results.xyxy[0].cpu().numpy()
            classes = results.names

        st.subheader("Detected Objects:")
        if len(detections) > 0:
            for detection in detections:
                class_id = int(detection[5])
                class_name = classes[class_id]
                confidence = detection[4]
                st.write(f"{class_name}: {confidence:.2%}")
        else:
            st.info("No objects detected in the image.")

if __name__ == "__main__":
    main()