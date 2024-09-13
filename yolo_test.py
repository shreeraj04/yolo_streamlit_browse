import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import requests

def load_model(version, size):
    model_name = f"yolo{version}{size}"
    return YOLO(model_name)

def detect_objects(model, image):
    results = model(image)
    return results[0]

def load_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            st.error("Failed to load image from URL.")
            return None
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

def is_valid_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png"]
    return any(filename.endswith(ext) for ext in valid_extensions)

def main():
    st.title("YOLO Object Detection App")

    col1, col2 = st.columns(2)
    with col1:
        version = st.selectbox("Select YOLO version", ["v8", "v9"])
    with col2:
        if version == "v8":
            size = st.selectbox("Select model size", ["n", "s", "m", "l"])
        elif version == "v9":
            size = st.selectbox("Select model size", ["t", "s", "m", "c"])

    model = load_model(version, size)

    # Two options: Upload image or provide URL
    option = st.radio("Choose an option to provide the image", ["Upload Image", "Image URL"])

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
        else:
            st.warning("Please provide a valid image URL with jpg, jpeg, or png extension.")

    if 'image' in locals() and st.button("Detect Objects"):
        results = detect_objects(model, image)

        st.image(results.plot(), caption="Detection Result", use_column_width=True)

        detections = results.boxes.data.cpu().numpy()
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
