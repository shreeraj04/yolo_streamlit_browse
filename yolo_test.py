import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import requests
import os

def load_model(version, size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if version in ["v8", "v9"]:
        model_name = f"yolo{version}{size}.pt"
        model_path = os.path.join(current_dir, model_name)
        return YOLO(model_path)
    
# @st.cache_data
def detect_objects(_model, image_bytes, max_size=640):
    # Convert bytes back to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize image if it's too large
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    
    results = _model(image)
    return results[0]

# @st.cache_data
def load_image_from_url(url, max_size=640):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            # Resize image if it's too large
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

    # Two options: Upload image or provide URL
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
        # Load model based on selected version and size
        with st.spinner("Loading model... This may take a moment."):
            st.text(f"Using YOLO {version}-{size}")
            model = load_model(version, size)

        with st.spinner("Detecting objects..."):
            # Convert image to bytes for caching
            image_bytes = image_to_bytes(image)
            results = detect_objects(model, image_bytes)

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