import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import pandas as pd

def load_model(version, size):
    model_name = f"yolo{version}{size}"
    return YOLO(model_name)

def detect_objects(model, image):
    results = model(image)
    return results[0]

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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            results = detect_objects(model, image)
            
            # Display the image with bounding boxes
            st.image(results.plot(), caption="Detection Result", use_column_width=True)
            
            # Extract detection results
            detections = results.boxes.data.cpu().numpy()
            classes = results.names
            
            # Create a list of detections with class names and confidences
            detection_list = []
            for detection in detections:
                class_id = int(detection[5])
                class_name = classes[class_id]
                confidence = detection[4]
                detection_list.append({
                    "Class": class_name,
                    "Confidence": f"{confidence:.2%}"
                })
            
            # Display the list of detections
            if detection_list:
                st.subheader("Detected Objects:")
                df = pd.DataFrame(detection_list)
                st.table(df)
            else:
                st.info("No objects detected in the image.")

if __name__ == "__main__":
    main()