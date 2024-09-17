import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.set_page_config(
    page_title="Object Detection App",  # Title of the web app
    page_icon="🚀",  # Optional: Favicon for the browser tab
    layout="centered"  # You can use "wide" for a wider layout
)

# Load the YOLO model
model = YOLO("best.pt")

# Streamlit app
st.title("YOLOv8 Object Detection")
st.write("Upload an image to detect objects with YOLOv8.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Run YOLOv8 prediction
    results = model.predict(source=img_np, show=False, conf=0.5)
    
    # Draw bounding boxes and labels on the image
    result_image = results[0].plot()
    
    # Convert back to PIL image for displaying in Streamlit
    result_image = Image.fromarray(result_image)
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.image(result_image, caption='Predicted Image', use_column_width=True)
