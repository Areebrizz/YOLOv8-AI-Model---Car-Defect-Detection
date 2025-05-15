import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Car Defect Detection",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model (ensure best.pt is in repo or add public path)
model = YOLO("best.pt")

# Sidebar
with st.sidebar:
    st.title("🔧 App Info")
    st.markdown("""
        **Model**: YOLOv8 custom  
        **Trained on**: Car defect images  
        **Defects Detected**:
        - Dent  
        - Scratch  
        - Crack  
        - Paint Damage  
        - Glass Break  
        
        💡 Upload a car image to get started.
    """)
    st.markdown("---")
    st.write("Made by [Muhammad Areeb Rizwan](https://www.linkedin.com/in/areebrizwan)")

# Header
st.markdown(
    "<h1 style='text-align: center; color: #0C2340;'>🚗 Car Defect Detection</h1>", 
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>AI-powered visual inspection using YOLOv8</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("📤 Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run model
    with st.spinner("🔍 Analyzing image..."):
        results = model.predict(source=temp_path, imgsz=640, conf=0.5)
        result = results[0]
        output_img = result.plot()
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # Tabs for Results
    tab1, tab2 = st.tabs(["📸 Prediction", "📊 Defect Report"])

    with tab1:
        st.image(output_img, caption="AI Prediction", use_column_width=True)

    with tab2:
        defect_counts = {}
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1

        if defect_counts:
            st.success("Defects detected:")
            for defect, count in defect_counts.items():
                st.markdown(f"- **{defect.capitalize()}**: {count}")
            st.markdown(f"### ✅ Total: `{sum(defect_counts.values())}` defects found")
        else:
            st.info("No visible defects detected.")

    # Clean temp
    os.remove(temp_path)

else:
    st.warning("Please upload an image to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>© 2025 Muhammad Areeb Rizwan | Powered by Streamlit & YOLOv8</p>",
    unsafe_allow_html=True
)
