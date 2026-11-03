
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("YOLO Object Detection")

model = YOLO("best.pt")

option = st.selectbox("Choose an option", ["Upload Image", "Open Camera"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(frame)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, channels="BGR")

elif option == "Open Camera":
    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = model(img)
            annotated_frame = results[0].plot()

            return annotated_frame

    webrtc_streamer(
        key="yolo-camera",
        video_transformer_factory=YOLOTransformer
    )
