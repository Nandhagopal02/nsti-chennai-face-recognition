import streamlit as st
import cv2
import face_recognition
from face_recognition_module import load_known_faces, encode_faces, recognize_faces

# Page setup
st.set_page_config(page_title="NSTI Chennai Face Recognition", layout="centered")
st.title("📸 NSTI Chennai - Face Recognition Attendance System")

st.markdown("""
This application is specifically trained with images of **NSTI Chennai staff and students**. 
Only registered faces from the institute will be recognized.
""")

# Load known faces
images, class_names = load_known_faces()
encoded_known_faces = encode_faces(images)

# Webcam state
if 'cam_active' not in st.session_state:
    st.session_state.cam_active = False

# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Webcam"):
        st.session_state.cam_active = True
with col2:
    if st.button("⏹ Stop Webcam"):
        st.session_state.cam_active = False

FRAME_WINDOW = st.image([])

# OpenCV webcam
if st.session_state.cam_active:
    camera = cv2.VideoCapture(0)
    st.success("✅ Webcam is active. Recognizing NSTI faces...")

    while st.session_state.cam_active:
        ret, frame = camera.read()
        if not ret:
            st.error("❌ Could not read from camera.")
            break

        recognized_faces = recognize_faces(frame, encoded_known_faces, class_names)

        for name, (top, right, bottom, left) in recognized_faces:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    camera.release()
    st.info("ℹ️ Webcam stopped.")

else:
    FRAME_WINDOW.empty()
