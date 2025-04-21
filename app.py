import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from datetime import datetime
from pathlib import Path
import os
import json

# Streamlit config
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📸 Face Recognition Attendance System")

# Initialize session state
for key in ["recognizing", "status_msg", "detected_name", "detected_image", "disabled_faces", "last_logged", "student_details"]:
    if key not in st.session_state:
        if key == "recognizing":
            st.session_state[key] = False
        elif key in ["status_msg", "detected_name", "detected_image"]:
            st.session_state[key] = None
        elif key == "disabled_faces":
            st.session_state[key] = set()
        else:
            st.session_state[key] = {}

# Load models
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_models()

# Load dataset embeddings and metadata
@st.cache_resource
def load_embeddings():
    dataset_dir = Path("dataset")
    embeddings, ids, student_details = [], [], {}

    for folder in dataset_dir.iterdir():
        if folder.is_dir():
            image_path = folder / "0.jpg"
            meta_path = folder / "meta.json"
            if image_path.exists() and meta_path.exists():
                img = Image.open(image_path)
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    emb = resnet(face_tensor.unsqueeze(0))
                    embeddings.append(emb.detach())
                    ids.append(folder.name)
                    with open(meta_path, "r") as f:
                        student_details[folder.name] = json.load(f)
    return embeddings, ids, student_details

stored_embeddings, stored_ids, student_details = load_embeddings()

# UI buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("✅ Start Attendance"):
        st.session_state.recognizing = True
        st.session_state.status_msg = ""
        st.session_state.detected_name = ""
        st.session_state.detected_image = None
        st.session_state.student_details = {}

with col2:
    if st.button("⛔ Stop Attendance"):
        st.session_state.recognizing = False
        st.session_state.status_msg = "Stopped attendance detection."

# Layout columns
cam_col, info_col = st.columns([2, 1])
with cam_col:
    frame_placeholder = st.empty()
with info_col:
    info_box = st.container()

# Camera logic
if st.session_state.recognizing:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with info_box:
        st.warning("📷 Starting webcam...")

    while st.session_state.recognizing:
        ret, frame = cap.read()
        if not ret:
            st.session_state.status_msg = "❌ Failed to capture frame."
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB")

        face = mtcnn(Image.fromarray(rgb))
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach()
            dists = [torch.norm(emb - db_emb).item() for db_emb in stored_embeddings]
            min_idx = int(np.argmin(dists))
            pred_id = stored_ids[min_idx]
            prob = 1 - dists[min_idx]

            now = datetime.now()
            if prob > 0.5:
                last_time = st.session_state.last_logged.get(pred_id)

                # Already marked logic
                if last_time and (now - last_time).total_seconds() < 30:
                    st.session_state.status_msg = f"⚠️ Your attendance is already marked"
                else:
                    # Mark attendance
                    with open("attendance_log.csv", "a") as f:
                        f.write(f"{pred_id},{now}\n")

                    st.session_state.last_logged[pred_id] = now
                    details = student_details.get(pred_id, {})
                    st.session_state.detected_name = details.get("name", pred_id)
                    st.session_state.student_details = {
                        "ID": pred_id,
                        "Department": details.get("department", "Unknown")
                    }
                    st.session_state.status_msg = f"✅ Attendance marked for {st.session_state.detected_name}"

                    dataset_img_path = Path("dataset") / pred_id / "0.jpg"
                    if dataset_img_path.exists():
                        st.session_state.detected_image = Image.open(dataset_img_path)

            else:
                st.session_state.status_msg = "❌ You are not registered"
                st.session_state.detected_name = None
                st.session_state.detected_image = None
                st.session_state.student_details = {}

            # Stop recognition for now
            st.session_state.recognizing = False
            cap.release()
            cv2.destroyAllWindows()
            break

# Display detected student info
with info_box:
    st.subheader("👤 Detected Student")
    if st.session_state.detected_image:
        st.image(st.session_state.detected_image, width=200)
    if st.session_state.detected_name:
        st.markdown(f"**Name:** `{st.session_state.detected_name}`")
    if st.session_state.student_details:
        st.markdown(f"**ID Number:** `{st.session_state.student_details['ID']}`")
        st.markdown(f"**Department:** `{st.session_state.student_details['Department']}`")
    
    # Safe status message handling
    if st.session_state.status_msg:
        if "✅" in st.session_state.status_msg:
            st.success(st.session_state.status_msg)
        elif "⚠️" in st.session_state.status_msg:
            st.warning(st.session_state.status_msg)
        elif "❌" in st.session_state.status_msg:
            st.error(st.session_state.status_msg)
        else:
            st.info(st.session_state.status_msg)
