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

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📸 Face Recognition Attendance System")

# Session state initialization
for key in ["recognizing", "status_msg", "detected_name", "detected_image", "disabled_faces", "last_logged", "student_details"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "recognizing" else None if key in ["status_msg", "detected_name", "detected_image"] else set() if key == "disabled_faces" else {} if key == "last_logged" else {}

# Load model
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_models()

# Load dataset from structured folders
@st.cache_resource
def load_embeddings():
    dataset_dir = Path("dataset")
    embeddings = []
    ids = []
    student_details = {}

    for student_folder in dataset_dir.iterdir():
        if student_folder.is_dir():
            image_path = student_folder / "0.jpg"
            meta_path = student_folder / "meta.json"

            if image_path.exists() and meta_path.exists():
                try:
                    img = Image.open(image_path)
                    face_tensor = mtcnn(img)
                    if face_tensor is not None:
                        emb = resnet(face_tensor.unsqueeze(0))
                        embeddings.append(emb.detach())
                        ids.append(student_folder.name)

                        with open(meta_path, "r") as f:
                            student_details[student_folder.name] = json.load(f)
                except Exception as e:
                    print(f"[!] Error processing {student_folder.name}: {e}")

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
    if st.button("⛔️ Stop Attendance"):
        st.session_state.recognizing = False
        st.session_state.status_msg = "Stopped attendance detection."

# Two-column layout for camera feed and detected student
cam_col, info_col = st.columns([2, 1])

with cam_col:
    frame_placeholder = st.empty()

with info_col:
    info_box = st.container()

if st.session_state.recognizing:
    cap = cv2.VideoCapture(0)
    with info_box:
        st.warning("📷 Starting webcam...")

    while st.session_state.recognizing:
        ret, frame = cap.read()
        if not ret:
            st.session_state.status_msg = "Failed to capture frame."
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB")

        face = mtcnn(Image.fromarray(rgb))
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach()
            dists = [torch.norm(emb - db_emb).item() for db_emb in stored_embeddings]
            min_idx = int(np.argmin(dists))
            pred = stored_ids[min_idx]
            prob = 1 - dists[min_idx]

            if prob > 0.5 and pred not in st.session_state.disabled_faces:
                now = datetime.now()
                with open("attendance_log.csv", "a") as f:
                    f.write(f"{pred},{now}\n")
                st.session_state.last_logged[pred] = now

                details = student_details.get(pred, {})
                st.session_state.detected_name = details.get("name", pred)
                st.session_state.student_details = {
                    "ID": pred,
                    "Department": details.get("department", "Unknown")
                }
                st.session_state.status_msg = f"✅ Attendance marked for {st.session_state.detected_name}"

                # Load image
                face_img = face.permute(1, 2, 0).cpu().numpy()
                face_img = np.clip(face_img * 255, 0, 255).astype(np.uint8)
                st.session_state.detected_image = Image.fromarray(face_img)

                st.session_state.disabled_faces.add(pred)
                st.session_state.recognizing = False
                cap.release()
                cv2.destroyAllWindows()
                break

# Detected Student Info Display
with info_box:
    st.subheader("👤 Detected Student")
    if st.session_state.detected_image:
        st.image(st.session_state.detected_image, width=200)
    if st.session_state.detected_name:
        st.markdown(f"**Name:** `{st.session_state.detected_name}`")
    if st.session_state.student_details:
        st.markdown(f"**ID Number:** `{st.session_state.student_details['ID']}`")
        st.markdown(f"**Department:** `{st.session_state.student_details['Department']}`")
    if st.session_state.status_msg:
        st.success(st.session_state.status_msg)