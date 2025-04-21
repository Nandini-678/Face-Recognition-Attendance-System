import cv2
import os
import mediapipe as mp
import numpy as np
import random

INPUT_DIR = "dataset"
OUTPUT_DIR = "augmented_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def augment(img):
    aug_images = []
    for _ in range(50):
        temp = img.copy()

        # Flip
        if random.random() > 0.5:
            temp = cv2.flip(temp, 1)

        # Rotate
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((temp.shape[1] // 2, temp.shape[0] // 2), angle, 1)
        temp = cv2.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))

        # Brightness/Contrast
        alpha = random.uniform(1.0, 1.2)
        beta = random.randint(5, 20)
        temp = cv2.convertScaleAbs(temp, alpha=alpha, beta=beta)

        # Mild Gaussian blur (optional)
        if random.random() > 0.8:
            temp = cv2.GaussianBlur(temp, (3, 3), 0)

        aug_images.append(temp)
    return aug_images

# Walk through student folders
for student_id in os.listdir(INPUT_DIR):
    student_folder = os.path.join(INPUT_DIR, student_id)
    if not os.path.isdir(student_folder):
        continue

    for filename in os.listdir(student_folder):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(student_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                h, w, _ = image.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                face = image[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (160, 160))

                augmented = augment(face_resized)

                for i, img in enumerate(augmented):
                    out_path = os.path.join(OUTPUT_DIR, f"{student_id}_{i}.jpg")
                    cv2.imwrite(out_path, img)

                print(f"[✓] Augmented 50 images for {student_id}")
            else:
                print(f"[✗] No face found in {image_path}")
