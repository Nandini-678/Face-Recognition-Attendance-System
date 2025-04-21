import cv2
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from datetime import datetime, timedelta

# Load classifier
with open("model_svm.pkl", "rb") as f:
    model = pickle.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20).to(device)

# Attendance memory
last_logged = {}

cap = cv2.VideoCapture(0)
print("[INFO] Starting Face Recognition Attendance System...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)

    if face is not None:
        try:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()

            pred = model.predict(emb)[0]
            prob = model.predict_proba(emb).max()

            if prob > 0.5:
                now = datetime.now()
                name = pred
                message = ""
                last_time = last_logged.get(name)

                if not last_time or (now - last_time).total_seconds() >= 30:
                    # ✅ Mark attendance
                    with open("attendance_log.csv", "a") as f:
                        f.write(f"{name},{now}\n")
                    last_logged[name] = now
                    message = f"✅ Attendance marked for {name}"
                else:
                    # ❗ Already marked recently
                    remaining = 30 - int((now - last_time).total_seconds())
                    message = f"⚠️ {name}, try again in {remaining}s"

                # Display on screen
                cv2.putText(frame, message, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if "marked" in message else (0, 0, 255), 2)
                print(message)

        except Exception as e:
            print("[✗] Error:", str(e))
    else:
        print("[!] No face detected")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
