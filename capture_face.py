import os
import json
import cv2

student_id = input("Enter Student ID: ")
name = input("Enter Full Name: ")
department = input("Enter Department: ")

folder = os.path.join("dataset", student_id)
os.makedirs(folder, exist_ok=True)

# Save metadata
with open(os.path.join(folder, "meta.json"), "w") as f:
    json.dump({"id": student_id, "name": name, "department": department}, f)

# Capture face
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture Face - Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("s"):
        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), frame)
        count += 1
        print(f"Captured {count} images")
    elif key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
