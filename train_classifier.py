import os
import pickle
from sklearn.svm import SVC
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

X, y = [], []
for file in os.listdir("augmented_dataset"):
    path = os.path.join("augmented_dataset", file)
    img = Image.open(path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor).cpu().numpy()[0]
    X.append(embedding)
    y.append(file.split("_")[0])

clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

with open("model_svm.pkl", "wb") as f:
    pickle.dump(clf, f)

print("[✓] Model trained and saved.")
