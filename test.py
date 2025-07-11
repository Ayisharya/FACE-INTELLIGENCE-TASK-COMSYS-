import argparse
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------- ARGUMENT PARSER -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--task_a_path', type=str, required=True)
parser.add_argument('--task_b_path', type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- TASK A -----------------
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Sequential(nn.Linear(base.fc.in_features, 1), nn.Sigmoid())
        self.model = base

    def forward(self, x):
        return self.model(x)

transform_task_a = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

task_a_dataset = ImageFolder(args.task_a_path, transform=transform_task_a)
task_a_loader = DataLoader(task_a_dataset, batch_size=32, shuffle=True)

model_a = GenderClassifier().to(device)
criterion_a = nn.BCELoss()
optimizer_a = optim.Adam(model_a.model.fc.parameters(), lr=1e-4)

# Train Task A
for epoch in range(15):
    model_a.train()
    total_loss = 0
    for images, labels in task_a_loader:
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        optimizer_a.zero_grad()
        outputs = model_a(images)
        loss = criterion_a(outputs, labels)
        loss.backward()
        optimizer_a.step()
        total_loss += loss.item()
    print(f"[Task A] Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Eval Task A
model_a.eval()
y_true_a, y_pred_a = [], []
with torch.no_grad():
    for images, labels in task_a_loader:
        images = images.to(device)
        outputs = model_a(images)
        preds = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        y_true_a.extend(labels.numpy())
        y_pred_a.extend(preds)

print("\n🔍 Evaluating Task A (Gender Classification)...")
print("Accuracy: ", accuracy_score(y_true_a, y_pred_a))
print("Precision:", precision_score(y_true_a, y_pred_a))
print("Recall:   ", recall_score(y_true_a, y_pred_a))
print("F1-score: ", f1_score(y_true_a, y_pred_a))


# ----------------- TASK B -----------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        dist = torch.abs(f1 - f2)
        return self.classifier(dist)

class FaceConSiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.person_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        for person_dir in self.person_folders:
            frontal_img = None
            for file in os.listdir(person_dir):
                if file.lower().endswith(('.jpg', '.png')) and "frontal" in file.lower():
                    frontal_img = os.path.join(person_dir, file)
                    break
                elif file.lower().endswith(('.jpg', '.png')):
                    frontal_img = os.path.join(person_dir, file)

            distortion_dir = os.path.join(person_dir, "distortion")
            if not frontal_img or not os.path.exists(distortion_dir):
                continue

            distorted_images = [
                os.path.join(distortion_dir, f)
                for f in os.listdir(distortion_dir)
                if f.lower().endswith(('.jpg', '.png'))
            ]

            for dimg in distorted_images:
                pairs.append((frontal_img, dimg, 1))

            neg_person = random.choice([p for p in self.person_folders if p != person_dir])
            neg_dist_dir = os.path.join(neg_person, "distortion")
            if os.path.exists(neg_dist_dir):
                neg_imgs = [os.path.join(neg_dist_dir, f) for f in os.listdir(neg_dist_dir) if f.lower().endswith(('.jpg', '.png'))]
                if neg_imgs:
                    neg_img = random.choice(neg_imgs)
                    pairs.append((frontal_img, neg_img, 0))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

transform_task_b = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

task_b_dataset = FaceConSiameseDataset(args.task_b_path, transform=transform_task_b)
task_b_loader = DataLoader(task_b_dataset, batch_size=16, shuffle=True)

model_b = SiameseNetwork().to(device)
criterion_b = nn.BCEWithLogitsLoss()
optimizer_b = optim.Adam(model_b.parameters(), lr=1e-4)

# Train Task B
for epoch in range(15):
    model_b.train()
    total_loss = 0
    for x1, x2, labels in task_b_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
        optimizer_b.zero_grad()
        outputs = model_b(x1, x2).squeeze()
        loss = criterion_b(outputs, labels)
        loss.backward()
        optimizer_b.step()
        total_loss += loss.item()
    print(f"[Task B] Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Eval Task B
model_b.eval()
y_true_b, y_pred_b = [], []
with torch.no_grad():
    for x1, x2, labels in task_b_loader:
        x1, x2 = x1.to(device), x2.to(device)
        outputs = model_b(x1, x2).squeeze()
        preds = torch.sigmoid(outputs) > 0.5
        y_true_b.extend(labels.cpu().numpy())
        y_pred_b.extend(preds.cpu().numpy())

print("\n🔍 Evaluating Task B (Face Verification)...")
print("Accuracy: ", accuracy_score(y_true_b, y_pred_b))
print("Precision:", precision_score(y_true_b, y_pred_b))
print("Recall:   ", recall_score(y_true_b, y_pred_b))
print("F1-score: ", f1_score(y_true_b, y_pred_b))
