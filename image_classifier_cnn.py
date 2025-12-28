## Adaptive CNN for Image Classification ##

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

## USER INPUTS 

data_dir = input("Enter the dataset folder path (containing 'train' and 'test'): ")

image_size = int(input("Enter image size (e.g., 64 or 128): "))
if image_size < 32:
    print("Image size too small. Setting default image size = 64")
    image_size = 64

batch_size = int(input("Enter batch size (e.g., 32): "))
num_epochs = int(input("Enter number of epochs: "))
learning_rate = 0.001  # default

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## CHECK FOLDERS

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Train or Test folder not found inside the dataset directory!")

## DATA TRANSFORMS & LOADERS

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Detect classes automatically
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Detected classes: {class_names}")

## DEFINING CNN 

class AdaptiveCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdaptiveCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

## INITIALIZE MODEL,LOSS,OPTIMIZER

model = AdaptiveCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## TRAINING LOOPS

for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    ## Validation

    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_acc = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

## SAVING  MODEL

torch.save(model.state_dict(), "adaptive_cnn_updated.pth")
print("Model saved as adaptive_cnn_updated.pth")

## PREDICTING SINGLE IMAGE

image_path = input("Enter path of an image to predict (or leave blank to skip): ")
if image_path:
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted_class = torch.max(output, 1)

    print(f"Predicted class: {class_names[predicted_class.item()]}")
