import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

MODEL_CACHE_DIR = r"..\model"   
DATA_DIR        = r"..\data"  

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

from transformers import ResNetForImageClassification

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_name = "microsoft/resnet-18"
model = ResNetForImageClassification.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_DIR
)

num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features, 10)
model.config.num_labels = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on MNIST test set: {accuracy:.2f}%")