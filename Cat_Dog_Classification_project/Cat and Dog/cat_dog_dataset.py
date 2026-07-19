import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def build_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model

def predict_image(img_path, model_path, device, val_transform):
    infer_model = build_model().to(device)
    infer_model.load_state_dict(torch.load(model_path, map_location=device))
    infer_model.eval()
    img = Image.open(img_path).convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = infer_model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
    label_map = {0: "Cat", 1: "Dog"}
    return label_map[pred_idx]

if __name__ == "__main__":
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    IMG_SIZE = (224, 224)
    DATA_ROOT = "./cat_dog_dataset"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    SAVE_MODEL_PATH = "./cat_dog_resnet18.pth"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset Class
    class CatDogDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.img_paths = []
            self.labels = []
            class_map = {"cat": 0, "dog": 1}
            for cls_name, label in class_map.items():
                cls_folder = os.path.join(data_dir, cls_name)
                if not os.path.exists(cls_folder):
                    raise FileNotFoundError(f"Missing folder: {cls_folder}")
                for img_name in os.listdir(cls_folder):
                    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.img_paths.append(os.path.join(cls_folder, img_name))
                        self.labels.append(label)

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            label = self.labels[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

    # Load datasets
    train_dataset = CatDogDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = CatDogDataset(VAL_DIR, transform=val_transform)

    # num_workers=0 fix Windows error
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Train one epoch
    def train_one_epoch():
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / total_samples
        acc = correct / total_samples
        return avg_loss, acc

    # Validate
    def validate():
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += torch.sum(preds == labels).item()
                total_samples += images.size(0)
        avg_loss = total_loss / total_samples
        acc = correct / total_samples
        return avg_loss, acc

    # Training loop
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"Saved best model, Best Val Acc: {best_val_acc:.4f}")

    # Plot curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Test prediction
    # result = predict_image("test.jpg", SAVE_MODEL_PATH, device, val_transform)
    # print("Prediction:", result)