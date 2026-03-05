!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="ComS0M3qw6ReVt6LDr2f")
project = rf.workspace("first-orenm").project("cervical-cancer-cell-classificat-i93u6")
version = project.version(3)
dataset = version.download("folder")

import os, warnings, logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# =============================
# SETTINGS
# =============================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
logger.info(f"⚡ Using {DEVICE}")

BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 224   # DenseNet expects 224x224
PATIENCE = 7

DATA_DIR = "/content/Cervical-Cancer-Cell-Classificat-3"
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")
# =====================
# TRANSFORMS
# =====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# =====================
# DATASETS
# =====================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VALID_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

num_classes = len(train_dataset.classes)
print(f"📂 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"📌 Classes: {train_dataset.classes}")


# =============================
# CLASS BALANCING
# =============================
train_labels   = train_dataset.targets
class_counts   = np.bincount(train_labels)
class_weights  = 1. / class_counts
samples_weight = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
logger.info(f"⚖️ Class Weights: {class_weights_t.cpu().numpy()}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =============================
# MODEL (DenseNet121)
# =============================
logger.info("🚀 Using DenseNet121 Pretrained on ImageNet")
model = models.densenet121(weights="IMAGENET1K_V1")

in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# =============================
# LOSS & OPTIMIZER
# =============================
criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

# =====================
# TRAIN LOOP
# =====================
best_f1 = 0
patience_counter = 0
train_acc_list, val_acc_list, test_acc_list = [], [], []
train_loss_list, val_loss_list, test_loss_list = [], [], []
test_f1_list = []

print("\n📢 Starting Training...")
for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    total_loss, correct, total = 0,0,0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (preds==labels).sum().item()
        total += labels.size(0)
    train_acc = correct/total
    train_loss = total_loss/len(train_loader)

    # ---- Validation ----
    model.eval(); val_loss, correct,total = 0,0,0
    y_true,y_pred = [],[]
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            val_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    val_acc = correct/total
    val_loss /= len(val_loader)
    val_f1 = f1_score(y_true,y_pred,average="macro")

    # ---- Test ----
    test_loss, correct,total = 0,0,0
    y_true,y_pred = [],[]
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            test_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    test_acc = correct/total
    test_loss /= len(test_loader)
    test_f1 = f1_score(y_true,y_pred,average="macro")

      # Logging
    train_acc_list.append(train_acc); val_acc_list.append(val_acc); test_acc_list.append(test_acc)
    train_loss_list.append(train_loss); val_loss_list.append(val_loss); test_loss_list.append(test_loss)
    test_f1_list.append(test_f1)

    scheduler.step(val_f1)
    print(f"[Epoch {epoch+1}] "
          f"Train: Acc={train_acc:.4f} Loss={train_loss:.4f} | "
          f"Val: Acc={val_acc:.4f} Loss={val_loss:.4f} F1={val_f1:.4f} | "
          f"Test: Acc={test_acc:.4f} Loss={test_loss:.4f} F1={test_f1:.4f}")


# =============================
# FINAL LOGGING
# =============================
logger.info(f"\n📊 FINAL TEST RESULTS\n=============================")
logger.info(f"✅ Test Accuracy : {test_acc*100:.2f}%")
logger.info(f"📌 Test Loss    : {test_loss:.4f}")
logger.info(f"📈 Test F1 Score: {test_f1:.4f}")



# =====================
# FINAL TEST EVALUATION
# =====================
print("\n📊 Evaluating Final Model on Test Set...")

# Load the VGG16 model and the saved state dictionary
model = models.vgg16(weights=None) # Load a VGG16 model with no pretrained weights
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, num_classes) # Adjust the final layer
model = model.to(DEVICE)

model.load_state_dict(torch.load("/content/drive/MyDrive/best_model.pth"))

test_acc, test_loss, test_f1, test_prec, test_rec, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion)

print("\n📊 FINAL TEST RESULTS")
print("=============================")
print(f"✅ Accuracy : {test_acc*100:.2f}%")
print(f"📉 Loss     : {test_loss:.4f}")
print(f"📈 F1-Score : {test_f1:.4f}")
print(f"🎯 Precision: {test_prec:.4f}")
print(f"❤️ Recall   : {test_rec:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes,
            cmap="Blues")
plt.title("Confusion Matrix — DenseNet121")
plt.show()

# =====================
# ROC CURVES
# =====================
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
colors = cycle(["red","green","blue","orange","purple","brown","cyan"])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{test_dataset.classes[i]} (AUC={roc_auc[i]:.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — DenseNet121")
plt.legend()
plt.show()

gc.collect()
torch.cuda.empty_cache()

import matplotlib.pyplot as plt

# Accuracy Curve
plt.figure(figsize=(7,5))
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(val_acc_list, label="Validation Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(); plt.grid(True)
plt.show()

# Loss Curve
plt.figure(figsize=(7,5))
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Validation Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(); plt.grid(True)
plt.show()

# f1 curve
plt.plot(test_f1_list,label="Test F1")
plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.legend(); plt.title("F1 Curves")
plt.show()
