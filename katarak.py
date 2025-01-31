import os
import cv2
import glob
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_fscore_support
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Blur, Resize
from albumentations.pytorch import ToTensorV2
from xml.etree import ElementTree as ET

# Configuration
BATCH_SIZE = 8
RESIZE_TO = 640
NUM_EPOCHS = 200
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_IMG = 'input/Katarak2/train/Image'
TRAIN_ANNOT = 'input/Katarak2/train/Annotation'
VALID_IMG = 'input/Katarak2/test/Image'
VALID_ANNOT = 'input/Katarak2/test/Annotation'
CLASSES = ['__background__', 'Immature', 'Mature', 'Normal']
NUM_CLASSES = len(CLASSES)
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, img_path, annot_path, width, height, classes, transforms=None):
        self.img_path = img_path
        self.annot_path = annot_path
        self.width = width
        self.height = height
        self.classes = classes
        self.transforms = transforms
        self.all_images = sorted(glob.glob(os.path.join(img_path, "*.jpg")))

    def __getitem__(self, idx):
        image_name = os.path.basename(self.all_images[idx])
        image_path = os.path.join(self.img_path, image_name)
        annot_path = os.path.join(self.annot_path, image_name.replace('.jpg', '.xml'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            label = self.classes.index(obj.find("name").text)
            labels.append(label)
            bbox = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(int, [bbox.find(tag).text for tag in ["xmin", "ymin", "xmax", "ymax"]])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transforms:
            sample = self.transforms(image=image, bboxes=boxes.numpy(), labels=labels.numpy())
            image = sample["image"]
            boxes = torch.tensor(sample["bboxes"], dtype=torch.float32)

        return image, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return Compose([
        Resize(RESIZE_TO, RESIZE_TO),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Model
def create_model(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_features, num_anchors, num_classes
    )
    return model

# Utilities
class Averager:
    def __init__(self):
        self.reset()

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        return self.current_total / self.iterations if self.iterations > 0 else 0

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def save_loss_plot(out_dir, train_loss_list):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/train_loss.png")
    plt.close()

# IoU Calculation
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def compute_mean_iou(targets, predictions):
    ious = []
    for t_box, p_box in zip(targets, predictions):
        ious.append(compute_iou(t_box, p_box))
    return np.mean(ious) if ious else 0.0

# Training Script
train_dataset = CustomDataset(
    img_path=TRAIN_IMG, annot_path=TRAIN_ANNOT, width=RESIZE_TO, height=RESIZE_TO,
    classes=CLASSES, transforms=get_train_transform()
)

valid_dataset = CustomDataset(
    img_path=VALID_IMG, annot_path=VALID_ANNOT, width=RESIZE_TO, height=RESIZE_TO,
    classes=CLASSES, transforms=get_valid_transform()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = create_model(NUM_CLASSES).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
scheduler = MultiStepLR(optimizer=optimizer, milestones=[20], gamma=0.1)

train_loss_hist = Averager()
train_losses = []
metrics_list = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    model.train()
    train_loss_hist.reset()
    
    for images, targets in tqdm(train_loader, desc="Training"):
        images = [img.to(DEVICE).float() for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        train_loss_hist.send(losses.item())

    train_losses.append(train_loss_hist.value)
    save_loss_plot(OUT_DIR, train_losses)

    # Validation
    model.eval()
    all_targets, all_predictions = [], []
    mean_ious = []

    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc="Validating"):
            images = [img.to(DEVICE).float() for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Abaikan jika tidak ada prediksi
                if len(output["boxes"]) == 0:
                    continue
                
                # Ambil label target dan prediksi
                true_labels = targets[i]["labels"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                # Tambahkan ke daftar semua target dan prediksi
                all_targets.extend(true_labels)
                all_predictions.extend(pred_labels)

                # Hitung Mean IoU untuk gambar ini
                mean_iou = compute_mean_iou(
                    targets[i]["boxes"].cpu().numpy(), output["boxes"].cpu().numpy()
                )
                mean_ious.append(mean_iou)

    # Handle kasus ketika tidak ada prediksi atau target
    if len(all_predictions) == 0 or len(all_targets) == 0:
        precision, recall, f1 = 0.0, 0.0, 0.0
        print("No predictions made or no targets found. Metrics set to 0.")
    else:
        # Hitung metrik menggunakan sklearn
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro'
        )

    # Hitung Mean IoU rata-rata untuk semua gambar
    mean_iou_epoch = np.mean(mean_ious) if mean_ious else 0.0

    # Print metrik
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Mean IoU: {mean_iou_epoch:.4f}")

    # Save metrics to list
    metrics_list.append({
        "Epoch": epoch + 1,
        "Train Loss": train_loss_hist.value,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Mean IoU": mean_iou_epoch,
    })

# Save metrics to Excel
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_excel(f"{OUT_DIR}/metrics.xlsx", index=False)
