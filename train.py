import os
import torch
from tqdm import tqdm
from config import (
    DEVICE, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, OUT_DIR, RESIZE_TO,
    TRAIN_IMG, TRAIN_ANNOT, VALID_IMG, VALID_ANNOT, CLASSES
)
from custom_utils import Averager, save_model, get_train_transform, get_valid_transform, save_loss_plot, save_mAP, SaveBestModel
from datasets import CustomDataset, collate_fn
from model import create_model
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import RMSprop

# Ensure outputs directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize datasets and loaders
train_dataset = CustomDataset(
    img_path=TRAIN_IMG,
    annot_path=TRAIN_ANNOT,
    width=RESIZE_TO,
    height=RESIZE_TO,
    classes=CLASSES,
    transforms=get_train_transform(),
)

valid_dataset = CustomDataset(
    img_path=VALID_IMG,
    annot_path=VALID_ANNOT,
    width=RESIZE_TO,
    height=RESIZE_TO,
    classes=CLASSES,
    transforms=get_valid_transform(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")

# Initialize model
model = create_model(NUM_CLASSES, min_size=RESIZE_TO, max_size=RESIZE_TO).to(DEVICE)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
#print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
p.numel() for p in model.parameters() if p.requires_grad)
#print(f"{total_trainable_params:,} training parameters.")
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.RMSprop(
    model.parameters(), 
    lr=0.01, 
    alpha=0.99, 
    weight_decay=0.0005
)


# Initialize loss tracker and metrics
train_loss_hist = Averager()
map_metric = MeanAveragePrecision()
save_best_model = SaveBestModel()

# DataFrame to store metrics
metrics_data = []

# Lists to store mAP metrics
map_50_list = []
map_95_list = []
precision_list = []
recall_list = []
f1_list = []

# Training and validation loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # Training phase
    train_loss_hist.reset()
    model.train()
    for images, targets in tqdm(train_loader, desc="Training"):
        images = [img.to(DEVICE).float() for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        train_loss_hist.send(losses.item())

    print(f"Epoch {epoch + 1} Training Loss: {train_loss_hist.value:.4f}")

    # Validation phase
    model.eval()
    map_metric.reset()

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for images, targets_batch in tqdm(valid_loader, desc="Validating"):
            images = [img.to(DEVICE).float() for img in images]
            outputs = model(images)

            for i in range(len(images)):
                # Ground truth
                true_labels = targets_batch[i]["labels"].detach().cpu()
                true_boxes = targets_batch[i]["boxes"].detach().cpu()
                all_true_labels.extend(true_labels.numpy())

                # Predictions
                pred_labels = outputs[i]["labels"].detach().cpu()
                pred_scores = outputs[i]["scores"].detach().cpu()
                pred_boxes = outputs[i]["boxes"].detach().cpu()

                # Apply confidence threshold
                high_conf_mask = pred_scores > 0.5
                filtered_boxes = pred_boxes[high_conf_mask]
                filtered_scores = pred_scores[high_conf_mask]
                filtered_labels = pred_labels[high_conf_mask]
                all_pred_labels.extend(filtered_labels.numpy())

                # Update metrics
                map_metric.update(
                    preds=[{"boxes": filtered_boxes, "scores": filtered_scores, "labels": filtered_labels}],
                    target=[{"boxes": true_boxes, "labels": true_labels}],
                )

    # Ensure consistent lengths
    min_length = min(len(all_true_labels), len(all_pred_labels))
    all_true_labels = all_true_labels[:min_length]
    all_pred_labels = all_pred_labels[:min_length]

    # Compute metrics
    map_result = map_metric.compute()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average="weighted", zero_division=0
    )

    print(f"Epoch {epoch + 1} mAP@0.5: {map_result['map_50']:.4f}")
    print(f"Epoch {epoch + 1} mAP@0.5:0.95: {map_result['map']:.4f}")
    print(f"Epoch {epoch + 1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Append metrics to lists
    map_50_list.append(map_result["map_50"].item())
    map_95_list.append(map_result["map"].item())
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Append metrics to list
    metrics_data.append({
        "Epoch": epoch + 1,
        "Train Loss": train_loss_hist.value,
        "mAP@0.5": map_result["map_50"].item(),
        "mAP@0.5:0.95": map_result["map"].item(),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    })

    # Save plots
    train_losses = [entry["Train Loss"] for entry in metrics_data]
    save_loss_plot(OUT_DIR, train_losses, save_name="train_loss")
    save_mAP(OUT_DIR, map_50_list, map_95_list, save_name="map_plot")

    # Save the best model only
    save_best_model(model, float(map_result['map']), epoch, 'outputs')
    #save_model(epoch, model, optimizer)



# Save all metrics to a single Excel file
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel(f"{OUT_DIR}/all_metrics.xlsx", index=False)

print("Training complete!")
