import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

# Averager class for tracking metrics
class Averager:
    
    def __init__(self):
        self.reset()

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        return self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

# SaveBestModel class with enhanced file naming
class SaveBestModel:
    def __init__(self, best_valid_map=0):
        self.best_valid_map = best_valid_map

    def __call__(self, model, current_valid_map, epoch, OUT_DIR):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            file_name = f"{OUT_DIR}/best_model_epoch_{epoch+1}_mAP_{current_valid_map:.2f}.pth"
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, file_name)
            print(f"\nSaved Best Model: {file_name}")

# Transform functions
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Display transformed images
def show_transformed_image(loader, classes, device):
    images, targets = next(iter(loader))
    image = images[0].to(device).permute(1, 2, 0).cpu().numpy()
    boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
    labels = targets[0]['labels'].cpu().numpy()

    for box, label in zip(boxes, labels):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(image, classes[label], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()
import torch

def save_model(epoch, model, optimizer, path="outputs/last_model.pth"):
   
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved at {path}")

def save_loss_plot(
    out_dir, 
    train_loss_list, 
    x_label='Iterations', 
    y_label='Training Loss', 
    save_name='train_loss'
):
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Training Loss Plot")
    plt.legend(loc='upper right')
    plot_path = f"{out_dir}/{save_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot at {plot_path}")
def save_loss_plot(
    out_dir, 
    train_loss_list, 
    x_label='Iterations', 
    y_label='Training Loss', 
    save_name='train_loss'
):
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Training Loss Plot")
    plt.legend(loc='upper right')
    plot_path = f"{out_dir}/{save_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot at {plot_path}")

def get_train_transform():
   
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
   
    return A.Compose([
        A.Resize(640, 640),  # Ensure resize is added to avoid warnings
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def save_mAP(out_dir, map_50, map, save_name="map"):
    plt.figure(figsize=(10, 7))
    plt.plot(map_50, label='mAP@0.5', color='orange')
    plt.plot(map, label='mAP@0.5:0.95', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP Over Epochs')
    plt.legend(loc='upper right')
    plot_path = f"{out_dir}/{save_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"mAP plot saved to {plot_path}")