import os
import cv2
import torch
import glob
import numpy as np
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset, DataLoader

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

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # Ensure float32
        image /= 255.0  # Normalize image

        # Parse annotations
        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes, labels = [], []

        for obj in root.findall("object"):
            label = self.classes.index(obj.find("name").text)
            labels.append(label)
            bbox = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(int, [bbox.find(tag).text for tag in ["xmin", "ymin", "xmax", "ymax"]])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32).clone().detach()  # Use clone().detach()
        labels = torch.tensor(labels, dtype=torch.int64).clone().detach()  # Use clone().detach()

        # Transform
        if self.transforms:
            sample = self.transforms(image=image, bboxes=boxes.numpy(), labels=labels.numpy())
            image = sample["image"]
            boxes = torch.tensor(sample["bboxes"], dtype=torch.float32).clone().detach()

        return image, {"boxes": boxes, "labels": labels}


    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    """
    Collate function to handle variable-sized targets.
    """
    return tuple(zip(*batch))
