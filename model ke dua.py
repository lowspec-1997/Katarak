import torchvision
import torch
from functools import partial
from torchvision.models.detection.fcos import FCOSClassificationHead

def create_model(num_classes=4, min_size=640, max_size=640):
    model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(
        in_channels=256, num_anchors=num_anchors, num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    for param in model.parameters():
        param.requires_grad = True

    return model

import torchvision

import torch

def create_model(num_classes=4, min_size=640, max_size=640):
    # Load Faster R-CNN model dengan backbone ResNet-50 FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Mengganti head klasifikasi agar sesuai dengan jumlah kelas
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Menetapkan ukuran input gambar (min_size dan max_size)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    # Memastikan semua parameter bisa dioptimasi
    for param in model.parameters():
        param.requires_grad = True

    return model
