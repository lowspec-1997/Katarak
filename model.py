import torchvision

def create_model(num_classes=4, min_size=640, max_size=640):
    # Load RetinaNet model dengan backbone ResNet-50 FPN
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    
    # Mengganti jumlah kelas sesuai kebutuhan
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_features, num_anchors, num_classes
    )

    return model
