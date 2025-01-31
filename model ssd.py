import torchvision

def create_model(num_classes=4, min_size=640, max_size=640):
    # Load SSD model with VGG16 backbone
    model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")

    # Get the number of input features for the classification head
    in_features = model.head.classification_head.conv1.in_channels
    num_anchors = model.head.classification_head.num_anchors_per_location()

    # Replace the classification head with a custom one
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
        in_features, num_anchors, num_classes
    )

    return model
