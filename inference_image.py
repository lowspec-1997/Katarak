import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import matplotlib.pyplot as plt

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

np.random.seed(42)

# Path to the directory containing test images
DIR_TEST = r"D:\LD\Katarak\katarak1\Dataset_v2\20231204_Fine_Tuning_FCOS_using_PyTorch\DATA\images"

os.makedirs('inference_outputs/images', exist_ok=True)

# Generate random colors for classes.
COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype="uint8").tolist()

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES, min_size=640, max_size=640)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Validate the directory and find images
if not os.path.exists(DIR_TEST):
    raise ValueError(f"Input directory '{DIR_TEST}' does not exist.")

# Support multiple image formats
test_images = glob.glob(f"{DIR_TEST}/*.*")
test_images = [img for img in test_images if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

print(f"Test instances: {len(test_images)}")
if len(test_images) == 0:
    print("No images found in the directory. Please check the input path and supported formats.")
    exit()

frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

for i, image_path in enumerate(test_images):
    # Get the image file name for saving output later on.
    image_name = os.path.basename(image_path).split('.')[0]
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Resize if necessary (optional hardcoding for example purposes)
    imgsz = 640  # Example of resizing, can be None to skip resizing
    if imgsz is not None:
        image = cv2.resize(image, (imgsz, imgsz))
    print(image.shape)

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).to(DEVICE)
    image_input = torch.unsqueeze(image_input, 0)

    start_time = time.time()
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current fps.
    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to detection_threshold
        threshold = 0.25  # Example detection threshold
        boxes = boxes[scores >= threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # Draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            # Rescale boxes to match original image dimensions
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color[::-1], 3)
            cv2.putText(orig_image, 
                        class_name, 
                        (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        color[::-1], 
                        2, 
                        lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    print(f"Image {i+1}/{len(test_images)} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
