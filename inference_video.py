import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='DATA/hihi.mp4'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
args = parser.parse_args()

# Create output directory
os.makedirs('inference_outputs/videos', exist_ok=True)

# Define colors for visualization
COLORS = [
    [0, 0, 0],      # Background
    [255, 0, 0],    # Class 1
    [0, 255, 0],    # Class 2
    [0, 0, 255],    # Class 3
    [255, 255, 0],  # Class 4
    [255, 0, 255],  # Class 5
    [0, 255, 255],  # Class 6
]

# Validate that COLORS list matches the number of classes
assert len(COLORS) >= len(CLASSES), "The COLORS list must have at least as many entries as CLASSES."

# Load the best model and trained weights
model = create_model(
    num_classes=NUM_CLASSES, min_size=640, max_size=640
)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
print("Model loaded successfully.")

# Open video file
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise FileNotFoundError(f"Error opening video file: {args.input}")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
save_name = str(pathlib.Path(args.input)).split(os.path.sep)[-1].split('.')[0]
out = cv2.VideoWriter(
    f"inference_outputs/videos/{save_name}.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (frame_width, frame_height)
)

frame_count = 0  # Total frames processed
total_fps = 0    # Total FPS for averaging

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = frame.copy()
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz, args.imgsz))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).unsqueeze(0).to(DEVICE)

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_input)
    end_time = time.time()

    # Calculate FPS
    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    # Process outputs
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) > 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        # Filter boxes by score threshold
        boxes = boxes[scores >= args.threshold].astype(np.int32)
        labels = labels[scores >= args.threshold]

        for j, box in enumerate(boxes):
            class_name = CLASSES[labels[j]]

            # Skip if class_name is invalid
            if class_name not in CLASSES:
                print(f"Warning: Detected class '{class_name}' is not in CLASSES. Skipping.")
                continue

            # Get color for the class
            color = COLORS[CLASSES.index(class_name)]

            # Draw bounding box and label
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color[::-1], 2)
            cv2.putText(
                frame,
                f"{class_name} {scores[j]:.2f}",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color[::-1],
                2,
                lineType=cv2.LINE_AA
            )

    # Add FPS to the frame
    cv2.putText(
        frame,
        f"{fps:.2f} FPS",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA
    )

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print average FPS
if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"Processed {frame_count} frames. Average FPS: {avg_fps:.2f}")
else:
    print("No frames processed.")
