# Object Detection Training Pipeline

This repository contains a PyTorch-based object detection training pipeline using `torchvision.models.detection`. The project supports various object detection models, including Faster R-CNN, SSD, and FCOS.

## Features
- **Pretrained Model Support**: Uses `torchvision` models with fine-tuning.
- **Flexible Data Handling**: Supports Pascal VOC-style annotations.
- **Efficient Training**: Includes AdamW optimizer with StepLR scheduler.
- **Evaluation Metrics**: Computes mAP (Mean Average Precision), Precision, Recall, and F1-Score.
- **Best Model Saving**: Stores only the best-performing model.
- **Training Visualization**: Loss and mAP plots generated after training.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/object-detection-training.git
   cd object-detection-training
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure PyTorch and torchvision are installed:
   ```sh
   pip install torch torchvision
   ```

## Project Structure
```
object-detection-training/
│── config.py           # Configuration settings
│── model.py            # Model creation
│── train.py            # Training loop
│── datasets.py         # Dataset handling
│── custom_utils.py     # Helper functions (e.g., saving model, plotting)
│── outputs/            # Saved models and results
│── README.md           # Project documentation
```

## Dataset

You can download the dataset from the following link:

[[Your Dataset Link Here](https://drive.google.com/drive/folders/1SDd7RRkTR6-qAKhGqUUudZp0FwaHxaBd?usp=sharing)]

## Training the Model

1. Prepare the dataset:

   a. Organize images and annotation files.
   
   b. Update `config.py` with dataset paths.

   c. Run `model.py`

   d. after that `custom_utils.py`

   e. and then `datasets`

3. Start training:
   ```sh
   python train.py
   ```

4. Monitor loss and performance:
   - Loss and mAP plots will be saved in `outputs/`.
   - The best-performing model is automatically stored.

## Evaluating the Model

After training, use `train.py` to evaluate the model:
```sh
python train.py --eval
```
This computes mAP, Precision, Recall, and F1-Score on the validation dataset.

## Customization

### Changing the Model
Modify `create_model()` in `model.py` to switch between models:
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
```

### Changing Optimizer & Scheduler
Modify `train.py` to use a different optimizer:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
```
And switch the scheduler:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

## Current Result 
- Images
 [[Your Current Images Link Here](https://drive.google.com/drive/folders/115o8JUMa-L8Uw3hDuYzDc9IRUg3_U_KY?usp=sharing)] 
- Video
  [[Your Current Videos Link Here](https://drive.google.com/drive/folders/1YyFjdbCpXENRFIiicaj6gtG1BxzjaWg9?usp=sharing)]
- Model
  [[Your Current Model Link Here](https://drive.google.com/drive/folders/1NAp8JGcUZUDB2LvVDNZ1jqWVthen3Xdz?usp=sharing)]
  

## License
This project is licensed under the MIT License.

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Albumentations](https://albumentations.ai/) for image augmentation.
