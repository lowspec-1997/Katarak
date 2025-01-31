import os
import torch

BATCH_SIZE = 8 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 8 # Number of parallel workers for data loading.

# Device setup
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_IMG = 'input/Katarak2/train/Image'
TRAIN_ANNOT = 'input/Katarak2/train/Annotation'
# Validation images and XML files directory.
VALID_IMG = 'input/Katarak2/test/Image'
VALID_ANNOT = 'input/Katarak2/test/Annotation'

# Classes
CLASSES = ['__background__', 'Immature', 'Mature', 'Normal']

NUM_CLASSES = len(CLASSES)
# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
# Location to save model and plots.
OUT_DIR = 'outputs'